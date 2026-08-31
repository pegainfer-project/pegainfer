//! The seam between the server binary and a model crate.
//!
//! A model line is everything south of the engine contract: config probing,
//! its own CLI knobs, and a `launch` that starts scheduler threads and hands
//! back a [`LaunchedEngine`]. The server binary holds a feature-gated
//! [`ModelLineRegistry`] and does pure dispatch — it never names a model
//! crate's option types.
//!
//! Onboarding a model line means implementing [`ModelLine`] in the model
//! crate and adding the instance to the registry in the server binary. No
//! other server edits: argument routing, consume-or-reject validation, and
//! config detection all derive from the trait.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;
use std::path::PathBuf;

use crate::engine::LaunchedEngine;
use crate::vllm::LoraModule;

/// Model detection failed. The server branches on [`DetectError::NoMatch`]
/// to append a feature-gate hint for families that exist but were compiled
/// out; everything else just renders.
#[derive(Debug, thiserror::Error)]
pub enum DetectError {
    #[error("model config claimed by both {first} and {second}")]
    Conflict {
        first: &'static str,
        second: &'static str,
    },
    #[error(
        "no compiled-in model line claims this config (model_type={model_type}, architectures={architectures}); rejections: {}",
        rejections.join("; ")
    )]
    NoMatch {
        /// `model_type` from config.json, rendered verbatim (or `missing`).
        model_type: String,
        /// `architectures` from config.json, rendered verbatim (or `missing`).
        architectures: String,
        /// One `<line>: <reason>` entry per compiled-in line.
        rejections: Vec<String>,
    },
}

/// A CLI-level rejection: the flag set is well-formed for clap but invalid
/// for the detected model line.
#[derive(Debug, thiserror::Error)]
pub enum CliError {
    /// A provided flag is neither core, nor shared-and-consumed, nor one of
    /// the detected line's own flags.
    #[error("--{flag} is not used by {line}")]
    UnconsumedFlag { flag: String, line: &'static str },
    /// A line-specific cross-flag rule was violated. The message is the
    /// user-facing explanation; rules are prose, not a taxonomy.
    #[error("{0}")]
    Rule(String),
}

impl CliError {
    pub fn rule(message: impl Into<String>) -> Self {
        Self::Rule(message.into())
    }
}

/// Flags accepted for every model line regardless of detected type.
const CORE_ARGS: &[&str] = &["model_path", "served_model_name", "port"];

/// A source argument id and the argument ids it requires.
pub type ArgRequirement = (&'static str, &'static [&'static str]);

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");

/// Shared CLI selector for model lines that can overlap prefill and decode.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, clap::ValueEnum)]
pub enum CliDecodeOverlap {
    /// One stream; prefill and decode serialize.
    #[default]
    Off,
    /// Two CUDA streams sharing all SMs.
    Stream,
    /// Green Context SM partition (SM-pinned streams).
    #[value(name = "green-ctx")]
    GreenCtx,
}

// CLI flags shared by more than one model line. Each line declares the
// subset it reads via `ModelLine::consumed_shared_args`; providing a flag
// outside that subset fails validation with a per-line error.
// (Regular comment: a doc comment here would override the command about.)
#[derive(Debug, clap::Args)]
#[allow(clippy::struct_excessive_bools)] // independent CLI flags, not a state machine
pub struct SharedArgs {
    /// Model directory containing config, tokenizer, and safetensor shards
    #[arg(long, default_value = DEFAULT_MODEL_PATH)]
    pub model_path: PathBuf,

    /// Public model ID returned by the OpenAI API (/v1/models, completion `model`).
    /// Defaults to the model path when omitted.
    #[arg(long)]
    pub served_model_name: Option<String>,

    /// Port to listen on
    #[arg(long, default_value_t = 8000)]
    pub port: u16,

    /// Enable CUDA Graph capture/replay on decode path (`--cuda-graph=false` to
    /// disable). Rejected for GLM5.2; forced off in Qwen3 LoRA mode; Qwen3.5
    /// always captures and rejects `false`; dense Gemma 4 captures per batch
    /// bucket while routed checkpoints warn and serve eagerly.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub cuda_graph: bool,

    /// Dump a live rank-0 decode CUDA Graph during startup. Qwen3 exports its
    /// batch-1 SplitKv graph; GLM5.2 exports EP bucket 1 selected by
    /// `--moe-topo`. Writes a complete sibling `.dot` for machine inspection
    /// and a folded Graphviz PNG at this path. Requires CUDA driver API 12.3
    /// or newer for kernel-name inspection.
    #[arg(long)]
    pub dump_graph_png: Option<PathBuf>,

    /// CUDA device ordinal for single-GPU Qwen3/Qwen3.5 loads
    #[arg(long, default_value_t = 0)]
    pub device_ordinal: usize,

    /// Tensor-parallel world size. GLM5.2 supports TP1/EP8 today; TP4/GB300
    /// bring-up uses --tp-size=4 --moe-topo=tp4.
    #[arg(long, default_value_t = 1)]
    pub tp_size: usize,

    /// Data-parallel world size. Kimi-K2 and GLM5.2 EP8 default to 8;
    /// GLM5.2 TP4 defaults to 1.
    #[arg(long)]
    pub dp_size: Option<usize>,

    /// Enable pegaflow KV offload (host-tier "L2" cache): single-GPU Qwen3,
    /// or GLM5.2 DP8 (one pool shared by all 8 ranks under one namespace).
    /// Sealed KV blocks are saved to host pinned memory and restored into
    /// HBM before prefill when a prompt's prefix has fallen out of the GPU
    /// cache. GLM5.2 requires the prefix cache: incompatible with
    /// --no-prefix-cache and speculative decoding.
    #[arg(long, default_value_t = false)]
    pub kv_offload: bool,

    /// Host pinned-memory pool size for the KV offload tier, in GiB. pegaflow
    /// allocates the whole pool up front, so RSS reflects this at startup.
    #[arg(long, default_value_t = 8.0, value_parser = parse_offload_gib)]
    pub kv_offload_host_gib: f64,

    /// Back the KV offload pool with 2 MiB hugepages. The box must hold a
    /// reservation covering the pool (`HugePages_Total` in /proc/meminfo;
    /// `echo N > /proc/sys/vm/nr_hugepages` as root) — allocation fails at
    /// startup otherwise.
    #[arg(long, default_value_t = false)]
    pub kv_offload_hugepages: bool,

    /// Join the cross-instance KV P2P mesh: pegaflow MetaServer gRPC address
    /// (e.g. `http://127.0.0.1:50056`). Saved block hashes register there and
    /// missing prefixes are pulled from peer instances over RDMA — the P/D
    /// disaggregation data plane. Requires --kv-offload, --kv-p2p-advertise-addr
    /// and --kv-p2p-nics.
    #[arg(long)]
    pub kv_p2p_metaserver_addr: Option<String>,

    /// This instance's routable IP:port for KV P2P — a literal socket address
    /// (it is also the embedded transfer-service bind address, so hostnames
    /// are rejected at startup). Peers dial it for RDMA handshakes and block
    /// queries. Must be reachable by every peer; not 0.0.0.0.
    #[arg(long)]
    pub kv_p2p_advertise_addr: Option<String>,

    /// RDMA NIC device names for KV P2P (e.g. `mlx5_0`), comma-separated.
    #[arg(long, value_delimiter = ',')]
    pub kv_p2p_nics: Vec<String>,

    /// vLLM-style no-prefix-cache. Without --kv-offload it disables prefix
    /// matching outright (every prefill recomputes the full prompt). With
    /// --kv-offload it is the pure-L2 mode: no cross-request HBM reuse, so every
    /// prefix is restored from the host tier — for measuring the L2 TTFT win.
    #[arg(long, default_value_t = false)]
    pub no_prefix_cache: bool,

    /// Speculative drafter model path: Qwen3 DFlash/DSpark decoding, or the
    /// GLM5.2 DSpark drafter (greedy AND sampled requests speculate;
    /// per-request accept stats logged). For Qwen3: single-GPU greedy only;
    /// incompatible with --enable-lora and --kv-offload, and forces the
    /// prefix cache off (it needs clean target hidden states).
    #[arg(long = "dflash-draft-model-path")]
    pub dflash_draft_model_path: Option<PathBuf>,

    /// Cap on total prompt tokens forwarded in one scheduler step. Qwen3 and
    /// Qwen3.5 only (rejected for other model lines); when omitted, they use
    /// their own crate defaults.
    #[arg(long)]
    pub max_prefill_tokens: Option<usize>,

    /// How prefill and decode share the GPU. Qwen3 supports `off`, `stream`,
    /// and `green-ctx`; Qwen3.5 supports `off` and `stream`.
    #[arg(long, value_enum, default_value_t = CliDecodeOverlap::Off)]
    pub decode_overlap: CliDecodeOverlap,

    /// Percent of SMs pinned to decode in `--decode-overlap green-ctx` (the rest
    /// go to prefill); rejected if set in any other mode.
    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u32).range(1..=99))]
    pub decode_sm_pct: u32,
}

const SHARED_ARG_REQUIREMENTS: &[ArgRequirement] = &[
    ("kv_offload_host_gib", &["kv_offload"]),
    ("kv_offload_hugepages", &["kv_offload"]),
    (
        "kv_p2p_metaserver_addr",
        &["kv_offload", "kv_p2p_advertise_addr", "kv_p2p_nics"],
    ),
    ("kv_p2p_advertise_addr", &["kv_p2p_metaserver_addr"]),
    ("kv_p2p_nics", &["kv_p2p_metaserver_addr"]),
];

impl SharedArgs {
    /// Interactions between shared flags that hold for every consumer.
    /// Line-specific rules live in each line's [`ModelLine::validate`].
    pub fn validate(&self, provided: &BTreeSet<String>) -> Result<(), CliError> {
        if self.dump_graph_png.is_some() && !self.cuda_graph {
            return Err(CliError::rule(
                "--dump-graph-png requires --cuda-graph=true",
            ));
        }
        if provided.contains("device_ordinal") && self.tp_size > 1 {
            return Err(CliError::rule(
                "--device-ordinal is ignored under tensor parallelism; tp_size>1 uses devices 0..tp_size",
            ));
        }
        if provided.contains("decode_sm_pct") && self.decode_overlap != CliDecodeOverlap::GreenCtx {
            return Err(CliError::rule(
                "--decode-sm-pct only applies with --decode-overlap=green-ctx",
            ));
        }
        Ok(())
    }
}

fn parse_offload_gib(value: &str) -> Result<f64, String> {
    let gib = value
        .parse::<f64>()
        .map_err(|error| format!("invalid --kv-offload-host-gib: {error}"))?;
    if gib.is_finite() && gib > 0.0 {
        Ok(gib)
    } else {
        Err("--kv-offload-host-gib must be a positive, finite number of GiB".to_owned())
    }
}

/// Everything `launch` and `serve_plan` may read.
pub struct LaunchContext<'a> {
    /// Model directory containing `config.json` and weights.
    pub model_path: &'a Path,
    /// Parsed `config.json` — the same value `probe` accepted.
    pub config: &'a serde_json::Value,
    /// Shared flags (the line reads only its consumed subset).
    pub shared: &'a SharedArgs,
    /// Full parse of the merged command; the line recovers its exclusive
    /// flags with `<Cli as clap::FromArgMatches>::from_arg_matches`.
    pub matches: &'a clap::ArgMatches,
}

/// What the frontend must know before (and independently of) the engine
/// finishing its load.
pub struct ServePlan {
    /// Scheduler partitions (logical DP ranks) the launched engine will
    /// expose. The HTTP frontend registers one engine identity per partition
    /// *while the engine is still loading*, so this must be derivable from
    /// CLI flags alone. Checked against
    /// [`EngineHandle::scheduler_partition_count`] after launch.
    pub scheduler_partition_count: usize,
    /// Serve the prefill-only route contract (GLM5.2 TP4 P/D prefill role).
    pub prefill_only: bool,
    /// `Some` enables the LoRA routes, preloading the listed adapters.
    pub lora_modules: Option<Vec<LoraModule>>,
}

impl Default for ServePlan {
    fn default() -> Self {
        Self {
            scheduler_partition_count: 1,
            prefill_only: false,
            lora_modules: None,
        }
    }
}

/// A servable model family. Implemented by each model crate.
pub trait ModelLine: Send + Sync {
    /// Family name used in logs, errors, and `--help` (e.g. `"Qwen3"`).
    fn name(&self) -> &'static str;

    /// Claim or reject a model directory by its parsed `config.json`.
    /// Return `Err` with the reason when the architecture doesn't match;
    /// exactly one registered line must accept a given config.
    fn probe(&self, config: &serde_json::Value) -> Result<(), String>;

    /// Append this line's exclusive CLI flags to the server command.
    fn augment_cli(&self, cmd: clap::Command) -> clap::Command {
        cmd
    }

    /// The [`SharedArgs`] ids this line reads. Providing any other shared
    /// flag with this line detected is an error.
    fn consumed_shared_args(&self) -> &'static [&'static str] {
        &[]
    }

    /// Unconditional requirements from this line's flags. Declare them here,
    /// not in Clap attributes, so release builds validate every referenced id.
    fn arg_requirements(&self) -> &'static [ArgRequirement] {
        &[]
    }

    /// Cross-flag rules beyond [`ModelLine::arg_requirements`]. Runs after the
    /// registry's consume-or-reject pass.
    fn validate(
        &self,
        _ctx: &LaunchContext<'_>,
        _provided: &BTreeSet<String>,
    ) -> Result<(), CliError> {
        Ok(())
    }

    /// Frontend-facing launch facts (partition count, prefill-only, LoRA).
    fn serve_plan(&self, _ctx: &LaunchContext<'_>) -> Result<ServePlan, CliError> {
        Ok(ServePlan::default())
    }

    /// Start the engine: spawn scheduler threads, build the handle with its
    /// metadata (`with_kv_capacity`, `with_metrics_watch`, ...), return it.
    /// Failures here are deep context chains (CUDA, weights, topology), not
    /// something callers branch on — hence `anyhow`.
    fn launch(&self, ctx: &LaunchContext<'_>) -> anyhow::Result<LaunchedEngine>;
}

struct LineEntry {
    line: &'static dyn ModelLine,
    own_ids: BTreeSet<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ArgOwner {
    Shared,
    Line(&'static str),
}

impl std::fmt::Display for ArgOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shared => f.write_str("SharedArgs"),
            Self::Line(name) => write!(f, "model line {name}"),
        }
    }
}

#[derive(Default)]
struct RegistryClaims {
    ids: BTreeMap<String, ArgOwner>,
    spellings: BTreeMap<String, ArgOwner>,
}

impl RegistryClaims {
    fn register_command(&mut self, command: &clap::Command, owner: ArgOwner) -> BTreeSet<String> {
        command
            .get_arguments()
            .map(|arg| self.register_arg(arg, owner))
            .collect()
    }

    fn register_arg(&mut self, arg: &clap::Arg, owner: ArgOwner) -> String {
        let id = arg.get_id().to_string();
        self.claim_id(id.clone(), owner);
        for long in arg
            .get_long()
            .into_iter()
            .chain(arg.get_all_aliases().into_iter().flatten())
        {
            self.claim_spelling(format!("--{long}"), owner);
        }
        for short in arg
            .get_short()
            .into_iter()
            .chain(arg.get_all_short_aliases().into_iter().flatten())
        {
            self.claim_spelling(format!("-{short}"), owner);
        }
        id
    }

    fn claim_id(&mut self, id: String, owner: ArgOwner) {
        if let Some(previous) = self.ids.get(&id) {
            if *previous == ArgOwner::Shared {
                if let ArgOwner::Line(name) = owner {
                    panic!(
                        "model line {name} redeclares SharedArgs flag id {id:?}; list it in consumed_shared_args instead"
                    );
                }
            }
            panic!("{previous} and {owner} both define flag id {id:?}");
        }
        self.ids.insert(id, owner);
    }

    fn claim_spelling(&mut self, spelling: String, owner: ArgOwner) {
        if let Some(previous) = self.spellings.get(&spelling) {
            panic!("{previous} and {owner} both define CLI spelling {spelling:?}");
        }
        self.spellings.insert(spelling, owner);
    }

    fn is_shared_id(&self, id: &str) -> bool {
        self.ids.get(id) == Some(&ArgOwner::Shared)
    }

    fn validate_requirement(&self, owner: ArgOwner, source: &str, targets: &[&str]) {
        assert!(
            self.ids.get(source) == Some(&owner),
            "{owner} declares requirements for unowned flag id {source:?}"
        );
        for &target in targets {
            assert!(source != target, "flag id {source:?} cannot require itself");
            let target_owner = self.ids.get(target);
            assert!(
                target_owner == Some(&ArgOwner::Shared) || target_owner == Some(&owner),
                "{owner} flag id {source:?} requires unknown or foreign argument id {target:?}"
            );
        }
    }
}

fn register_line(
    line: &'static dyn ModelLine,
    owner: ArgOwner,
    claims: &mut RegistryClaims,
) -> LineEntry {
    for id in line.consumed_shared_args() {
        assert!(
            claims.is_shared_id(id),
            "model line {} consumes unknown SharedArgs flag id {id:?}",
            line.name()
        );
    }
    let command = line.augment_cli(clap::Command::new("probe"));
    let own_ids = claims.register_command(&command, owner);
    LineEntry { line, own_ids }
}

/// The server binary's fixed list of compiled-in model lines.
pub struct ModelLineRegistry {
    entries: Vec<LineEntry>,
    requirements: BTreeMap<&'static str, &'static [&'static str]>,
}

impl ModelLineRegistry {
    /// Validates CLI ownership explicitly because Clap's equivalent schema
    /// assertions are disabled in release builds.
    pub fn new(lines: Vec<&'static dyn ModelLine>) -> Self {
        let shared_cmd = SharedArgs::augment_args(clap::Command::new("probe"));
        let mut claims = RegistryClaims::default();
        claims.register_command(&shared_cmd, ArgOwner::Shared);
        for id in CORE_ARGS {
            assert!(
                claims.is_shared_id(id),
                "CORE_ARGS contains unknown SharedArgs flag id {id:?}"
            );
        }
        let mut pending_requirements = SHARED_ARG_REQUIREMENTS
            .iter()
            .map(|&(source, targets)| (ArgOwner::Shared, source, targets))
            .collect::<Vec<_>>();
        let mut entries = Vec::with_capacity(lines.len());
        let mut line_names = BTreeSet::new();
        for line in lines {
            let name = line.name();
            assert!(
                line_names.insert(name),
                "duplicate model line name {name:?}"
            );
            let owner = ArgOwner::Line(name);
            entries.push(register_line(line, owner, &mut claims));
            pending_requirements.extend(
                line.arg_requirements()
                    .iter()
                    .map(|&(source, targets)| (owner, source, targets)),
            );
        }
        let mut requirements = BTreeMap::new();
        for (owner, source, targets) in pending_requirements {
            claims.validate_requirement(owner, source, targets);
            assert!(
                requirements.insert(source, targets).is_none(),
                "{owner} declares requirements more than once for flag id {source:?}"
            );
        }
        Self {
            entries,
            requirements,
        }
    }

    /// The full server command: shared flags plus every line's exclusive
    /// flags, so `--help` and parsing don't depend on the detected model.
    pub fn build_command(&self, cmd: clap::Command) -> clap::Command {
        let mut cmd = SharedArgs::augment_args(cmd);
        for entry in &self.entries {
            cmd = entry.line.augment_cli(cmd);
        }
        // `mut_arg` re-appends each source and would reorder `--help`.
        cmd.mut_args(|arg| {
            let Some(targets) = self.requirements.get(arg.get_id().as_str()) else {
                return arg;
            };
            arg.requires_all(targets.iter().copied())
        })
    }

    /// Find the unique line that claims this `config.json`. On
    /// [`DetectError::NoMatch`] the caller may prepend a hint for families
    /// that exist but were compiled out.
    pub fn detect(
        &self,
        config: &serde_json::Value,
    ) -> Result<&'static dyn ModelLine, DetectError> {
        let mut rejections = Vec::new();
        let mut claimed: Option<&'static dyn ModelLine> = None;
        for entry in &self.entries {
            match entry.line.probe(config) {
                Ok(()) => match claimed {
                    None => claimed = Some(entry.line),
                    Some(prev) => {
                        return Err(DetectError::Conflict {
                            first: prev.name(),
                            second: entry.line.name(),
                        });
                    }
                },
                Err(reason) => rejections.push(format!("{}: {reason}", entry.line.name())),
            }
        }
        claimed.ok_or_else(|| {
            let render = |key: &str| {
                config
                    .get(key)
                    .map_or_else(|| "missing".to_string(), std::string::ToString::to_string)
            };
            DetectError::NoMatch {
                model_type: render("model_type"),
                architectures: render("architectures"),
                rejections,
            }
        })
    }

    /// Consume-or-reject: every explicitly provided flag must be a core flag,
    /// a shared flag the detected line consumes, or one of the line's own.
    pub fn validate_provided(
        &self,
        detected: &'static dyn ModelLine,
        provided: &BTreeSet<String>,
        cmd: &clap::Command,
    ) -> Result<(), CliError> {
        let entry = self
            .entries
            .iter()
            .find(|entry| std::ptr::eq(entry.line, detected))
            .expect("detected line came from this registry");
        for id in provided {
            let id = id.as_str();
            if CORE_ARGS.contains(&id)
                || detected.consumed_shared_args().contains(&id)
                || entry.own_ids.contains(id)
            {
                continue;
            }
            return Err(CliError::UnconsumedFlag {
                flag: long_flag(cmd, id),
                line: detected.name(),
            });
        }
        Ok(())
    }
}

fn long_flag(cmd: &clap::Command, id: &str) -> String {
    cmd.get_arguments()
        .find(|arg| arg.get_id() == id)
        .and_then(clap::Arg::get_long)
        .map_or_else(|| id.to_owned(), str::to_owned)
}

/// Arg ids the user set explicitly (command line or env), for consume-or-reject.
pub fn provided_args(matches: &clap::ArgMatches, cmd: &clap::Command) -> BTreeSet<String> {
    // matches.ids() also yields clap's synthetic group ids; keep only real args.
    let real: BTreeSet<&str> = cmd
        .get_arguments()
        .map(|arg| arg.get_id().as_str())
        .collect();
    matches
        .ids()
        .map(clap::Id::as_str)
        .filter(|id| real.contains(id))
        .filter(|id| {
            matches!(
                matches.value_source(id),
                Some(
                    clap::parser::ValueSource::CommandLine | clap::parser::ValueSource::EnvVariable
                )
            )
        })
        .map(str::to_owned)
        .collect()
}

use clap::Args as _;

/// Test helper: parse `argv` against shared flags plus one line's flags, the
/// way the server binary does for a detected line. Exposed for model-crate
/// unit tests; not a serving entry point.
pub fn parse_for_line(
    line: &'static dyn ModelLine,
    argv: &[&str],
) -> anyhow::Result<(SharedArgs, clap::ArgMatches, BTreeSet<String>)> {
    use clap::FromArgMatches;
    let registry = ModelLineRegistry::new(vec![line]);
    let cmd = registry.build_command(clap::Command::new("pegainfer"));
    let matches = cmd.clone().try_get_matches_from(argv)?;
    let shared = SharedArgs::from_arg_matches(&matches)?;
    let provided = provided_args(&matches, &cmd);
    registry.validate_provided(line, &provided, &cmd)?;
    shared.validate(&provided)?;
    Ok((shared, matches, provided))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubLine {
        name: &'static str,
        claims: &'static str,
        augment: Option<fn(clap::Command) -> clap::Command>,
        consumed: &'static [&'static str],
        requirements: &'static [ArgRequirement],
    }

    impl StubLine {
        const fn new(
            name: &'static str,
            claims: &'static str,
            augment: Option<fn(clap::Command) -> clap::Command>,
        ) -> Self {
            Self {
                name,
                claims,
                augment,
                consumed: &["tp_size"],
                requirements: &[],
            }
        }
    }

    impl ModelLine for StubLine {
        fn name(&self) -> &'static str {
            self.name
        }

        fn probe(&self, config: &serde_json::Value) -> Result<(), String> {
            let model_type = config.get("model_type").and_then(serde_json::Value::as_str);
            if model_type == Some(self.claims) {
                Ok(())
            } else {
                Err(format!(
                    "model_type {model_type:?} is not {:?}",
                    self.claims
                ))
            }
        }

        fn augment_cli(&self, cmd: clap::Command) -> clap::Command {
            match self.augment {
                Some(augment) => augment(cmd),
                None => cmd,
            }
        }

        fn consumed_shared_args(&self) -> &'static [&'static str] {
            self.consumed
        }

        fn arg_requirements(&self) -> &'static [ArgRequirement] {
            self.requirements
        }

        fn launch(&self, _ctx: &LaunchContext<'_>) -> anyhow::Result<LaunchedEngine> {
            unreachable!("registry tests never launch")
        }
    }

    static LINE_A: StubLine = StubLine::new(
        "LineA",
        "model_a",
        Some(|cmd| {
            cmd.arg(
                clap::Arg::new("line_a_flag")
                    .long("line-a-flag")
                    .num_args(0),
            )
        }),
    );
    static LINE_B: StubLine = StubLine::new(
        "LineB",
        "model_b",
        Some(|cmd| {
            cmd.arg(
                clap::Arg::new("line_b_flag")
                    .long("line-b-flag")
                    .num_args(0),
            )
        }),
    );
    static LINE_B_TWIN: StubLine = StubLine::new("LineBTwin", "model_b", None);
    static LINE_SHARED_COLLISION: StubLine = StubLine::new(
        "LineSharedCollision",
        "model_c",
        Some(|cmd| cmd.arg(clap::Arg::new("tp_size").long("tp-size-again").num_args(0))),
    );
    static LINE_A_COPYCAT: StubLine = StubLine::new(
        "LineACopycat",
        "model_d",
        Some(|cmd| {
            cmd.arg(
                clap::Arg::new("line_a_flag")
                    .long("line-a-flag-copy")
                    .num_args(0),
            )
        }),
    );
    static LINE_LONG_COPYCAT: StubLine = StubLine::new(
        "LineLongCopycat",
        "model_e",
        Some(|cmd| {
            cmd.arg(
                clap::Arg::new("different_id")
                    .long("line-a-flag")
                    .num_args(0),
            )
        }),
    );
    static LINE_UNKNOWN_SHARED: StubLine = StubLine {
        consumed: &["not_a_shared_arg"],
        ..StubLine::new("LineUnknownShared", "model_f", None)
    };
    static LINE_SHARED_ALIAS: StubLine = StubLine::new(
        "LineSharedAlias",
        "model_g",
        Some(|cmd| {
            cmd.arg(
                clap::Arg::new("shared_alias")
                    .long("shared-alias")
                    .alias("tp-size")
                    .num_args(0),
            )
        }),
    );
    static LINE_SHORT_A: StubLine = StubLine::new(
        "LineShortA",
        "model_h",
        Some(|cmd| cmd.arg(clap::Arg::new("short_a").short('q').num_args(0))),
    );
    static LINE_SHORT_ALIAS: StubLine = StubLine::new(
        "LineShortAlias",
        "model_i",
        Some(|cmd| {
            cmd.arg(
                clap::Arg::new("short_alias")
                    .short('r')
                    .short_alias('q')
                    .num_args(0),
            )
        }),
    );
    static LINE_DANGLING_REQUIREMENT: StubLine = StubLine {
        requirements: &[("trigger", &["missing_target"])],
        ..StubLine::new(
            "LineDanglingRequirement",
            "model_j",
            Some(|cmd| cmd.arg(clap::Arg::new("trigger").long("trigger").num_args(0))),
        )
    };

    fn registry() -> ModelLineRegistry {
        ModelLineRegistry::new(vec![&LINE_A, &LINE_B])
    }

    #[test]
    fn detect_finds_the_unique_claimant() {
        let config = serde_json::json!({"model_type": "model_b"});
        let line = registry().detect(&config).expect("model_b should detect");
        assert_eq!(line.name(), "LineB");
    }

    #[test]
    fn detect_conflict_names_both_lines() {
        let registry = ModelLineRegistry::new(vec![&LINE_B, &LINE_B_TWIN]);
        let config = serde_json::json!({"model_type": "model_b"});
        let error = registry
            .detect(&config)
            .map(ModelLine::name)
            .expect_err("two claimants");
        assert!(matches!(error, DetectError::Conflict { .. }));
        let message = error.to_string();
        assert!(
            message.contains("LineB") && message.contains("LineBTwin"),
            "{message}"
        );
    }

    #[test]
    fn detect_no_match_renders_identity_fields_verbatim() {
        let config = serde_json::json!({"model_type": 123, "architectures": "Foo"});
        let error = registry()
            .detect(&config)
            .map(ModelLine::name)
            .expect_err("nothing claims this");
        let message = error.to_string();
        assert!(message.contains("model_type=123"), "{message}");
        assert!(message.contains("architectures=\"Foo\""), "{message}");
        assert!(
            message.contains("LineA") && message.contains("LineB"),
            "{message}"
        );
    }

    #[test]
    fn detect_no_match_reports_missing_fields() {
        let error = registry()
            .detect(&serde_json::json!({}))
            .map(ModelLine::name)
            .expect_err("empty config");
        assert!(error.to_string().contains("model_type=missing"));
    }

    #[test]
    fn validate_provided_allows_core_shared_and_own_flags() {
        let registry = registry();
        let cmd = registry.build_command(clap::Command::new("test"));
        let provided: BTreeSet<String> = ["port", "tp_size", "line_a_flag"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        registry
            .validate_provided(&LINE_A, &provided, &cmd)
            .expect("core + consumed shared + own flags all pass");
    }

    #[test]
    fn validate_provided_rejects_another_lines_flag_by_long_name() {
        let registry = registry();
        let cmd = registry.build_command(clap::Command::new("test"));
        let provided: BTreeSet<String> = [String::from("line_b_flag")].into_iter().collect();
        let error = registry
            .validate_provided(&LINE_A, &provided, &cmd)
            .expect_err("LineB's flag must be rejected for LineA");
        assert!(matches!(error, CliError::UnconsumedFlag { .. }));
        assert_eq!(error.to_string(), "--line-b-flag is not used by LineA");
    }

    #[test]
    fn validate_provided_rejects_unconsumed_shared_flag() {
        let registry = registry();
        let cmd = registry.build_command(clap::Command::new("test"));
        let provided: BTreeSet<String> = [String::from("kv_offload")].into_iter().collect();
        let error = registry
            .validate_provided(&LINE_A, &provided, &cmd)
            .expect_err("a shared flag outside consumed_shared_args must be rejected");
        assert_eq!(error.to_string(), "--kv-offload is not used by LineA");
    }

    #[test]
    #[should_panic(expected = "both define flag id")]
    fn registry_rejects_duplicate_exclusive_flag_ids() {
        let _ = ModelLineRegistry::new(vec![&LINE_A, &LINE_A_COPYCAT]);
    }

    #[test]
    #[should_panic(expected = "both define CLI spelling \"--line-a-flag\"")]
    fn registry_rejects_duplicate_exclusive_flag_spellings() {
        let _ = ModelLineRegistry::new(vec![&LINE_A, &LINE_LONG_COPYCAT]);
    }

    #[test]
    #[should_panic(expected = "list it in consumed_shared_args instead")]
    fn registry_rejects_shared_flag_id_collisions() {
        let _ = ModelLineRegistry::new(vec![&LINE_SHARED_COLLISION]);
    }

    #[test]
    #[should_panic(
        expected = "SharedArgs and model line LineSharedAlias both define CLI spelling \"--tp-size\""
    )]
    fn registry_rejects_shared_flag_alias_collisions() {
        let _ = ModelLineRegistry::new(vec![&LINE_SHARED_ALIAS]);
    }

    #[test]
    #[should_panic(
        expected = "model line LineShortA and model line LineShortAlias both define CLI spelling \"-q\""
    )]
    fn registry_rejects_short_alias_collisions() {
        let _ = ModelLineRegistry::new(vec![&LINE_SHORT_A, &LINE_SHORT_ALIAS]);
    }

    #[test]
    #[should_panic(expected = "consumes unknown SharedArgs flag id \"not_a_shared_arg\"")]
    fn registry_rejects_unknown_consumed_shared_arg() {
        let _ = ModelLineRegistry::new(vec![&LINE_UNKNOWN_SHARED]);
    }

    #[test]
    #[should_panic(expected = "requires unknown or foreign argument id")]
    fn registry_rejects_unknown_requirement_target() {
        let _ = ModelLineRegistry::new(vec![&LINE_DANGLING_REQUIREMENT]);
    }

    #[test]
    fn provided_args_reports_only_explicitly_set_flags() {
        let registry = registry();
        let cmd = registry.build_command(clap::Command::new("test"));
        let matches = cmd
            .clone()
            .try_get_matches_from(["test", "--tp-size", "2", "--line-a-flag"])
            .expect("parse");
        let provided = provided_args(&matches, &cmd);
        assert!(provided.contains("tp_size"));
        assert!(provided.contains("line_a_flag"));
        assert!(!provided.contains("port"));
    }
}
