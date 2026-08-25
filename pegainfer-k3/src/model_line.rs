//! K3's [`ModelLine`] implementation: the config probe, the K3-exclusive CLI
//! section, the frontend serve plan (one scheduler partition per EP rank),
//! and launch.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context as _;
use clap::Args as ClapArgs;
use clap::FromArgMatches;
use log::info;
use log::warn;
use pegainfer_frontend::engine::LaunchedEngine;
use pegainfer_frontend::model_line::CliError;
use pegainfer_frontend::model_line::LaunchContext;
use pegainfer_frontend::model_line::ModelLine;
use pegainfer_frontend::model_line::ServePlan;
use serde_json::Value;

use crate::config::K3_HIDDEN;
use crate::config::K3_LAYERS;
use crate::config::K3_SUPPORTED_ROUTED_EXPERTS;
use crate::executor::K3Executor;
use crate::executor::K3ExecutorConfig;
use crate::executor::cp::K3_WHALE_SEGMENT_FLOOR;
use crate::executor::cp::K3CpGroup;
use crate::executor::ep::K3EpRendezvous;
use crate::scheduler::K3CpGang;
use crate::scheduler::K3CpServing;
use crate::scheduler::K3SchedulerConfig;
use crate::scheduler::K3WhaleHub;
use crate::scheduler::K3WhaleServing;
use crate::scheduler::whale_hub::TcpWhaleHub;
use crate::weights::K3_WEIGHT_FILL_THREADS;

pub static MODEL_LINE: K3Line = K3Line;

pub struct K3Line;

/// Identity strings from the checkpoint's `config.json`. The wrapper config is
/// multimodal, so the language-model fields sit one level down under
/// `text_config` and carry their own (`kimi_linear`) identity.
const MODEL_TYPE: &str = "kimi_k3";
const ARCHITECTURE: &str = "KimiK3ForConditionalGeneration";
const TEXT_MODEL_TYPE: &str = "kimi_linear";

/// EP world sizes the decode topology is defined for. Experts shard whole
/// across ranks, and the fused MegaMoE kernel that carries the routed experts
/// is AOT-instantiated per (expert count, width) world — so this list is the
/// union of widths any checkpoint supports, not a set of arithmetically valid
/// shardings; the exact world is validated at launch against the kernel's own
/// matrix (`k3_mega_world_supported`). Kept independent of the checkpoint on
/// purpose: the frontend needs the partition count from the CLI alone, before
/// any config or weights are read.
const EP_SIZES: &[usize] = &[1, 4, 8, 16, 32, 64];

// K3-exclusive CLI flags.
#[derive(ClapArgs)]
struct K3Cli {
    /// K3 expert-parallel world size: one rank (and one scheduler partition)
    /// per GPU, routed experts split whole across ranks. One of 1, 4, 8, 16,
    /// 32 or 64; widths above 4 span machines and shard per checkpoint (the
    /// 224-expert dev checkpoint stops at 16, the full 896-expert model needs
    /// at least 8).
    #[arg(long, default_value_t = 1)]
    k3_ep_size: usize,

    /// K3 global EP ranks this process hosts, `start..end` (e.g. `4..8`).
    /// Default: the whole world (single-process, one machine). A partial
    /// range is the multi-process cross-machine shape: every machine runs the
    /// same binary over its own ranks, and requires `--k3-rendezvous`.
    #[arg(long)]
    k3_ranks: Option<String>,

    /// K3 bootstrap rendezvous address (`host:port`): the process hosting
    /// rank 0 binds it, collects every rank's NVLink-fabric slab handle, and
    /// serves the world's table back. A one-time handshake — after it the
    /// engines never talk again (the MegaMoE kernel pairs the ranks itself,
    /// over the rack-wide NVLink domain).
    #[arg(long)]
    k3_rendezvous: Option<String>,

    /// Load hosted K3 ranks concurrently and stage checkpoint bytes through
    /// pinned double buffers. This can substantially accelerate
    /// warm-page-cache loads; leave off for cold network-filesystem starts.
    #[arg(long)]
    k3_weight_staging: bool,
}

/// Parse a `start..end` rank range.
fn parse_rank_range(spec: &str) -> Result<std::ops::Range<usize>, String> {
    let (start, end) = spec
        .split_once("..")
        .ok_or_else(|| format!("expected start..end, got {spec:?}"))?;
    let parse = |raw: &str| {
        raw.trim()
            .parse::<usize>()
            .map_err(|_| format!("expected start..end with integer bounds, got {spec:?}"))
    };
    let range = parse(start)?..parse(end)?;
    if range.is_empty() {
        return Err(format!("rank range {spec:?} is empty"));
    }
    Ok(range)
}

fn cli(ctx: &LaunchContext<'_>) -> K3Cli {
    K3Cli::from_arg_matches(ctx.matches).expect("K3Cli parses from the merged command")
}

/// Validate `--k3-ep-size` where it is read, so an unsupported width fails
/// with the real problem instead of surfacing later as a topology mismatch.
fn ep_size(cli: &K3Cli) -> Result<usize, CliError> {
    if EP_SIZES.contains(&cli.k3_ep_size) {
        Ok(cli.k3_ep_size)
    } else {
        Err(CliError::rule(format!(
            "--k3-ep-size must be one of {EP_SIZES:?}, got {}",
            cli.k3_ep_size
        )))
    }
}

/// Read a language-model field: `text_config` first (where this family keeps
/// them), then the top level, so a text-only export probes the same way.
fn text_field<'a>(config: &'a Value, key: &str) -> Option<&'a Value> {
    config
        .get("text_config")
        .and_then(|text| text.get(key))
        .or_else(|| config.get(key))
}

fn expect_usize(config: &Value, key: &str, expected: usize) -> Result<(), String> {
    match text_field(config, key).and_then(Value::as_u64) {
        Some(value) if value == expected as u64 => Ok(()),
        other => Err(format!("{key} {other:?} is not {expected}")),
    }
}

fn claims_architecture(config: &Value) -> bool {
    config
        .get("architectures")
        .and_then(Value::as_array)
        .is_some_and(|list| {
            list.iter()
                .any(|entry| entry.as_str() == Some(ARCHITECTURE))
        })
}

/// End-of-sequence ids the scheduler stops on. The field is a scalar in this
/// family's config; arrays appear in other exports of the same architecture,
/// so accept both.
fn eos_token_ids(config: &Value) -> Vec<u32> {
    let field = config
        .get("eos_token_id")
        .or_else(|| text_field(config, "eos_token_id"));
    match field {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_u64)
            .map(|id| id as u32)
            .collect(),
        Some(value) => value.as_u64().map(|id| id as u32).into_iter().collect(),
        None => Vec::new(),
    }
}

impl ModelLine for K3Line {
    fn name(&self) -> &'static str {
        "K3"
    }

    fn probe(&self, config: &Value) -> Result<(), String> {
        let model_type = config.get("model_type").and_then(Value::as_str);
        if model_type != Some(MODEL_TYPE) {
            return Err(format!("model_type {model_type:?} is not {MODEL_TYPE:?}"));
        }
        if !claims_architecture(config) {
            return Err(format!(
                "architectures {:?} does not contain {ARCHITECTURE:?}",
                config.get("architectures")
            ));
        }
        let text_model_type = text_field(config, "model_type").and_then(Value::as_str);
        if text_model_type != Some(TEXT_MODEL_TYPE) {
            return Err(format!(
                "text_config.model_type {text_model_type:?} is not {TEXT_MODEL_TYPE:?}"
            ));
        }
        // Cheap shape gates against the architecture constants — enough to
        // catch a mislabelled directory. The exhaustive check every field
        // gets is the loader's `config::probe_config_json`, at launch.
        expect_usize(config, "hidden_size", K3_HIDDEN)?;
        expect_usize(config, "num_hidden_layers", K3_LAYERS)?;
        match text_field(config, "num_experts").and_then(Value::as_u64) {
            Some(experts) if K3_SUPPORTED_ROUTED_EXPERTS.contains(&(experts as usize)) => Ok(()),
            other => Err(format!(
                "num_experts {other:?} is not one of {K3_SUPPORTED_ROUTED_EXPERTS:?}"
            )),
        }
    }

    fn augment_cli(&self, cmd: clap::Command) -> clap::Command {
        K3Cli::augment_args(cmd)
    }

    fn consumed_shared_args(&self) -> &'static [&'static str] {
        // `model_path`, `served_model_name` and `port` are core flags every
        // line accepts; listed here too so this is the whole accepted set at
        // a glance. `no_prefix_cache` is accepted and inert: K3's KDA
        // recurrent state is not reconstructible from tokens, so prefix
        // caching is already off.
        &[
            "model_path",
            "served_model_name",
            "port",
            "device_ordinal",
            "no_prefix_cache",
            "dflash_draft_model_path",
        ]
    }

    fn validate(
        &self,
        ctx: &LaunchContext<'_>,
        provided: &BTreeSet<String>,
    ) -> Result<(), CliError> {
        let cli = cli(ctx);
        let ep_size = ep_size(&cli)?;
        if ep_size > 1 && provided.contains("device_ordinal") {
            return Err(CliError::rule(
                "--device-ordinal applies to single-rank K3 only; --k3-ep-size>1 uses devices 0..local_ranks",
            ));
        }
        let ranks = local_ranks(&cli, ep_size)?;
        if ranks.len() < ep_size && cli.k3_rendezvous.is_none() {
            return Err(CliError::rule(
                "--k3-ranks hosts a slice of the EP world, which needs --k3-rendezvous for the \
                 fleet's fabric-handle exchange",
            ));
        }
        if ranks.len() == ep_size && cli.k3_rendezvous.is_some() {
            return Err(CliError::rule(
                "--k3-rendezvous is for a multi-process fleet; this process hosts the whole EP \
                 world already",
            ));
        }
        Ok(())
    }

    fn serve_plan(&self, ctx: &LaunchContext<'_>) -> Result<ServePlan, CliError> {
        let cli = cli(ctx);
        let ep_size = ep_size(&cli)?;
        Ok(ServePlan {
            // One partition per EP rank THIS PROCESS hosts, and CLI-derivable:
            // the frontend registers engine identities before the weights are
            // loaded.
            scheduler_partition_count: local_ranks(&cli, ep_size)?.len(),
            prefill_only: false,
            lora_modules: None,
        })
    }

    fn launch(&self, ctx: &LaunchContext<'_>) -> anyhow::Result<LaunchedEngine> {
        let cli = cli(ctx);
        let ep_size = ep_size(&cli)?;
        let ranks = local_ranks(&cli, ep_size)?;
        let eos_token_ids = eos_token_ids(ctx.config);
        let mut config = K3ExecutorConfig::default().from_env().for_ep(ep_size);
        config.weight_staging = cli.k3_weight_staging;
        warn_if_loader_cpu_starved(ranks.len(), config.weight_staging);
        info!(
            "K3 engine starting: ep_size={ep_size}, ranks={ranks:?}, \
             eos_token_ids={eos_token_ids:?}, slots={}, ctx={}, layers={}, cuda_graph={}, \
             moe=mega, weight_staging={}",
            config.max_batch,
            config.max_ctx,
            config.num_layers,
            config.cuda_graph,
            config.weight_staging,
        );
        // One executor per EP rank this process hosts, on devices
        // 0..ranks.len(). A single-rank run honours --device-ordinal, which
        // `validate` already restricted.
        //
        // Every local rank's weights are resident before any rank is stepped:
        // a rank publishes its symmetric slab at load, but reads the world's
        // table back on its own scheduler thread at its first step — which is
        // after `start_with_executors` below, and so after every load. That
        // ordering is the point: the table read blocks, so one rank running
        // out of memory mid-load must not be able to strand its peers waiting.
        // Across a fleet the same story holds per process, and the bootstrap's
        // long timeouts absorb the machines' load skew.
        let rendezvous = match (&cli.k3_rendezvous, ep_size > 1) {
            (_, false) => None,
            (None, true) => Some(K3EpRendezvous::new(ep_size)),
            (Some(addr), true) => {
                Some(K3EpRendezvous::fleet(ep_size, ranks.clone(), addr.clone())?)
            }
        };
        let load_started = Instant::now();
        let mut executors = load_rank_executors(
            ctx.model_path,
            ctx.shared.dflash_draft_model_path.as_deref(),
            ctx.shared.device_ordinal,
            ranks.clone(),
            ep_size,
            config,
            rendezvous.as_ref(),
        )?;
        info!(
            "K3 local rank startup complete: ranks={}, mode={}, critical_path={:.2}s",
            executors.len(),
            if config.weight_staging {
                "parallel"
            } else {
                "serial"
            },
            load_started.elapsed().as_secs_f64(),
        );
        let dspark_armed = ctx.shared.dflash_draft_model_path.is_some();
        let chunk_tokens = executors
            .first()
            .context("K3 EP serving needs at least one local rank")?
            .chunk_tokens();
        let cp = cp_serving(ranks.len(), dspark_armed, chunk_tokens)?;
        let whale = whale_serving(
            &mut executors,
            ep_size,
            ranks.clone(),
            dspark_armed,
            cp.is_some(),
            chunk_tokens,
        )?;
        Ok(LaunchedEngine::Stepped(
            crate::scheduler::start_with_executors(
                executors,
                &K3SchedulerConfig {
                    eos_token_ids,
                    kv_capacity: None,
                    cp,
                    whale,
                },
            ),
        ))
    }
}

/// Arm the fleet whale lane from `PEGAINFER_K3_WHALE` (the rendezvous
/// address — the process hosting global rank 0 binds it and runs the
/// sequencer, everyone else connects) and `PEGAINFER_K3_WHALE_MIN` (admission
/// floor in prompt tokens, at least twice [`K3_WHALE_SEGMENT_FLOOR`] — the
/// narrowest prompt a width-2 gang can serve — which is also the default).
///
/// Arming stands the whole data plane up before any engine steps: every
/// local rank allocates its fabric slab, the hub's startup exchange moves the
/// world's handles (doubling as the fleet's startup barrier), and each
/// executor imports the table. From then on a committed whale needs no
/// further setup — the scheduler consults the rendezvous at every launch
/// boundary and enters supersteps at the committed launch.
fn whale_serving(
    executors: &mut [K3Executor],
    world: usize,
    ranks: std::ops::Range<usize>,
    dspark_armed: bool,
    cp_armed: bool,
    chunk_tokens: usize,
) -> anyhow::Result<Option<K3WhaleServing>> {
    let Some(raw) = std::env::var_os("PEGAINFER_K3_WHALE") else {
        return Ok(None);
    };
    let addr = raw.to_string_lossy().into_owned();
    anyhow::ensure!(
        !dspark_armed,
        "PEGAINFER_K3_WHALE does not compose with the dspark draft lane yet; disarm one of them"
    );
    anyhow::ensure!(
        !cp_armed,
        "PEGAINFER_K3_CP and PEGAINFER_K3_WHALE are exclusive: the in-process gang and the fleet \
         whale lane coordinate the same superstep"
    );
    anyhow::ensure!(
        world >= 2,
        "PEGAINFER_K3_WHALE needs an EP world of at least two ranks; a whale gang is never \
         narrower than two"
    );
    let narrowest = 2 * K3_WHALE_SEGMENT_FLOOR;
    let min_tokens = match std::env::var_os("PEGAINFER_K3_WHALE_MIN") {
        None => narrowest,
        Some(raw) => {
            let raw = raw.to_string_lossy();
            raw.parse()
                .ok()
                .filter(|&min| min >= narrowest)
                .with_context(|| {
                    format!(
                        "PEGAINFER_K3_WHALE_MIN={raw} must be a token count of at least \
                         {narrowest} — no gang serves a shorter prompt"
                    )
                })?
        }
    };
    let arm_started = Instant::now();
    let armed = executors
        .iter_mut()
        .map(|executor| executor.arm_whale_slab(world))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let slabs: Vec<_> = armed.iter().map(|&(_, wire)| wire).collect();
    let local: Vec<(usize, u64)> = armed
        .iter()
        .enumerate()
        .map(|(offset, &(base, _))| (ranks.start + offset, base))
        .collect();
    let slabs_armed = Instant::now();
    let (hub, table) = if ranks.start == 0 {
        TcpWhaleHub::host(&addr, world, chunk_tokens, 0, ranks.len(), slabs)
            .context("host the K3 whale hub")?
    } else {
        TcpWhaleHub::connect(&addr, ranks.start, ranks.len(), slabs)
            .context("join the K3 whale hub")?
    };
    let world_exchanged = Instant::now();
    let bases = executors[0].import_whale_world(&table, &local)?;
    for executor in executors.iter_mut() {
        executor.install_whale_gang(bases.clone())?;
    }
    info!(
        "K3 whale lane armed: world={world}, local_ranks={ranks:?}, min_tokens={min_tokens}, \
         chunk_tokens={chunk_tokens}, rendezvous={addr}, slab_alloc={:.2}s, \
         slab_exchange={:.2}s (fleet startup barrier), import_install={:.2}s",
        (slabs_armed - arm_started).as_secs_f64(),
        (world_exchanged - slabs_armed).as_secs_f64(),
        world_exchanged.elapsed().as_secs_f64(),
    );
    Ok(Some(K3WhaleServing {
        hub: K3WhaleHub::Tcp(hub),
        world,
        first_local: ranks.start,
        min_tokens,
        chunk_tokens,
    }))
}

fn warn_if_loader_cpu_starved(local_ranks: usize, weight_staging: bool) {
    let loader_threads = if weight_staging { local_ranks } else { 1 };
    let fill_threads = if weight_staging {
        K3_WEIGHT_FILL_THREADS * loader_threads
    } else {
        0
    };
    let wanted = loader_threads + fill_threads;
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    if available < wanted {
        warn!(
            "K3 weight startup is CPU-starved: affinity exposes {available} CPU(s), but \
             weight_staging={weight_staging} can use {wanted} \
             ({loader_threads} rank loader(s) + {fill_threads} pinned-fill worker(s)); \
             widen the process CPU affinity or expect serialized host fills"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn load_rank_executors(
    model_path: &Path,
    draft_path: Option<&Path>,
    single_rank_device: usize,
    ranks: std::ops::Range<usize>,
    ep_size: usize,
    config: K3ExecutorConfig,
    rendezvous: Option<&Arc<K3EpRendezvous>>,
) -> anyhow::Result<Vec<K3Executor>> {
    let placements = rank_placements(single_rank_device, ranks, ep_size);

    if !config.weight_staging || placements.len() == 1 {
        return placements
            .into_iter()
            .map(|(device, rank)| {
                load_rank_executor(
                    model_path,
                    draft_path,
                    device,
                    rank,
                    ep_size,
                    config,
                    rendezvous.cloned(),
                )
            })
            .collect();
    }

    std::thread::scope(|scope| {
        let handles = placements
            .into_iter()
            .map(|(device, rank)| {
                let rendezvous = rendezvous.cloned();
                (
                    rank,
                    scope.spawn(move || {
                        load_rank_executor(
                            model_path, draft_path, device, rank, ep_size, config, rendezvous,
                        )
                    }),
                )
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|(rank, handle)| {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("K3 rank {rank} loader thread panicked"))?
            })
            .collect()
    })
}

fn rank_placements(
    single_rank_device: usize,
    ranks: std::ops::Range<usize>,
    ep_size: usize,
) -> Vec<(usize, usize)> {
    ranks
        .enumerate()
        .map(|(local_device, rank)| {
            let device = if ep_size == 1 {
                single_rank_device
            } else {
                local_device
            };
            (device, rank)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn load_rank_executor(
    model_path: &Path,
    draft_path: Option<&Path>,
    device: usize,
    rank: usize,
    ep_size: usize,
    config: K3ExecutorConfig,
    rendezvous: Option<Arc<K3EpRendezvous>>,
) -> anyhow::Result<K3Executor> {
    let started = Instant::now();
    let target_started = Instant::now();
    let executor = match rendezvous {
        Some(rendezvous) => K3Executor::load_ep(model_path, device, rank, config, rendezvous),
        None => K3Executor::load(model_path, device, rank, ep_size, config),
    };
    let mut executor = executor
        .with_context(|| format!("loading K3 rank {rank} of {ep_size} onto device {device}"))?;
    let target_secs = target_started.elapsed().as_secs_f64();

    let draft_started = Instant::now();
    if let Some(draft_path) = draft_path {
        // The DSpark draft lane is rank-local and collective-free, so arming
        // it per rank adds no cross-rank coupling.
        executor
            .load_dspark(draft_path)
            .with_context(|| format!("arming the K3 dspark draft lane on rank {rank}"))?;
    }
    let draft_secs = draft_path
        .is_some()
        .then(|| draft_started.elapsed().as_secs_f64());
    info!(
        "K3 rank {rank} complete startup: total={:.2}s, target={target_secs:.2}s, dspark={}",
        started.elapsed().as_secs_f64(),
        draft_secs.map_or_else(|| "off".to_owned(), |secs| format!("{secs:.2}s")),
    );
    Ok(executor)
}

/// Arm the CP prefill lane from `PEGAINFER_K3_CP` (the CP width — must equal
/// this process's local rank count) and `PEGAINFER_K3_CP_MIN` (admission
/// floor in prompt tokens, default 2048). Bad values refuse to start rather
/// than serving something other than what was asked.
fn cp_serving(
    local_ranks: usize,
    dspark_armed: bool,
    chunk_tokens: usize,
) -> anyhow::Result<Option<K3CpServing>> {
    let Some(raw) = std::env::var_os("PEGAINFER_K3_CP") else {
        return Ok(None);
    };
    anyhow::ensure!(
        !dspark_armed,
        "PEGAINFER_K3_CP does not compose with the dspark draft lane yet; disarm one of them"
    );
    let raw = raw.to_string_lossy();
    let cp_size: usize = raw
        .parse()
        .ok()
        .filter(|&size| size > 1)
        .with_context(|| format!("PEGAINFER_K3_CP={raw} is not a CP width > 1"))?;
    anyhow::ensure!(
        cp_size == local_ranks,
        "PEGAINFER_K3_CP={cp_size} must equal this process's local rank count ({local_ranks}): \
         the gang spans the process's scheduler partitions, and every one of them must serve. \
         On a fleet each process runs its own gang over its local GPUs — remote EP ranks pad \
         the chunk steps like any other step"
    );
    let min_tokens = match std::env::var_os("PEGAINFER_K3_CP_MIN") {
        None => 2048,
        Some(raw) => {
            let raw = raw.to_string_lossy();
            raw.parse()
                .ok()
                .filter(|&min| min > 0)
                .with_context(|| format!("PEGAINFER_K3_CP_MIN={raw} is not a token count"))?
        }
    };
    info!(
        "K3 CP prefill lane armed: cp_size={cp_size}, min_tokens={min_tokens}, \
         chunk_tokens={chunk_tokens}"
    );
    Ok(Some(K3CpServing {
        gang: K3CpGang::new(K3CpGroup::new(cp_size)?),
        min_tokens,
        chunk_tokens,
    }))
}

/// The global ranks this process hosts: `--k3-ranks` when given, the whole
/// world otherwise. Validated where it is read, like the EP width.
fn local_ranks(cli: &K3Cli, ep_size: usize) -> Result<std::ops::Range<usize>, CliError> {
    let Some(spec) = &cli.k3_ranks else {
        return Ok(0..ep_size);
    };
    if ep_size == 1 {
        return Err(CliError::rule(
            "--k3-ranks partitions an EP world; it needs --k3-ep-size > 1",
        ));
    }
    let ranks =
        parse_rank_range(spec).map_err(|err| CliError::rule(format!("--k3-ranks: {err}")))?;
    if ranks.end > ep_size {
        return Err(CliError::rule(format!(
            "--k3-ranks {spec} does not fit an EP world of {ep_size}"
        )));
    }
    Ok(ranks)
}

#[cfg(test)]
mod tests {
    use pegainfer_frontend::model_line::parse_for_line;

    use super::*;

    /// A config shaped like the checkpoint's: multimodal wrapper on the
    /// outside, language-model fields under `text_config`.
    fn reference_config(experts: usize) -> Value {
        serde_json::json!({
            "model_type": MODEL_TYPE,
            "architectures": [ARCHITECTURE],
            "eos_token_id": 163_586,
            "text_config": {
                "model_type": TEXT_MODEL_TYPE,
                "architectures": ["KimiLinearForCausalLM"],
                "hidden_size": K3_HIDDEN,
                "num_hidden_layers": K3_LAYERS,
                "num_experts": experts,
                "num_experts_per_token": 16,
            },
        })
    }

    fn plan_for(argv: &[&str]) -> Result<ServePlan, CliError> {
        let (shared, matches, provided) =
            parse_for_line(&MODEL_LINE, argv).map_err(|error| CliError::rule(error.to_string()))?;
        let config = reference_config(224);
        let ctx = LaunchContext {
            model_path: std::path::Path::new("unused"),
            config: &config,
            shared: &shared,
            matches: &matches,
        };
        MODEL_LINE.validate(&ctx, &provided)?;
        MODEL_LINE.serve_plan(&ctx)
    }

    /// The scheduler partition count a CLI produces, or the rule it broke.
    /// (`ServePlan` is not `Debug`, so error assertions go through this.)
    fn partitions_for(argv: &[&str]) -> Result<usize, CliError> {
        plan_for(argv).map(|plan| plan.scheduler_partition_count)
    }

    #[test]
    fn probe_accepts_both_expert_counts() {
        for experts in K3_SUPPORTED_ROUTED_EXPERTS {
            MODEL_LINE
                .probe(&reference_config(experts))
                .unwrap_or_else(|error| {
                    panic!("K3 should claim a {experts}-expert config: {error}")
                });
        }
    }

    #[test]
    fn probe_rejects_another_family() {
        let error = MODEL_LINE
            .probe(&serde_json::json!({"model_type": "kimi_k2"}))
            .expect_err("K3 must not claim another family's config");
        assert!(error.contains("kimi_k3"), "{error}");
    }

    #[test]
    fn probe_rejects_a_mismatched_shape() {
        let mut config = reference_config(224);
        config["text_config"]["hidden_size"] = serde_json::json!(4096);
        let error = MODEL_LINE
            .probe(&config)
            .expect_err("a K3-labelled config with the wrong hidden size must be refused");
        assert!(error.contains("hidden_size"), "{error}");

        let mut config = reference_config(224);
        config["text_config"]["num_experts"] = serde_json::json!(64);
        let error = MODEL_LINE
            .probe(&config)
            .expect_err("an unsupported expert count must be refused");
        assert!(error.contains("num_experts"), "{error}");
    }

    #[test]
    fn probe_rejects_a_missing_architecture_claim() {
        let mut config = reference_config(224);
        config["architectures"] = serde_json::json!(["SomethingElseForCausalLM"]);
        let error = MODEL_LINE
            .probe(&config)
            .expect_err("the architecture claim is part of the identity");
        assert!(error.contains("architectures"), "{error}");
    }

    #[test]
    fn eos_ids_come_from_either_spelling() {
        assert_eq!(eos_token_ids(&reference_config(224)), vec![163_586]);
        assert_eq!(
            eos_token_ids(&serde_json::json!({"eos_token_id": [1, 2]})),
            vec![1, 2]
        );
        assert!(eos_token_ids(&serde_json::json!({})).is_empty());
    }

    #[test]
    fn serve_plan_defaults_to_one_partition() {
        let plan = plan_for(&["pegainfer"]).expect("defaults should validate");
        assert_eq!(plan.scheduler_partition_count, 1);
        assert!(!plan.prefill_only && plan.lora_modules.is_none());
    }

    #[test]
    fn serve_plan_counts_one_partition_per_ep_rank() {
        for ep_size in EP_SIZES {
            let partitions = partitions_for(&["pegainfer", "--k3-ep-size", &ep_size.to_string()])
                .expect("a supported EP width should validate");
            assert_eq!(partitions, *ep_size);
        }
    }

    #[test]
    fn weight_staging_flag_is_k3_owned() {
        let partitions = partitions_for(&["pegainfer", "--k3-ep-size", "4", "--k3-weight-staging"])
            .expect("K3 should accept its weight staging flag");
        assert_eq!(partitions, 4);
    }

    #[test]
    fn every_ep_width_maps_hosted_ranks_to_local_device_ordinals() {
        assert_eq!(rank_placements(7, 0..1, 1), vec![(7, 0)]);
        for ep_size in [4, 8, 16, 32, 64] {
            let start = ep_size - 4;
            assert_eq!(
                rank_placements(7, start..ep_size, ep_size),
                vec![(0, start), (1, start + 1), (2, start + 2), (3, start + 3)],
                "EP{ep_size}",
            );
        }
    }

    #[test]
    fn rejects_an_unsupported_ep_size() {
        let error = partitions_for(&["pegainfer", "--k3-ep-size", "6"])
            .expect_err("EP6 has no MegaMoE instantiation for any checkpoint");
        assert!(
            error.to_string().contains("--k3-ep-size must be one of"),
            "{error}"
        );
    }

    #[test]
    fn a_partial_fleet_counts_its_own_ranks() {
        let partitions = partitions_for(&[
            "pegainfer",
            "--k3-ep-size",
            "16",
            "--k3-ranks",
            "4..8",
            "--k3-rendezvous",
            "10.0.0.1:19300",
        ])
        .expect("a partial fleet slice should validate");
        assert_eq!(partitions, 4);
    }

    #[test]
    fn a_partial_fleet_requires_the_rendezvous() {
        let error = partitions_for(&["pegainfer", "--k3-ep-size", "16", "--k3-ranks", "4..8"])
            .expect_err("a fleet slice cannot pair without the bootstrap");
        assert!(error.to_string().contains("--k3-rendezvous"), "{error}");
    }

    #[test]
    fn a_whole_world_refuses_a_rendezvous() {
        let error = partitions_for(&[
            "pegainfer",
            "--k3-ep-size",
            "4",
            "--k3-rendezvous",
            "10.0.0.1:19300",
        ])
        .expect_err("an in-process world has nobody to exchange handles with");
        assert!(error.to_string().contains("--k3-rendezvous"), "{error}");
    }

    #[test]
    fn rank_ranges_are_validated() {
        let error = partitions_for(&["pegainfer", "--k3-ranks", "0..2"])
            .expect_err("--k3-ranks without a wider world is meaningless");
        assert!(error.to_string().contains("--k3-ep-size"), "{error}");

        let error = partitions_for(&[
            "pegainfer",
            "--k3-ep-size",
            "8",
            "--k3-ranks",
            "6..10",
            "--k3-rendezvous",
            "10.0.0.1:19300",
        ])
        .expect_err("ranks past the world must be refused");
        assert!(error.to_string().contains("does not fit"), "{error}");

        let error = partitions_for(&[
            "pegainfer",
            "--k3-ep-size",
            "8",
            "--k3-ranks",
            "4",
            "--k3-rendezvous",
            "10.0.0.1:19300",
        ])
        .expect_err("a bound without .. must be refused");
        assert!(error.to_string().contains("start..end"), "{error}");
    }

    #[test]
    fn accepts_a_device_ordinal_only_for_a_single_rank() {
        partitions_for(&["pegainfer", "--device-ordinal", "2"])
            .expect("a single-rank K3 run picks its GPU");
        let error = partitions_for(&["pegainfer", "--device-ordinal", "2", "--k3-ep-size", "4"])
            .expect_err("--device-ordinal is meaningless across EP ranks");
        assert!(error.to_string().contains("--device-ordinal"), "{error}");
    }

    #[test]
    fn rejects_a_shared_flag_this_line_does_not_read() {
        let error = partitions_for(&["pegainfer", "--kv-offload"])
            .expect_err("K3 serves no KV offload tier");
        assert!(error.to_string().contains("--kv-offload"), "{error}");
    }
}
