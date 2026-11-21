use std::collections::BTreeSet;

use clap::Args;
use clap::FromArgMatches;
use pegainfer_frontend::engine::Engine;
use pegainfer_frontend::engine::EngineInfo;
use pegainfer_frontend::engine::KvCapacity;
use pegainfer_frontend::engine::LaunchedEngine;
use pegainfer_frontend::engine::spawn_scheduler;
use pegainfer_frontend::model_line::CliError;
use pegainfer_frontend::model_line::LaunchContext;
use pegainfer_frontend::model_line::ModelLine;

use super::Model;
use super::ModelOptions;
use super::scheduler::AscendScheduler;
pub static MODEL_LINE: AscendQwen3Line = AscendQwen3Line;
pub struct AscendQwen3Line;
#[derive(Args)]
struct AscendArgs {
    /// Maximum prompt + completion length per request, reserved in every slot.
    #[arg(long, default_value_t = 4096)]
    ascend_max_context: usize,
    /// Number of simultaneous requests; each owns a fixed KV reservation.
    #[arg(long, default_value_t = 4)]
    ascend_slots: usize,
    /// Maximum rows per eager prefill/decode step.
    #[arg(long, default_value_t = 128)]
    ascend_chunk_size: usize,
}
impl ModelLine for AscendQwen3Line {
    fn name(&self) -> &'static str {
        "Qwen3 Ascend"
    }
    fn probe(&self, value: &serde_json::Value) -> Result<(), String> {
        super::config::probe_config_json(value).map_err(|e| e.to_string())
    }
    fn augment_cli(&self, cmd: clap::Command) -> clap::Command {
        <AscendArgs as clap::Args>::augment_args(cmd)
    }
    fn consumed_shared_args(&self) -> &'static [&'static str] {
        &["device_ordinal", "tp_size", "cuda_graph", "no_prefix_cache"]
    }
    fn serve_plan(
        &self,
        _ctx: &LaunchContext<'_>,
    ) -> Result<pegainfer_frontend::model_line::ServePlan, CliError> {
        Ok(pegainfer_frontend::model_line::ServePlan {
            extend_router: Some(pegainfer_frontend::vllm::greedy_text_routes),
            ..Default::default()
        })
    }
    fn validate(
        &self,
        ctx: &LaunchContext<'_>,
        provided: &BTreeSet<String>,
    ) -> Result<(), CliError> {
        if ctx.shared.tp_size != 1 {
            return Err(CliError::rule("Ascend first release requires --tp-size=1"));
        }
        if provided.contains("cuda_graph") && ctx.shared.cuda_graph {
            return Err(CliError::rule(
                "Ascend first release is eager; use --cuda-graph=false",
            ));
        }
        Ok(())
    }
    fn launch(&self, ctx: &LaunchContext<'_>) -> anyhow::Result<LaunchedEngine> {
        let args = AscendArgs::from_arg_matches(ctx.matches)?;
        let o = ModelOptions {
            device: ctx.shared.device_ordinal,
            max_context: args.ascend_max_context,
            slots: args.ascend_slots,
            chunk_size: args.ascend_chunk_size,
        };
        let model = Model::load(&ctx.shared.model_path, o)?;
        if ctx.shared.cuda_graph {
            log::info!(
                "Using eager execution: the shared cuda_graph default is ignored by the Ascend backend"
            );
        }
        log::info!(
            "Ascend profile: BF16, TP1, eager, greedy, fixed KV slots; prefix reuse disabled"
        );
        if ctx.shared.no_prefix_cache {
            log::info!(
                "--no-prefix-cache is redundant for this backend: prefix caching is always disabled"
            );
        }
        let scheduler = spawn_scheduler("qwen3-ascend", AscendScheduler::new(model));
        Ok(Engine {
            schedulers: vec![scheduler],
            info: EngineInfo {
                kv_capacity: Some(KvCapacity {
                    total_blocks: o.slots * o.max_context,
                    block_size: 1,
                }),
                servable_len: Some(o.max_context as u32),
            },
            lora: None,
        }
        .into())
    }
}
