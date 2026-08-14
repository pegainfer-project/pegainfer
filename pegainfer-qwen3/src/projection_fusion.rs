use pegainfer_kernels::ops::NumericPolicy;

use crate::DecodeOverlap;
use crate::config::Config;
use crate::config::TensorParallelConfig;

/// The measured production policy for Qwen3 projection fusion.
///
/// Fusion is one atomic decode topology: QKV and gate/up are either both
/// fused or both split. Unsupported model/runtime topologies fail closed to
/// the established split-GEMM path. The policy is intentionally GPU-agnostic;
/// each device still tunes its own cuBLASLt projection shapes at startup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Qwen3ProjectionFusionPlan {
    decode: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ProjectionFusionEnvironment {
    pub(crate) numeric_policy: NumericPolicy,
    pub(crate) decode_overlap: DecodeOverlap,
    pub(crate) dflash_enabled: bool,
}

impl Qwen3ProjectionFusionPlan {
    pub(crate) fn resolve(
        config: &Config,
        tensor_parallel: TensorParallelConfig,
        environment: ProjectionFusionEnvironment,
    ) -> Self {
        Self {
            decode: is_qwen3_4b(config)
                && tensor_parallel.world_size == 1
                && environment.numeric_policy == NumericPolicy::Tuned
                && matches!(environment.decode_overlap, DecodeOverlap::Off)
                && !environment.dflash_enabled,
        }
    }

    pub(crate) const fn decode(self) -> bool {
        self.decode
    }
}

const fn is_qwen3_4b(config: &Config) -> bool {
    config.hidden_size == 2560
        && config.intermediate_size == 9728
        && config.num_hidden_layers == 36
        && config.num_attention_heads == 32
        && config.num_key_value_heads == 8
        && config.head_dim == 128
        && config.vocab_size == 151_936
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen3_4b() -> Config {
        Config {
            hidden_size: 2560,
            intermediate_size: 9728,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 151_936,
            rms_norm_eps: 1.0e-6,
            rope_theta: 1.0e6,
            max_position_embeddings: 40_960,
            eos_token_id: 151_645,
            tie_word_embeddings: true,
            stop_token_ids: vec![151_645],
        }
    }

    fn environment() -> ProjectionFusionEnvironment {
        ProjectionFusionEnvironment {
            numeric_policy: NumericPolicy::Tuned,
            decode_overlap: DecodeOverlap::Off,
            dflash_enabled: false,
        }
    }

    #[test]
    fn enables_only_qualified_tp1_decode() {
        let plan = Qwen3ProjectionFusionPlan::resolve(
            &qwen3_4b(),
            TensorParallelConfig::default(),
            environment(),
        );
        assert!(plan.decode());
    }

    #[test]
    fn fails_closed_outside_qualified_environment() {
        let mut other_model = qwen3_4b();
        other_model.hidden_size = 4096;

        let mut pin = environment();
        pin.numeric_policy = NumericPolicy::Pin;
        let mut per_token = environment();
        per_token.numeric_policy = NumericPolicy::PerToken;
        let mut overlap = environment();
        overlap.decode_overlap = DecodeOverlap::SharedSm;
        let mut green_ctx = environment();
        green_ctx.decode_overlap = DecodeOverlap::GreenCtx { decode_pct: 20 };
        let mut dflash = environment();
        dflash.dflash_enabled = true;

        let cases = [
            Qwen3ProjectionFusionPlan::resolve(
                &qwen3_4b(),
                TensorParallelConfig {
                    rank: 0,
                    world_size: 2,
                },
                environment(),
            ),
            Qwen3ProjectionFusionPlan::resolve(
                &other_model,
                TensorParallelConfig::default(),
                environment(),
            ),
            Qwen3ProjectionFusionPlan::resolve(&qwen3_4b(), TensorParallelConfig::default(), pin),
            Qwen3ProjectionFusionPlan::resolve(
                &qwen3_4b(),
                TensorParallelConfig::default(),
                per_token,
            ),
            Qwen3ProjectionFusionPlan::resolve(
                &qwen3_4b(),
                TensorParallelConfig::default(),
                overlap,
            ),
            Qwen3ProjectionFusionPlan::resolve(
                &qwen3_4b(),
                TensorParallelConfig::default(),
                green_ctx,
            ),
            Qwen3ProjectionFusionPlan::resolve(
                &qwen3_4b(),
                TensorParallelConfig::default(),
                dflash,
            ),
        ];

        assert!(cases.into_iter().all(|plan| !plan.decode()));
    }
}
