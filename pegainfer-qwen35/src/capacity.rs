//! Qwen3.5 serving capacity model: accounts upfront for decode_width, max_resident,
//! recurrent state pool, dual-cohort graph buffers, prefix snapshots, KV pages, and
//! runtime workspace; fails fast at startup when memory is insufficient.

use anyhow::Result;
use log::info;

use super::batch_decode_graph;
use super::config::Config35;
use super::config::LocalGeometry;
use super::decode_buffers::BatchDecodeBuffers35;
use super::recurrent_state;

/// Default per-step decode width (matches the maximum CUDA Graph bucket).
pub(crate) const DEFAULT_DECODE_WIDTH: usize = batch_decode_graph::MAX_BATCH;

/// Maximum number of cohorts on a single GPU (two 64-row decode graphs A/B).
pub(crate) const MAX_SINGLE_GPU_COHORTS: usize = 2;

/// Default max resident requests on a single GPU (2 x 64 = 128).
pub(crate) const DEFAULT_SINGLE_GPU_MAX_RESIDENT: usize =
    DEFAULT_DECODE_WIDTH * MAX_SINGLE_GPU_COHORTS;

/// Full-attention KV page size (tokens per page).
pub(crate) const KV_PAGE_SIZE: usize = 16;

/// Runtime workspace reservation for cuBLASLt / FlashInfer / CUDA Graph (512 MiB).
pub(crate) const RUNTIME_WORKSPACE_BYTES: usize = 512 * 1024 * 1024;

/// Maximum fraction of remaining free VRAM allocated to the KV pool, leaving headroom for driver fragmentation.
const KV_HEADROOM_FRACTION: f64 = 0.90;

/// Full memory and concurrency capacity ledger for a Qwen3.5 instance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServingCapacity {
    /// Per-step decode width limit (1..=64).
    pub decode_width: usize,
    /// Maximum concurrent resident requests (single-GPU default 128, TP default 64).
    pub max_resident: usize,
    /// Number of pre-allocated request recurrent state slots (equals `max_resident`).
    pub state_slots: usize,
    /// Number of shared CUDA Graph padding state slots (1 on single GPU, 0 on TP).
    pub padding_state_slots: usize,
    /// Extra transient/graph state slots reserved for TP single-cohort mode (0 on single GPU).
    pub tp_extra_state_slots: usize,
    /// Number of active cohorts (2 when single-GPU `max_resident > decode_width`, else 1; always 1 on TP).
    pub cohort_count: usize,
    /// Resident request capacity per cohort (0 for unused entries).
    pub cohort_widths: [usize; MAX_SINGLE_GPU_COHORTS],
    /// CUDA Graph bucket capacity per cohort (0 for unused entries).
    pub cohort_graph_buckets: [usize; MAX_SINGLE_GPU_COHORTS],
    /// Maximum sequence length declared by the model config (e.g. 262144).
    pub max_model_len: usize,
    /// KV page size (tokens per page).
    pub kv_page_size: usize,
    /// Total KV pool pages (including 1 padding page).
    pub kv_pages: usize,
    /// KV pages available to requests (`kv_pages - 1`).
    pub kv_request_pages: usize,
    /// KV pool size in bytes.
    pub kv_bytes: usize,
    /// Recurrent state pool size in bytes (including padding/TP reservations).
    pub recurrent_state_bytes: usize,
    /// Number of prefix snapshot slots (0 when disabled).
    pub prefix_snapshot_slots: usize,
    /// Prefix snapshot pool size in bytes.
    pub prefix_snapshot_bytes: usize,
    /// Total graph decode buffers and pointer tables size across all cohorts (bytes).
    pub graph_buffers_bytes: usize,
    /// Prefill scratch reservation in bytes.
    pub prefill_scratch_bytes: usize,
    /// Runtime workspace reservation in bytes.
    pub runtime_workspace_bytes: usize,
    /// Free device memory observed at capacity planning time (bytes).
    pub free_bytes_at_plan: usize,
}

impl ServingCapacity {
    /// Returns the default `max_resident` for the given TP topology.
    #[must_use]
    pub fn default_max_resident(decode_width: usize, tp_world_size: usize) -> usize {
        if tp_world_size > 1 {
            decode_width
        } else {
            decode_width.saturating_mul(MAX_SINGLE_GPU_COHORTS)
        }
    }

    /// Validates requested `(decode_width, max_resident)` against the TP topology without silent downgrade.
    pub fn validate_request(
        decode_width: usize,
        max_resident: usize,
        tp_world_size: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            tp_world_size >= 1,
            "Qwen3.5 tp_size must be >= 1, got {tp_world_size}"
        );
        anyhow::ensure!(
            (1..=DEFAULT_DECODE_WIDTH).contains(&decode_width),
            "Qwen3.5 decode_width must be in 1..={DEFAULT_DECODE_WIDTH}, got {decode_width}"
        );
        anyhow::ensure!(
            max_resident >= 1,
            "Qwen3.5 max_resident must be >= 1, got {max_resident}"
        );
        if tp_world_size > 1 {
            anyhow::ensure!(
                max_resident <= decode_width,
                "Qwen3.5 TP supports a single cohort only: max_resident ({max_resident}) must be <= decode_width ({decode_width})"
            );
        } else {
            let max_allowed = decode_width * MAX_SINGLE_GPU_COHORTS;
            anyhow::ensure!(
                max_resident <= max_allowed,
                "Qwen3.5 single-GPU max_resident ({max_resident}) exceeds {MAX_SINGLE_GPU_COHORTS} cohorts x decode_width ({decode_width}) = {max_allowed}"
            );
        }
        Ok(())
    }

    /// Plans full memory budgets from actual free VRAM after weight loading; fails fast if insufficient.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn plan(
        config: &Config35,
        geometry: LocalGeometry,
        decode_width: usize,
        max_resident: usize,
        enable_cuda_graph: bool,
        split_decode_max_batch: usize,
        prefill_scratch_bytes: usize,
        prefix_snapshot_budget_bytes: usize,
        free_bytes: usize,
    ) -> Result<Self> {
        let world_size = geometry.world_size();
        Self::validate_request(decode_width, max_resident, world_size)?;

        let is_tp = world_size > 1;
        let (cohort_count, cohort_widths) = if is_tp || max_resident <= decode_width {
            (1, [max_resident, 0])
        } else {
            (2, [decode_width, max_resident - decode_width])
        };

        let mut cohort_graph_buckets = [0usize; MAX_SINGLE_GPU_COHORTS];
        for i in 0..cohort_count {
            cohort_graph_buckets[i] = batch_decode_graph::bucket_for(cohort_widths[i]);
        }

        let per_request_state_bytes = recurrent_state::bytes_per_request(config, geometry);
        let state_slots = max_resident;
        let padding_state_slots = usize::from(!is_tp);
        // TP keeps the single-cohort architecture: in graph mode it pre-allocates `slot_bucket` graph states
        // in addition to per-request transient states.
        let tp_extra_state_slots =
            if is_tp && enable_cuda_graph && config.decode_group_is_compiled() {
                cohort_graph_buckets[0]
            } else {
                0
            };
        let total_state_slots = state_slots + padding_state_slots + tp_extra_state_slots;
        let recurrent_state_bytes = total_state_slots
            .checked_mul(per_request_state_bytes)
            .ok_or_else(|| {
                anyhow::anyhow!("Qwen3.5 recurrent-state byte accounting overflowed usize")
            })?;

        let prefix_snapshot_slots = prefix_snapshot_budget_bytes
            .checked_div(per_request_state_bytes)
            .unwrap_or(0);
        anyhow::ensure!(
            prefix_snapshot_budget_bytes == 0 || prefix_snapshot_slots > 0,
            "Qwen3.5 prefix-cache budget is {} MiB, but one recurrent/conv snapshot requires {:.3} MiB",
            to_mib(prefix_snapshot_budget_bytes),
            per_request_state_bytes as f64 / 1024.0 / 1024.0,
        );
        let prefix_snapshot_bytes = prefix_snapshot_slots
            .checked_mul(per_request_state_bytes)
            .ok_or_else(|| {
                anyhow::anyhow!("Qwen3.5 prefix-snapshot byte accounting overflowed usize")
            })?;

        let num_full_layers = config.num_full_attention_layers();
        let kv_layout = pegainfer_kv_cache::KvLayout::new(
            num_full_layers,
            geometry.local_num_key_value_heads(),
            config.head_dim,
            KV_PAGE_SIZE,
        );
        let bytes_per_page = kv_layout
            .page_stride
            .checked_mul(std::mem::size_of::<half::bf16>())
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 KV page byte size overflowed usize"))?;

        // Per-cohort decode buffers + linear pointer tables (including `page_indices_d`).
        let graph_buffers_bytes: usize = (0..cohort_count)
            .map(|i| {
                let bucket = cohort_graph_buckets[i];
                BatchDecodeBuffers35::estimate_bytes(
                    config,
                    geometry,
                    bucket,
                    KV_PAGE_SIZE,
                    split_decode_max_batch,
                ) + recurrent_state::LinearStatePointerTables::estimate_bytes(config, bucket)
            })
            .sum();

        let runtime_workspace_bytes = RUNTIME_WORKSPACE_BYTES;
        let fixed_bytes = recurrent_state_bytes
            .checked_add(prefix_snapshot_bytes)
            .and_then(|v| v.checked_add(graph_buffers_bytes))
            .and_then(|v| v.checked_add(prefill_scratch_bytes))
            .and_then(|v| v.checked_add(runtime_workspace_bytes))
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 fixed byte accounting overflowed usize"))?;

        let max_model_len = config.max_position_embeddings;
        let min_request_pages = max_model_len.div_ceil(KV_PAGE_SIZE);
        // 1 extra page reserved for CUDA Graph padding.
        let min_kv_pages = min_request_pages
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 minimum KV page count overflowed usize"))?;
        let min_kv_bytes = min_kv_pages.checked_mul(bytes_per_page).ok_or_else(|| {
            anyhow::anyhow!("Qwen3.5 minimum KV byte accounting overflowed usize")
        })?;

        let required_bytes = fixed_bytes
            .checked_add(min_kv_bytes)
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 required byte accounting overflowed usize"))?;
        let mode = if is_tp {
            format!("tp{}-rank{}", world_size, geometry.rank())
        } else {
            "single-gpu".to_string()
        };
        anyhow::ensure!(
            free_bytes >= required_bytes,
            "insufficient device memory for Qwen3.5 ({mode}): {} MiB free, need at least {} MiB \
             [recurrent_state={} MiB ({state_slots} slots + {padding_state_slots} padding + {tp_extra_state_slots} tp_graph), \
             prefix_snapshots={} MiB ({prefix_snapshot_slots} slots), \
             graph_buffers={} MiB ({cohort_count} cohort(s) buckets={:?}), \
             prefill_scratch={} MiB, workspace={} MiB, \
             min_kv={} MiB ({min_kv_pages} pages for max_model_len={max_model_len})]; \
             refusing to silently downgrade max_resident={max_resident} or max_model_len={max_model_len}",
            to_mib(free_bytes),
            to_mib(required_bytes),
            to_mib(recurrent_state_bytes),
            to_mib(prefix_snapshot_bytes),
            to_mib(graph_buffers_bytes),
            &cohort_graph_buckets[..cohort_count],
            to_mib(prefill_scratch_bytes),
            to_mib(runtime_workspace_bytes),
            to_mib(min_kv_bytes),
        );

        let available_for_kv = free_bytes - fixed_bytes;
        let target_kv_bytes = ((available_for_kv as f64) * KV_HEADROOM_FRACTION) as usize;
        let budgeted_kv_bytes = target_kv_bytes.max(min_kv_bytes).min(available_for_kv);
        let kv_pages = (budgeted_kv_bytes / bytes_per_page).max(min_kv_pages);
        let kv_request_pages = kv_pages - 1;
        let kv_bytes = kv_pages
            .checked_mul(bytes_per_page)
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 KV pool byte accounting overflowed usize"))?;

        Ok(Self {
            decode_width,
            max_resident,
            state_slots,
            padding_state_slots,
            tp_extra_state_slots,
            cohort_count,
            cohort_widths,
            cohort_graph_buckets,
            max_model_len,
            kv_page_size: KV_PAGE_SIZE,
            kv_pages,
            kv_request_pages,
            kv_bytes,
            recurrent_state_bytes,
            prefix_snapshot_slots,
            prefix_snapshot_bytes,
            graph_buffers_bytes,
            prefill_scratch_bytes,
            runtime_workspace_bytes,
            free_bytes_at_plan: free_bytes,
        })
    }

    /// Largest graph bucket width across all cohorts (upper bound for cublasLt decode tuning).
    pub(crate) fn max_graph_bucket(&self) -> usize {
        self.cohort_graph_buckets[..self.cohort_count]
            .iter()
            .copied()
            .max()
            .unwrap_or(1)
    }

    /// Logs the startup capacity ledger so every item can be verified against measured VRAM.
    pub(crate) fn log_ledger(&self, geometry: LocalGeometry) {
        let mode = if geometry.world_size() > 1 {
            format!("tp{}-rank{}", geometry.world_size(), geometry.rank())
        } else {
            "single-gpu".to_string()
        };
        let total_planned = self.kv_bytes
            + self.recurrent_state_bytes
            + self.prefix_snapshot_bytes
            + self.graph_buffers_bytes
            + self.prefill_scratch_bytes
            + self.runtime_workspace_bytes;
        info!(
            "Qwen3.5 capacity ({mode}): decode_width={}, max_resident={}, cohorts={} ({:?}, buckets={:?}), \
             state_slots={} (+{} padding, +{} tp_graph, {} MiB), prefix_snapshots={} ({} MiB), \
             kv={} pages ({} req pages, {} tokens, {} MiB), \
             graph_buffers={} MiB, prefill_scratch={} MiB, workspace={} MiB, \
             max_model_len={}, planned={} MiB / free={} MiB",
            self.decode_width,
            self.max_resident,
            self.cohort_count,
            &self.cohort_widths[..self.cohort_count],
            &self.cohort_graph_buckets[..self.cohort_count],
            self.state_slots,
            self.padding_state_slots,
            self.tp_extra_state_slots,
            to_mib(self.recurrent_state_bytes),
            self.prefix_snapshot_slots,
            to_mib(self.prefix_snapshot_bytes),
            self.kv_pages,
            self.kv_request_pages,
            self.kv_request_pages * self.kv_page_size,
            to_mib(self.kv_bytes),
            to_mib(self.graph_buffers_bytes),
            to_mib(self.prefill_scratch_bytes),
            to_mib(self.runtime_workspace_bytes),
            self.max_model_len,
            to_mib(total_planned),
            to_mib(self.free_bytes_at_plan),
        );
    }
}

fn to_mib(bytes: usize) -> usize {
    bytes / (1024 * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LayerType;
    use crate::config::TensorParallelConfig;
    use crate::decode_buffers::SPLIT_DECODE_BASE_MAX_BATCH;
    use crate::prefill::SCRATCH_ESTIMATE_SEQ;
    use crate::prefill_buffers::prefill_scratch_bytes;

    fn qwen35_4b_config() -> Config35 {
        let mut layer_types = Vec::with_capacity(32);
        for i in 0..32 {
            layer_types.push(if i % 4 == 3 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            });
        }
        Config35 {
            hidden_size: 2560,
            intermediate_size: 9216,
            num_hidden_layers: 32,
            vocab_size: 248_320,
            rms_norm_eps: 1e-6,
            eos_token_ids: vec![248_044],
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 256,
            linear_num_key_heads: 16,
            linear_key_head_dim: 128,
            linear_num_value_heads: 32,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            rope_theta: 10_000_000.0,
            rotary_dim: 64,
            max_position_embeddings: 262_144,
            layer_types,
            tie_word_embeddings: true,
            selection_vocab: 248_320,
            decodable_vocab: 248_077,
        }
    }

    fn qwen35_27b_config() -> Config35 {
        let mut layer_types = Vec::with_capacity(64);
        for i in 0..64 {
            layer_types.push(if i % 4 == 3 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            });
        }
        Config35 {
            hidden_size: 5120,
            intermediate_size: 17408,
            num_hidden_layers: 64,
            vocab_size: 248_320,
            rms_norm_eps: 1e-6,
            eos_token_ids: vec![248_044],
            num_attention_heads: 24,
            num_key_value_heads: 4,
            head_dim: 256,
            linear_num_key_heads: 16,
            linear_key_head_dim: 128,
            linear_num_value_heads: 48,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            rope_theta: 10_000_000.0,
            rotary_dim: 64,
            max_position_embeddings: 262_144,
            layer_types,
            tie_word_embeddings: false,
            selection_vocab: 248_320,
            decodable_vocab: 248_077,
        }
    }

    #[test]
    fn single_gpu_default_plans_two_64_cohorts_and_262k_kv() {
        let cfg = qwen35_4b_config();
        let geom = LocalGeometry::try_new(&cfg, TensorParallelConfig::default()).unwrap();
        let scratch = prefill_scratch_bytes(&cfg, geom, SCRATCH_ESTIMATE_SEQ);
        let free_bytes = 85 * 1024 * 1024 * 1024;
        let cap = ServingCapacity::plan(
            &cfg,
            geom,
            DEFAULT_DECODE_WIDTH,
            DEFAULT_SINGLE_GPU_MAX_RESIDENT,
            true,
            SPLIT_DECODE_BASE_MAX_BATCH,
            scratch,
            0,
            free_bytes,
        )
        .expect("4B on H20 must fit dual 64-row cohorts");

        assert_eq!(cap.decode_width, 64);
        assert_eq!(cap.max_resident, 128);
        assert_eq!(cap.state_slots, 128);
        assert_eq!(cap.padding_state_slots, 1);
        assert_eq!(cap.cohort_count, 2);
        assert_eq!(cap.cohort_widths, [64, 64]);
        assert_eq!(cap.cohort_graph_buckets, [64, 64]);
        assert_eq!(cap.max_model_len, 262_144);
        assert!(cap.kv_request_pages * cap.kv_page_size >= 262_144);
        assert!(
            cap.kv_bytes
                + cap.recurrent_state_bytes
                + cap.prefix_snapshot_bytes
                + cap.graph_buffers_bytes
                + cap.prefill_scratch_bytes
                + cap.runtime_workspace_bytes
                <= free_bytes
        );
    }

    #[test]
    fn single_gpu_27b_tp1_accounting_and_fail_fast_boundary() {
        let cfg = qwen35_27b_config();
        let geom = LocalGeometry::try_new(&cfg, TensorParallelConfig::default()).unwrap();
        let scratch = prefill_scratch_bytes(&cfg, geom, SCRATCH_ESTIMATE_SEQ);

        // 27B TP1 has 48 local linear value heads (~146.8 MiB/slot): 128 resident slots +
        // 16K prefill scratch + one 262144-token KV sequence needs ~43.0 GiB, so a 96 GiB
        // H20 (~42 GiB free after weights) must fail fast at 128 and succeed at 64.
        let h20_free_bytes = 42 * 1024 * 1024 * 1024;
        let err = ServingCapacity::plan(
            &cfg,
            geom,
            DEFAULT_DECODE_WIDTH,
            DEFAULT_SINGLE_GPU_MAX_RESIDENT,
            true,
            SPLIT_DECODE_BASE_MAX_BATCH,
            scratch,
            0,
            h20_free_bytes,
        )
        .expect_err("27B TP1 with 42 GiB free must fail fast at max_resident=128")
        .to_string();
        assert!(err.contains("insufficient device memory"), "got: {err}");

        let cap64 = ServingCapacity::plan(
            &cfg,
            geom,
            DEFAULT_DECODE_WIDTH,
            64,
            true,
            SPLIT_DECODE_BASE_MAX_BATCH,
            scratch,
            0,
            h20_free_bytes,
        )
        .expect("27B TP1 with 42 GiB free must fit max_resident=64");
        assert_eq!(cap64.cohort_count, 1);
        assert_eq!(cap64.max_resident, 64);
        assert!(cap64.kv_request_pages * cap64.kv_page_size >= 262_144);

        // With 50 GiB free (e.g. 141 GiB H200), 27B TP1 fits dual 64-row cohorts + 262k KV.
        let h200_free_bytes = 50 * 1024 * 1024 * 1024;
        let cap128 = ServingCapacity::plan(
            &cfg,
            geom,
            DEFAULT_DECODE_WIDTH,
            DEFAULT_SINGLE_GPU_MAX_RESIDENT,
            true,
            SPLIT_DECODE_BASE_MAX_BATCH,
            scratch,
            0,
            h200_free_bytes,
        )
        .expect("27B TP1 with 50 GiB free must fit dual cohorts + one 262144 sequence");
        assert_eq!(cap128.decode_width, 64);
        assert_eq!(cap128.max_resident, 128);
        assert_eq!(cap128.cohort_count, 2);
        assert!(cap128.kv_request_pages * cap128.kv_page_size >= 262_144);
    }

    #[test]
    fn tp2_plans_single_cohort_64_and_rejects_128() {
        let cfg = qwen35_27b_config();
        let tp2 = TensorParallelConfig::try_from((0, 2)).unwrap();
        let geom = LocalGeometry::try_new(&cfg, tp2).unwrap();
        let scratch = prefill_scratch_bytes(&cfg, geom, SCRATCH_ESTIMATE_SEQ);
        let free_bytes = 68 * 1024 * 1024 * 1024;
        let cap = ServingCapacity::plan(
            &cfg,
            geom,
            64,
            64,
            false,
            SPLIT_DECODE_BASE_MAX_BATCH,
            scratch,
            0,
            free_bytes,
        )
        .expect("TP2 must fit single 64-slot cohort");
        assert_eq!(cap.cohort_count, 1);
        assert_eq!(cap.max_resident, 64);
        assert_eq!(cap.state_slots, 64);
        assert_eq!(cap.padding_state_slots, 0);

        let err = ServingCapacity::plan(
            &cfg,
            geom,
            64,
            128,
            false,
            SPLIT_DECODE_BASE_MAX_BATCH,
            scratch,
            0,
            free_bytes,
        )
        .expect_err("TP2 must fail-fast on max_resident > decode_width")
        .to_string();
        assert!(err.contains("single cohort"), "unexpected error: {err}");
    }

    #[test]
    fn fails_fast_when_memory_cannot_hold_one_max_model_len_sequence() {
        let cfg = qwen35_4b_config();
        let geom = LocalGeometry::try_new(&cfg, TensorParallelConfig::default()).unwrap();
        let scratch = prefill_scratch_bytes(&cfg, geom, SCRATCH_ESTIMATE_SEQ);
        let free_bytes = 10 * 1024 * 1024 * 1024;
        let err = ServingCapacity::plan(
            &cfg,
            geom,
            64,
            128,
            true,
            SPLIT_DECODE_BASE_MAX_BATCH,
            scratch,
            0,
            free_bytes,
        )
        .expect_err("must fail-fast instead of silently downgrading")
        .to_string();
        assert!(err.contains("insufficient device memory"), "got: {err}");
        assert!(err.contains("max_model_len=262144"), "got: {err}");
        assert!(err.contains("refusing to silently downgrade"), "got: {err}");
    }

    #[test]
    fn single_gpu_sub_width_uses_one_cohort() {
        let cfg = qwen35_4b_config();
        let geom = LocalGeometry::try_new(&cfg, TensorParallelConfig::default()).unwrap();
        let scratch = prefill_scratch_bytes(&cfg, geom, SCRATCH_ESTIMATE_SEQ);
        let free_bytes = 85 * 1024 * 1024 * 1024;
        let cap = ServingCapacity::plan(
            &cfg,
            geom,
            8,
            8,
            true,
            SPLIT_DECODE_BASE_MAX_BATCH,
            scratch,
            0,
            free_bytes,
        )
        .expect("8/8 should use one cohort");
        assert_eq!(cap.cohort_count, 1);
        assert_eq!(cap.cohort_widths, [8, 0]);
        assert_eq!(cap.cohort_graph_buckets, [8, 0]);
    }

    #[test]
    fn rejects_invalid_decode_width_and_max_resident() {
        assert!(ServingCapacity::validate_request(0, 64, 1).is_err());
        assert!(ServingCapacity::validate_request(65, 65, 1).is_err());
        assert!(ServingCapacity::validate_request(64, 0, 1).is_err());
        assert!(ServingCapacity::validate_request(64, 129, 1).is_err());
        assert!(ServingCapacity::validate_request(32, 65, 1).is_err());
        assert!(ServingCapacity::validate_request(64, 65, 2).is_err());
    }
}
