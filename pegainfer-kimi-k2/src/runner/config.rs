use std::path::PathBuf;

use pegainfer_frontend::parallel::ParallelConfig;

use crate::runner::affinity::KimiRankThreadPlacementPlan;
use crate::runner::worker::KimiK2RankPlacement;
use crate::weights::KimiRankSlicedLoadPlan;
use crate::weights::KimiRankWeightNames;

#[derive(Clone, Debug)]
pub(crate) struct KimiK2RunnerConfig {
    pub model_path: PathBuf,
    pub parallel: ParallelConfig,
    pub local_dims: crate::config::KimiLocalDims,
    pub rank_weight_names: Vec<KimiRankWeightNames>,
    pub rank_sliced_load_plans: Vec<KimiRankSlicedLoadPlan>,
    pub placements: Vec<KimiK2RankPlacement>,
    pub(crate) thread_placement: KimiRankThreadPlacementPlan,
    pub enable_cuda_graph: bool,
    /// KV pool size in pages per rank. Both the per-rank physical MLA pool
    /// and the scheduler's logical `BlockPool` are sized from this, so the
    /// block accounting and the GPU buffers can never disagree.
    pub kv_pool_pages: usize,
}
