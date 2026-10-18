use std::path::PathBuf;

use anyhow::Context as _;
use pegainfer_core::cuda_graph::CudaGraphDumpSummary;
use pegainfer_kv_store::BlockPool;

use super::plan::padding_step_kv;
use super::slot::GLM52_PADDING_STEP;
use crate::Glm52MoeTopo;
use crate::model::GLM52_DECODE_BUCKETS;
use crate::model::GLM52_MAX_BATCH_PER_RANK;
use crate::model::GLM52_MAX_STEP_ROWS;
use crate::model::Glm52StepShape;
use crate::runner::Glm52StepFlags;
use crate::runner::Glm52Worker;

pub(super) type GraphDumpRequest = (
    PathBuf,
    crossbeam_channel::Sender<anyhow::Result<CudaGraphDumpSummary>>,
);

/// Pre-capture every whole-step bucket graph at bootstrap, while the fleet
/// is idle: every engine captures the same fixed bucket sequence, so the
/// collectives inside capture pair across the fleet by entry order (the
/// pre-capture IS the bootstrap rendezvous). A launch-ahead replay must
/// always find its graph already captured — capture records without
/// executing, so a leased replay that hit an uncaptured shape would execute
/// nothing — and pre-capturing also removes the old mid-serving capture
/// stall. Every row is a padding write into the pool's padding page.
pub(super) fn precapture_step_graphs(
    workers: &[Glm52Worker],
    pools: &[BlockPool],
    table_width: usize,
    mirrored: bool,
) -> anyhow::Result<()> {
    for bucket in GLM52_DECODE_BUCKETS {
        let mut shape = Glm52StepShape {
            bucket,
            slots: [0; GLM52_MAX_STEP_ROWS],
            active_rows: 0,
        };
        // Slot ids are row % slots: a bucket may hold more rows than slots
        // (verify spans), and every row here is padding anyway.
        for (row, dst) in shape.slots.iter_mut().enumerate().take(bucket) {
            *dst = (row % GLM52_MAX_BATCH_PER_RANK) as u8;
        }
        let inputs = [(GLM52_PADDING_STEP.token, GLM52_PADDING_STEP.position); GLM52_MAX_STEP_ROWS];
        let flags = Glm52StepFlags::plain();
        let responses = workers
            .iter()
            .enumerate()
            .map(|(rank, worker)| {
                let pool = &pools[if mirrored { 0 } else { rank }];
                let kv = padding_step_kv(bucket, table_width, pool.padding_block_id(), &inputs);
                worker.step_async(inputs, shape, kv, flags, Vec::new(), 0)
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .context("GLM5.2 graph pre-capture submit")?;
        for (rank, response) in responses.into_iter().enumerate() {
            response
                .recv()
                .map_err(|_| anyhow::anyhow!("rank dropped its pre-capture response"))
                .and_then(|result| result)
                .with_context(|| {
                    format!("GLM5.2 graph pre-capture (bucket {bucket}) on rank {rank}")
                })?;
        }
    }
    log::info!(
        "GLM5.2 whole-step graphs pre-captured: {} buckets",
        GLM52_DECODE_BUCKETS.len()
    );
    Ok(())
}

/// Export rank 0's serving graph after the all-rank pre-capture barrier.
pub(super) fn dump_rank0_decode_graph(
    workers: &[Glm52Worker],
    moe_topo: Glm52MoeTopo,
    png_path: PathBuf,
) -> anyhow::Result<CudaGraphDumpSummary> {
    let rank0 = workers
        .first()
        .context("GLM5.2 graph export requires rank 0")?;
    let bucket = graph_dump_bucket();
    let topology = match moe_topo {
        Glm52MoeTopo::Ep4 => "DP4/EP4",
        Glm52MoeTopo::Ep8 => "DP8/EP8",
        Glm52MoeTopo::Ep16 => "DP16/EP16",
        Glm52MoeTopo::Ep32 => "DP32/EP32",
        Glm52MoeTopo::Ep64 => "DP64/EP64",
        Glm52MoeTopo::Tp4 => "MoE TP4 · mirrored",
    };
    let title =
        format!("GLM5.2 whole-step decode CUDA Graph · rank 0 · {topology} · bucket {bucket}");
    rank0
        .dump_decode_graph_async(bucket, png_path, title)?
        .recv()
        .map_err(|_| anyhow::anyhow!("GLM5.2 rank 0 dropped its graph export response"))?
}

pub(super) fn graph_dump_bucket() -> usize {
    1
}
