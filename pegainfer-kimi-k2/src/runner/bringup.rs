use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use bytesize::ByteSize;
use crossbeam_channel::bounded;
use log::debug;
use log::info;
use pegainfer_frontend::engine::EngineHandle;
use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::EpBackend;
use pegainfer_frontend::engine::SubmittedRequest;
use pegainfer_frontend::parallel::ParallelConfig;
use pegainfer_kv_cache::BlockPool;
use tokio::sync::mpsc;

use crate::config::KIMI_K2_VOCAB;
use crate::config::KimiK2ParallelShape;
use crate::runner::affinity::pin_scheduler_thread;
use crate::runner::config::KimiK2RunnerConfig;
use crate::runner::executor::ForwardExecutor;
use crate::runner::executor::Tp1Dp8ForwardExecutor;
use crate::runner::executor::Tp8Dp1ForwardExecutor;
use crate::runner::load_balancer::DpLoadBalancer;
use crate::runner::scheduler::KimiK2Scheduler;
use crate::runner::scheduler::dp::DpCoordinator;
use crate::runner::worker::KIMI_KV_PAGE_SIZE;
use crate::runner::worker::KimiRankWeightLoadReport;
use crate::runner::worker::KimiRankWorker;
use crate::runner::worker::build_placements;
use crate::weights::KimiRankGpuContext;
use crate::weights::KimiRankSlicedLoadPlan;
use crate::weights::ensure_text_only_model_index;

/// TP8 replicates the KV pool on every rank: 8192 pages × 16 tokens ×
/// (576 ckv + 64 kpe) bf16 ≈ 9.2 GiB per rank — the same footprint as the
/// old static bs64 arena, now shared across requests instead of sliced
/// into fixed 2048-token slots.
const KIMI_TP8_KV_POOL_PAGES: usize = 8192;
/// DP shards requests across ranks, so each rank only backs its own slots:
/// 1024 pages ≈ 1.15 GiB per rank (the old per-rank arena footprint).
const KIMI_DP_KV_POOL_PAGES_PER_RANK: usize = 1024;

pub(crate) fn start_engine(model_path: &Path, options: &EngineLoadOptions) -> Result<EngineHandle> {
    let parallel = resolve_parallel_config(options);
    info!(
        "resolving engine startup: model_path={}, tp_size={}, dp_size={}, ep_size={}, ep_backend={:?}, devices={:?}",
        model_path.display(),
        parallel.tp_world(),
        parallel.dp_world(),
        parallel.ep_world(),
        options.ep_backend,
        options.device_ordinals
    );
    ensure!(
        options.device_ordinals.len() == parallel.ep_world(),
        "Kimi-K2 {:?} requires {} devices, got {:?}",
        parallel,
        parallel.ep_world(),
        options.device_ordinals
    );

    match (parallel.tp_world(), parallel.dp_world()) {
        (8, 1) => start_engine_tp8_dp1(model_path, options, parallel),
        (1, 8) => start_engine_tp1_dp8(model_path, options, parallel),
        _ => bail!(
            "Kimi-K2 TP{}/DP{} not yet supported (v1: TP8DP1 or TP1DP8)",
            parallel.tp_world(),
            parallel.dp_world()
        ),
    }
}

fn resolve_parallel_config(options: &EngineLoadOptions) -> ParallelConfig {
    options
        .parallel_config
        .unwrap_or_else(|| ParallelConfig::new(8, 1))
}

fn build_runner_config(
    model_path: &Path,
    options: &EngineLoadOptions,
    parallel: ParallelConfig,
    shape: KimiK2ParallelShape,
    kv_pool_pages: usize,
) -> Result<KimiK2RunnerConfig> {
    let started = Instant::now();
    info!("start build runner config");
    let mut weight_manifest = ensure_text_only_model_index(model_path)?;
    weight_manifest = weight_manifest.with_parallel_shape(shape)?;
    let placements = build_placements(&options.device_ordinals)?;
    let thread_placement = crate::runner::affinity::KimiRankThreadPlacementPlan::for_devices(
        &options.device_ordinals,
    )?;
    let rank_weight_names = (0..placements.len())
        .map(|rank| weight_manifest.rank_weight_names(rank))
        .collect::<Result<Vec<_>>>()?;
    let rank_sliced_load_plans = (0..placements.len())
        .map(|rank| weight_manifest.rank_sliced_load_plan(rank))
        .collect::<Result<Vec<_>>>()?;
    let config = KimiK2RunnerConfig {
        model_path: model_path.to_path_buf(),
        parallel,
        local_dims: shape.local_dims(),
        rank_weight_names,
        rank_sliced_load_plans,
        placements,
        thread_placement,
        enable_cuda_graph: options.enable_cuda_graph,
        kv_pool_pages,
    };
    info!(
        "build runner config cost {:.2}s: ranks={}",
        started.elapsed().as_secs_f64(),
        config.placements.len()
    );
    debug!(
        "runner config detail: tensors_per_rank={:?}",
        config
            .rank_sliced_load_plans
            .iter()
            .map(|plan| plan.tensor_count)
            .collect::<Vec<_>>()
    );
    Ok(config)
}

fn start_engine_tp8_dp1(
    model_path: &Path,
    options: &EngineLoadOptions,
    parallel: ParallelConfig,
) -> Result<EngineHandle> {
    info!("starting TP8/DP1 engine");
    ensure!(
        options.ep_backend == EpBackend::Nccl,
        "Kimi-K2 TP8/DP1 routes MoE through the NCCL backend; --ep-backend={:?} has no TP8 path",
        options.ep_backend
    );
    let config = build_runner_config(
        model_path,
        options,
        parallel,
        KimiK2ParallelShape::tp8_ep8(),
        KIMI_TP8_KV_POOL_PAGES,
    )?;
    let stop_token_ids = pegainfer_core::eos::eos_token_ids(model_path, KIMI_K2_VOCAB)?;
    info!("stop tokens: {stop_token_ids:?}");
    let executor = build_tp8_dp1_executor(&config)?;
    let pool = BlockPool::new(KIMI_KV_PAGE_SIZE, config.kv_pool_pages)?;

    let (submit_tx, submit_rx) = mpsc::unbounded_channel::<SubmittedRequest>();
    let (init_tx, init_rx) = bounded::<Result<()>>(1);
    let scheduler_handle = thread::Builder::new()
        .name("kimi-k2-scheduler".into())
        .spawn(move || {
            pin_scheduler_thread(&config.thread_placement);
            let mut scheduler = match KimiK2Scheduler::new(executor, stop_token_ids, pool) {
                Ok(scheduler) => scheduler,
                Err(err) => {
                    let _ = init_tx.send(Err(err));
                    return;
                }
            };
            let _ = init_tx.send(Ok(()));
            scheduler.run(submit_rx);
        })
        .map_err(|err| anyhow::anyhow!("failed to spawn Kimi-K2 scheduler thread: {err}"))?;
    init_rx
        .recv()
        .map_err(|err| anyhow::anyhow!("Kimi-K2 scheduler init channel closed: {err}"))??;
    Ok(EngineHandle::new_with_join_handle(
        submit_tx,
        scheduler_handle,
    ))
}

fn start_engine_tp1_dp8(
    model_path: &Path,
    options: &EngineLoadOptions,
    parallel: ParallelConfig,
) -> Result<EngineHandle> {
    info!("starting TP1/DP8 engine");
    ensure!(
        options.ep_backend == EpBackend::DeepEp,
        "Kimi-K2 TP1/DP8 requires --ep-backend=deepep"
    );
    let dp_world = parallel.dp_world();
    let config = build_runner_config(
        model_path,
        options,
        parallel,
        KimiK2ParallelShape::tp1_dp8(),
        KIMI_DP_KV_POOL_PAGES_PER_RANK,
    )?;
    let stop_token_ids = pegainfer_core::eos::eos_token_ids(model_path, KIMI_K2_VOCAB)?;
    info!("stop tokens: {stop_token_ids:?}");
    let executors = build_tp1_dp8_executors(&config)?;
    let pools = (0..dp_world)
        .map(|_| BlockPool::new(KIMI_KV_PAGE_SIZE, config.kv_pool_pages))
        .collect::<Result<Vec<_>>>()?;
    let coordinator = DpCoordinator::new(executors, stop_token_ids, options.seed, pools);
    let lb = DpLoadBalancer::new(dp_world);

    let (submit_tx, submit_rx) = mpsc::unbounded_channel::<SubmittedRequest>();
    let (init_tx, init_rx) = bounded::<Result<()>>(1);
    let coord_handle = thread::Builder::new()
        .name("kimi-k2-dp-coord".into())
        .spawn(move || {
            let _ = init_tx.send(Ok(()));
            coordinator.run(submit_rx, lb);
        })
        .map_err(|err| anyhow::anyhow!("failed to spawn Kimi-K2 DP coordinator: {err}"))?;
    init_rx
        .recv()
        .map_err(|err| anyhow::anyhow!("Kimi-K2 DP coordinator init failed: {err}"))??;

    info!("TP1 DP{dp_world} coordinated engine started");
    Ok(EngineHandle::new_with_join_handle(submit_tx, coord_handle))
}

fn build_tp8_dp1_executor(config: &KimiK2RunnerConfig) -> Result<Box<dyn ForwardExecutor + Send>> {
    let started = Instant::now();
    info!("start build TP8/DP1 executor");
    let workers = spawn_workers(config)?;
    let weight_reports =
        maybe_load_rank_weights(&config.model_path, &config.rank_sliced_load_plans, &workers)?;
    init_tp_nccl(&workers)?;
    let executor: Box<dyn ForwardExecutor + Send> = Box::new(Tp8Dp1ForwardExecutor {
        workers,
        weight_reports,
    });
    info!(
        "build TP8/DP1 executor cost {:.2}s",
        started.elapsed().as_secs_f64()
    );
    Ok(executor)
}

fn build_tp1_dp8_executors(
    config: &KimiK2RunnerConfig,
) -> Result<Vec<Box<dyn ForwardExecutor + Send>>> {
    let started = Instant::now();
    info!("start build TP1/DP8 executors");
    let workers = spawn_workers(config)?;
    let weight_reports =
        maybe_load_rank_weights(&config.model_path, &config.rank_sliced_load_plans, &workers)?;
    install_deepep_backends(&workers)?;

    let mut executors: Vec<Box<dyn ForwardExecutor + Send>> =
        Vec::with_capacity(config.parallel.dp_world());
    for (worker, weight_report) in workers.into_iter().zip(weight_reports) {
        executors.push(Box::new(Tp1Dp8ForwardExecutor {
            worker,
            weight_report,
        }));
    }
    info!(
        "build TP1/DP8 executors cost {:.2}s",
        started.elapsed().as_secs_f64()
    );
    Ok(executors)
}

fn maybe_load_rank_weights(
    model_path: &Path,
    load_plans: &[KimiRankSlicedLoadPlan],
    workers: &[KimiRankWorker],
) -> Result<Vec<KimiRankWeightLoadReport>> {
    let started = Instant::now();
    info!("start load rank weights: ranks={}", workers.len());
    ensure_weight_payload_available(model_path, load_plans)?;
    let receivers = workers
        .iter()
        .map(|worker| worker.load_sliced_weights_async(model_path))
        .collect::<Result<Vec<_>>>()?;
    let mut reports = Vec::with_capacity(workers.len());
    for (worker, receiver) in workers.iter().zip(receivers) {
        let rank = worker.placement().rank;
        let report = receiver
            .recv()
            .map_err(|_| {
                anyhow::anyhow!(
                    "Kimi-K2 rank {} dropped weight load response",
                    worker.placement().rank
                )
            })?
            .with_context(|| {
                format!(
                    "Kimi-K2 rank {} sliced weight load failed",
                    worker.placement().rank
                )
            })?;
        debug!(
            "rank {rank} weights loaded: tensors={}, bytes={}, expert_layers={}",
            report.tensor_count,
            ByteSize(report.total_bytes as u64),
            report.expert_kernel_layers
        );
        reports.push(report);
    }
    info!(
        "load rank weights cost {:.2}s: ranks={}",
        started.elapsed().as_secs_f64(),
        reports.len()
    );
    Ok(reports)
}

fn spawn_workers(config: &KimiK2RunnerConfig) -> Result<Vec<KimiRankWorker>> {
    let started = Instant::now();
    let n = config.placements.len();
    info!("start spawn rank workers: ranks={n}");
    ensure!(
        config.rank_weight_names.len() == n && config.rank_sliced_load_plans.len() == n,
        "Kimi-K2 names/sliced counts must match {} placements",
        n
    );
    let contexts = config
        .placements
        .iter()
        .map(|placement| KimiRankGpuContext::new(placement.device_ordinal))
        .collect::<Result<Vec<_>>>()?;
    let collective_barrier = Arc::new(Barrier::new(config.parallel.tp_world()));
    let mut workers = Vec::with_capacity(n);
    for (((&placement, weight_names), sliced_load_plan), ctx) in config
        .placements
        .iter()
        .zip(config.rank_weight_names.iter().cloned())
        .zip(config.rank_sliced_load_plans.iter().cloned())
        .zip(contexts)
    {
        let thread_placement = config.thread_placement.rank(placement.rank)?;
        let worker = KimiRankWorker::spawn(
            placement,
            weight_names,
            sliced_load_plan,
            thread_placement,
            config.local_dims,
            ctx,
            Arc::clone(&collective_barrier),
            config.enable_cuda_graph,
            config.kv_pool_pages,
        )?;
        debug_assert_eq!(worker.placement(), placement);
        workers.push(worker);
    }
    info!(
        "spawn rank workers cost {:.2}s: ranks={}",
        started.elapsed().as_secs_f64(),
        workers.len()
    );
    Ok(workers)
}

fn init_tp_nccl(workers: &[KimiRankWorker]) -> Result<()> {
    let started = Instant::now();
    info!("start TP NCCL init: ranks={}", workers.len());
    let nccl_id = cudarc::nccl::safe::Id::new()
        .map_err(|err| anyhow::anyhow!("Kimi TP NCCL unique id creation failed: {err:?}"))?;
    let comm_receivers = workers
        .iter()
        .map(|worker| worker.init_tp_comm_async(nccl_id, workers.len()))
        .collect::<Result<Vec<_>>>()?;
    for (rank, receiver) in comm_receivers.into_iter().enumerate() {
        receiver
            .recv()
            .map_err(|_| anyhow::anyhow!("Kimi rank {rank} dropped TP comm init response"))?
            .with_context(|| format!("Kimi rank {rank} TP comm init"))?;
    }
    info!(
        "TP NCCL init cost {:.2}s: ranks={}",
        started.elapsed().as_secs_f64(),
        workers.len()
    );
    Ok(())
}

/// Collective DeepEP bootstrap: rank 0's unique id fans out to every rank
/// worker, which all enter the NCCL communicator + symmetric-window create
/// together. Send to all ranks before waiting on any response — each worker
/// blocks inside the collective until the last rank joins.
fn install_deepep_backends(workers: &[KimiRankWorker]) -> Result<()> {
    let started = Instant::now();
    info!("start install DeepEP EP backend: ranks={}", workers.len());
    let unique_id = pegainfer_kernels::ops::deepep_unique_id()?;
    let receivers = workers
        .iter()
        .map(|worker| worker.enable_deepep_async(unique_id, workers.len()))
        .collect::<Result<Vec<_>>>()?;
    for (rank, receiver) in receivers.into_iter().enumerate() {
        receiver
            .recv()
            .map_err(|_| anyhow::anyhow!("Kimi rank {rank} dropped DeepEP enable response"))?
            .with_context(|| format!("Kimi rank {rank} DeepEP enable"))?;
    }
    info!(
        "DeepEP EP backend install cost {:.2}s: ranks={}",
        started.elapsed().as_secs_f64(),
        workers.len()
    );
    Ok(())
}

fn ensure_weight_payload_available(
    model_path: &Path,
    load_plans: &[KimiRankSlicedLoadPlan],
) -> Result<()> {
    let shards = load_plans
        .iter()
        .flat_map(|plan| plan.shards.iter().map(|shard| shard.shard.as_str()))
        .collect::<BTreeSet<_>>();
    let existing = shards
        .iter()
        .filter(|shard| model_path.join(shard).exists())
        .count();
    if existing != shards.len() {
        bail!(
            "Kimi-K2 weight payload under {} is incomplete: found {existing}/{} planned shards",
            model_path.display(),
            shards.len()
        );
    }
    Ok(())
}
