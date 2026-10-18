use std::collections::BTreeSet;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use pegainfer_core::cpu_topology::CpuId;
use pegainfer_core::cpu_topology::RankCpuSlice;
use pegainfer_core::cpu_topology::RankNumaNode;
use pegainfer_core::cpu_topology::cuda_device_numa_node;
use pegainfer_core::cpu_topology::current_allowed_cpus;
use pegainfer_core::cpu_topology::pin_current_thread_to_cpu;
use pegainfer_core::cpu_topology::read_numa_cpu_pool;
use pegainfer_core::cpu_topology::split_rank_cpu_slices;

const SYSTEM_RESERVED_CPU: usize = 0;
const SCHEDULER_CPU: usize = 1;

#[derive(Clone, Debug)]
pub(super) struct KimiRankThreadPlacement {
    pub(super) rank: usize,
    rank_worker_cpu: CpuId,
}

#[derive(Clone, Debug)]
pub(crate) struct KimiRankThreadPlacementPlan {
    scheduler_cpu: Option<CpuId>,
    ranks: Vec<KimiRankThreadPlacement>,
}

impl KimiRankThreadPlacementPlan {
    pub(super) fn for_devices(devices: &[usize]) -> Result<Self> {
        let scheduler_cpu = scheduler_cpu()?;
        let allowed_cpus = current_allowed_cpus()?;
        let reserved_cpus = [CpuId::new(SYSTEM_RESERVED_CPU)?, CpuId::new(SCHEDULER_CPU)?];

        let mut rank_nodes = Vec::with_capacity(devices.len());
        let mut numa_nodes = BTreeSet::new();
        for (rank, &device_ordinal) in devices.iter().enumerate() {
            let numa_node = cuda_device_numa_node(device_ordinal).with_context(|| {
                format!("read NUMA node for Kimi rank {rank} cuda:{device_ordinal}")
            })?;
            rank_nodes.push(RankNumaNode { rank, numa_node });
            numa_nodes.insert(numa_node);
        }

        let pools = numa_nodes
            .iter()
            .map(|&node| read_numa_cpu_pool(node))
            .collect::<Result<Vec<_>>>()?;
        let slices = split_rank_cpu_slices(&pools, &rank_nodes, &allowed_cpus, &reserved_cpus)?;
        ensure!(
            slices.len() == devices.len(),
            "built {} Kimi CPU slices for {} devices",
            slices.len(),
            devices.len()
        );

        let mut ranks = Vec::with_capacity(devices.len());
        for rank in 0..devices.len() {
            let slice = slices
                .iter()
                .find(|slice| slice.rank == rank)
                .with_context(|| format!("missing Kimi CPU slice for rank {rank}"))?;
            ranks.push(rank_thread_placement(slice)?);
        }
        Ok(Self {
            scheduler_cpu,
            ranks,
        })
    }

    fn scheduler_cpu(&self) -> Option<CpuId> {
        self.scheduler_cpu
    }

    pub(super) fn rank(&self, rank: usize) -> Result<KimiRankThreadPlacement> {
        self.ranks
            .get(rank)
            .cloned()
            .with_context(|| format!("missing Kimi thread placement for rank {rank}"))
    }
}

fn rank_thread_placement(slice: &RankCpuSlice) -> Result<KimiRankThreadPlacement> {
    let rank_worker_cpu = *slice
        .cpus
        .first()
        .with_context(|| format!("Kimi rank {} has empty CPU slice", slice.rank))?;
    Ok(KimiRankThreadPlacement {
        rank: slice.rank,
        rank_worker_cpu,
    })
}

pub(super) fn pin_scheduler_thread(placement: &KimiRankThreadPlacementPlan) {
    let Some(cpu) = placement.scheduler_cpu() else {
        return;
    };
    pin_current_thread_to_cpu(cpu)
        .unwrap_or_else(|err| panic!("failed to pin Kimi-K2 scheduler to CPU {cpu}: {err:#}"));
}

pub(super) fn pin_rank_worker_thread(placement: &KimiRankThreadPlacement) {
    pin_current_thread_to_cpu(placement.rank_worker_cpu).unwrap_or_else(|err| {
        panic!(
            "failed to pin Kimi rank worker {} to CPU {}: {err:#}",
            placement.rank, placement.rank_worker_cpu
        )
    });
}

fn scheduler_cpu() -> Result<Option<CpuId>> {
    let cpu = CpuId::new(SCHEDULER_CPU)?;
    let allowed = current_allowed_cpus()?;
    Ok(allowed.contains(&cpu).then_some(cpu))
}
