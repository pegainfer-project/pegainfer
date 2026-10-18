use std::collections::BTreeMap;
use std::collections::HashSet;
use std::ffi::CStr;
use std::ffi::c_char;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::result as cuda_driver_result;
use cudarc::driver::sys as cuda_driver_sys;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CpuId(u16);

impl CpuId {
    pub fn new(cpu: usize) -> Result<Self> {
        let cpu =
            u16::try_from(cpu).with_context(|| format!("CPU id {cpu} does not fit in u16"))?;
        Ok(Self(cpu))
    }

    #[cfg(test)]
    fn as_u16(self) -> u16 {
        self.0
    }

    fn as_usize(self) -> usize {
        usize::from(self.0)
    }
}

impl std::fmt::Display for CpuId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NumaCpuPool {
    node: usize,
    cpus: Vec<CpuId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RankNumaNode {
    pub rank: usize,
    pub numa_node: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RankCpuSlice {
    pub rank: usize,
    numa_node: usize,
    pub cpus: Vec<CpuId>,
}

fn parse_cpu_list(cpulist: &str) -> Result<Vec<CpuId>> {
    let mut cpus = Vec::new();
    for part in cpulist.trim().split(',').filter(|part| !part.is_empty()) {
        if let Some((start, end)) = part.split_once('-') {
            let start = start
                .parse::<usize>()
                .with_context(|| format!("invalid CPU range start in {part:?}"))?;
            let end = end
                .parse::<usize>()
                .with_context(|| format!("invalid CPU range end in {part:?}"))?;
            ensure!(start <= end, "invalid descending CPU range {part:?}");
            for cpu in start..=end {
                cpus.push(CpuId::new(cpu)?);
            }
        } else {
            let cpu = part
                .parse::<usize>()
                .with_context(|| format!("invalid CPU id {part:?}"))?;
            cpus.push(CpuId::new(cpu)?);
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    Ok(cpus)
}

#[cfg(test)]
fn format_cpu_list(cpus: &[CpuId]) -> String {
    if cpus.is_empty() {
        return "[]".to_string();
    }
    let mut sorted = cpus.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    let mut ranges = Vec::new();
    let mut start = sorted[0].as_u16();
    let mut prev = start;
    for cpu in sorted.iter().skip(1).map(|cpu| cpu.as_u16()) {
        if cpu == prev + 1 {
            prev = cpu;
            continue;
        }
        push_cpu_range(&mut ranges, start, prev);
        start = cpu;
        prev = cpu;
    }
    push_cpu_range(&mut ranges, start, prev);
    ranges.join(",")
}

#[cfg(test)]
fn push_cpu_range(ranges: &mut Vec<String>, start: u16, end: u16) {
    if start == end {
        ranges.push(start.to_string());
    } else {
        ranges.push(format!("{start}-{end}"));
    }
}

pub fn current_allowed_cpus() -> Result<Vec<CpuId>> {
    unsafe {
        let mut allowed: libc::cpu_set_t = std::mem::zeroed();
        let set_size = std::mem::size_of::<libc::cpu_set_t>();
        let ret = libc::sched_getaffinity(0, set_size, std::ptr::addr_of_mut!(allowed));
        ensure!(ret == 0, "sched_getaffinity failed with errno {}", errno());

        let mut cpus = Vec::new();
        for cpu in 0..libc::CPU_SETSIZE as usize {
            if libc::CPU_ISSET(cpu, &allowed) {
                cpus.push(CpuId::new(cpu)?);
            }
        }
        ensure!(!cpus.is_empty(), "current CPU affinity mask is empty");
        Ok(cpus)
    }
}

pub fn pin_current_thread_to_cpu(cpu: CpuId) -> Result<()> {
    unsafe {
        let mut target: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut target);
        libc::CPU_SET(cpu.as_usize(), &mut target);
        let set_size = std::mem::size_of::<libc::cpu_set_t>();
        let ret = libc::pthread_setaffinity_np(
            libc::pthread_self(),
            set_size,
            std::ptr::addr_of!(target),
        );
        ensure!(
            ret == 0,
            "pthread_setaffinity_np failed for CPU {cpu} with errno {ret}"
        );
        Ok(())
    }
}

fn cuda_device_pci_bus_id(device_ordinal: usize) -> Result<String> {
    let mut buf = [c_char::default(); 32];
    cuda_driver_result::init().map_err(|err| anyhow::anyhow!("{err:?}"))?;
    let device = cuda_driver_result::device::get(device_ordinal as i32)
        .map_err(|err| anyhow::anyhow!("{err:?}"))?;
    unsafe {
        cuda_driver_sys::cuDeviceGetPCIBusId(buf.as_mut_ptr(), buf.len() as i32, device)
            .result()
            .map_err(|err| anyhow::anyhow!("{err:?}"))?;
        CStr::from_ptr(buf.as_ptr())
            .to_str()
            .map(str::to_ascii_lowercase)
            .map_err(anyhow::Error::msg)
    }
}

fn pci_numa_node(pci_bus_id: &str) -> Result<usize> {
    let raw = std::fs::read_to_string(format!("/sys/bus/pci/devices/{pci_bus_id}/numa_node"))
        .with_context(|| format!("read NUMA node for PCI device {pci_bus_id}"))?;
    let node = raw
        .trim()
        .parse::<isize>()
        .with_context(|| format!("parse NUMA node for PCI device {pci_bus_id}: {raw:?}"))?;
    usize::try_from(node).with_context(|| format!("PCI device {pci_bus_id} has NUMA node {node}"))
}

pub fn cuda_device_numa_node(device_ordinal: usize) -> Result<usize> {
    let pci_bus_id = cuda_device_pci_bus_id(device_ordinal)
        .with_context(|| format!("read PCI bus id for CUDA device {device_ordinal}"))?;
    pci_numa_node(&pci_bus_id)
        .with_context(|| format!("read NUMA node for CUDA device {device_ordinal} ({pci_bus_id})"))
}

pub fn read_numa_cpu_pool(numa_node: usize) -> Result<NumaCpuPool> {
    let cpulist =
        std::fs::read_to_string(format!("/sys/devices/system/node/node{numa_node}/cpulist"))
            .with_context(|| format!("read CPU list for NUMA node {numa_node}"))?;
    let cpus = parse_cpu_list(&cpulist)
        .with_context(|| format!("parse CPU list for NUMA node {numa_node}: {cpulist:?}"))?;
    ensure!(!cpus.is_empty(), "NUMA node {numa_node} CPU list is empty");
    Ok(NumaCpuPool {
        node: numa_node,
        cpus,
    })
}

pub fn split_rank_cpu_slices(
    pools: &[NumaCpuPool],
    ranks: &[RankNumaNode],
    allowed_cpus: &[CpuId],
    reserved_cpus: &[CpuId],
) -> Result<Vec<RankCpuSlice>> {
    let allowed = allowed_cpus.iter().copied().collect::<HashSet<_>>();
    let reserved = reserved_cpus.iter().copied().collect::<HashSet<_>>();
    let pool_by_node = pools
        .iter()
        .map(|pool| (pool.node, pool))
        .collect::<BTreeMap<_, _>>();

    let mut ranks_by_node = BTreeMap::<usize, Vec<RankNumaNode>>::new();
    for rank in ranks {
        ranks_by_node.entry(rank.numa_node).or_default().push(*rank);
    }

    let mut out = Vec::with_capacity(ranks.len());
    for (node, mut node_ranks) in ranks_by_node {
        node_ranks.sort_by_key(|rank| rank.rank);
        let pool = pool_by_node
            .get(&node)
            .with_context(|| format!("missing CPU pool for NUMA node {node}"))?;
        let cpus = pool
            .cpus
            .iter()
            .copied()
            .filter(|cpu| allowed.contains(cpu))
            .collect::<Vec<_>>();
        ensure!(
            cpus.len() >= node_ranks.len(),
            "NUMA node {node} has {} allowed CPUs for {} ranks",
            cpus.len(),
            node_ranks.len(),
        );

        let rank_count = node_ranks.len();
        for (idx, rank) in node_ranks.iter().enumerate() {
            let start = idx * cpus.len() / rank_count;
            let end = (idx + 1) * cpus.len() / rank_count;
            let slice = cpus[start..end]
                .iter()
                .copied()
                .filter(|cpu| !reserved.contains(cpu))
                .collect::<Vec<_>>();
            ensure!(
                !slice.is_empty(),
                "rank {} got an empty CPU slice on NUMA node {node}",
                rank.rank,
            );
            out.push(RankCpuSlice {
                rank: rank.rank,
                numa_node: node,
                cpus: slice,
            });
        }
    }

    out.sort_by_key(|slice| slice.rank);
    Ok(out)
}

fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpus(range: std::ops::Range<usize>) -> Vec<CpuId> {
        range.map(|cpu| CpuId::new(cpu).unwrap()).collect()
    }

    #[test]
    fn parses_linux_cpu_list() {
        let parsed = parse_cpu_list("0-3,8,10-12\n").unwrap();
        assert_eq!(
            parsed,
            [0, 1, 2, 3, 8, 10, 11, 12]
                .into_iter()
                .map(|cpu| CpuId::new(cpu).unwrap())
                .collect::<Vec<_>>()
        );
        assert_eq!(format_cpu_list(&parsed), "0-3,8,10-12");
    }

    #[test]
    fn splits_single_numa_single_rank_after_reserved_cpus() {
        let pools = [NumaCpuPool {
            node: 0,
            cpus: cpus(0..8),
        }];
        let ranks = [RankNumaNode {
            rank: 0,
            numa_node: 0,
        }];
        let slices = split_rank_cpu_slices(
            &pools,
            &ranks,
            &cpus(0..8),
            &[CpuId::new(0).unwrap(), CpuId::new(1).unwrap()],
        )
        .unwrap();
        assert_eq!(slices[0].cpus, cpus(2..8));
    }

    #[test]
    fn splits_one_numa_evenly_before_removing_reserved_cpus() {
        let pools = [NumaCpuPool {
            node: 0,
            cpus: cpus(0..64),
        }];
        let ranks = (0..4)
            .map(|rank| RankNumaNode { rank, numa_node: 0 })
            .collect::<Vec<_>>();
        let slices = split_rank_cpu_slices(
            &pools,
            &ranks,
            &cpus(0..64),
            &[CpuId::new(0).unwrap(), CpuId::new(1).unwrap()],
        )
        .unwrap();
        assert_eq!(slices[0].cpus, cpus(2..16));
        assert_eq!(slices[1].cpus, cpus(16..32));
        assert_eq!(slices[2].cpus, cpus(32..48));
        assert_eq!(slices[3].cpus, cpus(48..64));
    }
}
