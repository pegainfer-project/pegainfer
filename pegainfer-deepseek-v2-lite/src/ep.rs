use std::ops::Range;

use anyhow::Result;
use anyhow::ensure;

use crate::Config;

const SUPPORTED_EP_SIZE: usize = 2;
pub(crate) const SUPPORTED_ROUTED_EXPERTS: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ExpertParallelConfig {
    rank: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExpertParallelLayout {
    rank: usize,
    ep_size: usize,
    experts_per_rank: usize,
    owned: Range<usize>,
}

impl ExpertParallelConfig {
    pub(crate) fn ep2(rank: usize) -> Self {
        Self { rank }
    }

    pub(crate) fn validate_for(self, config: &Config) -> Result<ExpertParallelLayout> {
        ensure!(
            self.rank < SUPPORTED_EP_SIZE,
            "DeepSeek-V2-Lite EP rank {} must be < ep_size {}",
            self.rank,
            SUPPORTED_EP_SIZE
        );
        ensure!(
            config.n_routed_experts == SUPPORTED_ROUTED_EXPERTS,
            "DeepSeek-V2-Lite EP gate expects 64 routed experts, got {}",
            config.n_routed_experts
        );
        ensure!(
            config.n_routed_experts.is_multiple_of(SUPPORTED_EP_SIZE),
            "n_routed_experts={} must divide evenly by ep_size={}",
            config.n_routed_experts,
            SUPPORTED_EP_SIZE
        );
        let experts_per_rank = config.n_routed_experts / SUPPORTED_EP_SIZE;
        ensure!(
            experts_per_rank == 32,
            "DeepSeek-V2-Lite EP=2 expects 32 local routed experts, got {}",
            experts_per_rank
        );
        let start = self.rank * experts_per_rank;
        Ok(ExpertParallelLayout {
            rank: self.rank,
            ep_size: SUPPORTED_EP_SIZE,
            experts_per_rank,
            owned: start..start + experts_per_rank,
        })
    }
}

impl ExpertParallelLayout {
    pub(crate) fn rank(&self) -> usize {
        self.rank
    }

    pub(crate) fn ep_size(&self) -> usize {
        self.ep_size
    }

    pub(crate) fn experts_per_rank(&self) -> usize {
        self.experts_per_rank
    }

    pub(crate) fn owned_experts(&self) -> Range<usize> {
        self.owned.clone()
    }

    fn owns(&self, expert: usize) -> bool {
        self.owned.contains(&expert)
    }

    pub(crate) fn owner_rank(&self, expert: usize) -> Result<usize> {
        ensure!(
            expert < SUPPORTED_ROUTED_EXPERTS,
            "routed expert {expert} out of range 0..{}",
            SUPPORTED_ROUTED_EXPERTS
        );
        Ok(expert / self.experts_per_rank)
    }

    pub(crate) fn local_expert(&self, expert: usize) -> Result<usize> {
        ensure!(
            self.owns(expert),
            "routed expert {expert} is owned by rank {}, not local rank {}",
            self.owner_rank(expert)?,
            self.rank
        );
        Ok(expert - self.owned.start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::test_lite_config;

    #[test]
    fn ep2_expert_ranges_are_fixed() {
        let config = test_lite_config();
        config.validate_lite().unwrap();
        let rank0 = ExpertParallelConfig::ep2(0).validate_for(&config).unwrap();
        let rank1 = ExpertParallelConfig::ep2(1).validate_for(&config).unwrap();
        assert_eq!(rank0.owned_experts(), 0..32);
        assert_eq!(rank1.owned_experts(), 32..64);
        assert_eq!(rank0.local_expert(31).unwrap(), 31);
        assert_eq!(rank1.local_expert(32).unwrap(), 0);
        assert!(rank0.local_expert(32).is_err());
    }

    #[test]
    fn owner_rank_rejects_out_of_range_experts() {
        let config = test_lite_config();
        let rank0 = ExpertParallelConfig::ep2(0).validate_for(&config).unwrap();
        assert_eq!(rank0.owner_rank(0).unwrap(), 0);
        assert_eq!(rank0.owner_rank(31).unwrap(), 0);
        assert_eq!(rank0.owner_rank(32).unwrap(), 1);
        assert_eq!(rank0.owner_rank(63).unwrap(), 1);
        assert!(rank0.owner_rank(64).is_err());
    }
}
