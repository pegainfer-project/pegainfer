use super::scheduler::dp::DpRankState;

#[derive(Clone, Copy)]
pub(super) struct DpLoadBalancer {
    dp_world: usize,
}

impl DpLoadBalancer {
    pub(super) fn new(dp_world: usize) -> Self {
        Self { dp_world }
    }

    pub(super) fn pick_rank(self, ranks: &[DpRankState]) -> Option<usize> {
        debug_assert_eq!(ranks.len(), self.dp_world);
        ranks
            .iter()
            .enumerate()
            .filter(|(_, r)| r.has_free_slot())
            .max_by_key(|(_, r)| r.free_slot_count())
            .map(|(i, _)| i)
    }
}
