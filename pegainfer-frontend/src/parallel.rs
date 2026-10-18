//! Model-agnostic parallel topology types.

/// Pure parallel topology. No model-specific fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(clippy::struct_field_names)]
pub struct ParallelConfig {
    tp_size: usize,
    dp_size: usize,
    ep_size: usize,
}

impl ParallelConfig {
    #[must_use]
    pub fn new(tp_world: usize, dp_world: usize) -> Self {
        assert!(tp_world > 0 && dp_world > 0);
        // TODO: support topologies without EP over every rank.
        Self {
            tp_size: tp_world,
            dp_size: dp_world,
            ep_size: tp_world * dp_world,
        }
    }

    #[must_use]
    pub fn tp_world(&self) -> usize {
        self.tp_size
    }

    #[must_use]
    pub fn dp_world(&self) -> usize {
        self.dp_size
    }

    #[must_use]
    pub fn ep_world(&self) -> usize {
        self.ep_size
    }

    #[must_use]
    pub fn tp_rank(&self, global_rank: usize) -> usize {
        assert!(global_rank < self.ep_size);
        global_rank % self.tp_size
    }

    #[must_use]
    pub fn ep_rank(&self, global_rank: usize) -> usize {
        assert!(global_rank < self.ep_size);
        global_rank
    }
}
