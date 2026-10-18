use anyhow::Result;

use crate::ep::ExpertParallelLayout;

#[derive(Clone, Debug)]
pub(super) struct MoeRouteEntry {
    pub(super) token: usize,
    pub(super) global_expert: usize,
    pub(super) owner_rank: usize,
    pub(super) weight: f32,
}

#[derive(Clone, Debug)]
pub(super) struct MoeRoutePlan {
    entries: Vec<MoeRouteEntry>,
    // Counts are relative to the rank carried by the layout used to build the plan.
    local_routes: usize,
    remote_routes: usize,
}

impl MoeRoutePlan {
    pub(super) fn from_topk_routes(
        routes: &[Vec<(usize, f32)>],
        layout: &ExpertParallelLayout,
    ) -> Result<Self> {
        let route_count = routes.iter().map(Vec::len).sum();
        let mut entries = Vec::with_capacity(route_count);
        let mut local_routes = 0usize;
        let mut remote_routes = 0usize;

        for (token, token_routes) in routes.iter().enumerate() {
            for &(global_expert, weight) in token_routes {
                let owner_rank = layout.owner_rank(global_expert)?;
                if owner_rank == layout.rank() {
                    local_routes += 1;
                } else {
                    remote_routes += 1;
                }
                entries.push(MoeRouteEntry {
                    token,
                    global_expert,
                    owner_rank,
                    weight,
                });
            }
        }

        Ok(Self {
            entries,
            local_routes,
            remote_routes,
        })
    }

    pub(super) fn entries(&self) -> &[MoeRouteEntry] {
        &self.entries
    }

    pub(super) fn route_count(&self) -> usize {
        self.entries.len()
    }

    pub(super) fn local_routes(&self) -> usize {
        self.local_routes
    }

    pub(super) fn remote_routes(&self) -> usize {
        self.remote_routes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::test_lite_config;
    use crate::ep::ExpertParallelConfig;

    #[test]
    fn route_plan_preserves_token_major_expert_order() {
        let config = test_lite_config();
        let layout = ExpertParallelConfig::ep2(0).validate_for(&config).unwrap();
        let routes = vec![vec![(7, 0.7), (33, 0.3)], vec![(2, 0.4), (41, 0.6)]];

        let plan = MoeRoutePlan::from_topk_routes(&routes, &layout).unwrap();
        let entries: Vec<_> = plan
            .entries()
            .iter()
            .map(|entry| {
                (
                    entry.token,
                    entry.global_expert,
                    entry.owner_rank,
                    entry.weight,
                )
            })
            .collect();

        assert_eq!(
            entries,
            vec![
                (0, 7, 0, 0.7),
                (0, 33, 1, 0.3),
                (1, 2, 0, 0.4),
                (1, 41, 1, 0.6)
            ]
        );
        assert_eq!(plan.route_count(), 4);
        assert_eq!(plan.local_routes(), 2);
        assert_eq!(plan.remote_routes(), 2);
    }

    #[test]
    fn route_plan_rejects_out_of_range_experts() {
        let config = test_lite_config();
        let layout = ExpertParallelConfig::ep2(0).validate_for(&config).unwrap();
        let err = MoeRoutePlan::from_topk_routes(&[vec![(64, 1.0)]], &layout).unwrap_err();

        assert!(
            err.to_string().contains("out of range"),
            "unexpected error: {err:#}"
        );
    }
}
