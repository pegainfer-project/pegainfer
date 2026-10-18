use super::ActiveRequestState;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DecodePositionGroupPlan {
    pub(super) position: usize,
    pub(super) indices: Vec<usize>,
}

pub(super) struct DecodePositionGroup {
    pub(super) position: usize,
    pub(super) rows: Vec<(usize, ActiveRequestState)>,
}

pub(super) fn decode_position_groups_for_positions(
    positions: &[usize],
) -> Vec<DecodePositionGroupPlan> {
    let mut groups: Vec<DecodePositionGroupPlan> = Vec::new();
    for (idx, position) in positions.iter().copied().enumerate() {
        if let Some(group) = groups.iter_mut().find(|group| group.position == position) {
            group.indices.push(idx);
        } else {
            groups.push(DecodePositionGroupPlan {
                position,
                indices: vec![idx],
            });
        }
    }
    groups
}

pub(super) fn common_decode_position(active: &[ActiveRequestState]) -> Option<usize> {
    let first = active.first()?.next_decode_position();
    active
        .iter()
        .all(|state| state.next_decode_position() == first)
        .then_some(first)
}

pub(super) fn take_decode_position_groups(
    active: &mut Vec<ActiveRequestState>,
) -> Vec<DecodePositionGroup> {
    let positions: Vec<_> = active
        .iter()
        .map(ActiveRequestState::next_decode_position)
        .collect();
    let plans = decode_position_groups_for_positions(&positions);
    let mut rows: Vec<_> = active.drain(..).map(Some).collect();
    plans
        .into_iter()
        .map(|plan| DecodePositionGroup {
            position: plan.position,
            rows: plan
                .indices
                .into_iter()
                .map(|idx| {
                    (
                        idx,
                        rows[idx]
                            .take()
                            .expect("decode position group indices come from active rows"),
                    )
                })
                .collect(),
        })
        .collect()
}

pub(super) fn restore_surviving_rows(
    mut survivors: Vec<(usize, ActiveRequestState)>,
) -> Vec<ActiveRequestState> {
    survivors.sort_by_key(|(idx, _)| *idx);
    survivors
        .into_iter()
        .map(|(_, state)| state)
        .collect::<Vec<_>>()
}
