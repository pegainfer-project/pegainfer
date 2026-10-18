//! One rank's native-MTP draft round.

use anyhow::Context as _;

use super::RankSlots;
use crate::model::GLM52_DECODE_BUCKETS;
use crate::runner::Glm52MtpAppend;
use crate::runner::Glm52MtpRound;
use crate::runner::Glm52Worker;

/// Native MTP is an EP collective, unlike DSpark, and its collective chain
/// is FIXED: every rank runs the full round (one context forward + the
/// proposal iterations) every step, sized by its OWN appends/proposals —
/// a rank with no work enters with padding rows. No round-kind negotiation,
/// no cross-rank bucket agreement: the per-step collective count is a
/// constant of the code, never a function of fleet state (the free-running
/// fixed-chain discipline, `docs/models/glm52/free-running-dp.md` §4).
pub(super) fn run_mtp_round(
    rank: usize,
    worker: &Glm52Worker,
    slots: &mut RankSlots,
    source_bucket: usize,
    pending_resets: &mut Vec<usize>,
    appends: Vec<Glm52MtpAppend>,
    proposals: Vec<(usize, u32, usize)>,
) -> anyhow::Result<()> {
    let pick_bucket = |rows: usize| {
        GLM52_DECODE_BUCKETS
            .into_iter()
            .find(|&bucket| bucket >= rows.max(1))
            .with_context(|| format!("GLM5.2 native MTP row count {rows} exceeds bucket capacity"))
    };
    #[cfg(test)]
    let probe = crate::freerun_probe::enabled().then(|| {
        (
            appends.is_empty() && proposals.is_empty(),
            std::time::Instant::now(),
        )
    });

    let slots_for_rank: Vec<usize> = proposals.into_iter().map(|(slot, _, _)| slot).collect();
    let resets = std::mem::take(pending_resets);
    let round = Glm52MtpRound {
        source_bucket,
        context_bucket: pick_bucket(appends.len())?,
        draft_bucket: pick_bucket(slots_for_rank.len())?,
        resets,
        appends,
        proposal_slots: slots_for_rank.clone(),
    };
    let rx = worker
        .mtp_draft_async(round)
        .map_err(|err| err.context(format!("GLM5.2 rank {rank} MTP draft submission")))?;
    let spans = rx
        .recv()
        .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} MTP draft dropped its response"))?
        .map_err(|err| err.context(format!("GLM5.2 rank {rank} MTP draft")))?;
    anyhow::ensure!(
        spans.len() == slots_for_rank.len(),
        "GLM5.2 rank {rank} MTP draft returned {} spans for {} proposals",
        spans.len(),
        slots_for_rank.len()
    );
    #[cfg(test)]
    if let Some((empty, started)) = probe {
        crate::freerun_probe::record_mtp_round(rank, empty, started.elapsed());
    }
    for (slot_id, span) in slots_for_rank.into_iter().zip(spans) {
        if let Some(active) = slots[slot_id].as_mut() {
            #[cfg(test)]
            super::slot::record_mtp_proposal(active.req.request_id.as_deref(), &span);
            active
                .state
                .set_drafts(span.to_vec(), crate::mtp::glm52_mtp_draft_len());
        }
    }
    Ok(())
}
