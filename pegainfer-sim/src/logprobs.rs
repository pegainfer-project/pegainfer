use pegainfer_frontend::engine::PromptEcho;
use pegainfer_frontend::engine::TokenLogprob;

const PROMPT_LOGPROB: f32 = -2.5;
const PROMPT_RANK: u32 = 3;
const COMPLETION_LOGPROB: f32 = -0.25;
const FIRST_ALTERNATIVE_LOGPROB: f32 = -1.0;

pub(super) fn prompt(ids: &[u32], top_k: usize) -> PromptEcho {
    let logprobs = ids
        .iter()
        .enumerate()
        .map(|(index, &id)| {
            (index > 0).then(|| TokenLogprob {
                logprob: PROMPT_LOGPROB,
                rank: PROMPT_RANK,
                top_logprobs: alternatives(id, top_k),
            })
        })
        .collect();
    PromptEcho {
        ids: ids.to_vec(),
        logprobs,
    }
}

pub(super) fn completion(id: u32, top_k: usize) -> TokenLogprob {
    TokenLogprob {
        logprob: COMPLETION_LOGPROB,
        rank: 1,
        top_logprobs: alternatives(id, top_k),
    }
}

/// Protocol fixtures deliberately keep the scored token out of the alternatives.
/// These scores are not model accuracy evidence.
fn alternatives(scored_id: u32, top_k: usize) -> Vec<(u32, f32)> {
    (0..top_k)
        .map(|index| {
            (
                scored_id.wrapping_add(index as u32 + 1),
                FIRST_ALTERNATIVE_LOGPROB - index as f32,
            )
        })
        .collect()
}
