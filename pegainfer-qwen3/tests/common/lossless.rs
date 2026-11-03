//! Greedy losslessness check shared by the speculative gates.
//!
//! A speculative run commits most tokens from a verify forward and the rest
//! from prefill or batched decode, while the plain-decode baseline produces each
//! position from its own forward. They run in bf16 with different kernels and
//! GEMM row counts, so where the model's two best tokens sit within a bf16 step
//! of each other the two forwards can round to opposite winners, and an exact
//! `spec == baseline` match false-fails. At the first divergence the check reads
//! the forward row that committed the speculative token from the executor's
//! commit trace and accepts the divergence only as such a tie: the speculative
//! token is that row's argmax, the baseline token is its runner up, and the two
//! forwards' margins between the two tokens differ by at most [`TIE_STEPS`]
//! bf16 steps at the logit's magnitude.

// Each integration test crate compiles this module independently and uses a
// different slice of it.
#![allow(dead_code)]

use pegainfer_frontend::engine::RequestId;
use pegainfer_frontend::sampler::SamplingParams;
use vllm_text::tokenizer::DynTokenizer;

use super::harness::EngineHarness;
use super::harness::Outcome;
use super::harness::committed_row;
use super::harness::request;

/// Top-K logprobs requested from the baseline; wide enough that the speculative
/// pick is in the set on any real tie (a pick outside the top-K is itself a red
/// flag the gate should catch).
pub(crate) const LOGPROBS: usize = 20;
/// Widest disagreement between the two forwards, in bf16 steps at the logit's
/// magnitude, that still counts as a tie: the committing forward's lead for
/// the speculative token plus the decode forward's lead for the baseline's.
/// Over 3769 agreeing positions from every lever configuration of the DFlash
/// and EAGLE-3 gates, the two forwards' margins between the same two tokens
/// differed by at most 5 steps, 99.7% of them by 3 or fewer.
pub(crate) const TIE_STEPS: f32 = 6.0;

/// One decoded position: the chosen token and (when requested) the top-K
/// `(token, logprob)` distribution that produced it.
pub(crate) struct Step {
    pub(crate) id: u32,
    pub(crate) top_logprobs: Vec<(u32, f32)>,
}

/// One request's decoded steps and the id the engine served it under, which
/// locates the forward rows behind its committed tokens.
pub(crate) struct Run {
    pub(crate) id: RequestId,
    pub(crate) steps: Vec<Step>,
}

impl std::ops::Deref for Run {
    type Target = [Step];

    fn deref(&self) -> &[Step] {
        &self.steps
    }
}

/// Fold a finished request's outcome into decoded steps: tokens paired with
/// their requested top-K distributions (empty when logprobs were off).
pub(crate) fn to_steps(outcome: Outcome) -> Vec<Step> {
    let Outcome {
        tokens, logprobs, ..
    } = outcome;
    tokens
        .into_iter()
        .enumerate()
        .map(|(i, id)| Step {
            id,
            top_logprobs: logprobs
                .get(i)
                .and_then(Option::as_ref)
                .map(|lp| lp.top_logprobs.clone())
                .unwrap_or_default(),
        })
        .collect()
}

/// [`to_steps`] keeping the request id.
pub(crate) fn to_run(outcome: Outcome) -> Run {
    let id = outcome.id;
    Run {
        id,
        steps: to_steps(outcome),
    }
}

/// Submit one greedy request and collect the decoded steps until it finishes.
pub(crate) fn generate(
    engine: &EngineHarness,
    prompt_tokens: Vec<u32>,
    logprobs: usize,
    max_tokens: usize,
) -> Run {
    let mut req = request(prompt_tokens, SamplingParams::default(), max_tokens);
    req.logprobs = (logprobs > 0).then_some(logprobs);
    to_run(engine.submit(req).expect_finished())
}

/// The bf16 step at `logit`'s magnitude: bf16 keeps eight significant bits.
fn bf16_ulp(logit: f32) -> f32 {
    (logit.abs().max(f32::MIN_POSITIVE).log2().floor() - 7.0).exp2()
}

/// Compare one prompt's speculative `spec` run against its plain-greedy `base`,
/// tolerating only a bf16 tie between the forward that committed the
/// speculative token and the baseline's decode forward. `Ok(())` ⇒ lossless or
/// a benign tie; `Err(diagnostic)` ⇒ a real spec bug. The commit trace is read
/// from the engine the harness started last, so the test must have called
/// `init_capture_logging` before that engine served the run.
pub(crate) fn check_lossless(
    tokenizer: &DynTokenizer,
    i: usize,
    prompt: &str,
    prompt_tokens: &[u32],
    base: &[Step],
    spec: &Run,
) -> Result<(), String> {
    let matched = base
        .iter()
        .zip(spec.iter())
        .take_while(|(b, s)| b.id == s.id)
        .count();

    // A matching prefix alone does not prove the full continuation was produced.
    if matched == base.len().min(spec.len()) {
        if base.len() != spec.len() {
            return Err(format!(
                "prompt {i}: matching prefix but different output lengths: baseline={}, spec={}",
                base.len(),
                spec.len()
            ));
        }
        eprintln!(
            "prompt {i} ({prompt:?}): {matched}/{} tokens identical (100% lossless)",
            base.len()
        );
        return Ok(());
    }

    let spec_id = spec[matched].id;
    let base_id = base[matched].id;

    // Diagnostic: show the exact branch point.
    {
        let lo = matched.saturating_sub(2);
        let hi = (matched + 3).min(base.len()).min(spec.len());
        let base_ids: Vec<u32> = base[..hi].iter().map(|s| s.id).collect();
        eprintln!("  [diag] prompt {i} matched={matched}");
        eprintln!(
            "  [diag] context+gen base ids {:?} = {:?}",
            base_ids,
            tokenizer.decode(&base_ids, false).unwrap_or_default()
        );
        eprintln!(
            "  [diag] base[{lo}..{hi}] = {:?}",
            base[lo..hi]
                .iter()
                .map(|s| (s.id, tokenizer.decode(&[s.id], false).unwrap_or_default()))
                .collect::<Vec<_>>()
        );
        eprintln!(
            "  [diag] spec[{lo}..{hi}] = {:?}",
            spec[lo..hi]
                .iter()
                .map(|s| (s.id, tokenizer.decode(&[s.id], false).unwrap_or_default()))
                .collect::<Vec<_>>()
        );
    }

    let pos = prompt_tokens.len() + matched;
    let Some(row) = committed_row(spec.id, pos) else {
        return Err(format!(
            "prompt {i}: at token {matched} spec chose {spec_id} and the baseline {base_id}, but \
             the commit trace has no row for position {pos}; call init_capture_logging before \
             the speculative engine starts"
        ));
    };
    let [(top_id, top_logit), (second_id, second_logit)] = row.top;
    if top_id != spec_id {
        return Err(format!(
            "prompt {i}: at token {matched} spec committed {spec_id}, but the forward that \
             committed it ranks {top_id} first ({top_logit}) — the commit does not follow its \
             forward — real spec bug"
        ));
    }
    if second_id != base_id {
        return Err(format!(
            "prompt {i}: at token {matched} the baseline chose {base_id}, which is not the \
             committing forward's runner up ({second_id} at {second_logit}, after {spec_id} at \
             {top_logit}) — the two forwards disagree beyond a tie — real spec bug"
        ));
    }
    let ulp = bf16_ulp(top_logit);
    let commit_gap = (top_logit - second_logit) / ulp;
    let logprob = |token: u32| {
        base[matched]
            .top_logprobs
            .iter()
            .find(|(t, _)| *t == token)
            .map(|(_, lp)| *lp)
    };
    let decode_gap = logprob(base_id)
        .zip(logprob(spec_id))
        .map(|(base_lp, spec_lp)| (base_lp - spec_lp) / ulp);
    match decode_gap {
        Some(decode_gap) if commit_gap + decode_gap <= TIE_STEPS => {
            eprintln!(
                "prompt {i} ({prompt:?}): tie flip at token {matched} — committing forward→{spec_id} \
                 by {commit_gap:.1} steps, decode→{base_id} by {decode_gap:.1} steps (one step = \
                 {ulp} at logit {top_logit}). Not a spec bug."
            );
            Ok(())
        }
        _ => Err(format!(
            "prompt {i}: at token {matched} spec chose {spec_id} by {commit_gap:.1} steps in the \
             forward that committed it and the baseline chose {base_id} by {decode_gap:?} steps \
             (one step = {ulp} at logit {top_logit}, tie band {TIE_STEPS} steps in all) — real \
             spec bug"
        )),
    }
}
