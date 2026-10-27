//! Greedy losslessness check shared by the speculative gates.
//!
//! The verify path runs the prefill attention kernel over the draft span while a
//! plain decode runs the decode kernel, and the two differ by about one bf16
//! ULP, so an exact `spec == baseline` match false-fails on a benign tie flip.
//! At the first divergence the check re-prefills the shared context and accepts
//! the speculative pick only when it sits within `MARGIN_TOL` of that prefill
//! argmax.

// Each integration test crate compiles this module independently and uses a
// different slice of it.
#![allow(dead_code)]

use pegainfer_frontend::sampler::SamplingParams;
use vllm_text::tokenizer::DynTokenizer;

use super::harness::EngineHarness;
use super::harness::Outcome;
use super::harness::request;

/// Top-K logprobs requested from the baseline; wide enough that the speculative
/// pick is in the set on any real tie (a pick outside the top-K is itself a red
/// flag the gate should catch).
pub(crate) const LOGPROBS: usize = 20;
/// Max acceptable regret: how far below the baseline's argmax (in the baseline's
/// own logprobs) the speculative pick may sit at the divergence point. ~3 bf16
/// ULP at typical logit magnitudes — mirrors `hf_golden_gate`'s `MARGIN_TOL`.
pub(crate) const MARGIN_TOL: f32 = 0.20;

/// One decoded position: the chosen token and (when requested) the top-K
/// `(token, logprob)` distribution that produced it.
pub(crate) struct Step {
    pub(crate) id: u32,
    pub(crate) top_logprobs: Vec<(u32, f32)>,
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

/// Submit one greedy request and collect the decoded steps until it finishes.
pub(crate) fn generate(
    engine: &EngineHarness,
    prompt_tokens: Vec<u32>,
    logprobs: usize,
    max_tokens: usize,
) -> Vec<Step> {
    let mut req = request(prompt_tokens, SamplingParams::default(), max_tokens);
    req.logprobs = (logprobs > 0).then_some(logprobs);
    to_steps(engine.submit(req).expect_finished())
}

/// Prefill `context` (prompt scoring) and return the next-token distribution the *prefill*
/// kernel produces — the kernel the speculative verify path also uses. This is
/// the reference the spec pick should match (vs the plain-decode baseline, whose
/// kernel resolves bifurcation ties to the other side). Returns the first
/// generated token's `(id, top_logprobs)`.
fn prefill_next(engine: &EngineHarness, context: Vec<u32>, logprobs: usize) -> Step {
    let mut req = request(context, SamplingParams::default(), 1);
    req.logprobs = (logprobs > 0).then_some(logprobs);
    req.prompt_logprobs = Some(logprobs);
    let mut generated = to_steps(engine.submit(req).expect_finished());
    assert!(
        !generated.is_empty(),
        "prompt-scoring prefill finished without a token"
    );
    generated.swap_remove(0)
}

/// Compare one prompt's speculative `spec` steps against its plain-greedy `base`,
/// tolerating only the benign prefill-vs-decode kernel-gap tie flip (the spec
/// pick sits within `MARGIN_TOL` of the prefill kernel's own argmax, measured in
/// the prefill distribution the verify path actually runs). `Ok(())` ⇒ lossless
/// or a benign tie; `Err(diagnostic)` ⇒ a real spec bug. `engine` must be the
/// live speculative engine — at a divergence it re-prefills the shared context
/// (`prompt_tokens` + the matched prefix) to read that prefill-kernel reference.
pub(crate) fn check_lossless(
    engine: &EngineHarness,
    tokenizer: &DynTokenizer,
    i: usize,
    prompt: &str,
    prompt_tokens: &[u32],
    base: &[Step],
    spec: &[Step],
) -> Result<(), String> {
    let matched = base
        .iter()
        .zip(spec)
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
    let decode_argmax = base[matched].top_logprobs[0].0;

    // Diagnostic: show the exact branch point.
    {
        let lo = matched.saturating_sub(2);
        let hi = (matched + 3).min(base.len()).min(spec.len());
        let base_ids: Vec<u32> = base[..hi].iter().map(|s| s.id).collect();
        let spec_ids: Vec<u32> = spec[..hi].iter().map(|s| s.id).collect();
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
        let _ = spec_ids;
    }

    // The verify path runs the prefill kernel, so the right reference for the
    // spec pick is a plain *prefill* of the same shared context — not the
    // plain-decode baseline, whose kernel resolves a bifurcation tie to the
    // other side and amplifies the gap.
    let mut context = prompt_tokens.to_vec();
    context.extend(base[..matched].iter().map(|s| s.id));
    let prefill_ref = prefill_next(engine, context, LOGPROBS);

    if prefill_ref.id == spec_id {
        // Spec faithfully reproduced the prefill-kernel greedy pick; the
        // divergence is purely the pre-existing prefill-vs-decode kernel gap.
        let decode_lp = base[matched]
            .top_logprobs
            .iter()
            .find(|(t, _)| *t == spec_id)
            .map(|(_, lp)| base[matched].top_logprobs[0].1 - lp);
        eprintln!(
            "prompt {i} ({prompt:?}): kernel-gap flip at token {matched} — verify(prefill)→{spec_id}, \
             decode→{decode_argmax}; spec matches prefill greedy (decode-margin {:?}). Not a spec bug.",
            decode_lp
        );
        return Ok(());
    }

    // Spec's greedy pick differs from the prefill-kernel argmax too. The verify
    // path builds its committed KV incrementally across batched speculative
    // spans while this reference prefill builds it in one shot; the two differ by
    // a few bf16 ULP. Within MARGIN_TOL of the prefill argmax ⇒ a benign tie
    // flip; clearly worse ⇒ the verify/accept/capture logic picked a token the
    // forward never favored — a real bug.
    let prefill_regret = prefill_ref
        .top_logprobs
        .iter()
        .find(|(t, _)| *t == spec_id)
        .map(|(_, lp)| prefill_ref.top_logprobs[0].1 - lp);

    if let Some(regret) = prefill_regret {
        if regret <= MARGIN_TOL {
            eprintln!(
                "prompt {i} ({prompt:?}): tie flip at token {matched} — \
                 verify(prefill)→{}, spec→{spec_id}, decode→{decode_argmax}; \
                 spec pick is #2 in the prefill distribution (regret {regret:.3} ≤ {MARGIN_TOL}). \
                 Not a spec bug.",
                prefill_ref.id,
            );
            return Ok(());
        }
    }

    // Either the spec pick is outside the prefill top-K entirely, or it sits
    // clearly below the prefill argmax — neither is a benign tie.
    let decode_regret = base[matched]
        .top_logprobs
        .iter()
        .find(|(t, _)| *t == spec_id)
        .map(|(_, lp)| base[matched].top_logprobs[0].1 - lp);
    Err(format!(
        "prompt {i}: at token {matched} spec chose {spec_id} but prefill greedy says {} and \
         decode greedy says {decode_argmax} (spec regret in prefill dist: {prefill_regret:?} > \
         {MARGIN_TOL}; in decode dist: {decode_regret:?}) — real spec bug",
        prefill_ref.id,
    ))
}
