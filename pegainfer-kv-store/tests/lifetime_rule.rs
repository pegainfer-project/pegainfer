//! Lifetime-rule oracle for issue #1120: what does a request actually draw
//! over its lifetime?
//!
//! Drives kv-store's `schedule_*`/`apply_*` sequences (whole and chunked
//! prefill, plain decode, speculative verify) and measures the peak resident
//! draw (`base - available_blocks()`), then compares it with the two
//! competing rules:
//!
//! - `production`: `pegainfer_kv_store::lifetime_blocks` — the rule under
//!   test. The oracle calls it directly and must never re-implement it.
//! - `old`: `ceil((prompt + max_tokens) / bs)` — the pre-change formula, kept
//!   only as the comparison arm.
//!
//! Not modeled here: prefix matching (it only lowers the draw), external
//! anchor promotion and pad-to-boundary (covered by pool unit tests), and a
//! real scheduler's speculative span planning (the bound still holds because
//! `generated + span <= max_tokens`).
//!
//! The one-token question is whether kvbm provisions a page for the dangling
//! first generated token when the request finishes after prefill. The tests
//! print the boundary table and assert which rule is a safe upper bound and
//! which is tight; a production violation means the union must be the old
//! rule.

use pegainfer_kv_store::BlockPool;
use pegainfer_kv_store::lifetime_blocks;

const BS: usize = 16;

#[derive(Clone, Copy, Debug)]
struct Path {
    /// Prefill chunk size; `0` means one whole-prompt chunk.
    chunk: usize,
    /// Speculative draft span per decode step; `None` is a plain decode.
    draft: Option<usize>,
    /// Accepted drafts per spec step (clamped to the remaining tokens).
    accept: usize,
}

impl Path {
    const PLAIN: Self = Self {
        chunk: 0,
        draft: None,
        accept: 0,
    };

    fn chunked(chunk: usize) -> Self {
        Self {
            chunk,
            ..Self::PLAIN
        }
    }

    fn spec(draft: usize, accept: usize) -> Self {
        Self {
            draft: Some(draft),
            accept,
            ..Self::PLAIN
        }
    }
}

/// Runs one request through `path` with a pool of `total_blocks` blocks
/// (padding block included) and returns the peak draw, or the first failure.
fn run(prompt: usize, max_tokens: usize, path: Path, total_blocks: usize) -> Result<usize, String> {
    let pool = BlockPool::new(BS, total_blocks);
    let base = pool.available_blocks();
    let mut peak = 0;
    let mut kv = pool.new_request(vec![7; prompt], max_tokens, None);

    let chunk = if path.chunk == 0 {
        prompt.max(1)
    } else {
        path.chunk
    };
    let mut pos = 0;
    while pos < prompt {
        let n = chunk.min(prompt - pos);
        kv.schedule_prefill(n, &pool).map_err(|e| {
            format!("prompt={prompt} max={max_tokens} {path:?}: prefill schedule: {e}")
        })?;
        peak = peak.max(base - pool.available_blocks());
        pos += n;
        if pos == prompt {
            kv.apply_prefill(100, &pool).map_err(|e| {
                format!("prompt={prompt} max={max_tokens} {path:?}: prefill apply: {e}")
            })?;
        } else {
            kv.apply_prefill_chunk(&pool).map_err(|e| {
                format!("prompt={prompt} max={max_tokens} {path:?}: chunk apply: {e}")
            })?;
        }
        peak = peak.max(base - pool.available_blocks());
    }

    // Prefill emits the first generated token.
    let mut generated = 1usize;
    while generated < max_tokens {
        if let Some(draft) = path.draft {
            let remaining = max_tokens - generated;
            let span = draft.min(remaining);
            kv.schedule_speculative(span, &pool).map_err(|e| {
                format!("prompt={prompt} max={max_tokens} {path:?}: spec schedule: {e}")
            })?;
            peak = peak.max(base - pool.available_blocks());
            let accepted = path.accept.clamp(1, span);
            let tokens: Vec<u32> = (0..accepted).map(|i| 200 + i as u32).collect();
            kv.apply_speculative(&tokens, &pool).map_err(|e| {
                format!("prompt={prompt} max={max_tokens} {path:?}: spec apply: {e}")
            })?;
            generated += accepted;
        } else {
            kv.schedule_decode(&pool).map_err(|e| {
                format!("prompt={prompt} max={max_tokens} {path:?}: decode schedule: {e}")
            })?;
            peak = peak.max(base - pool.available_blocks());
            kv.apply_decode(300 + generated as u32, &pool)
                .map_err(|e| {
                    format!("prompt={prompt} max={max_tokens} {path:?}: decode apply: {e}")
                })?;
            generated += 1;
        }
        peak = peak.max(base - pool.available_blocks());
    }

    kv.release()
        .map_err(|e| format!("prompt={prompt} max={max_tokens} {path:?}: release: {e}"))?;
    if pool.available_blocks() != base {
        return Err(format!(
            "prompt={prompt} max={max_tokens} {path:?}: leaked blocks ({} -> {})",
            base,
            pool.available_blocks()
        ));
    }
    Ok(peak)
}

/// The rule every scheduler used before the shared function: always
/// `prompt + max_tokens`. Kept only as the comparison arm of the oracle —
/// production must be measured through `lifetime_blocks`, never a copy.
fn old_prompt_plus_max(prompt: usize, max_tokens: usize) -> usize {
    (prompt + max_tokens).div_ceil(BS)
}

/// The pool needs one padding block on top of the measured draw.
fn pool_for(peak: usize) -> usize {
    peak + 1
}

#[test]
fn peak_table_at_the_boundaries() {
    println!("bs={BS}");
    println!("prompt max | peak | production | old | winner");
    for (prompt, max_tokens) in [
        (15, 1),
        (16, 1),
        (17, 1),
        (31, 1),
        (32, 1),
        (16, 2),
        (17, 2),
        (32, 2),
        (33, 2),
        (16, 17),
        (33, 100),
        (64, 1),
        (64, 65),
    ] {
        let peak = run(prompt, max_tokens, Path::PLAIN, 1024).expect("plain path");
        let production = lifetime_blocks(prompt, max_tokens, BS);
        let old = old_prompt_plus_max(prompt, max_tokens);
        let winner = if peak == production && peak == old {
            "both"
        } else if peak == production {
            "production"
        } else if peak == old {
            "old"
        } else {
            "neither"
        };
        println!(
            "{prompt:>6} {max_tokens:>3} | {peak:>4} | {production:>10} | {old:>3} | {winner}"
        );
    }
}

#[test]
fn production_rule_is_never_under_the_plain_peak() {
    for prompt in 1..=48 {
        for max_tokens in 1..=48 {
            let peak = run(prompt, max_tokens, Path::PLAIN, 1024).expect("plain path");
            let production = lifetime_blocks(prompt, max_tokens, BS);
            assert!(
                peak <= production,
                "production under-reserves: prompt={prompt} max={max_tokens} peak={peak} \
                 production={production} old={}",
                old_prompt_plus_max(prompt, max_tokens)
            );
        }
    }
}

#[test]
fn single_token_completion_never_provisions_the_dangling_page() {
    // The qwen3/Qwen3.5 claim: a one-token completion finishes after prefill,
    // so `schedule_decode` never provisions the page the dangling first
    // generated token would land in.
    for prompt in 1..=48 {
        let peak = run(prompt, 1, Path::PLAIN, 1024).expect("one-token completion");
        assert_eq!(
            peak,
            prompt.div_ceil(BS).max(1),
            "prompt={prompt}: a one-token completion drew {peak} blocks"
        );
    }
    // The boundary that makes the two rules disagree: production is tight,
    // the old rule over-reserves one block.
    assert_eq!(run(16, 1, Path::PLAIN, 1024).unwrap(), 1);
    assert_eq!(lifetime_blocks(16, 1, BS), 1);
    assert_eq!(old_prompt_plus_max(16, 1), 2);
}

#[test]
fn multi_token_completion_matches_both_rules() {
    // For max_tokens > 1 the last decode step pre-allocates the block its
    // dangling generated token will occupy, so the draw covers prompt + max —
    // production and the old rule coincide here.
    for prompt in 1..=48 {
        for max_tokens in 2..=48 {
            let peak = run(prompt, max_tokens, Path::PLAIN, 1024).expect("plain path");
            assert_eq!(
                peak,
                lifetime_blocks(prompt, max_tokens, BS),
                "production mismatch: prompt={prompt} max={max_tokens}"
            );
            assert_eq!(
                peak,
                old_prompt_plus_max(prompt, max_tokens),
                "old-rule mismatch: prompt={prompt} max={max_tokens}"
            );
        }
    }
}

#[test]
fn chunked_prefill_does_not_change_the_peak() {
    for chunk in [1usize, 3, 7, 16, 31, 64] {
        for prompt in 1..=48 {
            for max_tokens in [1usize, 2, 17] {
                let peak =
                    run(prompt, max_tokens, Path::chunked(chunk), 1024).expect("chunked path");
                let plain = run(prompt, max_tokens, Path::PLAIN, 1024).expect("plain path");
                assert_eq!(
                    peak, plain,
                    "chunk={chunk} prompt={prompt} max={max_tokens}"
                );
            }
        }
    }
}

#[test]
fn speculative_peak_stays_within_the_lifetime_rule() {
    for draft in [2usize, 4, 8] {
        for accept in 1..=draft {
            for prompt in [1usize, 15, 16, 17, 31, 32, 33] {
                for max_tokens in [1usize, 2, 5, 17, 33] {
                    let path = Path::spec(draft, accept);
                    let peak = run(prompt, max_tokens, path, 1024).expect("speculative path");
                    assert!(
                        peak <= old_prompt_plus_max(prompt, max_tokens),
                        "draft={draft} accept={accept} prompt={prompt} max={max_tokens}: \
                         peak={peak} old={}",
                        old_prompt_plus_max(prompt, max_tokens)
                    );
                    // Speculation runs only with max_tokens > 1 in every
                    // scheduler; assert the production rule still covers it.
                    assert!(
                        peak <= lifetime_blocks(prompt, max_tokens, BS),
                        "draft={draft} accept={accept} prompt={prompt} max={max_tokens}: \
                         peak={peak} production={}",
                        lifetime_blocks(prompt, max_tokens, BS)
                    );
                }
            }
        }
    }
}

#[test]
fn the_measured_peak_is_tight() {
    // A pool of exactly `peak + padding` blocks must run the request; one
    // block less must fail. This proves the oracle measures the real draw,
    // not just an upper bound.
    for (prompt, max_tokens) in [(16usize, 1usize), (16, 2), (32, 17), (33, 64)] {
        let peak = run(prompt, max_tokens, Path::PLAIN, 1024).expect("measure");
        let tight = pool_for(peak);
        run(prompt, max_tokens, Path::PLAIN, tight)
            .unwrap_or_else(|e| panic!("tight pool must fit: {e}"));
        // A pool needs one padding block, so a measured peak of 1 has no
        // representable "one block less" pool.
        if tight > 2 {
            assert!(
                run(prompt, max_tokens, Path::PLAIN, tight - 1).is_err(),
                "prompt={prompt} max={max_tokens}: peak={peak} but a pool of {} blocks fit",
                tight - 1
            );
        }
    }
}
