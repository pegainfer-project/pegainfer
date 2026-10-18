use std::time::Duration;

use anyhow::Result;
use anyhow::bail;
use pegainfer_frontend::engine::FinishReason;
use sha2::Digest;
use sha2::Sha256;

pub(super) fn token_sha256(tokens: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    hex::encode(hasher.finalize())
}

pub(super) fn ensure_same_prompt_batch_rows_match(rows: &[Vec<u32>]) -> Result<()> {
    let Some(first) = rows.first() else {
        return Ok(());
    };
    for (row_idx, row) in rows.iter().enumerate().skip(1) {
        if row != first {
            let first_diff = first
                .iter()
                .zip(row)
                .position(|(lhs, rhs)| lhs != rhs)
                .unwrap_or_else(|| first.len().min(row.len()));
            bail!(
                "same-prompt batched decode row {row_idx} differs from row 0 at generated token index {first_diff}: row0_sha256={}, row_sha256={}",
                token_sha256(first),
                token_sha256(row)
            );
        }
    }
    Ok(())
}

pub(super) fn duration_micros(duration: Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}

pub(super) fn append_generated_token(
    generated: &mut Vec<u32>,
    token: u32,
    eos_token_ids: &[u32],
    ignore_eos: bool,
) -> Option<FinishReason> {
    if !ignore_eos && eos_token_ids.contains(&token) {
        return Some(FinishReason::Stop);
    }
    generated.push(token);
    None
}
