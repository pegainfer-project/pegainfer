#![cfg(feature = "ascend")]
//! Manual probe: print top-k logits for a token sequence so they can be
//! compared with a transformers reference.
use anyhow::Result;
use pegainfer_ascend::qwen3::Model;
use pegainfer_ascend::qwen3::ModelOptions;
use pegainfer_ascend::qwen3::TokenInput;

#[test]
#[ignore = "requires an Ascend device and PEGAINFER_TEST_MODEL_PATH"]
fn print_topk_logits() -> Result<()> {
    let path = std::env::var("PEGAINFER_TEST_MODEL_PATH")?;
    let tokens: Vec<u32> = std::env::var("PG_TOKENS")
        .unwrap_or_else(|_| "9707".to_string())
        .split(',')
        .map(|x| x.trim().parse().unwrap())
        .collect();
    let mut model = Model::load(
        path,
        ModelOptions {
            device: 0,
            max_context: 64,
            slots: 1,
            chunk_size: 8,
        },
    )?;
    let input = tokens
        .iter()
        .enumerate()
        .map(|(i, &token)| TokenInput {
            token,
            slot: 0,
            position: i,
        })
        .collect::<Vec<_>>();
    println!(
        "vocab={} output_vocab={}",
        model.vocab_size(),
        model.output_vocab_size()
    );
    let ids = model.forward(&input, &[input.len() - 1])?;
    let logits = model.logits()?.remove(0);
    let mut top = (0..logits.len()).collect::<Vec<_>>();
    top.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    println!("TOKENS {tokens:?} argmax={} top:", ids[0]);
    for &i in top.iter().take(8) {
        println!("{i} {}", logits[i]);
    }
    Ok(())
}
