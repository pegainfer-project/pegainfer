use half::bf16;

use super::*;

#[test]
fn projection_and_graph_replay_match_scalar_selection() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let directory = tempfile::tempdir()?;
    let config = serde_json::json!({
        "hidden_size": 4, "intermediate_size": 8, "num_hidden_layers": 1,
        "num_attention_heads": 1, "num_key_value_heads": 1, "num_target_layers": 1,
        "head_dim": 4, "vocab_size": 16, "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0, "block_size": 3, "mask_token_id": 0,
        "target_layer_ids": [0], "selector_rank": 2
    });
    std::fs::write(directory.path().join("config.json"), config.to_string())?;
    let flat = DFlashConfig::from_file(directory.path().to_str().unwrap())?;
    let native = serde_json::json!({
        "transformer_layer_config": config,
        "block_size": 3, "mask_token_id": 0, "selector_rank": 2,
        "conv_kernel_size": 2, "conv_group_size": 2,
        "aux_hidden_state_layer_ids": [1], "sliding_window_non_causal": true
    });
    std::fs::write(directory.path().join("config.json"), native.to_string())?;
    let parsed = DFlashConfig::from_file(directory.path().to_str().unwrap())?;
    assert_eq!(parsed.target_layer_ids, vec![0], "HF hidden-state indexing");
    assert_eq!(parsed.selector_rank, flat.selector_rank);
    assert!(parsed.conv.is_some());
    let mut nested = config;
    nested["dflash_config"] = serde_json::json!({
        "block_size": nested.as_object_mut().unwrap().remove("block_size"),
        "mask_token_id": nested.as_object_mut().unwrap().remove("mask_token_id"),
        "target_layer_ids": nested.as_object_mut().unwrap().remove("target_layer_ids"),
        "selector_rank": nested.as_object_mut().unwrap().remove("selector_rank"),
        "selector_top_k": 16,
        "training_metadata": { "unused": true }
    });
    std::fs::write(directory.path().join("config.json"), nested.to_string())?;
    let config = DFlashConfig::from_file(directory.path().to_str().unwrap())?;
    assert_eq!(config.selector_rank, flat.selector_rank);
    assert_eq!(config.block_size, flat.block_size);

    let projection_weights: Vec<_> = (0..8)
        .map(|i| bf16::from_f32((i as f32 - 3.0) / 8.0))
        .collect();
    let predecessor: Vec<_> = (0..32)
        .map(|i| bf16::from_f32(((i % 7) as f32 - 3.0) / 8.0))
        .collect();
    let successor: Vec<_> = (0..32)
        .map(|i| bf16::from_f32(((i % 11) as f32 - 5.0) / 8.0))
        .collect();
    let head = SelectorHead {
        projection: DeviceMatrix::from_host(&ctx, &projection_weights, 2, 4)?,
        predecessor: DeviceMatrix::from_host(&ctx, &predecessor, 16, 2)?,
        successor: DeviceMatrix::from_host(&ctx, &successor, 16, 2)?,
        enabled: true,
    };
    let mut scratch = SelectorScratch::new(&ctx, &config, 2)?;
    let mut hidden = HiddenStates::zeros(&ctx, 4, 6)?;
    let mut logits = HiddenStates::zeros(&ctx, 16, 6)?;

    // Each batch gets eager warmup, capture and replay. Change both inputs and
    // anchors on every call while keeping the captured device pointers stable.
    for (iteration, batch) in [1, 1, 1, 2, 2, 2, 1].into_iter().enumerate() {
        let input_hidden: Vec<_> = (0..24)
            .map(|i| bf16::from_f32(((i + iteration) % 13) as f32 / 8.0 - 0.5))
            .collect();
        let input_logits: Vec<_> = (0..96)
            .map(|i| bf16::from_f32(((i + iteration * 3) % 9) as f32 / 16.0))
            .collect();
        let anchors: Vec<_> = (0..batch).map(|i| ((i + iteration) % 16) as u32).collect();
        ctx.stream.memcpy_htod(&input_hidden, &mut hidden.data)?;
        ctx.stream.memcpy_htod(&input_logits, &mut logits.data)?;
        hidden.seq_len = batch * 3;
        logits.seq_len = batch * 3;
        let projected: Vec<_> = (0..batch * 3 * 2)
            .map(|index| {
                let row = index / 2;
                let component = index % 2;
                bf16::from_f32(
                    (0..4)
                        .map(|k| {
                            input_hidden[row * 4 + k].to_f32()
                                * projection_weights[component * 4 + k].to_f32()
                        })
                        .sum(),
                )
            })
            .collect();
        let mut expected = Vec::new();
        for (request, &anchor) in anchors.iter().enumerate() {
            let mut previous = anchor as usize;
            for position in 1..3 {
                let row = request * 3 + position;
                let mut best = (0, f32::NEG_INFINITY);
                for token in 0..16 {
                    let pair: f32 = (0..2)
                        .map(|r| {
                            projected[row * 2 + r].to_f32()
                                * predecessor[previous * 2 + r].to_f32()
                                * successor[token * 2 + r].to_f32()
                        })
                        .sum();
                    let score = input_logits[row * 16 + token].to_f32() + pair;
                    if score > best.1 {
                        best = (token, score);
                    }
                }
                previous = best.0;
                expected.push(previous as u32);
            }
        }
        let selected = head.select(&ctx, &hidden, &logits, &anchors, 3, &mut scratch)?;
        assert_eq!(selected, expected, "iteration {iteration}, batch {batch}");
        assert_eq!(
            ctx.stream
                .clone_dtoh(&scratch.buffers.projected.data.slice(..projected.len()))?,
            projected,
            "projection layout/rounding"
        );
    }
    // A captured graph must still reject bad scores, then recover on reuse.
    logits.seq_len = 3;
    hidden.seq_len = 3;
    ctx.stream
        .memcpy_htod(&[bf16::NAN; 16], &mut logits.data.slice_mut(16..32))?;
    assert!(
        head.select(&ctx, &hidden, &logits, &[0], 3, &mut scratch)
            .is_err()
    );
    ctx.stream
        .memcpy_htod(&[bf16::ZERO; 16], &mut logits.data.slice_mut(16..32))?;
    assert!(
        head.select(&ctx, &hidden, &logits, &[0], 3, &mut scratch)
            .is_ok()
    );
    Ok(())
}
