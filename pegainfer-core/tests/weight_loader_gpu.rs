//! GPU, model-free: StagedWeightLoader state-machine contract.

use std::collections::HashMap;

use half::bf16;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::weight_loader::StagedWeightLoader;
use safetensors::Dtype;
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

#[test]
fn record_after_finish_is_rejected() {
    let ctx = DeviceContext::new().unwrap();
    let data: Vec<u8> = (0..4u16)
        .flat_map(|i| bf16::from_f32(f32::from(i)).to_bits().to_le_bytes())
        .collect();
    let view = TensorView::new(Dtype::BF16, vec![2, 2], &data).unwrap();
    let bytes = safetensors::serialize([("w".to_string(), view)], None).unwrap();
    let shards = vec![SafeTensors::deserialize(&bytes).unwrap()];
    let weight_map = HashMap::new();

    let mut loader = StagedWeightLoader::new(&ctx, &shards, &weight_map).unwrap();
    let slot = loader.matrix("w", 2, 2).unwrap();
    loader.finish().unwrap();

    assert!(
        loader.matrix("w", 2, 2).is_err(),
        "matrix record after finish must be rejected"
    );
    assert!(
        loader.vector("w", 4).is_err(),
        "vector record after finish must be rejected"
    );
    let taken = loader.take(slot);
    assert_eq!((taken.rows, taken.cols), (2, 2));
}

#[test]
fn unaligned_vector_payload_roundtrips() {
    let ctx = DeviceContext::new().unwrap();
    let values: Vec<bf16> = (0..4u16)
        .map(|i| bf16::from_f32(f32::from(i) + 0.5))
        .collect();
    let payload: Vec<u8> = values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    let view = TensorView::new(Dtype::BF16, vec![4], &payload).unwrap();
    let bytes = safetensors::serialize([("v".to_string(), view)], None).unwrap();

    // Stage the archive at both parities and keep the one whose payload
    // pointer is odd, forcing the owned-decode path.
    let mut backing = vec![0u8; bytes.len() + 1];
    let mut chosen = None;
    for start in [0usize, 1] {
        backing[start..start + bytes.len()].copy_from_slice(&bytes);
        let shard = SafeTensors::deserialize(&backing[start..start + bytes.len()]).unwrap();
        if shard.tensor("v").unwrap().data().as_ptr() as usize % 2 == 1 {
            chosen = Some(start);
            break;
        }
    }
    let start = chosen.expect("adjacent offsets have opposite payload parity");
    let shards = vec![SafeTensors::deserialize(&backing[start..start + bytes.len()]).unwrap()];
    let weight_map = HashMap::new();

    let mut loader = StagedWeightLoader::new(&ctx, &shards, &weight_map).unwrap();
    let slot = loader.vector("v", 4).unwrap();
    loader.finish().unwrap();

    let out = loader.take_vec(slot);
    let host: Vec<bf16> = ctx.stream.clone_dtoh(&out.data).unwrap();
    let got: Vec<u16> = host.iter().map(|v| v.to_bits()).collect();
    let want: Vec<u16> = values.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, want);
}
