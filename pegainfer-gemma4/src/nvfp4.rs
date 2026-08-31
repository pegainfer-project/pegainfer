//! Reading the NVFP4 tensors the routed experts are stored in.
//!
//! A checkpoint declares packed weights, block scales, a tensor scale, and an
//! input activation scale. The manifest validates all four. Production reads
//! the first three and leaves the weights packed; BF16 activations do not
//! consume the input scale. Test-only widening supports the arithmetic oracle.

use anyhow::Result;
use safetensors::SafeTensors;
use safetensors::tensor::TensorView;

use crate::manifest::schema::QuantMatrix;

/// Values per block scale.
pub(crate) const GROUP: usize = 16;
/// Values per stored byte.
pub(crate) const PER_BYTE: usize = 2;

/// One `e4m3` block scale. Bias 7, no infinity, and `0x7f`/`0xff` are NaN
/// rather than a finite magnitude.
pub(crate) fn decode_e4m3(bits: u8) -> f32 {
    let sign = if bits & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = (bits >> 3) & 0x0f;
    let mantissa_bits = bits & 0x07;
    let mantissa = f32::from(mantissa_bits);
    let magnitude = if exponent == 0 {
        // Subnormal: no implicit leading one, fixed exponent.
        mantissa / 8.0 * 2f32.powi(-6)
    } else if exponent == 0x0f && mantissa_bits == 7 {
        f32::NAN
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(i32::from(exponent) - 7)
    };
    sign * magnitude
}

/// One `e2m1` value of a packed row, the low nibble holding the even index.
/// Sign, one exponent bit pair and one mantissa bit, so the magnitudes are
/// exactly {0, .5, 1, 1.5, 2, 3, 4, 6}.
#[cfg(all(test, feature = "gemma4"))]
fn decode_e2m1(packed: &[u8], index: usize) -> f32 {
    let byte = packed[index / PER_BYTE];
    let nibble = if index.is_multiple_of(PER_BYTE) {
        byte & 0x0f
    } else {
        byte >> 4
    };
    let sign = if nibble & 0x08 != 0 { -1.0 } else { 1.0 };
    let magnitude = match nibble & 0x07 {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        3 => 1.5,
        4 => 2.0,
        5 => 3.0,
        6 => 4.0,
        _ => 6.0,
    };
    sign * magnitude
}

/// The three tensors one quantized projection consumes at runtime.
/// `input_scale` remains manifest-only because activations stay BF16.
pub(crate) struct QuantSource<'a> {
    packed: &'a [u8],
    scales: &'a [u8],
    tensor_scale: f32,
}

fn find<'a>(shards: &'a [SafeTensors<'a>], name: &str) -> Result<TensorView<'a>> {
    shards
        .iter()
        .find_map(|shard| shard.tensor(name).ok())
        .ok_or_else(|| anyhow::anyhow!("NVFP4: '{name}' is missing from every shard"))
}

fn scalar_f32(shards: &[SafeTensors], name: &str) -> Result<f32> {
    let view = find(shards, name)?;
    let bytes: [u8; 4] = view.data().try_into().map_err(|_| {
        anyhow::anyhow!(
            "NVFP4: '{name}' is {} bytes, not one f32",
            view.data().len()
        )
    })?;
    let value = f32::from_le_bytes(bytes);
    anyhow::ensure!(
        value.is_finite(),
        "NVFP4: '{name}' is {value}, which cannot scale anything"
    );
    Ok(value)
}

impl<'a> QuantSource<'a> {
    pub(crate) fn read(shards: &'a [SafeTensors<'a>], plan: &QuantMatrix) -> Result<Self> {
        Ok(Self {
            packed: find(shards, &plan.weight.name)?.data(),
            scales: find(shards, &plan.weight_scale.name)?.data(),
            tensor_scale: scalar_f32(shards, &plan.weight_scale_2.name)?,
        })
    }

    pub(crate) fn packed(&self) -> &[u8] {
        self.packed
    }

    /// The reference widening: every value decoded and scaled by its block
    /// and by the tensor. The GEMM never does this — it reads the packed form
    /// — so this exists for the gates that hold the GEMM to the arithmetic
    /// the checkpoint's format defines.
    #[cfg(all(test, feature = "gemma4"))]
    pub(crate) fn widen(&self, rows: usize, values: usize) -> Result<Vec<f32>> {
        anyhow::ensure!(
            values.is_multiple_of(GROUP),
            "NVFP4: a row of {values} values is not a whole number of {GROUP}-value blocks"
        );
        let wanted_packed = rows * values / PER_BYTE;
        let wanted_scales = rows * values / GROUP;
        anyhow::ensure!(
            self.packed.len() >= wanted_packed && self.scales.len() >= wanted_scales,
            "NVFP4: {rows}x{values} needs {wanted_packed} packed bytes and {wanted_scales} \
             scales, the tensors hold {} and {}",
            self.packed.len(),
            self.scales.len()
        );
        let mut out = vec![0.0f32; rows * values];
        for row in 0..rows {
            for value in 0..values {
                let at = row * values + value;
                let block = decode_e4m3(self.scales[at / GROUP]);
                out[at] = decode_e2m1(self.packed, at) * block * self.tensor_scale;
            }
        }
        Ok(out)
    }

    pub(crate) fn scales(&self) -> &[u8] {
        self.scales
    }

    pub(crate) fn tensor_scale(&self) -> f32 {
        self.tensor_scale
    }
}

#[cfg(test)]
mod tests {
    use safetensors::Dtype;
    use safetensors::tensor::TensorView;

    use super::*;

    /// Every value here is exactly representable, and `-0.0` has to stay
    /// distinct from `0.0`, so these compare bit patterns rather than values.
    fn assert_exact(got: f32, want: f32) {
        assert_eq!(got.to_bits(), want.to_bits(), "got {got}, want {want}");
    }

    #[test]
    fn e4m3_decodes_the_values_its_bias_implies() {
        assert_exact(decode_e4m3(0x00), 0.0);
        assert_exact(decode_e4m3(0x38), 1.0);
        assert_exact(decode_e4m3(0x3c), 1.5);
        assert_exact(decode_e4m3(0x40), 2.0);
        assert_exact(decode_e4m3(0xb8), -1.0);
        // Subnormal: no implicit leading one, so the step is 2^-9.
        assert_exact(decode_e4m3(0x01), 2f32.powi(-9));
    }

    fn toy_manifest() -> crate::manifest::schema::Manifest {
        let mut config = crate::manifest::schema::sample_config();
        config.hidden_size = 32;
        config.intermediate_size = 64;
        config.moe = Some(crate::config::MoeConfig {
            num_experts: 1,
            top_k: 1,
            intermediate_size: 16,
        });
        crate::manifest::schema::Manifest::from_config(&config).unwrap()
    }

    #[test]
    fn a_projection_is_resolved_from_a_later_shard() {
        let manifest = toy_manifest();
        let gate = &manifest.layers[0].moe.as_ref().unwrap().experts[0].gate;
        let (rows, values) = gate.geometry().unwrap();
        let unrelated = [0u8];
        let first = safetensors::serialize(
            [(
                "unrelated",
                TensorView::new(Dtype::U8, vec![1], &unrelated).unwrap(),
            )],
            None,
        )
        .unwrap();
        let packed = vec![0x21u8; rows * values / PER_BYTE];
        let scales = vec![0x38u8; rows * values / GROUP];
        let scale_2 = 3.0f32.to_le_bytes();
        let second = safetensors::serialize(
            [
                (
                    gate.weight.name.as_str(),
                    TensorView::new(Dtype::U8, vec![rows, values / PER_BYTE], &packed).unwrap(),
                ),
                (
                    gate.weight_scale.name.as_str(),
                    TensorView::new(Dtype::F8_E4M3, vec![rows, values / GROUP], &scales).unwrap(),
                ),
                (
                    gate.weight_scale_2.name.as_str(),
                    TensorView::new(Dtype::F32, vec![], &scale_2).unwrap(),
                ),
            ],
            None,
        )
        .unwrap();
        let shards = [
            SafeTensors::deserialize(&first).unwrap(),
            SafeTensors::deserialize(&second).unwrap(),
        ];
        let source = QuantSource::read(&shards, gate).unwrap();

        assert_eq!(source.packed(), packed);
        assert_eq!(source.scales(), scales);
        assert_exact(source.tensor_scale(), 3.0);
    }

    #[test]
    fn a_missing_runtime_tensor_names_itself() {
        let manifest = toy_manifest();
        let gate = &manifest.layers[0].moe.as_ref().unwrap().experts[0].gate;
        let (rows, values) = gate.geometry().unwrap();
        let packed = vec![0u8; rows * values / PER_BYTE];
        let scales = vec![0x38u8; rows * values / GROUP];
        let blob = safetensors::serialize(
            [
                (
                    gate.weight.name.as_str(),
                    TensorView::new(Dtype::U8, vec![rows, values / PER_BYTE], &packed).unwrap(),
                ),
                (
                    gate.weight_scale.name.as_str(),
                    TensorView::new(Dtype::F8_E4M3, vec![rows, values / GROUP], &scales).unwrap(),
                ),
            ],
            None,
        )
        .unwrap();
        let shards = [SafeTensors::deserialize(&blob).unwrap()];
        let Err(err) = QuantSource::read(&shards, gate) else {
            panic!("a missing tensor was accepted");
        };

        assert_eq!(
            err.to_string(),
            format!(
                "NVFP4: '{}' is missing from every shard",
                gate.weight_scale_2.name
            )
        );
    }

    #[test]
    fn a_tensor_scale_that_cannot_scale_anything_is_refused() {
        let manifest = toy_manifest();
        let gate = &manifest.layers[0].moe.as_ref().unwrap().experts[0].gate;
        let (rows, values) = gate.geometry().unwrap();
        let packed = vec![0u8; rows * values / PER_BYTE];
        let scales = vec![0x38u8; rows * values / GROUP];
        let scale_2 = f32::INFINITY.to_le_bytes().to_vec();
        let input_scale = 1.0f32.to_le_bytes().to_vec();
        let tensors = vec![
            (
                gate.weight.name.clone(),
                TensorView::new(Dtype::U8, vec![rows, values / PER_BYTE], &packed).unwrap(),
            ),
            (
                gate.weight_scale.name.clone(),
                TensorView::new(Dtype::F8_E4M3, vec![rows, values / GROUP], &scales).unwrap(),
            ),
            (
                gate.weight_scale_2.name.clone(),
                TensorView::new(Dtype::F32, vec![], &scale_2).unwrap(),
            ),
            (
                gate.input_scale.name.clone(),
                TensorView::new(Dtype::F32, vec![], &input_scale).unwrap(),
            ),
        ];
        let blob = safetensors::serialize(tensors, None).unwrap();
        let shard = SafeTensors::deserialize(&blob).unwrap();
        let shards = [shard];
        let Err(err) = QuantSource::read(&shards, gate) else {
            panic!("an infinite tensor scale was accepted");
        };
        assert!(err.to_string().contains("cannot scale anything"), "{err}");
    }
}
