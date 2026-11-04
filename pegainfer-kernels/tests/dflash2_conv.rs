//! Native conv arithmetic and request-boundary isolation against a BF16 oracle.

use anyhow::Result;
use half::bf16;
use pegainfer_kernels::ops::dflash2_grouped_conv_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;

#[test]
fn grouped_conv_matches_bf16_reference_across_blocks_and_sides() -> Result<()> {
    const BATCH: usize = 3;
    const BLOCK: usize = 5;
    const HIDDEN: usize = 48;
    const GROUP: usize = 16;
    const TAPS: usize = 2;
    const GROUPS: usize = HIDDEN / GROUP;
    const COEFFICIENTS: usize = 2 * TAPS * GROUPS;

    let ctx = DeviceContext::new()?;
    let hidden: Vec<_> = (0..BATCH * BLOCK * HIDDEN)
        .map(|i| bf16::from_f32(((i * 17 % 127) as f32 - 63.0) / 9.0))
        .collect();
    let coefficients: Vec<_> = (0..BATCH * BLOCK * COEFFICIENTS)
        .map(|i| bf16::from_f32(((i * 11 % 43) as f32 - 21.0) / 13.0))
        .collect();
    let base: Vec<_> = (0..2 * TAPS * HIDDEN)
        .map(|i| bf16::from_f32(((i * 7 % 31) as f32 - 15.0) / 11.0))
        .collect();
    let mut input = HiddenStates::from_host(&ctx, &hidden, HIDDEN, BATCH * BLOCK)?;
    let mut dynamic = HiddenStates::from_host(&ctx, &coefficients, COEFFICIENTS, BATCH * BLOCK)?;
    let device_base = ctx.stream.clone_htod(&base)?;
    let mut output = HiddenStates::zeros(&ctx, HIDDEN, BATCH * BLOCK)?;

    for batch in [3, 1, 2, 3] {
        input.seq_len = batch * BLOCK;
        dynamic.seq_len = batch * BLOCK;
        output.seq_len = batch * BLOCK;
        for side in 0..2 {
            let mut expected = Vec::with_capacity(batch * BLOCK * HIDDEN);
            for row in 0..batch * BLOCK {
                for channel in 0..HIDDEN {
                    let mut sum = bf16::ZERO;
                    for tap in 0..TAPS.min(row % BLOCK + 1) {
                        let delta = coefficients
                            [row * COEFFICIENTS + (side * TAPS + tap) * GROUPS + channel / GROUP];
                        let weight = base[(side * TAPS + tap) * HIDDEN + channel];
                        let coefficient = bf16::from_f32(weight.to_f32() + delta.to_f32());
                        let product = bf16::from_f32(
                            hidden[(row - tap) * HIDDEN + channel].to_f32() * coefficient.to_f32(),
                        );
                        sum = bf16::from_f32(sum.to_f32() + product.to_f32());
                    }
                    expected.push(sum);
                }
            }
            dflash2_grouped_conv_into(
                &ctx,
                &input,
                &dynamic,
                &device_base,
                BLOCK,
                GROUP,
                side,
                &mut output,
            )?;
            let actual = ctx
                .stream
                .clone_dtoh(&output.data.slice(..expected.len()))?;
            assert_eq!(actual, expected, "batch {batch}, conv side {side}");
        }
    }
    Ok(())
}
