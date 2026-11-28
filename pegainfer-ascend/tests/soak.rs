//! Manual device soak: a long decode loop must not retain per-step resources.
//!
//! Device gates are manual (see `docs/subsystems/runtime/ascend-cann-qwen3.md`):
//!
//! ```bash
//! PEGAINFER_TEST_MODEL_PATH=/opt/atomgit/models/Qwen3-0.6B \
//!   cargo test --release -p pegainfer-ascend --features ascend \
//!   --test soak -- --ignored --test-threads=1
//! ```
use anyhow::Result;
use anyhow::ensure;
use pegainfer_ascend::qwen3::Model;
use pegainfer_ascend::qwen3::ModelOptions;
use pegainfer_ascend::qwen3::TokenInput;

fn rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix("VmRSS:")?
            .split_whitespace()
            .next()?
            .parse()
            .ok()
    })
}

#[test]
#[ignore = "requires an explicitly selected Ascend device and model path"]
fn decode_steps_keep_host_memory_flat() -> Result<()> {
    let path = std::env::var("PEGAINFER_TEST_MODEL_PATH")
        .unwrap_or_else(|_| "models/Qwen3-0.6B".to_string());
    let steps = std::env::var("PEGAINFER_SOAK_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000usize);
    let context = 512usize;
    let mut model = Model::load(
        &path,
        ModelOptions {
            device: 0,
            max_context: context,
            slots: 1,
            chunk_size: 32,
        },
    )?;
    let warmup = 64.min(steps);
    let mut baseline = 0u64;
    let mut position = 0usize;
    for step in 0..steps {
        if position == context {
            model.reset_slot(0)?;
            position = 0;
        }
        model.forward(
            &[TokenInput {
                token: 1,
                slot: 0,
                position,
            }],
            &[0],
        )?;
        position += 1;
        if step + 1 == warmup {
            baseline = rss_kb().unwrap_or(0);
        }
        if (step + 1) % 200 == 0 {
            println!("step={} rss={}KB", step + 1, rss_kb().unwrap_or(0));
        }
    }
    let end = rss_kb().unwrap_or(0);
    if baseline != 0 && end != 0 {
        let growth = end.saturating_sub(baseline);
        println!("soak steps={steps} baseline={baseline}KB end={end}KB growth={growth}KB");
        ensure!(
            growth < 128 * 1024,
            "host RSS grew {growth}KB over the soak"
        );
    }
    Ok(())
}
