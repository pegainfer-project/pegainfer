//! TP=2 launch with CUDA Graph on — when rendering tools are available, startup
//! dumps rank 0's pre-captured bs1 graph, then decode replays captured graphs
//! under concurrent serving.
//! Worker threads fold each request stream; the main thread collects results
//! against a deadline so a deadlock fails instead of wedging the run.

use std::mem::ManuallyDrop;
use std::path::Path;
use std::process::Command;
use std::process::Stdio;
use std::time::Duration;
use std::time::Instant;

use pegainfer_frontend::engine::Terminal;
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_qwen3::DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES;
use pegainfer_qwen3::DEFAULT_KV_PAGE_SIZE;
use pegainfer_qwen3::DEFAULT_MAX_PREFILL_TOKENS;
use pegainfer_qwen3::DecodeOverlap;
use pegainfer_qwen3::Qwen3LaunchOptions;
use pegainfer_qwen3::Qwen3MemoryOptions;
use pegainfer_qwen3::Qwen3OffloadOptions;

mod common;

use common::harness::EngineHarness;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const REQUESTS: usize = 16;
const DEADLINE_SECS: u64 = 300;

fn model_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping tp2 concurrent decode: {MODEL_PATH}/config.json missing; set PEGAINFER_TEST_MODEL_PATH"
            );
            None
        }
    }
}

fn cuda_device_count() -> usize {
    cudarc::driver::CudaContext::device_count().map_or(0, |n| n.max(0) as usize)
}

fn graph_render_tools_available() -> bool {
    let succeeds = |program: &str, args: &[&str]| {
        Command::new(program)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    };
    succeeds("dot", &["-Tpng:cairo"]) && succeeds("c++filt", &["--version"])
}

#[test]
fn tp2_graph_dump_when_available_and_concurrent_decode_complete() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let gpus = cuda_device_count();
    if gpus < 2 {
        eprintln!("skipping tp2 concurrent decode: needs >=2 GPUs, have {gpus}");
        return;
    }
    let dump_dir = tempfile::tempdir().expect("create graph dump directory");
    let dump_png = dump_dir.path().join("tp2-decode.png");
    let dump_enabled = graph_render_tools_available();
    if dump_enabled {
        pegainfer_core::cuda_graph::validate_graph_dump_request(&dump_png)
            .expect("validate TP graph export request");
    } else {
        eprintln!("TP graph export coverage disabled: Graphviz Cairo or c++filt unavailable");
    }

    let options = Qwen3LaunchOptions {
        device_ordinal: 0,
        tp_size: 2,
        cuda_graph: true,
        dump_graph_png: dump_enabled.then(|| dump_png.clone()),
        offload: Qwen3OffloadOptions::disabled(),
        no_prefix_cache: false,
        max_prefill_tokens: DEFAULT_MAX_PREFILL_TOKENS,
        memory: Qwen3MemoryOptions::new(
            0.85,
            DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES,
            DEFAULT_KV_PAGE_SIZE,
        )
        .validate()
        .expect("valid memory options"),
        lora: None,
        decode_overlap: DecodeOverlap::Off,
        batch_invariant: false,
        dflash_draft_model_path: None,
    };
    // Dropping the harness joins the scheduler thread; on a panic the engine may
    // be wedged and the join would hang, so panics leak it — only the happy path
    // drops.
    let engine = ManuallyDrop::new(EngineHarness::new(
        pegainfer_qwen3::launch(Path::new(&model_path), options).expect("launch tp2 engine"),
    ));
    if dump_enabled {
        let dump_dot = dump_png.with_extension("dot");
        assert!(dump_png.is_file(), "TP graph PNG was not exported");
        assert!(dump_dot.is_file(), "TP graph DOT was not exported");
        let dot = std::fs::read_to_string(&dump_dot).expect("read TP graph DOT");
        assert!(dot.contains("dynamic_shared_mem_bytes="));
        assert!(dot.contains(" -> "), "TP graph DOT has no dependency edges");
    }

    let tokenizer = common::load_tokenizer(&model_path);
    // Submit all up front so they coexist in the engine and form real decode batches.
    let streams: Vec<_> = (0..REQUESTS)
        .map(|i| {
            let prompt = format!("Write a few sentences about topic {i}:");
            let prompt_tokens = tokenizer.encode(&prompt, false).expect("encode failed");
            engine.submit(common::harness::request(
                prompt_tokens,
                SamplingParams::default(),
                24 + (i % 4) * 24,
            ))
        })
        .collect();

    // Fold every stream on its own thread; the main thread enforces the
    // deadline so a wedged engine surfaces as a timeout, not a hang.
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let workers: Vec<_> = streams
        .into_iter()
        .enumerate()
        .map(|(i, stream)| {
            let done_tx = done_tx.clone();
            std::thread::spawn(move || {
                let _ = done_tx.send((i, stream.outcome()));
            })
        })
        .collect();
    drop(done_tx);

    let deadline = Instant::now() + Duration::from_secs(DEADLINE_SECS);
    for _ in 0..REQUESTS {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let Ok((i, outcome)) = done_rx.recv_timeout(remaining) else {
            panic!("no request completed within {DEADLINE_SECS}s, engine deadlocked");
        };
        match outcome.terminal {
            Terminal::Finished { .. } => {}
            Terminal::Rejected { reason, .. } => panic!("request {i} rejected: {reason}"),
            Terminal::Failed { message, .. } => panic!("request {i} failed: {message}"),
        }
        assert!(
            !outcome.tokens.is_empty(),
            "request {i} finished with zero decoded tokens"
        );
    }
    for worker in workers {
        worker.join().expect("stream worker panicked");
    }
    drop(ManuallyDrop::into_inner(engine));
}
