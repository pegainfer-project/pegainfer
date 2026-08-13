use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::LoadLoraAdapterRequest;
use pegainfer_frontend::engine::Terminal;
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_qwen3::lora_fixtures as fixtures;
use serde::Deserialize;
use vllm_text::tokenizer::DynTokenizer;

mod common;

use common::harness::EngineHarness;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");

#[derive(Deserialize)]
struct ModelConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

fn get_model_path() -> String {
    std::env::var("PEGAINFER_TEST_MODEL_PATH").unwrap_or_else(|_| MODEL_PATH.to_string())
}

fn get_device_ordinal() -> usize {
    std::env::var("PEGAINFER_TEST_DEVICE_ORDINAL")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn load_model_config(model_path: &str) -> ModelConfig {
    let config_path = Path::new(model_path).join("config.json");
    let content = fs::read_to_string(&config_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", config_path.display()));
    serde_json::from_str(&content)
        .unwrap_or_else(|err| panic!("failed to parse {}: {err}", config_path.display()))
}

fn write_zero_lora_adapter(path: &Path, config: &ModelConfig, rank: usize) {
    fixtures::write_adapter_config(path, rank, rank, &["q_proj", "v_proj"]);

    let mut tensors = BTreeMap::new();
    for layer_idx in 0..config.num_hidden_layers {
        fixtures::push_projection(
            &mut tensors,
            layer_idx,
            "self_attn.q_proj",
            rank,
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
        );
        fixtures::push_projection(
            &mut tensors,
            layer_idx,
            "self_attn.v_proj",
            rank,
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
        );
    }
    fixtures::write_adapter_tensors(path, tensors);
}

fn load_adapter(engine: &EngineHarness, adapter_name: &str, adapter_path: PathBuf) {
    let control = engine.lora_client();
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("build runtime")
        .block_on(control.load(LoadLoraAdapterRequest {
            lora_name: adapter_name.to_string(),
            lora_path: adapter_path,
            load_inplace: false,
        }))
        .expect("load LoRA adapter");
}

fn generate_tokens(
    engine: &EngineHarness,
    tokenizer: &DynTokenizer,
    prompt: &str,
    max_tokens: usize,
    lora_adapter: Option<String>,
) -> (Vec<u32>, FinishReason) {
    let prompt_tokens = tokenizer.encode(prompt, false).expect("encode failed");
    let mut request =
        common::harness::request(prompt_tokens, SamplingParams::default(), max_tokens);
    request.lora_adapter = lora_adapter;

    let outcome = engine.submit(request).expect_finished();
    let Terminal::Finished { reason, .. } = outcome.terminal else {
        unreachable!("expect_finished returned a non-Finished terminal");
    };
    (outcome.tokens, reason)
}

#[test]
#[ignore = "requires Qwen3-4B weights and a CUDA GPU"]
fn qwen3_lora_loads_adapter_and_generates() {
    qwen3_lora_loads_rank_and_generates(1, "zero-smoke");
}

#[test]
#[ignore = "requires Qwen3-4B weights and a CUDA GPU"]
fn qwen3_lora_loads_rank64_adapter_and_generates() {
    qwen3_lora_loads_rank_and_generates(64, "zero-rank64-smoke");
}

fn qwen3_lora_loads_rank_and_generates(rank: usize, adapter_name: &str) {
    let model_path = get_model_path();
    let config = load_model_config(&model_path);
    assert!(
        config.intermediate_size > config.hidden_size,
        "unexpected Qwen3 config dimensions"
    );

    let adapter_dir = tempfile::tempdir().expect("create temp adapter dir");
    write_zero_lora_adapter(adapter_dir.path(), &config, rank);

    let engine = EngineHarness::new(
        pegainfer_qwen3::start_engine_with_lora_control(
            Path::new(&model_path),
            EngineLoadOptions {
                enable_cuda_graph: false,
                device_ordinals: vec![get_device_ordinal()],
                seed: 42,
                ..EngineLoadOptions::default()
            },
            pegainfer_qwen3::Qwen3LoraOptions::default(),
            pegainfer_qwen3::Qwen3OffloadOptions::disabled(),
            false,
            pegainfer_qwen3::DEFAULT_MAX_PREFILL_TOKENS,
            pegainfer_qwen3::Qwen3MemoryOptions::default(),
            pegainfer_qwen3::DecodeOverlap::Off,
            false,
        )
        .expect("start LoRA-capable Qwen3 engine"),
    );

    load_adapter(&engine, adapter_name, adapter_dir.path().to_path_buf());

    let tokenizer = common::load_tokenizer(&model_path);
    let (tokens, finish_reason) = generate_tokens(
        &engine,
        &tokenizer,
        "Hello",
        4,
        Some(adapter_name.to_string()),
    );
    assert!(
        !tokens.is_empty(),
        "LoRA smoke generation returned no tokens"
    );
    assert_eq!(finish_reason, FinishReason::Length);
}
