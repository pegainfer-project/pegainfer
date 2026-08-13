//! Shared fakes for the scheduler mechanics tests and the frontend-adapter
//! contract tests.

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::Result;
use pegainfer_frontend::engine::Request;
use pegainfer_frontend::engine::UnloadLoraAdapterRequest;
use pegainfer_frontend::sampler::SamplingParams;

use super::blocks_needed;
use crate::executor::DecodePlan;
use crate::executor::DecodeRequestResult;
use crate::executor::ModelExecutor;
use crate::executor::PrefillPlan;
use crate::executor::PrefillRequestResult;
use crate::executor::PrefillResult;
use crate::executor::PrefillStepItem;
use crate::executor::RequestId;
use crate::executor::UnifiedPlan;
use crate::executor::UnifiedResult;

pub(crate) struct FakeExecutor {
    pub(crate) block_size: usize,
    max_request_blocks: usize,
    max_context_tokens: usize,
    available_blocks: usize,
    held_tokens: HashMap<RequestId, usize>,
    // Prompt progress of requests mid-chunked-prefill (mirrors the real
    // executor's kv_position so multi-chunk scheduling is exercised).
    prefill_positions: HashMap<RequestId, usize>,
    fail_decode_once: bool,
    decode_delay: Duration,
    loaded_lora_adapters: HashSet<String>,
    pub(crate) dropped: Arc<Mutex<Vec<u64>>>,
    pub(crate) prefetch_offers: Arc<Mutex<Vec<u64>>>,
    stop_token: Option<u32>,
}

impl FakeExecutor {
    pub(crate) fn new(max_request_blocks: usize, dropped: Arc<Mutex<Vec<u64>>>) -> Self {
        Self {
            block_size: 16,
            max_request_blocks,
            max_context_tokens: usize::MAX,
            available_blocks: max_request_blocks,
            held_tokens: HashMap::new(),
            prefill_positions: HashMap::new(),
            fail_decode_once: false,
            decode_delay: Duration::ZERO,
            loaded_lora_adapters: HashSet::new(),
            dropped,
            prefetch_offers: Arc::new(Mutex::new(Vec::new())),
            stop_token: None,
        }
    }

    pub(crate) fn with_stop_token(mut self, token: u32) -> Self {
        self.stop_token = Some(token);
        self
    }

    pub(crate) fn with_decode_failure(mut self) -> Self {
        self.fail_decode_once = true;
        self
    }

    pub(crate) fn with_decode_delay(mut self, delay: Duration) -> Self {
        self.decode_delay = delay;
        self
    }

    pub(crate) fn with_lora_adapters(mut self, names: &[&str]) -> Self {
        self.loaded_lora_adapters = names.iter().map(|name| (*name).to_string()).collect();
        self
    }

    /// Advance a request's prompt by one chunk, mirroring the real
    /// executor: clamp the scheduler's budget to the tokens remaining
    /// and report the new authoritative position.
    fn fake_prefill_result(&mut self, req: &PrefillStepItem) -> PrefillRequestResult {
        let start = self
            .prefill_positions
            .get(&req.request_id)
            .copied()
            .unwrap_or(0);
        let chunk = (req.prompt_tokens.len() - start).min(req.chunk_budget);
        let prefill_pos = start + chunk;
        let completed = prefill_pos == req.prompt_tokens.len();
        if completed {
            self.prefill_positions.remove(&req.request_id);
        } else {
            self.prefill_positions.insert(req.request_id, prefill_pos);
        }
        PrefillRequestResult {
            request_id: req.request_id,
            first_token: 100 + req.request_id.get() as u32,
            first_token_logprob: None,
            prompt_logprobs: None,
            cached_tokens: 0,
            completed,
            prefill_pos,
        }
    }

    pub(crate) fn ensure_request_tokens(
        &mut self,
        request_id: RequestId,
        token_count: usize,
    ) -> Result<()> {
        let current_tokens = self.held_tokens.get(&request_id).copied().unwrap_or(0);
        let current_blocks = blocks_needed(current_tokens, self.block_size);
        let needed_blocks = blocks_needed(token_count, self.block_size);
        let grow = needed_blocks.saturating_sub(current_blocks);
        if grow > self.available_blocks {
            anyhow::bail!("fake KV capacity exhausted");
        }
        self.available_blocks -= grow;
        self.held_tokens.insert(request_id, token_count);
        Ok(())
    }
}

impl ModelExecutor for FakeExecutor {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn max_request_blocks(&self) -> usize {
        self.max_request_blocks
    }

    fn max_context_tokens(&self) -> usize {
        self.max_context_tokens
    }

    fn max_decode_batch_size(&self) -> usize {
        64
    }

    fn available_blocks(&self) -> usize {
        self.available_blocks
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        self.stop_token == Some(token_id)
    }

    fn drop_request(&mut self, request_id: RequestId) -> Result<()> {
        if let Some(tokens) = self.held_tokens.remove(&request_id) {
            self.available_blocks += blocks_needed(tokens, self.block_size);
        }
        self.prefill_positions.remove(&request_id);
        self.dropped.lock().unwrap().push(request_id.get());
        Ok(())
    }

    fn begin_kv_prefetch(
        &mut self,
        request_id: RequestId,
        _prompt_tokens: &[u32],
        _lora_adapter: Option<&str>,
        _reserve_floor: usize,
    ) -> bool {
        self.prefetch_offers.lock().unwrap().push(request_id.get());
        false
    }

    fn list_lora_adapters(&self) -> Vec<String> {
        let mut names: Vec<_> = self.loaded_lora_adapters.iter().cloned().collect();
        names.sort();
        names
    }

    fn unload_lora_adapter(&mut self, request: &UnloadLoraAdapterRequest) -> Result<()> {
        anyhow::ensure!(
            self.loaded_lora_adapters.remove(&request.lora_name),
            "LoRA adapter is not loaded: {}",
            request.lora_name
        );
        Ok(())
    }

    fn execute_prefill(&mut self, plan: PrefillPlan<'_>) -> Result<PrefillResult> {
        for req in plan.requests {
            self.ensure_request_tokens(req.request_id, req.prompt_tokens.len())?;
        }
        Ok(PrefillResult {
            requests: plan
                .requests
                .iter()
                .map(|req| self.fake_prefill_result(req))
                .collect(),
            dflash_context_captured_requests: Vec::new(),
        })
    }

    fn execute_decode(&mut self, plan: DecodePlan<'_>) -> Result<crate::executor::DecodeResult> {
        if !self.decode_delay.is_zero() {
            std::thread::sleep(self.decode_delay);
        }
        if self.fail_decode_once {
            self.fail_decode_once = false;
            anyhow::bail!("fake decode KV capacity exhausted");
        }

        for req in plan.requests {
            let current_tokens = self
                .held_tokens
                .get(&req.request_id)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("missing fake request state"))?;
            self.ensure_request_tokens(req.request_id, current_tokens + 1)?;
        }

        Ok(crate::executor::DecodeResult {
            requests: plan
                .requests
                .iter()
                .map(|req| DecodeRequestResult {
                    request_id: req.request_id,
                    token: 200 + req.request_id.get() as u32,
                    logprob: None,
                })
                .collect(),
        })
    }

    fn execute_unified(&mut self, plan: UnifiedPlan<'_>) -> Result<UnifiedResult> {
        for req in plan.prefill_requests {
            self.ensure_request_tokens(req.request_id, req.prompt_tokens.len())?;
        }
        for req in plan.decode_requests {
            let current_tokens = self
                .held_tokens
                .get(&req.request_id)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("missing fake request state"))?;
            self.ensure_request_tokens(req.request_id, current_tokens + 1)?;
        }

        Ok(UnifiedResult {
            prefill_requests: plan
                .prefill_requests
                .iter()
                .map(|req| self.fake_prefill_result(req))
                .collect(),
            decode_requests: plan
                .decode_requests
                .iter()
                .map(|req| DecodeRequestResult {
                    request_id: req.request_id,
                    token: 200 + req.request_id.get() as u32,
                    logprob: None,
                })
                .collect(),
        })
    }
}

/// A minimal contract request: `prompt_len` filler tokens, default sampling.
pub(crate) fn request(prompt_len: usize, max_tokens: usize) -> Request {
    Request {
        prompt_tokens: vec![1; prompt_len],
        params: SamplingParams::default(),
        max_tokens,
        lora_adapter: None,
        kv_transfer_params: None,
        logprobs: 0,
        echo: false,
        trace_parent: None,
        client_label: None,
    }
}

pub(crate) fn request_with_lora(
    prompt_len: usize,
    max_tokens: usize,
    lora_adapter: Option<&str>,
) -> Request {
    Request {
        lora_adapter: lora_adapter.map(ToString::to_string),
        ..request(prompt_len, max_tokens)
    }
}
