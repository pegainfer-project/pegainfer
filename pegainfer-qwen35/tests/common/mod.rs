use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;

use pegainfer_frontend::engine::Engine;
use pegainfer_frontend::engine::EosPolicy;
use pegainfer_frontend::engine::Request;
use pegainfer_frontend::engine::RequestControl;
use pegainfer_frontend::engine::RequestId;
use pegainfer_frontend::engine::RequestUpdate;
use pegainfer_frontend::engine::SchedulerHandle;
use pegainfer_frontend::engine::SchedulerMetrics;
use pegainfer_frontend::engine::StepOutputs;
use pegainfer_frontend::engine::StepReceiver;
use pegainfer_frontend::engine::StopPolicy;
use pegainfer_frontend::sampler::SamplingParams;
use vllm_text::Error;
use vllm_text::Result;
use vllm_text::backend::hf::ResolvedModelFiles;
use vllm_text::backend::hf::TokenizerSource;
use vllm_text::tokenizer::DynTokenizer;
use vllm_text::tokenizer::HuggingFaceTokenizer;
use vllm_text::tokenizer::TekkenTokenizer;
use vllm_text::tokenizer::TiktokenTokenizer;

pub(crate) mod model_fixture;

pub(crate) use model_fixture::model_path_or_skip;

#[allow(dead_code)]
pub(crate) fn request(
    prompt_tokens: Vec<u32>,
    params: SamplingParams,
    max_tokens: usize,
) -> Request {
    Request {
        prompt_tokens,
        stop_policy: StopPolicy::new(
            if params.ignore_eos {
                EosPolicy::Ignore
            } else {
                EosPolicy::ModelDefault
            },
            Vec::new(),
        ),
        params,
        max_tokens,
        lora_adapter: None,
        kv_transfer_params: None,
        logprobs: None,
        prompt_logprobs: None,
        trace_parent: None,
        client_label: None,
    }
}

// Cancel on scope exit or assertion failure before the harness joins the scheduler.
pub(crate) struct RequestGuard(RequestControl);

#[allow(dead_code)]
impl RequestGuard {
    pub(crate) fn id(&self) -> RequestId {
        self.0.id()
    }

    pub(crate) fn abort(&self) {
        self.0.abort();
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[allow(dead_code)]
pub(crate) struct EngineHarness {
    handle: Option<SchedulerHandle>,
    join: Option<std::thread::JoinHandle<()>>,
    steps: StepReceiver,
    pending: HashMap<RequestId, VecDeque<RequestUpdate>>,
}

#[allow(dead_code)]
impl EngineHarness {
    pub(crate) fn new(mut engine: Engine) -> Self {
        assert_eq!(engine.schedulers.len(), 1);
        let mut scheduler = engine.schedulers.pop().unwrap();
        Self {
            steps: scheduler
                .handle
                .take_steps()
                .expect("new engine has a step stream"),
            handle: Some(scheduler.handle),
            join: Some(scheduler.join),
            pending: HashMap::new(),
        }
    }

    pub(crate) fn submit(&self, request: Request) -> RequestGuard {
        RequestGuard(self.handle.as_ref().unwrap().submit(request))
    }

    pub(crate) fn metrics(&self) -> SchedulerMetrics {
        self.handle.as_ref().unwrap().metrics()
    }

    pub(crate) fn next(&mut self, id: RequestId) -> RequestUpdate {
        loop {
            if let Some(update) = self.try_next(id) {
                return update;
            }
            let step = self
                .steps
                .blocking_recv()
                .expect("scheduler closed before terminal");
            self.buffer(step);
        }
    }

    pub(crate) fn try_next(&mut self, id: RequestId) -> Option<RequestUpdate> {
        while let Ok(step) = self.steps.try_recv() {
            self.buffer(step);
        }
        self.pending.get_mut(&id).and_then(VecDeque::pop_front)
    }

    fn buffer(&mut self, step: StepOutputs) {
        for update in step.updates {
            self.pending.entry(update.id).or_default().push_back(update);
        }
    }
}

impl Drop for EngineHarness {
    fn drop(&mut self) {
        drop(self.handle.take());
        if let Some(join) = self.join.take() {
            let result = join.join();
            if !std::thread::panicking() {
                result.expect("scheduler thread panicked");
            }
        }
    }
}

#[allow(dead_code)]
pub(crate) fn load_tokenizer(model_path: &str) -> DynTokenizer {
    try_load_tokenizer(model_path)
        .unwrap_or_else(|err| panic!("Failed to load tokenizer for {model_path}: {err}"))
}

// vllm-text exposes model-file resolution as async even though the local
// directory path (all we ever use) is synchronous; bridge it with a throwaway
// current-thread runtime.
#[allow(dead_code)]
fn try_load_tokenizer(model_path: &str) -> Result<DynTokenizer> {
    if tokio::runtime::Handle::try_current().is_ok() {
        return Err(Error::Tokenizer(
            "load_tokenizer cannot be called from inside an active Tokio runtime".to_string(),
        ));
    }
    let files = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| {
            Error::Tokenizer(format!("failed to build tokenizer resolver runtime: {err}"))
        })?
        .block_on(ResolvedModelFiles::new(model_path, None))?;
    match &files.tokenizer {
        TokenizerSource::HuggingFace(path) => Ok(Arc::new(HuggingFaceTokenizer::new(path)?)),
        TokenizerSource::Tiktoken(path) => Ok(Arc::new(TiktokenTokenizer::new(path)?)),
        TokenizerSource::Tekken(path) => Ok(Arc::new(TekkenTokenizer::new(path)?)),
    }
}

#[allow(dead_code)]
pub(crate) fn tp2_device_ordinals() -> Vec<usize> {
    const ENV: &str = "PEGAINFER_TEST_TP_DEVICES";
    let Ok(value) = std::env::var(ENV) else {
        return vec![0, 1];
    };

    let devices: Vec<usize> = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .unwrap_or_else(|err| panic!("{ENV} must be comma-separated CUDA ordinals: {err}"))
        })
        .collect();

    assert_eq!(
        devices.len(),
        2,
        "{ENV} must specify exactly two CUDA ordinals for TP2, e.g. 0,1 or 2,3"
    );
    assert_ne!(
        devices[0], devices[1],
        "{ENV} must specify two distinct CUDA ordinals for TP2"
    );
    devices
}
