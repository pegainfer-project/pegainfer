//! The model spread over several cards by layer. Each card runs a contiguous
//! run of the layers in a worker thread of its own, which owns the card's
//! context, its layers and their state for every slot; the first card also
//! embeds and the last reads out and picks the tokens. A pass hands the
//! stream from card to card through host memory.

use std::path::Path;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread::JoinHandle;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use half::bf16;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_kernels::ops::copy_hidden_token_range_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;
use pegainfer_sample::LogprobRequest;
use pegainfer_sample::SampleScratch;
use pegainfer_sample::select_batch;
use pegainfer_sample::token_logprobs_batch;

use super::Model;
use super::ModelState;
use super::Pass;
use crate::weights::Geometry;
use crate::weights::load_config;

/// What a pass hands back from the last card.
#[derive(Clone, Debug)]
pub enum Readout {
    /// The stream the last layer leaves, `[rows, stream_width]`.
    Stream,
    /// The next token of each sequence: from the last row of a prompt, or
    /// from every row of a decode step.
    Sample(Sampling),
}

/// One row's params and request local step per sequence, and a fresh seed
/// for the call, as `pegainfer_sample::select_batch` takes them, and per
/// sequence how many top logprobs to report with its token, if any.
#[derive(Clone, Debug)]
pub struct Sampling {
    pub params: Vec<SamplingParams>,
    pub steps: Vec<u64>,
    pub seed: u64,
    pub logprobs: Vec<Option<usize>>,
}

pub enum Output {
    Stream(Vec<bf16>),
    /// One token per sequence, and its logprob where the sequence asked.
    Tokens {
        ids: Vec<u32>,
        logprobs: Vec<Option<TokenLogprob>>,
    },
}

#[derive(Clone)]
enum Slots {
    Prefill(usize),
    Decode(Vec<usize>),
}

enum Work {
    Pass {
        slots: Slots,
        ids: Vec<u32>,
        /// The stream the previous card left; the first card embeds instead.
        stream: Option<Vec<bf16>>,
        /// What to hand back, on the last card only.
        readout: Option<Readout>,
    },
    Reset(usize),
}

/// One card's part: its layers, the bookends when it is the first or the
/// last, the state of its layers for every slot, and on the last card the
/// sampler's scratch.
struct Stage {
    ctx: DeviceContext,
    model: Model,
    state: ModelState,
    sampler: Option<SampleScratch>,
}

impl Stage {
    fn run(
        &mut self,
        slots: &Slots,
        ids: &[u32],
        stream: Option<Vec<bf16>>,
        readout: Option<Readout>,
    ) -> Result<Output> {
        let rows = ids.len();
        let ctx = &self.ctx;
        let stream = match stream {
            None => self.model.embed(ctx, ids)?,
            Some(host) => HiddenStates::from_host(ctx, &host, self.model.g.stream_width(), rows)?,
        };
        let pass = match slots {
            Slots::Prefill(slot) => Pass::Prefill(*slot),
            Slots::Decode(slots) => Pass::Decode(slots),
        };
        let out = self
            .model
            .advance(ctx, ids, stream, &mut self.state, pass)?;
        match readout {
            None | Some(Readout::Stream) => {
                let host = ctx.stream.clone_dtoh(&out.data)?;
                ctx.sync()?;
                Ok(Output::Stream(host))
            }
            Some(Readout::Sample(sampling)) => {
                let read = match slots {
                    Slots::Prefill(_) => {
                        let mut last = HiddenStates::zeros(ctx, out.hidden_dim, 1)?;
                        copy_hidden_token_range_into(ctx, &out, rows - 1, &mut last, 0, 1)?;
                        last
                    }
                    Slots::Decode(_) => out,
                };
                let (_, logits) = self.model.readout(ctx, &read)?;
                let scratch = self
                    .sampler
                    .as_mut()
                    .context("only the last card picks tokens")?;
                let params: Vec<&SamplingParams> = sampling.params.iter().collect();
                let ids = select_batch(
                    ctx,
                    &logits,
                    &params,
                    &sampling.steps,
                    sampling.seed,
                    scratch,
                )?;
                // From the raw logits, as vLLM reports them by default.
                let wanted: Vec<LogprobRequest> = sampling
                    .logprobs
                    .iter()
                    .zip(&ids)
                    .enumerate()
                    .filter_map(|(row, (top_k, &picked))| {
                        top_k.map(|top_k| LogprobRequest { row, picked, top_k })
                    })
                    .collect();
                let mut scored = token_logprobs_batch(ctx, &logits, &wanted)?.into_iter();
                let logprobs = sampling
                    .logprobs
                    .iter()
                    .map(|top_k| top_k.and_then(|_| scored.next()))
                    .collect();
                Ok(Output::Tokens { ids, logprobs })
            }
        }
    }

    fn work(&mut self, work: Work) -> Result<Option<Output>> {
        match work {
            Work::Pass {
                slots,
                ids,
                stream,
                readout,
            } => self.run(&slots, &ids, stream, readout).map(Some),
            Work::Reset(slot) => {
                self.state.reset(&self.ctx, slot)?;
                self.ctx.sync()?;
                Ok(None)
            }
        }
    }
}

struct StageHandle {
    work: Option<mpsc::Sender<Work>>,
    replies: mpsc::Receiver<Result<Option<Output>>>,
    join: Option<JoinHandle<()>>,
    device: usize,
}

impl StageHandle {
    fn ask(&self, work: Work) -> Result<Option<Output>> {
        self.work
            .as_ref()
            .expect("the work channel lives as long as the handle")
            .send(work)
            .ok()
            .with_context(|| format!("card {}'s worker stopped", self.device))?;
        self.replies
            .recv()
            .with_context(|| format!("card {}'s worker stopped", self.device))?
            .with_context(|| format!("on card {}", self.device))
    }
}

impl Drop for StageHandle {
    fn drop(&mut self) {
        // Closing the work channel ends the worker's loop.
        self.work.take();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

pub struct SplitModel {
    stages: Vec<StageHandle>,
}

impl SplitModel {
    /// Load `layers` (increasing checkpoint indices) over `devices`, one
    /// contiguous run per device in order, with state for `slots` sequences of
    /// `capacity` tokens each. The same device may appear more than once.
    pub fn load(
        model_dir: &Path,
        devices: &[usize],
        layers: &[usize],
        slots: usize,
        capacity: usize,
    ) -> Result<Self> {
        ensure!(
            !devices.is_empty() && devices.len() <= layers.len(),
            "{} devices for {} layers",
            devices.len(),
            layers.len()
        );
        ensure!(
            layers.windows(2).all(|pair| pair[0] < pair[1]),
            "layers must be loaded in increasing order, got {layers:?}"
        );
        let (base, extra) = (layers.len() / devices.len(), layers.len() % devices.len());
        let mut start = 0;
        let mut pending = Vec::with_capacity(devices.len());
        for (index, &device) in devices.iter().enumerate() {
            let end = start + base + usize::from(index < extra);
            let run = layers[start..end].to_vec();
            start = end;
            let part = Part {
                first: index == 0,
                last: index + 1 == devices.len(),
            };
            pending.push(spawn_stage(
                model_dir.to_path_buf(),
                device,
                run,
                part,
                slots,
                capacity,
            )?);
        }
        let mut stages = Vec::with_capacity(pending.len());
        for (stage, ready) in pending {
            ready
                .recv()
                .with_context(|| format!("card {}'s worker stopped while loading", stage.device))?
                .with_context(|| format!("loading card {}'s layers", stage.device))?;
            stages.push(stage);
        }
        Ok(Self { stages })
    }

    fn pass(&self, slots: &Slots, ids: &[u32], readout: Readout) -> Result<Output> {
        let mut stream = None;
        let mut readout = Some(readout);
        let last = self.stages.len() - 1;
        for (index, stage) in self.stages.iter().enumerate() {
            let output = stage
                .ask(Work::Pass {
                    slots: slots.clone(),
                    ids: ids.to_vec(),
                    stream: stream.take(),
                    readout: if index == last { readout.take() } else { None },
                })?
                .context("a pass hands back an output")?;
            if index == last {
                return Ok(output);
            }
            match output {
                Output::Stream(host) => stream = Some(host),
                Output::Tokens { .. } => {
                    bail!("card {} picked tokens before the last card", stage.device)
                }
            }
        }
        unreachable!("the last card returns")
    }

    /// One prompt in `slot`, from the slot's fresh state.
    pub fn prefill(&self, slot: usize, ids: &[u32], readout: Readout) -> Result<Output> {
        self.pass(&Slots::Prefill(slot), ids, readout)
    }

    /// One more token in each of `slots` as one batch: `ids[i]` belongs to
    /// `slots[i]`, and so does row `i` of the output.
    pub fn decode(&self, slots: &[usize], ids: &[u32], readout: Readout) -> Result<Output> {
        self.pass(&Slots::Decode(slots.to_vec()), ids, readout)
    }

    /// Return `slot` to a fresh state on every card.
    pub fn reset(&self, slot: usize) -> Result<()> {
        for stage in &self.stages {
            stage.ask(Work::Reset(slot))?;
        }
        Ok(())
    }
}

/// Where a card's run sits: the first card embeds, the last reads out and
/// picks the tokens.
#[derive(Clone, Copy)]
struct Part {
    first: bool,
    last: bool,
}

/// Destroys the cuBLAS handles a card's worker thread created.
struct CublasThreadGuard;

impl Drop for CublasThreadGuard {
    fn drop(&mut self) {
        unsafe { pegainfer_kernels::ffi::cublas_destroy() };
    }
}

/// Start one card's worker; the receiver reports whether its part loaded.
fn spawn_stage(
    model_dir: PathBuf,
    device: usize,
    layers: Vec<usize>,
    part: Part,
    slots: usize,
    capacity: usize,
) -> Result<(StageHandle, mpsc::Receiver<Result<()>>)> {
    let (work, work_rx) = mpsc::channel::<Work>();
    let (reply_tx, replies) = mpsc::channel();
    let (ready_tx, ready) = mpsc::channel();
    let join = std::thread::Builder::new()
        .name(format!("qwen4-card{device}"))
        .spawn(move || {
            // Declared first, so it drops after the part has released the card.
            let _cublas = CublasThreadGuard;
            // The context binds this thread to the card and gives it its own
            // cuBLAS handle, so the part loads and runs here.
            let load = || -> Result<Stage> {
                let ctx = DeviceContext::new_with_device(device)?;
                let config = load_config(&model_dir)?;
                let bookends = part.first || part.last;
                let model = Model::load(&ctx, &model_dir, &config, &layers, capacity, bookends)?;
                let state = model.new_state(&ctx, slots, capacity)?;
                let vocab = Geometry::from_config(&config).vocab;
                let sampler = part
                    .last
                    .then(|| SampleScratch::new(&ctx, vocab, slots))
                    .transpose()?;
                Ok(Stage {
                    ctx,
                    model,
                    state,
                    sampler,
                })
            };
            let mut stage = match load() {
                Ok(stage) => {
                    let _ = ready_tx.send(Ok(()));
                    stage
                }
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                    return;
                }
            };
            while let Ok(work) = work_rx.recv() {
                if reply_tx.send(stage.work(work)).is_err() {
                    return;
                }
            }
        })
        .context("spawning a card's worker")?;
    Ok((
        StageHandle {
            work: Some(work),
            replies,
            join: Some(join),
            device,
        },
        ready,
    ))
}
