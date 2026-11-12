//! The model spread over several cards by layer. Each card runs a contiguous
//! run of the layers in a worker thread of its own, which owns the card's
//! context, its layers and their state for every slot; the first card also
//! embeds and the last reads out. A pass hands the stream from card to card
//! through host memory.

use std::path::Path;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread::JoinHandle;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use half::bf16;
use pegainfer_kernels::ops::copy_hidden_token_range_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;

use super::Model;
use super::ModelState;
use super::Pass;
use crate::weights::Geometry;
use crate::weights::load_config;

/// What a pass hands back from the last card.
#[derive(Clone, Copy, Debug)]
pub enum Readout {
    /// The stream the last layer leaves, `[rows, stream_width]`.
    Stream,
    /// The logits of the last row, `[vocab]`.
    LastRow,
    /// The logits of every row, `[rows, vocab]`.
    EveryRow,
}

pub enum Output {
    Stream(Vec<bf16>),
    Logits(Vec<f32>),
}

#[derive(Clone)]
enum Slots {
    Prefill(usize),
    Decode(Vec<usize>),
}

struct Job {
    slots: Slots,
    ids: Vec<u32>,
    /// The stream the previous card left; the first card embeds instead.
    stream: Option<Vec<bf16>>,
    /// What to hand back, on the last card only.
    readout: Option<Readout>,
}

/// One card's part: its layers, the bookends when it is the first or the
/// last, and the state of its layers for every slot.
struct Stage {
    ctx: DeviceContext,
    model: Model,
    state: ModelState,
}

impl Stage {
    fn run(&mut self, job: Job) -> Result<Output> {
        let rows = job.ids.len();
        let ctx = &self.ctx;
        let stream = match job.stream {
            None => self.model.embed(ctx, &job.ids)?,
            Some(host) => HiddenStates::from_host(ctx, &host, self.model.g.stream_width(), rows)?,
        };
        let pass = match &job.slots {
            Slots::Prefill(slot) => Pass::Prefill(*slot),
            Slots::Decode(slots) => Pass::Decode(slots),
        };
        let out = self
            .model
            .advance(ctx, &job.ids, stream, &mut self.state, pass)?;
        let output = match job.readout {
            None | Some(Readout::Stream) => {
                let host = ctx.stream.clone_dtoh(&out.data)?;
                ctx.sync()?;
                Output::Stream(host)
            }
            Some(Readout::LastRow) => {
                let mut last = HiddenStates::zeros(ctx, out.hidden_dim, 1)?;
                copy_hidden_token_range_into(ctx, &out, rows - 1, &mut last, 0, 1)?;
                Output::Logits(self.model.readout(ctx, &last)?.1.to_host(ctx)?)
            }
            Some(Readout::EveryRow) => {
                Output::Logits(self.model.readout(ctx, &out)?.1.to_host(ctx)?)
            }
        };
        Ok(output)
    }
}

struct StageHandle {
    jobs: Option<mpsc::Sender<Job>>,
    outputs: mpsc::Receiver<Result<Output>>,
    join: Option<JoinHandle<()>>,
    device: usize,
}

impl Drop for StageHandle {
    fn drop(&mut self) {
        // Closing the job channel ends the worker's loop.
        self.jobs.take();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

pub struct SplitModel {
    stages: Vec<StageHandle>,
    pub vocab: usize,
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
        let vocab = Geometry::from_config(&load_config(model_dir)?).vocab;
        let (base, extra) = (layers.len() / devices.len(), layers.len() % devices.len());
        let mut start = 0;
        let mut pending = Vec::with_capacity(devices.len());
        for (index, &device) in devices.iter().enumerate() {
            let end = start + base + usize::from(index < extra);
            let run = layers[start..end].to_vec();
            start = end;
            let bookends = index == 0 || index + 1 == devices.len();
            pending.push(spawn_stage(
                model_dir.to_path_buf(),
                device,
                run,
                bookends,
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
        Ok(Self { stages, vocab })
    }

    fn pass(&self, slots: &Slots, ids: &[u32], readout: Readout) -> Result<Output> {
        let mut stream = None;
        let last = self.stages.len() - 1;
        for (index, stage) in self.stages.iter().enumerate() {
            let job = Job {
                slots: slots.clone(),
                ids: ids.to_vec(),
                stream: stream.take(),
                readout: (index == last).then_some(readout),
            };
            stage
                .jobs
                .as_ref()
                .expect("the job channel lives as long as the handle")
                .send(job)
                .ok()
                .with_context(|| format!("card {}'s worker stopped", stage.device))?;
            let output = stage
                .outputs
                .recv()
                .with_context(|| format!("card {}'s worker stopped", stage.device))?
                .with_context(|| format!("on card {}", stage.device))?;
            if index == last {
                return Ok(output);
            }
            match output {
                Output::Stream(host) => stream = Some(host),
                Output::Logits(_) => bail!("card {} read out before the last card", stage.device),
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
    bookends: bool,
    slots: usize,
    capacity: usize,
) -> Result<(StageHandle, mpsc::Receiver<Result<()>>)> {
    let (jobs, job_rx) = mpsc::channel::<Job>();
    let (output_tx, outputs) = mpsc::channel();
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
                let model = Model::load(&ctx, &model_dir, &config, &layers, capacity, bookends)?;
                let state = model.new_state(&ctx, slots, capacity)?;
                Ok(Stage { ctx, model, state })
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
            while let Ok(job) = job_rx.recv() {
                if output_tx.send(stage.run(job)).is_err() {
                    return;
                }
            }
        })
        .context("spawning a card's worker")?;
    Ok((
        StageHandle {
            jobs: Some(jobs),
            outputs,
            join: Some(join),
            device,
        },
        ready,
    ))
}
