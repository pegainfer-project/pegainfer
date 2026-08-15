//! The engine contract, split by cohesion.
//!
//! Step-driven contract (current; qwen3 is migrated, other lines follow):
//!
//! - [`step`] — the wire types: `Request`, `RequestId`, `StepOutputs` with
//!   one flat `RequestUpdate` per touched request per step.
//! - [`request_lifecycle`] — the typestate handles (`QueuedRequest` →
//!   `ActiveRequest` → consumed) that make the event protocol a move-checked
//!   state machine.
//! - [`emitter`] — `StepEmitter`, the scheduler-side single writer of the
//!   per-step buffer.
//! - [`wiring`] — `scheduler_pair` wiring, `SchedulerHandle`, and the
//!   `Engine`/`EngineInfo` bundle a model line's `launch` returns.
//! - [`driver`] — the `Scheduler` trait and the polling `drive` loop.
//! - [`control`] — the LoRA adapter capability: vocabulary + client, outside
//!   the scheduler contract (engines mint their own channel before spawn).
//!
//! Legacy per-token contract, kept until the remaining model lines migrate:
//!
//! - [`request`] / [`event`] — `GenerateRequest` and the `TokenEvent` stream.
//! - [`sink`] — the per-request `TokenSink` over the shared tagged channel.
//! - [`kv`] — KV-prefix resolution.
//! - [`handle`] — `EngineHandle`: routing, load feed, shutdown.
//!
//! Everything is re-exported flat here, so `pegainfer_frontend::engine::X`
//! paths are unchanged by the split.

mod control;
mod driver;
mod emitter;
mod event;
mod handle;
mod kv;
mod metrics;
mod request;
mod request_lifecycle;
mod sink;
mod step;
mod wiring;

pub use control::*;
pub use driver::*;
pub use emitter::*;
pub use event::*;
pub use handle::*;
pub use kv::*;
pub use metrics::*;
pub use request::*;
pub use request_lifecycle::*;
pub use sink::*;
pub use step::*;
pub use wiring::*;
