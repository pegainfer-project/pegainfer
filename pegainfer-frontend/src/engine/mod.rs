//! The engine request/event contract, split by cohesion:
//!
//! - [`request`] / [`event`] — the pure data both sides exchange.
//! - [`sink`] — the per-request `TokenSink` over the shared tagged channel.
//! - [`kv`] — the KV-prefix resolution passed in, and the KV block-event feed.
//! - [`control`] — the LoRA control-plane commands.
//! - [`handle`] — engine launch options plus the frontend's `EngineHandle`:
//!   routing, load feed, shutdown.
//!
//! Everything is re-exported flat here, so `pegainfer_frontend::engine::X`
//! paths are unchanged by the split.

mod control;
mod event;
mod handle;
mod kv;
mod request;
mod sink;

pub use control::*;
pub use event::*;
pub use handle::*;
pub use kv::*;
pub use request::*;
pub use sink::*;
