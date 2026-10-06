//! This line's [`ModelLine`] implementation.
//!
//! Slice A is the foundation only: the probe and the validated config. There is no
//! text graph, no weights loader and no step yet, so detection succeeds and
//! `launch` refuses with the reason rather than pretending to serve. Slice B
//! (#1127) adds the decoder layer, the forward and the capturable decode step;
//! slice C (#1128) takes it to multiple ranks.

use pegainfer_frontend::engine::LaunchedEngine;
use pegainfer_frontend::model_line::LaunchContext;
use pegainfer_frontend::model_line::ModelLine;

pub static MODEL_LINE: Qwen38FlashNextLine = Qwen38FlashNextLine;

pub struct Qwen38FlashNextLine;

impl ModelLine for Qwen38FlashNextLine {
    fn name(&self) -> &'static str {
        "Qwen3.8-Flash-Next"
    }

    fn probe(&self, config: &serde_json::Value) -> Result<(), String> {
        crate::probe::probe_config_json(config).map_err(|error| error.to_string())
    }

    fn launch(&self, _ctx: &LaunchContext<'_>) -> anyhow::Result<LaunchedEngine> {
        anyhow::bail!(
            "Qwen3.8-Flash-Next is detected and its config is validated, but the text graph is \
             not built yet (slice B). Refusing to start rather than serving a partial model. \
             Serving also depends on the GDN gated-norm activation being sigmoid rather than \
             silu, which this checkpoint's output_gate_type selects."
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_registry_entry_detects_the_frozen_revision() {
        let json: serde_json::Value =
            serde_json::from_str(include_str!("../tests/frozen_config.json")).unwrap();
        assert_eq!(MODEL_LINE.name(), "Qwen3.8-Flash-Next");
        MODEL_LINE.probe(&json).unwrap();
    }

    /// `pegainfer-server`'s `model_lines()` collects `&'static dyn ModelLine`, and
    /// the server cannot be compiled without a CUDA toolchain — so the coercion,
    /// the `'static` bound and the trait's `Send + Sync` supertraits are checked
    /// here instead, where they are the only part of that registration line that
    /// can fail to type-check.
    #[test]
    fn the_line_coerces_to_the_registry_object_type() {
        fn assert_object_safe(line: &'static dyn ModelLine) -> &'static str {
            line.name()
        }
        assert_eq!(
            assert_object_safe(&MODEL_LINE),
            "Qwen3.8-Flash-Next",
            "the server's registry entry must keep coercing to `&'static dyn ModelLine`"
        );
    }

    /// The line has no engine yet, so it must not claim any shared flag: with an
    /// empty consumed set the registry rejects every shared flag for this line,
    /// which is the right behaviour for a foundation-only slice and is what makes
    /// `launch`'s refusal the only reachable outcome.
    #[test]
    fn the_line_consumes_no_shared_args_yet() {
        assert!(MODEL_LINE.consumed_shared_args().is_empty());
    }
}
