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

use crate::config::Config;

pub static MODEL_LINE: Qwen4Line = Qwen4Line;

pub struct Qwen4Line;

impl ModelLine for Qwen4Line {
    fn name(&self) -> &'static str {
        "Qwen4"
    }

    /// Constructs the whole [`Config`], not just the structural probe. `detect`
    /// claims a config through this method, so the geometry the line advertises —
    /// the n-gram partition, the head groups — has to be derivable here as well,
    /// or detection would accept a config the geometry layer cannot lay out. This
    /// is also what keeps `probe_config_json` from being a weaker gate than the
    /// path the line actually runs.
    fn probe(&self, config: &serde_json::Value) -> Result<(), String> {
        Config::from_json(config)
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    fn launch(&self, _ctx: &LaunchContext<'_>) -> anyhow::Result<LaunchedEngine> {
        anyhow::bail!(
            "Qwen4 is detected and its config is validated, but the text graph is \
             not built yet (slice B). Refusing to start rather than serving a partial model. \
             The GDN gated-norm activation this checkpoint selects (sigmoid, not Qwen3.5's \
             silu) became selectable in the shared kernel in #1129; the missing piece is \
             wiring that call site, which lands with the text graph in slice B rather than \
             being an open kernel dependency."
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
        assert_eq!(MODEL_LINE.name(), "Qwen4");
        MODEL_LINE.probe(&json).unwrap();
    }

    /// `probe_config_json` on its own accepts any positive `ple_embed_dim`; it is
    /// the n-gram derivation that rejects one which cannot split across the heads.
    /// `detect` claims a config through `probe`, so a config only the geometry
    /// layer refuses must be refused here too — otherwise `detect` would claim it
    /// and `launch` would report it as "validated".
    #[test]
    fn the_probe_rejects_geometry_that_does_not_derive() {
        let mut json: serde_json::Value =
            serde_json::from_str(include_str!("../tests/frozen_config.json")).unwrap();
        json["text_config"]["ple_embed_dim"] = serde_json::json!(2561);
        let err = MODEL_LINE.probe(&json).unwrap_err();
        assert!(err.contains("ple_embed_dim"), "{err}");
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
            "Qwen4",
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
