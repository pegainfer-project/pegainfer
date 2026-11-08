//! This line's [`ModelLine`] implementation.
//!
//! There is no text graph, no weights loader and no step yet, so detection
//! succeeds and `launch` refuses rather than serving a partial model.

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

    /// Builds the whole [`Config`], so `detect` refuses a `qwen4_exp` config the
    /// geometry cannot lay out before the frontend starts.
    fn probe(&self, config: &serde_json::Value) -> Result<(), String> {
        Config::from_json(config)
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    fn launch(&self, _ctx: &LaunchContext<'_>) -> anyhow::Result<LaunchedEngine> {
        anyhow::bail!("this qwen4_exp checkpoint is detected, but inference is not implemented yet")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `detect` claims a config through `probe`, so a config only the geometry
    /// refuses must be refused here too.
    #[test]
    fn the_probe_rejects_geometry_that_does_not_derive() {
        let mut json: serde_json::Value =
            serde_json::from_str(include_str!("../tests/frozen_config.json")).unwrap();
        json["text_config"]["ple_embed_dim"] = serde_json::json!(2561);
        let err = MODEL_LINE.probe(&json).unwrap_err();
        assert!(err.contains("ple_embed_dim"), "{err}");
    }
}
