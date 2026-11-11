//! The model around the layers: the token embedding repeated into the
//! hyper-connection stream, decoder layers in order, the top-level mixer that
//! reads the stream out and the untied head. Any subset of the layers can be
//! loaded, in checkpoint order; that is what one card can hold against the
//! reference while serving still waits for all 48.

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use pegainfer_kernels::ops::HcMixScratch;
use pegainfer_kernels::ops::embedding_batch;
use pegainfer_kernels::ops::gemm_into_checked;
use pegainfer_kernels::ops::hc_mix_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;

use crate::config::Config;
use crate::layer::LayerState;
use crate::layer::LayerTrace;
use crate::layer::Rope;
use crate::layer::layer_decode;
use crate::layer::layer_prefill;
use crate::weights::Bookends;
use crate::weights::DecoderLayer;
use crate::weights::Geometry;
use crate::weights::load_bookends;
use crate::weights::load_layer;

#[cfg(test)]
mod golden_gate;

pub struct Model {
    pub g: Geometry,
    pub bookends: Bookends,
    /// Checkpoint index and weights of every loaded layer, in forward order.
    pub layers: Vec<(usize, DecoderLayer)>,
    pub rope: Rope,
}

/// One sequence's state across the loaded layers.
pub struct ModelState {
    pub layers: Vec<LayerState>,
}

/// Everything one pass produced, kept for the reference gate.
pub struct Forward {
    pub stream_in: HiddenStates,
    pub layers: Vec<LayerTrace>,
    pub mixed: HiddenStates,
    pub logits: HiddenStates,
}

impl Model {
    /// Load the bookends and the given layers, with RoPE tables for `positions`.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the model line is the next caller")
    )]
    pub(crate) fn load(
        ctx: &DeviceContext,
        model_dir: &std::path::Path,
        config: &Config,
        layers: &[usize],
        positions: usize,
    ) -> Result<Self> {
        ensure!(
            layers.windows(2).all(|pair| pair[0] < pair[1]),
            "layers must be loaded in increasing order, got {layers:?}"
        );
        let g = Geometry::from_config(config);
        let bookends = load_bookends(ctx, model_dir, config)?;
        let layers = layers
            .iter()
            .map(|&index| Ok((index, load_layer(ctx, model_dir, config, index)?)))
            .collect::<Result<Vec<_>>>()?;
        let rope = Rope::new(ctx, &g, positions)?;
        Ok(Self {
            g,
            bookends,
            layers,
            rope,
        })
    }

    pub fn new_state(&self, ctx: &DeviceContext, capacity: usize) -> Result<ModelState> {
        let layers = self
            .layers
            .iter()
            .map(|(_, layer)| LayerState::new(ctx, &self.g, layer, capacity))
            .collect::<Result<Vec<_>>>()?;
        Ok(ModelState { layers })
    }

    /// The tokens' embeddings repeated `hc_count` times each: the stream row of
    /// a token is its embedding row written `hc_count` times in a row, so one
    /// gather with every id repeated fills `[rows, hc_count * hidden]` exactly.
    pub fn embed(&self, ctx: &DeviceContext, ids: &[u32]) -> Result<HiddenStates> {
        ensure!(!ids.is_empty(), "the prompt is empty");
        let repeated: Vec<u32> = ids
            .iter()
            .flat_map(|&id| std::iter::repeat_n(id, self.g.hc_count))
            .collect();
        let ids_d: CudaSlice<u32> = ctx.stream.clone_htod(&repeated)?;
        let mut rows = HiddenStates::zeros(ctx, self.g.hidden, repeated.len())?;
        embedding_batch(ctx, &self.bookends.embed, &ids_d, &mut rows)?;
        Ok(HiddenStates {
            data: rows.data,
            hidden_dim: self.g.stream_width(),
            seq_len: ids.len(),
        })
    }

    /// The top-level mixer and the head: the stream read out to one hidden row
    /// per token (no final norm), then the logits.
    pub fn readout(
        &self,
        ctx: &DeviceContext,
        stream: &HiddenStates,
    ) -> Result<(HiddenStates, HiddenStates)> {
        let g = &self.g;
        let rows = stream.seq_len;
        let mut scratch = HcMixScratch::new(ctx, rows, g.hidden, g.hc_count, g.hc_lowrank, false)?;
        let mut mixed = HiddenStates::zeros(ctx, g.hidden, rows)?;
        let m = &self.bookends.mixer;
        hc_mix_into(
            ctx,
            &m.norm,
            &m.down,
            &m.up,
            None,
            stream,
            g.hc_count,
            g.rms_eps,
            &mut scratch,
            &mut mixed,
            None,
        )?;
        let mut logits = HiddenStates::zeros(ctx, g.vocab, rows)?;
        gemm_into_checked(ctx, &self.bookends.lm_head, &mixed, &mut logits)?;
        Ok((mixed, logits))
    }

    /// `ids` are the tokens `stream_in` embeds: the whole prompt, or the one
    /// token of a decode step. Only the n-gram block reads them.
    fn forward(
        &self,
        ctx: &DeviceContext,
        ids: &[u32],
        stream_in: HiddenStates,
        state: &mut ModelState,
        decode: bool,
    ) -> Result<Forward> {
        ensure!(
            state.layers.len() == self.layers.len(),
            "the state holds {} layers, the model {}",
            state.layers.len(),
            self.layers.len()
        );
        let mut traces: Vec<LayerTrace> = Vec::with_capacity(self.layers.len());
        for ((_, layer), layer_state) in self.layers.iter().zip(&mut state.layers) {
            let input = traces.last().map_or(&stream_in, |t| &t.stream_out);
            let trace = if decode {
                layer_decode(ctx, &self.g, layer, &self.rope, layer_state, ids[0], input)?
            } else {
                layer_prefill(ctx, &self.g, layer, &self.rope, layer_state, ids, input)?
            };
            traces.push(trace);
        }
        let last = traces.last().map_or(&stream_in, |t| &t.stream_out);
        let (mixed, logits) = self.readout(ctx, last)?;
        Ok(Forward {
            stream_in,
            layers: traces,
            mixed,
            logits,
        })
    }

    /// A whole prompt from a fresh state.
    pub fn prefill(
        &self,
        ctx: &DeviceContext,
        ids: &[u32],
        state: &mut ModelState,
    ) -> Result<Forward> {
        let stream = self.embed(ctx, ids)?;
        self.forward(ctx, ids, stream, state, false)
    }

    /// One more token on the carried state.
    pub fn decode(&self, ctx: &DeviceContext, id: u32, state: &mut ModelState) -> Result<Forward> {
        let stream = self.embed(ctx, &[id])?;
        self.forward(ctx, &[id], stream, state, true)
    }
}
