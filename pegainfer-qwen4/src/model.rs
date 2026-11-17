//! The model around the layers: the token embedding repeated into the
//! hyper-connection stream, decoder layers in order, the top-level mixer that
//! reads the stream out and the untied head. Any subset of the layers can be
//! loaded, in checkpoint order, and [`split`] spreads them over several cards.

use anyhow::Context as _;
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
pub mod split;

pub struct Model {
    pub g: Geometry,
    /// The embedding and the readout, on the part of the model that runs either.
    pub bookends: Option<Bookends>,
    /// Checkpoint index and weights of every loaded layer, in forward order.
    pub layers: Vec<(usize, DecoderLayer)>,
    pub rope: Rope,
}

/// The state of several sequences, one per slot, across the loaded layers.
pub struct ModelState {
    pub layers: Vec<LayerState>,
}

impl ModelState {
    /// Return `slot` to a fresh state on every loaded layer.
    pub fn reset(&mut self, ctx: &DeviceContext, slot: usize) -> Result<()> {
        self.layers
            .iter_mut()
            .try_for_each(|layer| layer.reset(ctx, slot))
    }
}

/// Which slots a forward advances: one prompt in a slot, or one token in each
/// of a batch of slots.
#[derive(Clone, Copy)]
enum Pass<'a> {
    Prefill(usize),
    Decode(&'a [usize]),
}

/// Everything one pass produced, kept for the reference gate.
pub struct Forward {
    pub stream_in: HiddenStates,
    pub layers: Vec<LayerTrace>,
    pub mixed: HiddenStates,
    pub logits: HiddenStates,
}

impl Model {
    /// Load the given layers, the bookends when asked, and RoPE tables for
    /// `positions`.
    pub(crate) fn load(
        ctx: &DeviceContext,
        model_dir: &std::path::Path,
        config: &Config,
        layers: &[usize],
        positions: usize,
        bookends: bool,
    ) -> Result<Self> {
        ensure!(
            layers.windows(2).all(|pair| pair[0] < pair[1]),
            "layers must be loaded in increasing order, got {layers:?}"
        );
        let g = Geometry::from_config(config);
        let bookends = bookends
            .then(|| load_bookends(ctx, model_dir, config))
            .transpose()?;
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

    /// Room for `slots` sequences of `capacity` tokens each.
    pub fn new_state(
        &self,
        ctx: &DeviceContext,
        slots: usize,
        capacity: usize,
    ) -> Result<ModelState> {
        let layers = self
            .layers
            .iter()
            .map(|(_, layer)| LayerState::new(ctx, &self.g, layer, slots, capacity))
            .collect::<Result<Vec<_>>>()?;
        Ok(ModelState { layers })
    }

    fn bookends(&self) -> Result<&Bookends> {
        self.bookends
            .as_ref()
            .context("this part of the model loads neither the embedding nor the head")
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
        embedding_batch(ctx, &self.bookends()?.embed, &ids_d, &mut rows)?;
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
        let bookends = self.bookends()?;
        let rows = stream.seq_len;
        let mut scratch = HcMixScratch::new(ctx, rows, g.hidden, g.hc_count, g.hc_lowrank, false)?;
        let mut mixed = HiddenStates::zeros(ctx, g.hidden, rows)?;
        let m = &bookends.mixer;
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
        gemm_into_checked(ctx, &bookends.lm_head, &mixed, &mut logits)?;
        Ok((mixed, logits))
    }

    fn check_state(&self, state: &ModelState) -> Result<()> {
        ensure!(
            state.layers.len() == self.layers.len(),
            "the state holds {} layers, the model {}",
            state.layers.len(),
            self.layers.len()
        );
        Ok(())
    }

    fn layer_pass(
        &self,
        ctx: &DeviceContext,
        layer: &DecoderLayer,
        layer_state: &mut LayerState,
        ids: &[u32],
        input: &HiddenStates,
        pass: Pass,
    ) -> Result<LayerTrace> {
        match pass {
            Pass::Prefill(slot) => layer_prefill(
                ctx,
                &self.g,
                layer,
                &self.rope,
                layer_state,
                slot,
                ids,
                input,
            ),
            Pass::Decode(slots) => layer_decode(
                ctx,
                &self.g,
                layer,
                &self.rope,
                layer_state,
                slots,
                ids,
                input,
            ),
        }
    }

    /// `ids` are the tokens `stream_in` embeds: the whole prompt, or one token
    /// per slot of a decode step. Only the n-gram block reads them.
    fn forward(
        &self,
        ctx: &DeviceContext,
        ids: &[u32],
        stream_in: HiddenStates,
        state: &mut ModelState,
        pass: Pass,
    ) -> Result<Forward> {
        self.check_state(state)?;
        let mut traces: Vec<LayerTrace> = Vec::with_capacity(self.layers.len());
        for ((_, layer), layer_state) in self.layers.iter().zip(&mut state.layers) {
            let input = traces.last().map_or(&stream_in, |t| &t.stream_out);
            traces.push(self.layer_pass(ctx, layer, layer_state, ids, input, pass)?);
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

    /// The stream the loaded layers leave after `pass`, each layer's stages
    /// dropped once the next layer has its input.
    fn advance(
        &self,
        ctx: &DeviceContext,
        ids: &[u32],
        stream_in: HiddenStates,
        state: &mut ModelState,
        pass: Pass,
    ) -> Result<HiddenStates> {
        self.check_state(state)?;
        let mut stream = stream_in;
        for ((_, layer), layer_state) in self.layers.iter().zip(&mut state.layers) {
            stream = self
                .layer_pass(ctx, layer, layer_state, ids, &stream, pass)?
                .stream_out;
        }
        Ok(stream)
    }

    /// A whole prompt in `slot`, from the slot's fresh state.
    pub fn prefill(
        &self,
        ctx: &DeviceContext,
        slot: usize,
        ids: &[u32],
        state: &mut ModelState,
    ) -> Result<Forward> {
        let stream = self.embed(ctx, ids)?;
        self.forward(ctx, ids, stream, state, Pass::Prefill(slot))
    }

    /// One more token in each of `slots` on its carried state, as one batch:
    /// `ids[i]` is the token of `slots[i]`, and row `i` of every output belongs
    /// to it.
    pub fn decode(
        &self,
        ctx: &DeviceContext,
        slots: &[usize],
        ids: &[u32],
        state: &mut ModelState,
    ) -> Result<Forward> {
        let stream = self.embed(ctx, ids)?;
        self.forward(ctx, ids, stream, state, Pass::Decode(slots))
    }
}
