//! DSpine code-bridge weights, vocabulary tables, and lane scratch.

use std::collections::BTreeSet;
use std::collections::HashMap;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_core::weight_loader::load_tensor_1d;
use pegainfer_core::weight_loader::load_tensor_2d;
use pegainfer_core::weight_loader::load_tensor_3d_layer;
use pegainfer_core::weight_loader::load_tensor_i64_host;
use pegainfer_kernels::ops;
use safetensors::SafeTensors;

use crate::config::DFlashConfig;
use crate::weights::Qwen3Model;

const BRIDGE_SUFFIX: &str = "selected_prefix_bridge.";

pub(crate) struct DSpineBridge {
    pub(crate) code_size: usize,
    pub(crate) rank: usize,
    pub(crate) eps: f32,
    pub(crate) head_topk: usize,
    pub(crate) feature_layers: Vec<usize>,
    pub(crate) codebook: DeviceMatrix,
    /// `[layers * code_size, hidden]`: early reads followed by the last read.
    read: DeviceMatrix,
    /// `[layers * hidden, rank]`.
    write: DeviceMatrix,
    /// `[3 * rank, code_size]` = gate / predecessor_gate / prev.
    packed: DeviceMatrix,
    pub(crate) gate_bias: DeviceVec,
    /// `[logits_rows, rank]`.
    pub(crate) tg: DeviceMatrix,
    /// `[vocab, 2 * rank]`.
    pub(crate) tsv: DeviceMatrix,
}

pub(crate) struct DSpineScratch {
    pub(crate) projected: HiddenStates,
    pub(crate) codes: HiddenStates,
    pub(crate) packed: HiddenStates,
    pub(crate) message: HiddenStates,
    pub(crate) rms: CudaSlice<f32>,
    pub(crate) candidates: CudaSlice<i32>,
    pub(crate) idx: CudaSlice<i32>,
    pub(crate) tokens: CudaSlice<u32>,
}

impl DSpineScratch {
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &DFlashConfig,
        max_decode_batch_size: usize,
    ) -> Result<Self> {
        let dspine = config
            .dspine
            .as_ref()
            .expect("DSpine scratch requires a bridge config");
        let rows = max_decode_batch_size * config.block_size;
        let proposals = max_decode_batch_size * (config.block_size - 1);
        Ok(Self {
            projected: HiddenStates::zeros(ctx, dspine.code_size, rows)?,
            codes: HiddenStates::zeros(ctx, dspine.code_size, rows)?,
            packed: HiddenStates::zeros(ctx, 3 * dspine.rank, rows)?,
            message: HiddenStates::zeros(ctx, dspine.rank, rows)?,
            rms: ctx.stream.alloc_zeros(rows)?,
            candidates: ctx.stream.alloc_zeros(proposals * dspine.head_topk)?,
            idx: ctx.stream.alloc_zeros(proposals * dspine.head_topk)?,
            tokens: ctx.stream.alloc_zeros(rows)?,
        })
    }

    pub(crate) fn activate(&mut self, rows: usize) {
        self.projected.seq_len = rows;
        self.codes.seq_len = rows;
        self.packed.seq_len = rows;
        self.message.seq_len = rows;
    }
}

impl DSpineBridge {
    pub(crate) fn load(
        ctx: &DeviceContext,
        shards: &[SafeTensors],
        weight_map: &HashMap<String, usize>,
        config: &DFlashConfig,
        target: &Qwen3Model,
        final_norm: &DeviceVec,
    ) -> Result<Self> {
        let dspine = config
            .dspine
            .as_ref()
            .context("DSpine load without a bridge config")?;
        let hidden = config.hidden_size;
        let layers = config.num_hidden_layers;
        let vocab = config.vocab_size;
        let code_size = dspine.code_size;
        let rank = dspine.rank;
        ensure!(
            layers > 0 && vocab > 0 && code_size > 0 && rank > 0,
            "DSpine bridge geometry must be non-zero"
        );
        let prefix = bridge_prefix(shards, weight_map)?;
        let name = |suffix: &str| format!("{prefix}{suffix}");

        let codebook =
            load_tensor_2d(ctx, shards, weight_map, &name("codebook"), vocab, code_size)?;
        let read_last = load_tensor_2d(
            ctx,
            shards,
            weight_map,
            &name("read_weight"),
            code_size,
            hidden,
        )?;
        let mut read_layers = Vec::with_capacity(layers);
        for layer in 0..layers - 1 {
            read_layers.push(load_tensor_3d_layer(
                ctx,
                shards,
                weight_map,
                &name("early_read_weights"),
                layers - 1,
                code_size,
                hidden,
                layer,
            )?);
        }
        read_layers.push(read_last);
        let read_refs: Vec<&DeviceMatrix> = read_layers.iter().collect();
        let read = DeviceMatrix::vstack(ctx, &read_refs)?;

        let mut write_layers = Vec::with_capacity(layers);
        for layer in 0..layers {
            write_layers.push(load_tensor_3d_layer(
                ctx,
                shards,
                weight_map,
                &name("feature_write_weights"),
                layers,
                hidden,
                rank,
                layer,
            )?);
        }
        let write_refs: Vec<&DeviceMatrix> = write_layers.iter().collect();
        let write = DeviceMatrix::vstack(ctx, &write_refs)?;

        let gate = load_tensor_2d(
            ctx,
            shards,
            weight_map,
            &name("gate.weight"),
            rank,
            code_size,
        )?;
        let predecessor = load_tensor_2d(
            ctx,
            shards,
            weight_map,
            &name("predecessor_gate.weight"),
            rank,
            code_size,
        )?;
        let prev = load_tensor_2d(
            ctx,
            shards,
            weight_map,
            &name("prev.weight"),
            rank,
            code_size,
        )?;
        let packed = DeviceMatrix::vstack(ctx, &[&gate, &predecessor, &prev])?;
        let gate_bias = load_tensor_1d(ctx, shards, weight_map, &name("gate.bias"))?;
        ensure!(
            gate_bias.len == rank,
            "DSpine gate bias length {} != rank {rank}",
            gate_bias.len
        );

        // Terminal-disabled checkpoints ship a zero out projection; keep the
        // load loud when a terminal bridge sneaks through.
        let out = load_tensor_2d(
            ctx,
            shards,
            weight_map,
            &name("out.weight"),
            code_size,
            rank,
        )?;
        let out_host = ctx.stream.clone_dtoh(&out.data)?;
        ensure!(
            out_host.iter().all(|&value| value.to_bits() == 0),
            "DSpine terminal-disabled checkpoint must ship a zero out.weight"
        );

        check_version(
            shards,
            weight_map,
            &name("independent_write_version"),
            "independent_write_version",
        )?;
        check_version(
            shards,
            weight_map,
            &name("terminal_disabled_version"),
            "terminal_disabled_version",
        )?;
        if let Some(causal) = find_name(shards, weight_map, "draft_block_causal_version") {
            check_version(shards, weight_map, &causal, "draft_block_causal_version")?;
        }
        if let Some(refine) = find_name(shards, weight_map, "last_write_refine_version") {
            check_version(shards, weight_map, &refine, "last_write_refine_version")?;
        }

        let logits_rows = target.output_projection().rows;
        anyhow::ensure!(
            logits_rows <= vocab,
            "target logits rows {logits_rows} exceed DSpine codebook rows {vocab}"
        );
        let mut lm_scaled = HiddenStates::zeros(ctx, hidden, logits_rows)?;
        ops::dspine_scale_cols_into(
            ctx,
            target.output_projection(),
            final_norm,
            &mut lm_scaled.data,
        )?;
        let mut tg_rows = HiddenStates::zeros(ctx, rank, logits_rows)?;
        let last = layers - 1;
        let write_last = write
            .data
            .slice(last * hidden * rank..(last + 1) * hidden * rank);
        ops::gemm_strided_batched_bf16(
            ctx,
            false,
            false,
            rank,
            logits_rows,
            hidden,
            &write_last,
            rank,
            hidden * rank,
            &lm_scaled.data,
            hidden,
            hidden * logits_rows,
            &mut tg_rows.data,
            rank,
            rank * logits_rows,
            1,
        )?;

        let mut codebook_scaled = HiddenStates::zeros(ctx, code_size, vocab)?;
        ops::dspine_scale_const_into(
            ctx,
            &codebook,
            &mut codebook_scaled.data,
            (code_size as f32).sqrt(),
        )?;
        let mut tsv_rows = HiddenStates::zeros(ctx, 2 * rank, vocab)?;
        ops::gemm_rows_into_checked(
            ctx,
            &packed,
            rank,
            2 * rank,
            &codebook_scaled,
            &mut tsv_rows,
        )?;

        Ok(Self {
            code_size,
            rank,
            eps: dspine.eps,
            head_topk: dspine.head_topk,
            feature_layers: dspine.feature_layers.clone(),
            codebook,
            read,
            write,
            packed,
            gate_bias,
            tg: DeviceMatrix {
                data: tg_rows.data,
                rows: logits_rows,
                cols: rank,
            },
            tsv: DeviceMatrix {
                data: tsv_rows.data,
                rows: vocab,
                cols: 2 * rank,
            },
        })
    }

    pub(crate) fn is_active_layer(&self, layer: usize) -> bool {
        self.feature_layers.contains(&layer)
    }

    /// Bridge write after one draft layer. `scratch.hidden` is the layer output
    /// (the pre-write raw); `scratch.o_buf` is free after the MLP.
    pub(crate) fn write_after_layer(
        &self,
        ctx: &DeviceContext,
        layer: usize,
        scratch: &mut crate::dflash::DFlashBatchScratch,
        next_norm: Option<&DeviceVec>,
        rms_norm_eps: f32,
        block_size: usize,
    ) -> Result<()> {
        let hidden = scratch.hidden.hidden_dim;
        let rows = scratch.hidden.seq_len;
        ensure!(
            rows > 0 && rows.is_multiple_of(block_size),
            "DSpine bridge rows {rows} are not whole blocks of {block_size}"
        );
        dspine_activate(scratch);
        let dspine = scratch
            .dspine
            .as_mut()
            .context("DSpine bridge write without scratch")?;
        ops::dspine_prepare_read_into(
            ctx,
            &scratch.hidden,
            &mut scratch.normed,
            &mut dspine.rms,
            self.eps,
        )?;
        ops::gemm_rows_into_checked(
            ctx,
            &self.read,
            layer * self.code_size,
            self.code_size,
            &scratch.normed,
            &mut dspine.projected,
        )?;
        ops::dspine_normalize_codes_into(
            ctx,
            &dspine.projected,
            &self.codebook,
            &scratch.token_ids_d,
            &mut dspine.codes,
            block_size,
            self.eps,
        )?;
        ops::gemm_rows_into_checked(
            ctx,
            &self.packed,
            0,
            3 * self.rank,
            &dspine.codes,
            &mut dspine.packed,
        )?;
        ops::dspine_feature_message_into(
            ctx,
            &dspine.packed,
            &self.gate_bias,
            &mut dspine.message,
            block_size,
        )?;
        ops::gemm_rows_into_checked(
            ctx,
            &self.write,
            layer * hidden,
            hidden,
            &dspine.message,
            &mut scratch.o_buf,
        )?;
        ops::dspine_feature_finish_into(
            ctx,
            &scratch.hidden,
            &scratch.o_buf,
            &dspine.rms,
            &mut scratch.hidden_out,
            block_size,
        )?;
        std::mem::swap(&mut scratch.hidden, &mut scratch.hidden_out);
        if let Some(norm) = next_norm {
            ops::rms_norm_batch_into(
                ctx,
                &scratch.hidden,
                norm,
                rms_norm_eps,
                &mut scratch.normed,
            );
        }
        Ok(())
    }
}

fn dspine_activate(scratch: &mut crate::dflash::DFlashBatchScratch) {
    if let Some(dspine) = scratch.dspine.as_mut() {
        dspine.activate(scratch.hidden.seq_len);
    }
}

fn bridge_prefix(shards: &[SafeTensors], weight_map: &HashMap<String, usize>) -> Result<String> {
    let mut prefixes: BTreeSet<String> = BTreeSet::new();
    let names = weight_map
        .keys()
        .map(String::as_str)
        .chain(shards.iter().flat_map(|shard| shard.names()));
    for name in names {
        if let Some(offset) = name.find(BRIDGE_SUFFIX) {
            prefixes.insert(name[..offset + BRIDGE_SUFFIX.len()].to_string());
        }
    }
    ensure!(
        prefixes.len() == 1,
        "expected exactly one DSpine bridge prefix, got {prefixes:?}"
    );
    Ok(prefixes.into_iter().next().expect("one prefix"))
}

fn check_version(
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    name: &str,
    what: &str,
) -> Result<()> {
    let values = load_tensor_i64_host(shards, weight_map, name)?;
    ensure!(
        values.len() == 1 && values[0] == 1,
        "DSpine {what} must be [1], got {values:?}"
    );
    Ok(())
}

fn find_name(
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    suffix: &str,
) -> Option<String> {
    weight_map
        .keys()
        .map(String::as_str)
        .chain(shards.iter().flat_map(|shard| shard.names()))
        .find(|name| name.ends_with(suffix))
        .map(str::to_owned)
}

/// Fixed bridge footprint: checkpoint weights/tables plus the lane scratch and
/// the load-time table-build transients.
pub(crate) fn reservation_bytes(
    config: &DFlashConfig,
    max_decode_batch_size: usize,
) -> Result<usize> {
    let Some(dspine) = config.dspine.as_ref() else {
        return Ok(0);
    };
    let hidden = config.hidden_size;
    let layers = config.num_hidden_layers;
    let vocab = config.vocab_size;
    let code = dspine.code_size;
    let rank = dspine.rank;
    let logits_rows = config.logits_rows;
    let rows = max_decode_batch_size * config.block_size;
    let proposals = max_decode_batch_size * (config.block_size - 1);
    let bf16 = 2;
    let f32b = 4;
    let weights = crate::sizing::sum(&[
        crate::sizing::product(&[vocab, code, bf16])?, // codebook
        crate::sizing::product(&[layers, code, hidden, bf16])?, // early + last read
        crate::sizing::product(&[layers, hidden, rank, bf16])?, // writes
        crate::sizing::product(&[3, rank, code, bf16])?, // packed
        crate::sizing::product(&[rank, bf16])?,        // gate bias
        crate::sizing::product(&[code, rank, bf16])?,  // out (checked, then freed)
        crate::sizing::product(&[logits_rows, rank, bf16])?, // TG
        crate::sizing::product(&[vocab, 2, rank, bf16])?, // TSV
    ])?;
    let transients = crate::sizing::sum(&[
        crate::sizing::product(&[logits_rows, hidden, bf16])?, // lm_head scaled
        crate::sizing::product(&[vocab, code, bf16])?,         // codebook scaled
        // Per-layer read/write pieces coexist with the stacked matrices.
        crate::sizing::product(&[layers, code, hidden, bf16])?,
        crate::sizing::product(&[layers, hidden, rank, bf16])?,
    ])?;
    let scratch = crate::sizing::sum(&[
        crate::sizing::product(&[rows, code, bf16])?, // projected
        crate::sizing::product(&[rows, code, bf16])?, // codes
        crate::sizing::product(&[rows, 3, rank, bf16])?, // packed
        crate::sizing::product(&[rows, rank, bf16])?, // message
        crate::sizing::product(&[rows, f32b])?,       // rms
        crate::sizing::product(&[proposals, dspine.head_topk, 4])?, // candidates
        crate::sizing::product(&[proposals, dspine.head_topk, 4])?, // idx
        crate::sizing::product(&[rows, 4])?,          // tokens
    ])?;
    crate::sizing::sum(&[weights, transients, scratch])
}
