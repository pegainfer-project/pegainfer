//! Fixed scratch for Qwen3.5 target-side speculative verification.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;

use crate::config::Config35;
use crate::ops::PrefillPagedPlan;
use crate::prefill_buffers::GdrChunkwiseScratch35;

pub(crate) struct VerifyBuffers35 {
    max_batch: usize,
    span: usize,
    max_rows: usize,
    token_ids_h: Vec<u32>,
    pub(crate) token_ids_d: CudaSlice<u32>,

    pub(crate) hidden: HiddenStates,
    pub(crate) hidden_next: HiddenStates,
    pub(crate) normed: HiddenStates,
    pub(crate) attn_results: HiddenStates,
    pub(crate) hidden_mid: HiddenStates,
    pub(crate) gate_up_out: HiddenStates,
    pub(crate) act_out: HiddenStates,
    pub(crate) mlp_out: HiddenStates,
    pub(crate) logits_normed: HiddenStates,
    pub(crate) logits: HiddenStates,
    pub(crate) q_full: HiddenStates,
    pub(crate) k_full: HiddenStates,
    pub(crate) v_full: HiddenStates,
    pub(crate) q_prepped: HiddenStates,
    pub(crate) attn_out_full: HiddenStates,

    pub(crate) qkv: HiddenStates,
    pub(crate) z: HiddenStates,
    pub(crate) b_proj: HiddenStates,
    pub(crate) a_proj: HiddenStates,
    pub(crate) gdr_out: HiddenStates,
    pub(crate) normed_gated: HiddenStates,
    pub(crate) compact_qkv: HiddenStates,
    pub(crate) compact_b: HiddenStates,
    pub(crate) compact_a: HiddenStates,
    pub(crate) compact_qkv_conv: HiddenStates,
    pub(crate) compact_gdr: HiddenStates,
    pub(crate) gdr_scratch: GdrChunkwiseScratch35,

    pub(crate) plan: PrefillPagedPlan,
    pub(crate) sample: pegainfer_sample::SampleScratch,
}

impl VerifyBuffers35 {
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &Config35,
        max_batch: usize,
        span: usize,
        max_total_pages: usize,
    ) -> Result<Self> {
        anyhow::ensure!(max_batch > 0, "Qwen3.5 verify buffers need max_batch > 0");
        anyhow::ensure!(span > 0, "Qwen3.5 verify buffers need span > 0");
        let max_rows = max_batch * span;
        let hidden = config.hidden_size;
        let q_proj_dim = config.full_attn_q_proj_dim();
        let q_dim = config.full_attn_q_dim();
        let kv_dim = config.full_attn_kv_dim();
        let qkv_dim = config.linear_attn_qkv_dim();
        let z_dim = config.linear_attn_z_dim();
        let group_size = config.num_attention_heads / config.num_key_value_heads;
        let max_tiles = max_batch * span * group_size.max(1);

        Ok(Self {
            max_batch,
            span,
            max_rows,
            token_ids_h: vec![0; max_rows],
            token_ids_d: ctx.stream.alloc_zeros(max_rows)?,

            hidden: HiddenStates::zeros(ctx, hidden, max_rows)?,
            hidden_next: HiddenStates::zeros(ctx, hidden, max_rows)?,
            normed: HiddenStates::zeros(ctx, hidden, max_rows)?,
            attn_results: HiddenStates::zeros(ctx, hidden, max_rows)?,
            hidden_mid: HiddenStates::zeros(ctx, hidden, max_rows)?,
            gate_up_out: HiddenStates::zeros(ctx, 2 * config.intermediate_size, max_rows)?,
            act_out: HiddenStates::zeros(ctx, config.intermediate_size, max_rows)?,
            mlp_out: HiddenStates::zeros(ctx, hidden, max_rows)?,
            logits_normed: HiddenStates::zeros(ctx, hidden, max_rows)?,
            logits: HiddenStates::zeros(ctx, config.selection_vocab, max_rows)?,
            q_full: HiddenStates::zeros(ctx, q_proj_dim, max_rows)?,
            k_full: HiddenStates::zeros(ctx, kv_dim, max_rows)?,
            v_full: HiddenStates::zeros(ctx, kv_dim, max_rows)?,
            q_prepped: HiddenStates::zeros(ctx, q_dim, max_rows)?,
            attn_out_full: HiddenStates::zeros(ctx, q_dim, max_rows)?,

            qkv: HiddenStates::zeros(ctx, qkv_dim, max_rows)?,
            z: HiddenStates::zeros(ctx, z_dim, max_rows)?,
            b_proj: HiddenStates::zeros(ctx, config.linear_num_value_heads, max_rows)?,
            a_proj: HiddenStates::zeros(ctx, config.linear_num_value_heads, max_rows)?,
            gdr_out: HiddenStates::zeros(ctx, z_dim, max_rows)?,
            normed_gated: HiddenStates::zeros(ctx, z_dim, max_rows)?,
            compact_qkv: HiddenStates::zeros(ctx, qkv_dim, span)?,
            compact_b: HiddenStates::zeros(ctx, config.linear_num_value_heads, span)?,
            compact_a: HiddenStates::zeros(ctx, config.linear_num_value_heads, span)?,
            compact_qkv_conv: HiddenStates::zeros(ctx, qkv_dim, span)?,
            compact_gdr: HiddenStates::zeros(ctx, z_dim, span)?,
            gdr_scratch: GdrChunkwiseScratch35::new(ctx, config, max_rows)?,

            plan: PrefillPagedPlan::new_preallocated(
                ctx,
                max_rows,
                max_total_pages,
                max_batch,
                max_tiles,
            )?,
            sample: pegainfer_sample::SampleScratch::new(ctx, config.selection_vocab, max_rows)?,
        })
    }

    pub(crate) fn max_batch(&self) -> usize {
        self.max_batch
    }

    pub(crate) fn set_rows(&mut self, rows: usize) {
        assert!(
            rows <= self.max_rows,
            "Qwen3.5 verify rows {rows} exceeds capacity {}",
            self.max_rows
        );
        self.hidden.seq_len = rows;
        self.hidden_next.seq_len = rows;
        self.normed.seq_len = rows;
        self.attn_results.seq_len = rows;
        self.hidden_mid.seq_len = rows;
        self.gate_up_out.seq_len = rows;
        self.act_out.seq_len = rows;
        self.mlp_out.seq_len = rows;
        self.logits_normed.seq_len = rows;
        self.logits.seq_len = rows;
        self.q_full.seq_len = rows;
        self.k_full.seq_len = rows;
        self.v_full.seq_len = rows;
        self.q_prepped.seq_len = rows;
        self.attn_out_full.seq_len = rows;
        self.qkv.seq_len = rows;
        self.z.seq_len = rows;
        self.b_proj.seq_len = rows;
        self.a_proj.seq_len = rows;
        self.gdr_out.seq_len = rows;
        self.normed_gated.seq_len = rows;
        self.gdr_scratch.set_rows(rows);
    }

    pub(crate) fn set_compact_rows(&mut self, rows: usize) {
        assert!(
            rows <= self.span,
            "Qwen3.5 compact verify rows {rows} exceeds span {}",
            self.span
        );
        self.compact_qkv.seq_len = rows;
        self.compact_b.seq_len = rows;
        self.compact_a.seq_len = rows;
        self.compact_qkv_conv.seq_len = rows;
        self.compact_gdr.seq_len = rows;
        self.gdr_scratch.set_rows(rows);
    }

    pub(crate) fn stage_tokens(&mut self, ctx: &DeviceContext, spans: &[&[u32]]) -> Result<usize> {
        anyhow::ensure!(
            spans.len() <= self.max_batch,
            "Qwen3.5 verify batch {} exceeds capacity {}",
            spans.len(),
            self.max_batch
        );
        let total_rows: usize = spans.iter().map(|span| span.len()).sum();
        anyhow::ensure!(
            total_rows <= self.max_rows,
            "Qwen3.5 verify rows {total_rows} exceeds capacity {}",
            self.max_rows
        );
        self.token_ids_h.clear();
        self.token_ids_h.reserve(total_rows);
        for span in spans {
            self.token_ids_h.extend_from_slice(span);
        }
        self.set_rows(total_rows);
        if total_rows > 0 {
            let mut token_ids_d = self.token_ids_d.slice_mut(..total_rows);
            ctx.stream
                .memcpy_htod(&self.token_ids_h, &mut token_ids_d)?;
        }
        Ok(total_rows)
    }
}
