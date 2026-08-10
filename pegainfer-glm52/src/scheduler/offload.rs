//! Disaggregated-prefill (native MTP P/D) protocol + the per-request KV
//! resolver. Storage orchestration lives in `pegainfer-kv-store`
//! (`resolve_prefix` / `seal` / `retire`); this module keeps only what is
//! GLM5.2 business: the handoff envelope, the padded naming chain, the KV
//! shape contract, and the zero-obligation resolver that turns an incoming
//! request into a scheduler-ready intake. The engine loop never blocks on
//! any of this: resolution runs on the store's runtime and completed
//! intakes arrive over a channel. Request-level allocation happens at
//! admission, never here.

use anyhow::Context as _;
use pegainfer_frontend::engine::GenerateRequest;
use pegainfer_frontend::engine::KvPrefix;
use pegainfer_kv_store::CacheScope;
use pegainfer_kv_store::CancelProbe;
use pegainfer_kv_store::KvStore;
use pegainfer_kv_store::PAD_TOKEN_ID;
use pegainfer_kv_store::ResolvePolicy;
use serde::Deserialize;
use serde::Serialize;

use super::PAGE;

/// The handoff envelope (v3), one struct for both sides. Boundary rule:
/// content shareable in the radix is a pure prompt function and travels
/// the KV data plane; anchor-dependent continuation state travels here.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct NativeMtpHandoff {
    /// Capability manifest (see [`handoff_fingerprint`]); a mismatch is an
    /// intake rejection carrying both strings.
    pub(super) fingerprint: String,
    pub(super) committed_len: usize,
    /// P's first sampled token; `None` = EOS — nothing to restore or decode.
    pub(super) anchor_token_id: Option<u32>,
    pub(super) draft_tokens: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
pub(super) struct PegaInferPdEnvelope {
    pub(super) pegainfer_pd: NativeMtpHandoff,
}

/// This engine's capability manifest. Any protocol evolution edits this
/// string; readable on purpose — a mismatch log names the divergence. The
/// page stride is the wire-layout identity: one page-first slab page carries
/// every layer's slices, so two engines agreeing on the stride agree on the
/// whole per-block byte layout.
pub(super) fn handoff_fingerprint() -> String {
    format!(
        "glm52-native-mtp/4/page:{}/salt:{}/drafts:{}",
        crate::model::GLM52_KV_PAGE_STRIDE,
        super::native_mtp_cache_salt(),
        crate::mtp::glm52_mtp_draft_len(),
    )
}

pub(super) fn native_mtp_handoff(
    req: &GenerateRequest,
) -> anyhow::Result<Option<NativeMtpHandoff>> {
    let Some(value) = req.kv_transfer_params.clone() else {
        return Ok(None);
    };
    if value.get("pegainfer_pd").is_none() {
        return Ok(None);
    }
    let envelope: PegaInferPdEnvelope =
        serde_json::from_value(value).context("invalid pegainfer native-MTP P/D metadata")?;
    let handoff = envelope.pegainfer_pd;
    let ours = handoff_fingerprint();
    anyhow::ensure!(
        handoff.fingerprint == ours,
        "native P/D fingerprint mismatch: peer {:?}, this engine {:?}",
        handoff.fingerprint,
        ours
    );
    anyhow::ensure!(
        req.prompt_tokens.len() == handoff.committed_len,
        "native P/D v3 expects the original prompt: committed_len {}, got {} prompt tokens",
        handoff.committed_len,
        req.prompt_tokens.len()
    );
    Ok(Some(handoff))
}

/// kvbm geometry of a native D request: the committed prompt plus P's
/// anchor as input; the anchor consumes the first client output position
/// (its sampling step ran on P).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct NativeKvShape {
    pub(super) input_tokens: usize,
    pub(super) max_output_tokens: usize,
}

pub(super) fn native_kv_shape(req: &GenerateRequest, handoff: &NativeMtpHandoff) -> NativeKvShape {
    NativeKvShape {
        input_tokens: handoff.committed_len + 1,
        max_output_tokens: req.max_tokens,
    }
}

/// The naming chain P sealed: its natural sequence at handoff (committed
/// prompt + dangling anchor) padded to the page boundary. A page-aligned
/// commit has no partial page and no pads — the anchor stays unnamed.
fn padded_chain(committed_prompt: &[u32], anchor: u32) -> Vec<u32> {
    let mut chain = committed_prompt.to_vec();
    if chain.len().is_multiple_of(PAGE) {
        return chain;
    }
    chain.push(anchor);
    chain.resize(chain.len().next_multiple_of(PAGE), PAD_TOKEN_ID);
    chain
}

/// A resolved intake, produced on the store runtime and drained by the
/// engine loop: the inbox holds only scheduler-ready requests.
pub(super) enum Resolved {
    /// Plain request: prefix resolution ran (or was skipped); admission does
    /// its normal match and drops the hold.
    Plain {
        req: GenerateRequest,
        prefix: KvPrefix,
    },
    /// Native P/D request: every page — boundary included — sits in the
    /// radix behind the prefix hold, zero-obligation. Admission asserts the
    /// hit and makes all authoritative allocations.
    Native {
        req: GenerateRequest,
        prefix: KvPrefix,
        handoff: NativeMtpHandoff,
    },
    /// Resolution failed terminally (missing checkpoint past the deadline):
    /// admission answers with the standard rejection.
    Failed {
        req: GenerateRequest,
        message: String,
    },
}

impl Resolved {
    /// Surrender the request, dropping any resolution state (a prefix hold
    /// releases via RAII).
    pub(super) fn into_request(self) -> GenerateRequest {
        match self {
            Resolved::Plain { req, .. }
            | Resolved::Native { req, .. }
            | Resolved::Failed { req, .. } => req,
        }
    }
}

/// The decode side of the handoff: one zero-obligation `resolve_prefix`
/// over the padded chain. Full pages and the boundary page restore into
/// the radix as shared blocks; no request-level allocation happens here.
/// A short hit is terminal — decode cannot recompute the miss — and
/// surfaces before any slot is occupied; the router retries against P.
pub(super) async fn native_pd_resolve(
    store: &KvStore,
    rank: usize,
    req: &GenerateRequest,
    handoff: &NativeMtpHandoff,
    anchor: u32,
    cancel: &dyn CancelProbe,
) -> anyhow::Result<KvPrefix> {
    let chain = padded_chain(&req.prompt_tokens, anchor);
    let req_id_owned;
    let req_id = match req.request_id.as_deref() {
        Some(id) => id,
        None => {
            req_id_owned = super::anon_resolve_key("native-pd", rank);
            &req_id_owned
        }
    };
    let t_start = std::time::Instant::now();
    let prefix = store
        .resolve_prefix(
            rank,
            req_id,
            &chain,
            CacheScope::default().cache_salt(super::native_mtp_cache_salt()),
            ResolvePolicy::default().wait_for_full_hit().full_pages(),
            cancel,
        )
        .await;
    anyhow::ensure!(
        prefix.hit_tokens() == chain.len(),
        "native P/D restore resolved {} of {} chain tokens before the deadline",
        prefix.hit_tokens(),
        chain.len()
    );
    log::info!(
        "native P/D resolve: rank={rank} committed_len={} chain={} pages={} took={}ms",
        handoff.committed_len,
        chain.len(),
        chain.len() / PAGE,
        t_start.elapsed().as_millis(),
    );
    Ok(prefix)
}
