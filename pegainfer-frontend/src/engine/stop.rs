use std::sync::Arc;

/// How a request treats end-of-sequence tokens.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EosPolicy {
    /// Do not stop on model EOS tokens.
    Ignore,
    /// Use the model executor's configured EOS set.
    #[default]
    ModelDefault,
}

/// Request-scoped token stopping policy.
///
/// EOS is kept separate from caller stop tokens because the vLLM protocol
/// reports them differently: EOS has no 'stop_reason', while a request stop
/// reports the actual matching token ID.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StopPolicy {
    pub eos: EosPolicy,
    /// Sorted and deduplicated explicit stop IDs.
    ///
    /// Requests clone this policy while building a plan and sending it to
    /// worker ranks. Keeping the normalized set behind an `Arc` makes those
    /// clones cheap and lets classification use binary search for large stop
    /// sets without regressing the common one-ID case.
    pub token_ids: Arc<[u32]>,
}

impl StopPolicy {
    /// Build a policy from wire-provided stop IDs.
    ///
    /// Normalization happens once at the request boundary. Internal copies can
    /// then share the immutable slice instead of repeatedly sorting, deduping,
    /// or cloning the caller's vector.
    #[must_use]
    pub fn new(eos: EosPolicy, mut token_ids: Vec<u32>) -> Self {
        token_ids.sort_unstable();
        token_ids.dedup();
        Self {
            eos,
            token_ids: token_ids.into(),
        }
    }

    /// Classify a token using vLLM's priority: EOS first, then the request's
    /// explicit stop-token set.
    #[must_use]
    pub fn classify(
        &self,
        token_id: u32,
        is_model_eos: impl FnOnce(u32) -> bool,
    ) -> Option<StopCause> {
        let is_eos = match self.eos {
            EosPolicy::Ignore => false,
            EosPolicy::ModelDefault => is_model_eos(token_id),
        };

        if is_eos {
            Some(StopCause::Eos(token_id))
        } else if self.token_ids.binary_search(&token_id).is_ok() {
            Some(StopCause::Token(token_id))
        } else {
            None
        }
    }
}

/// The token-level cause of a normal stop finish.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StopCause {
    /// A primary or model-default EOS token.
    Eos(u32),
    /// A token from the request's explicit stop-token set.
    Token(u32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_default_classifies_model_eos() {
        let policy = StopPolicy::default();

        assert_eq!(
            policy.classify(99, |token_id| token_id == 99),
            Some(StopCause::Eos(99))
        );
    }

    #[test]
    fn ignored_eos_does_not_disable_an_explicit_stop() {
        let policy = StopPolicy::new(EosPolicy::Ignore, vec![99]);

        assert_eq!(
            policy.classify(99, |token_id| token_id == 99),
            Some(StopCause::Token(99))
        );
    }

    #[test]
    fn normalizes_stop_sets_across_common_sizes() {
        let empty = StopPolicy::default();
        assert!(empty.classify(7, |_| false).is_none());

        let single = StopPolicy::new(EosPolicy::Ignore, vec![42]);
        assert_eq!(single.classify(42, |_| false), Some(StopCause::Token(42)));
        assert!(single.classify(43, |_| false).is_none());

        let policy = StopPolicy::new(EosPolicy::Ignore, vec![7, 3, 7, 1]);
        assert_eq!(policy.token_ids.as_ref(), [1, 3, 7]);
        assert!(policy.classify(7, |_| false).is_some());
        assert!(policy.classify(8, |_| false).is_none());

        let full = StopPolicy::new(EosPolicy::Ignore, (0..151_936).collect());
        assert_eq!(full.token_ids.len(), 151_936);
        assert!(full.classify(0, |_| false).is_some());
        assert!(full.classify(151_935, |_| false).is_some());
        assert!(full.classify(151_936, |_| false).is_none());
    }
}
