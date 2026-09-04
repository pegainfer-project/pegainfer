/// How a request treats end-of-sequence tokens.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EosPolicy {
    /// Do not stop on model EOS tokens.
    Ignore,
    /// Use the model executor's configured EOS set.
    #[default]
    ModelDefault,
    /// Stop only on this protocol-provided primary EOS token.
    Token(u32),
}

/// Request-scoped token stopping policy.
///
/// EOS is kept separate from caller stop tokens because the vLLM protocol
/// reports them differently: EOS has no 'stop_reason', while a request stop
/// reports the actual matching token ID.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StopPolicy {
    pub eos: EosPolicy,
    pub token_ids: Vec<u32>,
}

impl StopPolicy {
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
            EosPolicy::Token(eos_token_id) => token_id == eos_token_id,
        };

        if is_eos {
            Some(StopCause::Eos(token_id))
        } else if self.token_ids.contains(&token_id) {
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
        let policy = StopPolicy {
            eos: EosPolicy::Ignore,
            token_ids: vec![99],
        };

        assert_eq!(
            policy.classify(99, |token_id| token_id == 99),
            Some(StopCause::Token(99))
        );
    }

    #[test]
    fn eos_wins_when_the_same_id_is_also_an_explicit_stop() {
        let policy = StopPolicy {
            eos: EosPolicy::Token(99),
            token_ids: vec![99],
        };

        assert_eq!(policy.classify(99, |_| false), Some(StopCause::Eos(99)));
    }
}
