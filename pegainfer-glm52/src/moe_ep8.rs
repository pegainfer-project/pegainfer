//! GLM5.2 EP MoE layer weights (router + shared + local expert bank).
//!
//! The Hopper DeepGEMM masked routed chain is gone; production decode runs
//! the SM100 DeepGEMM path in `moe_ep`. This module only holds the layer
//! weight bundle still shared by the model builder.

use crate::moe_decode::Glm52MoeExpertBank;
use crate::moe_decode::Glm52MoeRouterWeights;
use crate::moe_decode::Glm52MoeSharedExpert;

/// One rank's weights for one EP MoE layer: router and shared expert run
/// where the token lives (every rank); the bank holds this rank's local
/// experts.
pub(crate) struct Glm52MoeEp8LayerWeights {
    pub(crate) router: Glm52MoeRouterWeights,
    pub(crate) shared: Glm52MoeSharedExpert,
    pub(crate) bank: Glm52MoeExpertBank,
}
