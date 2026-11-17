//! Checked sizing arithmetic for startup memory budgets.
//!
//! Budgets are computed from config-file dims before those dims are validated
//! against the weight tensors, so a corrupt config must fail loudly here rather
//! than wrap into an under-reservation.

use anyhow::Result;
use anyhow::anyhow;

pub(crate) fn product(factors: &[usize]) -> Result<usize> {
    factors
        .iter()
        .try_fold(1usize, |acc, &f| acc.checked_mul(f))
        .ok_or_else(|| anyhow!("sizing overflow in product {factors:?}"))
}

pub(crate) fn sum(terms: &[usize]) -> Result<usize> {
    terms
        .iter()
        .try_fold(0usize, |acc, &t| acc.checked_add(t))
        .ok_or_else(|| anyhow!("sizing overflow in sum {terms:?}"))
}
