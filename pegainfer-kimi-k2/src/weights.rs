//! Text-only Kimi-K2.6 weight loading.
//!
//! This file is the `crate::weights` module root. The sibling `weights/`
//! directory contains the implementation modules, following the repository's
//! flat Rust module layout (`foo.rs` + `foo/`, no `mod.rs`).

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use cudarc::driver::CudaContext;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::DeviceRepr;
use cudarc::driver::ValidAsZeroBits;
use cudarc::driver::result as cuda_result;
use half::bf16;
use log::debug;
use memmap2::Mmap;
use pegainfer_kernels::ffi;
use pegainfer_kernels::ops::KimiInt4ExpertRole;
use pegainfer_kernels::ops::KimiInt4NibbleOrder;
use pegainfer_kernels::ops::KimiInt4WeightManifest;
use pegainfer_kernels::ops::KimiMarlinFusedW13Int4Weight;
use pegainfer_kernels::ops::KimiMarlinInt4ExpertWeights;
use pegainfer_kernels::ops::KimiMarlinInt4Weight;
use pegainfer_kernels::ops::kimi_marlin_int4_fuse_w13;
use pegainfer_kernels::ops::kimi_marlin_int4_reorder_scale;
use pegainfer_kernels::ops::kimi_marlin_int4_reorder_weight;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::GpuWeight;
use safetensors::Dtype;
use safetensors::SafeTensors;
use serde_json::Value;

use crate::config::KIMI_K2_DENSE_INTERMEDIATE;
use crate::config::KIMI_K2_DENSE_LAYERS;
use crate::config::KIMI_K2_EXPERT_INTERMEDIATE;
use crate::config::KIMI_K2_HIDDEN;
use crate::config::KIMI_K2_INT4_GROUP_SIZE;
use crate::config::KIMI_K2_LAYERS;
use crate::config::KIMI_K2_MOE_LAYERS;
use crate::config::KIMI_K2_Q_HEAD_DIM;
use crate::config::KIMI_K2_QK_NOPE_HEAD_DIM;
use crate::config::KIMI_K2_ROUTED_EXPERTS;
use crate::config::KIMI_K2_V_HEAD_DIM;
use crate::config::KimiK2ParallelShape;

const KIMI_K2_WEIGHT_INDEX: &str = "model.safetensors.index.json";
const TEXT_PREFIX: &str = "language_model.";

mod context;
mod load;
mod manifest;
mod package;
#[cfg(test)]
mod tests;

pub(crate) use context::KimiRankGpuContext;
pub(crate) use load::KimiRankSlicedLoadPlan;
pub(crate) use load::ensure_text_only_model_index;
pub(crate) use load::load_rank_sliced_weights_to_gpu;
pub(crate) use manifest::KimiK2WeightManifest;
pub(crate) use manifest::KimiLayerWeightKindNames;
pub(crate) use manifest::KimiLayerWeightNames;
pub(crate) use manifest::KimiRankWeightNames;
pub(crate) use package::KimiGpuRawTensor;
pub(crate) use package::KimiRankExpertMarlinWeights;
pub(crate) use package::KimiRankGpuWeights;
pub(crate) use package::KimiRouterDeviceWeights;
pub(crate) use package::KimiRouterGpuWeights;
