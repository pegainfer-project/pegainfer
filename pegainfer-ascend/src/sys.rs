//! Raw AscendCL / ACLNN declarations.
//!
//! The declarations are hand-written so `cargo check` works on a machine
//! without a CANN install; only linking needs the libraries (see `build.rs`).
#![allow(non_camel_case_types)]

use std::ffi::c_char;
use std::ffi::c_void;

pub(crate) type aclError = i32;
pub(crate) type aclnnStatus = i32;

pub(crate) const ACL_SUCCESS: aclError = 0;

pub(crate) const ACL_FLOAT: u32 = 0;
pub(crate) const ACL_INT8: u32 = 2;
pub(crate) const ACL_INT32: u32 = 3;
pub(crate) const ACL_BF16: u32 = 27;

pub(crate) const ACL_FORMAT_ND: u32 = 2;
pub(crate) const ACL_FORMAT_FRACTAL_NZ: u32 = 29;

pub(crate) const ACL_MEMCPY_HOST_TO_DEVICE: u32 = 1;
pub(crate) const ACL_MEMCPY_DEVICE_TO_HOST: u32 = 2;

pub(crate) const ACL_MEM_MALLOC_HUGE_FIRST: u32 = 0;

#[repr(C)]
pub(crate) struct aclTensor {
    _private: [u8; 0],
}
#[repr(C)]
pub(crate) struct aclOpExecutor {
    _private: [u8; 0],
}
#[repr(C)]
pub(crate) struct aclScalar {
    _private: [u8; 0],
}
#[repr(C)]
pub(crate) struct aclIntArray {
    _private: [u8; 0],
}
#[repr(C)]
pub(crate) struct aclTensorList {
    _private: [u8; 0],
}

unsafe extern "C" {
    // ---- AscendCL runtime ----
    pub(crate) safe fn aclInit(config_path: *const c_char) -> aclError;
    pub(crate) safe fn aclFinalize() -> aclError;
    pub(crate) safe fn aclGetRecentErrMsg() -> *const c_char;

    pub(crate) safe fn aclrtSetDevice(device_id: i32) -> aclError;
    pub(crate) safe fn aclrtResetDevice(device_id: i32) -> aclError;
    pub(crate) safe fn aclrtCreateContext(context: *mut *mut c_void, device_id: i32) -> aclError;
    pub(crate) safe fn aclrtDestroyContext(context: *mut c_void) -> aclError;
    pub(crate) safe fn aclrtSetCurrentContext(context: *mut c_void) -> aclError;
    pub(crate) safe fn aclrtCreateStream(stream: *mut *mut c_void) -> aclError;
    pub(crate) safe fn aclrtDestroyStream(stream: *mut c_void) -> aclError;
    pub(crate) safe fn aclrtSynchronizeStream(stream: *mut c_void) -> aclError;
    pub(crate) safe fn aclrtMalloc(ptr: *mut *mut c_void, size: usize, policy: u32) -> aclError;
    pub(crate) safe fn aclrtFree(ptr: *mut c_void) -> aclError;
    pub(crate) safe fn aclrtMemcpy(
        dst: *mut c_void,
        dst_max: usize,
        src: *const c_void,
        count: usize,
        kind: u32,
    ) -> aclError;
    pub(crate) safe fn aclrtGetSocName() -> *const c_char;

    // ---- aclnn tensor descriptors ----
    pub(crate) safe fn aclCreateTensor(
        view_dims: *const i64,
        view_dims_num: usize,
        data_type: u32,
        stride: *const i64,
        offset: i64,
        format: u32,
        storage_dims: *const i64,
        storage_dims_num: usize,
        tensor_data: *mut c_void,
    ) -> *mut aclTensor;
    pub(crate) safe fn aclDestroyTensor(tensor: *const aclTensor) -> aclnnStatus;
    pub(crate) safe fn aclCreateScalar(value: *mut c_void, data_type: u32) -> *mut aclScalar;
    pub(crate) safe fn aclDestroyScalar(scalar: *const aclScalar) -> aclnnStatus;
    pub(crate) safe fn aclCreateIntArray(value: *const i64, size: usize) -> *mut aclIntArray;
    pub(crate) safe fn aclDestroyIntArray(array: *const aclIntArray) -> aclnnStatus;
    pub(crate) safe fn aclCreateTensorList(
        value: *const *const aclTensor,
        size: usize,
    ) -> *mut aclTensorList;
    pub(crate) safe fn aclDestroyTensorList(list: *const aclTensorList) -> aclnnStatus;

    pub(crate) safe fn aclSetAclOpExecutorRepeatable(executor: *mut aclOpExecutor) -> aclnnStatus;
    pub(crate) safe fn aclDestroyAclOpExecutor(executor: *mut aclOpExecutor) -> aclnnStatus;

    // ---- ACLNN operators ----
    pub(crate) safe fn aclnnMatmulGetWorkspaceSize(
        self_: *const aclTensor,
        mat2: *const aclTensor,
        out: *mut aclTensor,
        cube_math_type: i8,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;
    pub(crate) safe fn aclnnMatmul(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: *mut c_void,
    ) -> aclnnStatus;

    pub(crate) safe fn aclnnMatmulWeightNzGetWorkspaceSize(
        self_: *const aclTensor,
        mat2: *const aclTensor,
        out: *mut aclTensor,
        cube_math_type: i8,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;
    pub(crate) safe fn aclnnMatmulWeightNz(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: *mut c_void,
    ) -> aclnnStatus;

    pub(crate) safe fn aclnnEmbeddingGetWorkspaceSize(
        weight: *const aclTensor,
        indices: *const aclTensor,
        out: *const aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;
    pub(crate) safe fn aclnnEmbedding(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: *mut c_void,
    ) -> aclnnStatus;

    pub(crate) safe fn aclnnRmsNormGetWorkspaceSize(
        x: *const aclTensor,
        gamma: *const aclTensor,
        epsilon: f64,
        y_out: *const aclTensor,
        rstd_out: *const aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;
    pub(crate) safe fn aclnnRmsNorm(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: *mut c_void,
    ) -> aclnnStatus;

    pub(crate) safe fn aclnnAddGetWorkspaceSize(
        self_: *const aclTensor,
        other: *const aclTensor,
        alpha: *const aclScalar,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;
    pub(crate) safe fn aclnnAdd(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: *mut c_void,
    ) -> aclnnStatus;

    pub(crate) safe fn aclnnSiluGetWorkspaceSize(
        self_: *const aclTensor,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;
    pub(crate) safe fn aclnnSilu(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: *mut c_void,
    ) -> aclnnStatus;

    pub(crate) safe fn aclnnMulGetWorkspaceSize(
        self_: *const aclTensor,
        other: *const aclTensor,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;
    pub(crate) safe fn aclnnMul(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: *mut c_void,
    ) -> aclnnStatus;

    pub(crate) safe fn aclnnApplyRotaryPosEmbV2GetWorkspaceSize(
        query_ref: *mut aclTensor,
        key_ref: *mut aclTensor,
        cos: *const aclTensor,
        sin: *const aclTensor,
        layout: i64,
        rotary_mode: *mut c_char,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;
    pub(crate) safe fn aclnnApplyRotaryPosEmbV2(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: *mut c_void,
    ) -> aclnnStatus;

    pub(crate) safe fn aclnnIndexCopyGetWorkspaceSize(
        self_ref: *mut aclTensor,
        dim: i64,
        index: *const aclTensor,
        source: *const aclTensor,
        out_ref: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;
    pub(crate) safe fn aclnnIndexCopy(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: *mut c_void,
    ) -> aclnnStatus;

    pub(crate) safe fn aclnnIncreFlashAttentionV4GetWorkspaceSize(
        query: *const aclTensor,
        key: *const aclTensorList,
        value: *const aclTensorList,
        pse_shift: *const aclTensor,
        atten_mask: *const aclTensor,
        actual_seq_lengths: *const aclIntArray,
        dequant_scale1: *const aclTensor,
        quant_scale1: *const aclTensor,
        dequant_scale2: *const aclTensor,
        quant_scale2: *const aclTensor,
        quant_offset2: *const aclTensor,
        antiquant_scale: *const aclTensor,
        antiquant_offset: *const aclTensor,
        blocktable: *const aclTensor,
        kv_padding_size: *const aclTensor,
        num_heads: i64,
        scale_value: f64,
        input_layout: *mut c_char,
        num_key_value_heads: i64,
        block_size: i64,
        inner_precise: i64,
        attention_out: *const aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;
    pub(crate) safe fn aclnnIncreFlashAttentionV4(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: *mut c_void,
    ) -> aclnnStatus;

    pub(crate) safe fn aclnnPromptFlashAttentionV3GetWorkspaceSize(
        query: *const aclTensor,
        key: *const aclTensor,
        value: *const aclTensor,
        pse_shift: *const aclTensor,
        atten_mask: *const aclTensor,
        actual_seq_lengths: *const aclIntArray,
        actual_seq_lengths_kv: *const aclIntArray,
        deq_scale1: *const aclTensor,
        quant_scale1: *const aclTensor,
        deq_scale2: *const aclTensor,
        quant_scale2: *const aclTensor,
        quant_offset2: *const aclTensor,
        num_heads: i64,
        scale_value: f64,
        pre_tokens: i64,
        next_tokens: i64,
        input_layout: *mut c_char,
        num_key_value_heads: i64,
        sparse_mode: i64,
        inner_precise: i64,
        attention_out: *const aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;
    pub(crate) safe fn aclnnPromptFlashAttentionV3(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: *mut c_void,
    ) -> aclnnStatus;

    pub(crate) safe fn aclnnAddRmsNormGetWorkspaceSize(
        x1: *const aclTensor,
        x2: *const aclTensor,
        gamma: *const aclTensor,
        epsilon: f64,
        y_out: *const aclTensor,
        rstd_out: *const aclTensor,
        x_out: *const aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;
    pub(crate) safe fn aclnnAddRmsNorm(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: *mut c_void,
    ) -> aclnnStatus;

    pub(crate) safe fn aclnnClippedSwigluGetWorkspaceSize(
        x: *const aclTensor,
        group_index: *const aclTensor,
        dim: i64,
        alpha: f64,
        limit: f64,
        bias: f64,
        interleaved: bool,
        out: *const aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;
    pub(crate) safe fn aclnnClippedSwiglu(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: *mut c_void,
    ) -> aclnnStatus;

    pub(crate) safe fn aclnnArgMaxGetWorkspaceSize(
        self_: *const aclTensor,
        dim: i64,
        keepdim: bool,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> aclnnStatus;
    pub(crate) safe fn aclnnArgMax(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: *mut c_void,
    ) -> aclnnStatus;
}
