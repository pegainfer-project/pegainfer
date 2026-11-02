/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <flashinfer/trtllm/fmha/kernelParams.h>

namespace tensorrt_llm {
namespace kernels {

// Minimal subset of FlashInfer 0.7.0's generated metadata needed by
// GLM5.2 TP4 decode. The checked-in cubins carry the matching SHA-256 names.
struct TllmGenFmhaKernelMetaInfo {
  Data_type mDataTypeQ;
  Data_type mDataTypeKv;
  Data_type mDataTypeK;
  Data_type mDataTypeV;
  Data_type mDataTypeO;
  int mTileSizeQ;
  int mTileSizeKv;
  int mStepQ;
  int mStepKv;
  int mHeadDimPerCtaV;
  int mHeadDimQk;
  int mHeadDimV;
  int mSM;
  const unsigned char* mCubin;
  unsigned int mCubinSize;
  const char* mFuncName;
  int mSharedMemBytes;
  int mThreadsPerCTA;
  int mQkvLayout;
  int mNumTokensPerPage;
  int mMaskType;
  int mKernelType;
  int mTileScheduler;
  int mMultiCtasKvMode;
  int mNumEltsPerSageAttnBlkQ;
  int mNumEltsPerSageAttnBlkK;
  int mNumEltsPerSageAttnBlkP;
  int mNumEltsPerSageAttnBlkV;
  bool mGroupsHeadsQ;
  bool mGroupsTokensHeadsQ;
  bool mReuseSmemKForV;
  bool m2CtaMma;
  int mSparseAttn;
  bool mSkipsSoftmaxWhenPossible;
  bool mFp16Softmax;
  bool mUsesSpcompress;
  bool mEnablesBf16QFp8KvKOnlyTransform;
  bool mSeparateTransformedKv;
  bool mFusesDsv4InvRopeFp8Quant;
  bool mUsesDsv4Ue8m0ScaleO;
  const char* sha256;
};

static const TllmGenFmhaKernelMetaInfo sTllmGenFmhaKernelMetaInfos[] = {
    // Selector seed: isSupported() checks the unsplit V=512 shape before the
    // launch heuristic right-sizes it to VPerCta128 on GB300.
    {DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3,
     DATA_TYPE_BF16, 8, 128, 8, 128, 512, 576, 512, kSM_100f, nullptr, 0,
     "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen",
     166800, 512, 2, 1, 0, 2, 0, 1, 0, 0, 0, 0, true, false, false,
     false, 1, false, false, false, false, false, false, false,
     "97b3d2de9c2025e967e54ce3986aa3c88954b13e7dde4927c4a672e54f4b68a0"},
    {DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3,
     DATA_TYPE_BF16, 8, 128, 8, 128, 512, 576, 512, kSM_100f, nullptr, 0,
     "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1VarSeqQ8Kv128PersistentSwapsAbForGen",
     168960, 512, 2, 1, 0, 2, 1, 0, 0, 0, 0, 0, true, false, false,
     false, 1, false, false, false, false, false, false, false,
     "6544178cddeffe987d4d7774549a7e0d8823815afa66e1fdc2131381c050ed41"},
    {DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3,
     DATA_TYPE_BF16, 8, 128, 8, 128, 128, 576, 512, kSM_100f, nullptr, 0,
     "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta128PagedKvDenseStaticTokenSparseP1VarSeqQ8Kv128PersistentSwapsAbForGen",
     168960, 512, 2, 1, 0, 2, 1, 0, 0, 0, 0, 0, true, false, false,
     false, 1, false, false, false, false, false, false, false,
     "b1640b8068bc12de38d3f2950b7af7b53e818d9fb6d9a07b2e630bb968007df3"},
    {DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3,
     DATA_TYPE_BF16, 8, 128, 8, 128, 128, 576, 512, kSM_100f, nullptr, 0,
     "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta128PagedKvDenseStaticTokenSparseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen",
     166800, 512, 2, 1, 0, 2, 0, 1, 0, 0, 0, 0, true, false, false,
     false, 1, false, false, false, false, false, false, false,
     "eddd15b3ae26ac40468a61f1947a64f189c7aca0120ceb385f8ff1e1c634f08f"},
    {DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3,
     DATA_TYPE_BF16, 16, 128, 16, 128, 128, 576, 512, kSM_100f, nullptr, 0,
     "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta128PagedKvDenseStaticTokenSparseP1MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen",
     179472, 512, 2, 1, 0, 2, 0, 1, 0, 0, 0, 0, true, false, false,
     false, 1, false, false, false, false, false, false, false,
     "ca45c6611280408f82bd613411b0e25075833348ddac5dd0af24937cca2dd509"},
    {DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3,
     DATA_TYPE_BF16, 16, 128, 16, 128, 256, 576, 512, kSM_100f, nullptr, 0,
     "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta256PagedKvDenseStaticTokenSparseP1MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen",
     179472, 512, 2, 1, 0, 2, 0, 1, 0, 0, 0, 0, true, false, false,
     false, 1, false, false, false, false, false, false, false,
     "3e39ced03725e3a8d03f79a22ddc704d2d059cf3256692affae23759264c3db7"},
    {DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3,
     DATA_TYPE_BF16, 16, 128, 16, 128, 512, 576, 512, kSM_100f, nullptr, 0,
     "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen",
     179472, 512, 2, 1, 0, 2, 0, 1, 0, 0, 0, 0, true, false, false,
     false, 1, false, false, false, false, false, false, false,
     "2166c91e4c23c51222af9dfd30372daedf1a74c8727289168ffa8c21c552adeb"},
};

}  // namespace kernels
}  // namespace tensorrt_llm
