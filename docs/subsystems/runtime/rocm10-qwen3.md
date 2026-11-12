# ROCm 10.1 Qwen3 首版

**TL;DR：** 原生 Rust + HIP/rocBLAS 路径已在 Radeon AI PRO R9700（gfx1201）运行 Qwen3-0.6B/4B BF16，支持 TP1、greedy、eager、chunked prefill、混合批次及共享 HTTP 前端。两模型均通过 HF golden 和真实 HTTP 生命周期检查。gfx1100 仅完成内核交叉编译，缺少 RDNA 3 实机；这是一版有明确边界的 RDNA 4 预览实现，尚未完成双架构稳定版。

Last touched: 2026-11-12

## 支持边界

| 项目 | 本版状态 |
| --- | --- |
| 运行平台 | Linux x86_64；锁定的 ROCm 10.1.0 foundation；R9700 / gfx1201 |
| 模型 | Qwen3-0.6B、Qwen3-4B 原始 BF16 safetensors；其他尺寸尚未取得资格 |
| 执行 | 单 GPU，rocBLAS BF16 GEMM / FP32 累加，自有 HIP eager 算子 |
| 调度 | 最多 `--rocm-slots` 个活跃请求，额外请求排队；chunked prefill 与 decode 可同批 |
| 输出 | greedy；HTTP 显式设置 `temperature=0`，或 `top_k=1` |
| HTTP | `/v1/completions`、`/v1/chat/completions`、SSE、usage、停止词、EOS、取消、metrics |
| 上下文 | 默认 4096；配置范围 1–8192，且不能超过 checkpoint 上限；4B HTTP 边界实测 4096；8192 仅有算子资格 |
| 并发与 chunk | slots 默认 4、范围 1–16；chunk 默认 128、范围 1–512；本次服务资格为 slots=4/chunk=128 |
| 暂不支持 | HIP Graph、TP/RCCL、随机采样、logprobs/prompt scoring、prefix cache、paged KV、offload/P-D、LoRA、投机解码、量化及其他模型线 |
| 其他架构 | gfx1100 内核编译通过；gfx1101/1102/1200 是编译器可选 target，均无本项目运行资格 |

后端使用固定 KV 槽位，槽位回收后重新从 position 0 写入。它没有 CUDA 后端的共享 prefix/page/offload 状态，不能据此宣称已经迁移这些子系统。启动日志会显示实际 profile；显式要求 `--cuda-graph=true` 或 `--tp-size` 大于 1 时拒绝启动。HTTP 采样、评分、KV transfer 请求在打开 SSE 前返回带原因的 400；scheduler 也保留内部请求校验。

## 构建与运行

从本 worktree 根目录运行。工作区已有共享入口 `/home/zihaomu/zihaomu_workspace/rocm10-launch.sh` 和冻结的 foundation 镜像；本项目 launcher 每次先验证 foundation，再校验项目的 HIP/rocBLAS 文件 hash 与 compiler 身份。默认选择 HIP ordinal 2，即物理 GPU0（PCI `0000:03:00.0`）。本机 HIP 顺序与 amd-smi 物理编号不同；HIP ordinal 0 实际是物理 GPU3。项目入口在共享 verifier 执行 GPU 计算之前先做只读 PCI 校验，拒绝错误映射及物理 GPU3。

```bash
HIP_VISIBLE_DEVICES=2 scripts/amd/run_rocm10.sh bash scripts/amd/bootstrap.sh
HIP_VISIBLE_DEVICES=2 scripts/amd/run_rocm10.sh cargo build --locked --release \
  -p pegainfer-server --no-default-features --features rocm,qwen3
```

`bootstrap.sh` 将 Rust toolchain 和 OpenSSL 开发文件放到工作区 `artifacts/amd/toolchain`，不会修改基础镜像。Rust 版本由已有 `rust-toolchain.toml` 固定为 `nightly-2026-07-10`。首次构建需要联网获取 Cargo.lock 指定的依赖；OpenSSL 包、rustup 安装脚本会检查已固定的 SHA256。上游移除这些版本时需要重新取得环境资格，不应静默替换。

模型下载也固定 revision。下面两条命令已在本工作区执行，当前权重已存在：

```bash
HIP_VISIBLE_DEVICES=2 scripts/amd/run_rocm10.sh python scripts/amd/download_models.py \
  --model Qwen/Qwen3-0.6B --output ../models/Qwen3-0.6B
HIP_VISIBLE_DEVICES=2 scripts/amd/run_rocm10.sh python scripts/amd/download_models.py \
  --model Qwen/Qwen3-4B --output ../models/Qwen3-4B
```

实际服务参数记录在验收报告的 `http.4B.command` 中；使用项目 launcher 运行其中的二进制，可保证运行时库与编译环境一致。面向宿主机的服务启动命令见本文末尾。每次请求需要显式指定 greedy 参数。

## 实现结构与所有权

| 入口 | 职责 |
| --- | --- |
| `pegainfer-build/src/rocm.rs` | 独立 HIP 编译与归档；wheel SDK 的项目内 linker aliases；gfx 列表进入编译产物身份 |
| `pegainfer-device` | HIP device/stream、rocBLAS handle、typed Buffer、同步拷贝、跨线程重新绑定 |
| `pegainfer-kernels/csrc/hip/qwen3.cpp` | embedding、RMSNorm、残差、SiLU、RoPE、KV append、causal GQA attention、argmax、rocBLAS GEMM |
| `pegainfer-kernels/src/rocm.rs` | ABI 封装，校验 shape、容量、buffer 所属 device/stream |
| `pegainfer-qwen3/src/rocm/model.rs` | 复用 Qwen3 config/EOS/vocabulary；BF16 权重加载、内存预算、连续 KV append、ragged forward |
| `pegainfer-qwen3/src/rocm/scheduler.rs` | 复用 `RequestLedger` 与 step contract，负责排队、槽位、chunk、停止和取消 |
| `pegainfer-frontend` / `pegainfer-server` | 复用 vLLM Rust HTTP 前端；ServePlan 可指定请求校验 router，CUDA 默认行为保持原样 |
| `scripts/amd` | 项目环境锁、下载、构建环境、真实算子/golden/HTTP/CLI 资格检查与证据归档 |

CUDA 实现保存在 Qwen3 的 `cuda.rs`，在原 crate root 条件展开；原有符号路径保留。默认 feature 仍选 CUDA。ROCm build 在任何 CUDA toolkit 探测、子模块初始化前分流，且不编译 `cudarc`、CUPTI、CUDA sampling/KV/offload。CUDA 依赖的其他模型显式保留 CUDA feature，避免根 workspace 关闭默认依赖 feature 后改变它们的构建选择。CUDA 与 ROCm feature 互斥，因此不要用 `--all-features` 验收两后端，应建立分别运行的 CI job。

设备资源由 `Arc` 保持生命周期，stream 上的提交由 mutex 串行化。跨线程使用前重新绑定 HIP device；H2D/D2H 返回前完成 stream 同步，host 临时数据不会先于拷贝释放。Buffer drop 等待 stream 后释放分配，所有 Buffer 释放后才销毁 runtime。模型在 GPU 步骤开始时置为不可复用状态，仅完整成功后恢复；部分失败不能继续使用已被部分更新的 KV。

固定槽位的 KV 字节数为 `slots × context × layers × kv_heads × head_dim × 2(K,V) × 2(BF16)`。Qwen3-4B 的 slots=4/context=4096 约占 **2.25 GiB** KV；还需权重、scratch、库 workspace 和预留空间。启动前按实际可用显存检查预算，分配失败会保留错误上下文。metrics 的 KV 使用率表示**活跃槽位所占的预留容量比例**，不是已写 token 的比例；取消后保留底层 pool，复用槽位，不逐请求释放整块显存。

attention 是首版正确性基线：每个 query/head 对自己的连续 KV 做 FP32 score/softmax/归约，输出 BF16。没有采用 CUDA 的 warp/PTX 假设；wave 宽度由目标决定。它不是性能优化后的 paged/flash attention，长 prefill 的代价较高。后续优化必须继续通过同一数值和生命周期门限。

## 本次验证结果

环境身份来自 [environment.lock.json](../../../scripts/amd/environment.lock.json)。运行设备为物理 GPU0，PCI `0000:03:00.0`，UUID `29ff7551-0000-1000-80dc-b03ca15afb63`；容器内 ordinal 0。ROCm release 是 **10.1.0**，HIP component **7.16.26385**，AMD clang revision `d6f6cb691863e9258034ab48c1dfce21689d5606`；HIP 版本没有以 10 开头是该冻结环境的正常身份。

运行时不依赖 Python/Torch。CPU Torch 只用于原生算子的独立参考；foundation 的 Torch/Triton 配对保持不变。

### 数值

复用仓库既有 HF BF16 fixtures，48 条 prompt，长度 8–247，逐条 teacher-force 16 个 decode token。每条执行路径覆盖 **816 个位置 / 6528 个 top-8 logprob 比较**，并检查 GPU argmax 与 CPU tie-break 一致。维持原门限：mean ≤ 0.06、p99 ≤ 0.20、HF argmax regret ≤ 0.20，未放宽门限。

| 模型 | 路径 | mean | p99 |
| --- | --- | ---: | ---: |
| Qwen3-0.6B | sequential | 0.040036 | 0.148445 |
| Qwen3-0.6B | ragged batch4 | 0.039923 | 0.148720 |
| Qwen3-0.6B | mixed prefill/decode batch4 | 0.039452 | 0.145736 |
| Qwen3-4B | sequential | 0.031401 | 0.111609 |
| Qwen3-4B | ragged batch4 | 0.031099 | 0.108667 |
| Qwen3-4B | mixed prefill/decode batch4 | 0.031145 | 0.111609 |

每个模型另外执行固定 shape 的重复 logits 检查；mixed pass 实际执行 52 次 prefill/decode 同批步骤。共计两模型三条路径 **4896 个位置**。

独立 HIP 算子检查包含 9 项：RMSNorm、GEMM、残差、SiLU、RoPE、position 8191 内的高位置 RoPE、ragged causal attention、context=8192 attention、argmax ties。高位置 RoPE 最大绝对误差为 0.0078125，其余所测数值项为 0；这是这些固定输入的结果，不是对所有 shape 的精确等价声明。

**参考溯源限制：** 当前权重 revision 已固定，但上游提交的 golden metadata 没有 checkpoint revision/config hash。因此本次是对仓库既有 fixture 的兼容验收，不能宣称已证明 fixture 与当前 snapshot 的字节身份相同。qualification 保存原 fixture metadata/hash 和当前完整模型文件 hash，后续应重建带完整来源的参考，保留原 gate。

### 服务与资源

- 每模型 16 条流式请求，客户端 concurrency=8、服务槽位=4；输入长度不同，跨 chunk 边界，验证 SSE `[DONE]`、finish reason 和 usage。
- 非流式 completion、chat、stop string；4B 的 4094-token prompt + 2-token output 触达 4096 边界，超限返回 400。
- 4 个活跃请求 + 1 个等待请求时，分别断开等待连接及活跃连接；确认 running/waiting/KV 使用率全部归零，再成功接入 4 个请求。
- sampling、logprobs、prompt scoring、echo、KV transfer 均有真实 HTTP 400 检查。TP2、graph、非法容量、量化 checkpoint 有真实 CLI 拒绝检查。
- SIGINT 正常关闭并等待进程退出。设备层验证 Buffer 在创建者 Device 被 drop 后仍能跨线程读取，以及拷贝边界、零初始化和溢出拒绝。
- 共享 host 单测：core 9、frontend 97、Qwen3 config 7；server 4；release build、fmt、clippy 通过。clippy 仍有共享 config 在 ROCm 下未使用等 warning；未把 warning-free 当作验收结果。

HTTP 报告保留延迟与文本以便排障；没有执行配对性能基准或两小时稳定性实验，不能据此报告相对 CUDA/vLLM 的性能提升。

### 复现与证据

```bash
HIP_VISIBLE_DEVICES=2 scripts/amd/run_rocm10.sh bash scripts/amd/qualify.sh \
  ../artifacts/amd/qualification/rocm10-v1
```

此命令要求一个新的输出目录，防止旧日志混入新资格。重跑时改成新目录。验收脚本执行 release build、设备、原生数值、host/server 单测、CLI、两模型 golden/HTTP、fmt、clippy，全部通过后生成 `qualification.json`。

当前完整证据位于工作区 `artifacts/amd/qualification/rocm10-v1/`，包括 foundation report、每项日志、HTTP 请求/响应、tracked patch，以及修改源码、二进制、Cargo.lock、模型和 fixture 的 SHA256。验收发生在提交前，报告通过 baseline commit + dirty-source hashes 标识被测候选；提交前再次核对实现文件与这些 hash 一致。原始日志及模型文件保留在工作区，不随 Git 提交发布。

RDNA 3 的编译检查使用同一 ROCm 10.1 toolchain 与独立 target directory：

```bash
HIP_VISIBLE_DEVICES=2 PEGAINFER_ROCM_ARCH=gfx1100 scripts/amd/run_rocm10.sh \
  cargo build --release -p pegainfer-kernels --no-default-features --features rocm
```

对应 `artifacts/amd/qualification/rocm10-v1/cross-gfx1100.log`。该命令运行 foundation 的 gfx1201 环境检查，然后**仅编译** gfx1100 kernel/device 产物，没有执行 gfx1100 code object，也没有验证 RDNA 3 rocBLAS solution。实际设备架构必须在模型加载时与编译 target 列表匹配。

**设备选择事件：** 初轮测试误将 HIP ordinal 0 当成 amd-smi 物理 GPU0；最终记录实际 PCI 时发现它是被排除的物理 GPU3。初轮 `qualification` 已标为无效并移到 `rocm10-v1-invalid-device-selection`。本文表格对应新增 PCI 守卫后、在物理 GPU0 重新执行的整套验收。`artifacts/amd/INVALID_DEVICE_SELECTION.md` 标明早期 bootstrap/native/http 记录不作为最终资格；错误的 `HIP_VISIBLE_DEVICES=0` 现已验证会在任何 verifier GPU 运算前被拒绝。

## 与完整适配计划的关系和下一步

本版先完成可运行的原生 Qwen3 垂直路径。它复用 config、RequestLedger 和 HTTP contract，但尚未把 CUDA executor、paged KV、sampling 和统一设备 facade 全部迁到共同类型。固定槽位和 rocBLAS 是为尽早验证正确性而作的明确阶段选择；这些限制不等价于完整计划 M1–M3 已完成。

下一步优先在 gfx1100 真机执行相同 gate、补 CUDA GPU 回归与独立后端 CI，再把固定槽位换为可保留生命周期契约的 paged KV，补随机采样/logprobs，随后进入 HIP Graph 和 attention/GEMM 性能优化。long-context HF reference、GPU 失效注入、完整 OOM 恢复和更大并发资格也仍需补齐。

## 启动检查记录

以下命令在物理 GPU0 上启动了 4B 服务，宿主机访问成功。测试以外层超时发送 SIGINT 收尾，交互使用时直接运行下面命令，Ctrl-C 结束：

```bash
HIP_VISIBLE_DEVICES=2 ROCM10_NETWORK=host scripts/amd/run_rocm10.sh \
  ../artifacts/amd/target/rocm10.1-gfx1201/release/pegainfer \
  --model-path ../models/Qwen3-4B --served-model-name qwen3-rocm --port 18302 \
  --cuda-graph=false --no-prefix-cache --rocm-slots 4 \
  --rocm-max-context 4096 --rocm-chunk-size 128
```

服务使用共享前端的监听配置，host network 模式将端口暴露给宿主机网络。客户端验证：

```bash
curl --fail --silent --show-error http://127.0.0.1:18302/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-rocm","messages":[{"role":"user","content":"请用中文回答：2 加 2 等于几？"}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

实际回答为“2 加 2 等于 4。”。`host-server.log`、`host-response.json` 和 `host-curl-response.json` 保存在最终资格目录。本次交付不留下后台推理服务。
