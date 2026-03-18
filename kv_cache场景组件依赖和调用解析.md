# KV Cache 传输完整链式分析

## 文档概述

本文档整合 vLLM + PyTorch + Mooncake + CUDA 四层架构在 KV Cache 传输场景的完整链式分析，涵盖：

1. **架构总览与组件职责**
2. **五大传输场景详解**
3. **组件间接口契约**
4. **数据流与控制流分离**
5. **关键技术对比与最佳实践**
6. **完整代码位置索引**

---

## 一、架构总览与组件职责

### 1.1 四层架构模型

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                         KV Cache 传输四层架构模型                                              │
│                                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         Layer 1: vLLM (业务调度层)                                    │   │
│   │                                                                                      │   │
│   │  职责: 决定"何时传输"、"传输什么"、"传输到哪里"                                        │   │
│   │  ├── Scheduler: 判断 GPU 内存压力，决定 offload 策略 (LRU/ARC)                        │   │
│   │  ├── BlockManager: 管理 KV Cache block 的分配和映射                                  │   │
│   │  ├── OffloadingConnector: 协调本地传输请求，管理传输状态                              │   │
│   │  └── MooncakeConnector: 协调跨节点传输请求                                           │   │
│   │                                                                                      │   │
│   │  输出: TransferSpec = (src_block_ids, dst_block_ids, transfer_type)                 │   │
│   └──────────────────────────────────────┬──────────────────────────────────────────────┘   │
│                                          │                                                   │
│                                          │ TransferSpec (Python 数据结构)                     │
│                                          ▼                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         Layer 2: PyTorch (框架抽象层)                                 │   │
│   │                                                                                      │   │
│   │  职责: 提供"数据载体"和"CUDA 抽象"，但不执行传输                                      │   │
│   │  ├── torch.Tensor: KV Cache 的物理存储 (GPU/CPU)                                     │   │
│   │  │   ├── GPU Tensor: device="cuda", 存储 KV Cache block                             │   │
│   │  │   └── CPU Tensor: device="cpu", pin_memory=True (pinned memory)                  │   │
│   │  ├── torch.cuda.Stream: CUDA 流管理                                                  │   │
│   │  ├── torch.Event: CUDA 事件同步                                                      │   │
│   │  └── ATen C++ API: 与底层 CUDA 扩展的桥接                                            │   │
│   │                                                                                      │   │
│   │  输出: tensor.data_ptr(), tensor.device(), cudaStream_t                              │   │
│   │                                                                                      │   │
│   │  ★ PyTorch 不直接调用 cudaMemcpyAsync，而是通过自定义 CUDA 扩展间接调用               │   │
│   └──────────────────────────────────────┬──────────────────────────────────────────────┘   │
│                                          │                                                   │
│                                          │ 指针 + 设备信息 + CUDA Stream                     │
│                                          ▼                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         Layer 3: Mooncake (传输引擎层)                                │   │
│   │                                                                                      │   │
│   │  职责: 执行"跨节点"和"跨介质"的数据传输                                               │   │
│   │  ├── TransferEngine: 统一传输接口                                                    │   │
│   │  ├── RdmaTransport: GPU↔GPU 跨节点 RDMA 传输 (GDRDMA)                                │   │
│   │  ├── NVMeoFTransport: GPU↔SSD GDS 传输 (GPUDirect Storage)                           │   │
│   │  ├── GdsTransport: 新版 GDS 传输实现 (tent 版本)                                     │   │
│   │  └── TCPTransport: CPU↔CPU TCP 传输                                                  │   │
│   │                                                                                      │   │
│   │  输出: 底层 API 调用参数 (ibv_post_send, cuFileBatchIOSubmit)                         │   │
│   └──────────────────────────────────────┬──────────────────────────────────────────────┘   │
│                                          │                                                   │
│                                          │ 底层 API 调用                                     │
│                                          ▼                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         Layer 4: CUDA (底层执行层)                                    │   │
│   │                                                                                      │   │
│   │  职责: 执行"实际的 DMA 传输"                                                          │   │
│   │  ├── cudaMemcpyAsync: GPU↔CPU DMA 传输                                               │   │
│   │  ├── cuFile API: GPU↔SSD GDS 传输                                                    │   │
│   │  │   ├── cuFileDriverOpen(): 初始化驱动                                              │   │
│   │  │   ├── cuFileBufRegister(): 注册 buffer                                            │   │
│   │  │   ├── cuFileHandleRegister(): 注册文件句柄                                        │   │
│   │  │   └── cuFileBatchIOSubmit(): 批量 IO 提交                                        │   │
│   │  ├── ibv_post_send: RDMA 网络传输                                                    │   │
│   │  └── CUDA Driver: 管理 GPU 硬件资源                                                  │   │
│   │                                                                                      │   │
│   │  输出: 硬件 DMA 操作                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 组件职责边界

| 组件 | 职责范围 | 不负责什么 |
|------|----------|------------|
| **vLLM** | 业务调度、Block 管理、传输协调、LRU/ARC 淘汰策略 | 直接执行传输、管理 CUDA 资源 |
| **PyTorch** | Tensor 存储、CUDA 抽象、Stream/Event 管理 | 执行 cudaMemcpy、网络传输 |
| **Mooncake** | 跨节点/跨介质传输、RDMA/GDS 封装、批量 IO | 业务调度、Tensor 管理 |
| **CUDA** | 硬件 DMA、GPU 资源管理、cuFile/RDMA verbs | 业务逻辑、传输协调 |

### 1.3 KV Cache 分层存储金字塔

```
┌─────────────────────────────────────────────────────────────────┐
│                    KV Cache 分层存储金字塔                        │
│                                                                 │
│     ┌─────────────┐                                             │
│     │   GPU HBM   │  ← 热数据（当前请求）                        │
│     │   ~80 GB    │    延迟: ~100 ns, 带宽: ~3 TB/s             │
│     └──────┬──────┘                                             │
│            │ cudaMemcpyAsync (pinned memory)                    │
│            │ 场景: vLLM OffloadingConnector                     │
│            ▼                                                     │
│     ┌─────────────┐                                             │
│     │  CPU DRAM   │  ← 温数据（近期可能用到）                    │
│     │  ~512 GB    │    延迟: ~10 μs, 带宽: ~200 GB/s            │
│     └──────┬──────┘                                             │
│            │ GDS / cuFile (可选)                                │
│            │ 场景: 跨会话复用、持久化                            │
│            ▼                                                     │
│     ┌─────────────┐                                             │
│     │  NVMe SSD   │  ← 冷数据（持久化）                          │
│     │   ~4 TB     │    延迟: ~100 μs, 带宽: ~12 GB/s            │
│     └─────────────┘                                             │
│                                                                 │
│  ★ 推荐: HBM → DRAM 是主要 offload 路径，SSD 仅用于持久化 ★    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、五大传输场景详解

### 2.1 传输场景总览

| 场景 | 传输路径 | 传输技术 | 实现组件 | 底层 API | 典型带宽 |
|------|----------|----------|----------|----------|----------|
| **场景1** | HBM → DRAM | CUDA DMA | vLLM OffloadingConnector | cudaMemcpyAsync | ~32 GB/s |
| **场景2** | DRAM → HBM | CUDA DMA | vLLM OffloadingConnector | cudaMemcpyAsync | ~32 GB/s |
| **场景3** | HBM → SSD | GDS | Mooncake NVMeoFTransport | cuFileBatchIOSubmit | ~12-24 GB/s |
| **场景4** | SSD → HBM | GDS | Mooncake NVMeoFTransport | cuFileBatchIOSubmit | ~12-24 GB/s |
| **场景5** | HBM → HBM (跨节点) | GDRDMA | Mooncake RdmaTransport | ibv_post_send | ~25 GB/s |

---

### 2.2 场景1: HBM → DRAM 卸载路径 (本地)

#### 完整调用链

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  Step 1: vLLM Scheduler 决策                                                                 │
│  ─────────────────────────────────────────────────────────────────────────────────────────  │
│  组件: vLLM                                                                                  │
│  文件: vllm/v1/kv_offload/lru_manager.py, arc_manager.py                                    │
│  函数: LRUManager.touch() / ARCManager.touch()                                              │
│                                                                                              │
│  输入:                                                                                       │
│  ├── request: Request 对象 (包含 block_hashes)                                              │
│  └── gpu_memory_usage: 当前 GPU 内存使用率                                                   │
│                                                                                              │
│  处理逻辑:                                                                                    │
│  ├── 检查 block 的 LRU 时间戳                                                                │
│  ├── 计算 evict_score = access_count * recency_weight                                       │
│  └── 选择 evict_score 最低的 blocks 进行 offload                                             │
│                                                                                              │
│  输出: offload_block_ids: List[int] - 需要卸载的 GPU block IDs                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ offload_block_ids
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  Step 2: OffloadingConnectorScheduler 构建传输规格                                           │
│  ─────────────────────────────────────────────────────────────────────────────────────────  │
│  组件: vLLM                                                                                  │
│  文件: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:358-475         │
│  函数: update_state_after_alloc(), _get_reqs_to_store()                                     │
│                                                                                              │
│  处理逻辑:                                                                                    │
│  ├── src_spec = GPULoadStoreSpec(src_block_ids)    # GPU block IDs                          │
│  ├── dst_spec = manager.prepare_store(block_hashes) # CPU block IDs                         │
│  └── transfer_spec = (src_spec, dst_spec)                                                   │
│                                                                                              │
│  输出: TransferSpec = (GPULoadStoreSpec, CPULoadStoreSpec)                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ TransferSpec
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  Step 3: OffloadingConnectorWorker 提交传输任务                                              │
│  ─────────────────────────────────────────────────────────────────────────────────────────  │
│  组件: vLLM                                                                                  │
│  文件: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:629-651         │
│  函数: start_kv_transfers(), prepare_store_kv()                                             │
│                                                                                              │
│  处理逻辑:                                                                                    │
│  ├── job_id = self._generate_job_id()                                                      │
│  ├── self._jobs[job_id] = (req_id, is_store=True)                                          │
│  └── self.worker.transfer_async(job_id, transfer_spec)  # 调用 OffloadingWorker            │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ (job_id, TransferSpec)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  Step 4: OffloadingWorker 路由到 Handler                                                     │
│  ─────────────────────────────────────────────────────────────────────────────────────────  │
│  组件: vLLM                                                                                  │
│  文件: vllm/v1/kv_offload/worker/worker.py:114-146                                          │
│  函数: OffloadingWorker.transfer_async()                                                    │
│                                                                                              │
│  处理逻辑:                                                                                    │
│  ├── transfer_type = (src_spec.medium(), dst_spec.medium())  # ("GPU", "CPU")              │
│  ├── handler = self.transfer_type_to_handler[transfer_type]                                │
│  └── return handler.transfer_async(job_id, spec)                                           │
│                                                                                              │
│  Handler 选择: ("GPU", "CPU") → SingleDirectionOffloadingHandler(gpu_to_cpu=True)           │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ handler.transfer_async(job_id, spec)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  Step 5: SingleDirectionOffloadingHandler 准备 PyTorch 资源                                  │
│  ─────────────────────────────────────────────────────────────────────────────────────────  │
│  组件: PyTorch (框架层)                                                                       │
│  文件: vllm/v1/kv_offload/worker/cpu_gpu.py:119-192                                         │
│  函数: SingleDirectionOffloadingHandler.transfer_async()                                    │
│                                                                                              │
│  PyTorch 资源准备:                                                                            │
│  ├── 1. stream = torch.cuda.Stream()  # PyTorch API                                         │
│  ├── 2. start_event = torch.Event(enable_timing=True)                                       │
│  │   end_event = torch.Event(enable_timing=True)                                            │
│  ├── 3. src_tensors: list[torch.Tensor]  # GPU tensors, device="cuda"                       │
│  │   dst_tensors: list[torch.Tensor]  # CPU tensors, device="cpu", pin_memory=True          │
│  └── 4. stream.wait_stream(torch.cuda.current_stream())  # 等模型计算完成                   │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ PyTorch 资源已就绪
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  Step 6: 在 PyTorch Stream 中调用自定义 CUDA 扩展                                            │
│  ─────────────────────────────────────────────────────────────────────────────────────────  │
│  组件: PyTorch → vLLM CUDA Extension                                                         │
│  文件: vllm/v1/kv_offload/worker/cpu_gpu.py:165-178                                         │
│                                                                                              │
│  代码:                                                                                       │
│  with torch.cuda.stream(stream):  # PyTorch context manager                                 │
│      start_event.record(stream)                                                             │
│      for src_tensor, dst_tensor, block_size_in_bytes in zip(...):                          │
│          ops.swap_blocks(  # ★ 调用 vLLM 自定义 CUDA 扩展                                    │
│              src_tensor,      # torch.Tensor (GPU)                                          │
│              dst_tensor,      # torch.Tensor (CPU, pinned)                                  │
│              block_size_in_bytes,                                                           │
│              src_to_dst_tensor  # torch.Tensor (block mapping)                              │
│          )                                                                                  │
│      end_event.record(stream)                                                               │
│                                                                                              │
│  ★ 关键: ops.swap_blocks 是 vLLM 自定义 CUDA 扩展，通过 PyTorch ATen C++ API 获取底层数据   │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ ops.swap_blocks(src_tensor, dst_tensor, ...)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  Step 7: vLLM CUDA Extension 提取 PyTorch Tensor 信息                                        │
│  ─────────────────────────────────────────────────────────────────────────────────────────  │
│  组件: vLLM CUDA Extension (C++)                                                             │
│  文件: vllm/csrc/cache_kernels.cu, csrc/attention/swap_blocks.cu                            │
│  函数: swap_blocks()                                                                         │
│                                                                                              │
│  通过 PyTorch ATen API 提取信息:                                                              │
│  ├── 1. torch::Device src_device = src.device();  // ATen API                              │
│  │     torch::Device dst_device = dst.device();  // ATen API                               │
│  ├── 2. if (src_device.is_cuda() && dst_device.is_cpu())                                   │
│  │         memcpy_type = cudaMemcpyDeviceToHost;  // GPU → CPU                              │
│  ├── 3. char* src_ptr = static_cast<char*>(src.data_ptr());  // GPU 地址                    │
│  │     char* dst_ptr = static_cast<char*>(dst.data_ptr());  // CPU 地址 (pinned)            │
│  └── 4. const cudaStream_t stream = at::cuda::getCurrentCUDAStream();  // PyTorch API       │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ cudaMemcpyAsync(dst_ptr, src_ptr, size, type, stream)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  Step 8: CUDA Runtime 执行 DMA 传输                                                          │
│  ─────────────────────────────────────────────────────────────────────────────────────────  │
│  组件: CUDA Runtime                                                                          │
│  API: cudaMemcpyAsync()                                                                      │
│                                                                                              │
│  硬件操作:                                                                                    │
│  ├── 1. GPU Driver 检查 src_ptr 是否在 GPU HBM 中                                           │
│  ├── 2. GPU Driver 检查 dst_ptr 是否是 pinned memory (通过 cuMemHostRegister 注册)          │
│  ├── 3. 设置 PCIe DMA 引擎                                                                   │
│  │   ├── 源地址: GPU HBM 物理地址                                                           │
│  │   └── 目标地址: CPU DRAM 物理地址 (pinned)                                               │
│  ├── 4. 执行 DMA 传输: GPU HBM ──PCIe──► CPU DRAM                                           │
│  └── 5. 记录完成事件到 stream                                                                │
│                                                                                              │
│  ★ 关键优化: Pinned Memory 允许 GPU 直接 DMA 访问 CPU 内存，无需通过 OS 页缓存               │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ DMA 传输完成
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  Step 9: PyTorch Event 同步                                                                  │
│  ─────────────────────────────────────────────────────────────────────────────────────────  │
│  组件: PyTorch                                                                               │
│  文件: vllm/v1/kv_offload/worker/cpu_gpu.py:194-214                                         │
│  函数: SingleDirectionOffloadingHandler.get_finished()                                      │
│                                                                                              │
│  同步检查:                                                                                    │
│  ├── 1. while self._transfers and self._transfers[0].end_event.query():                    │
│  │         # query() 是 PyTorch API，非阻塞检查 CUDA event 是否完成                         │
│  ├── 2. transfer_time = start_event.elapsed_time(end_event) * 1e-3  # PyTorch API           │
│  └── 3. 返回 TransferResult(job_id, success=True, transfer_size, transfer_time)             │
│                                                                                              │
│  资源回收:                                                                                    │
│  ├── self._stream_pool.append(stream)  # 回收 stream                                        │
│  └── self._event_pool.extend([start_event, end_event])  # 回收 events                       │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### CPU Tensor 分配 (Pinned Memory)

```python
# 文件: vllm/v1/kv_offload/worker/cpu_gpu.py:292-310

pin_memory = is_pin_memory_available()
logger.info("Allocating %d CPU tensors...", len(parsed_gpu_tensors))

for gpu_tensor, split_k_and_v in parsed_gpu_tensors:
    cpu_shape = list(gpu_tensor.shape)
    cpu_shape[1 if split_k_and_v else 0] = num_cpu_kernel_blocks

    cpu_tensor = torch.zeros(
        cpu_shape,
        dtype=gpu_tensor.dtype,
        device="cpu",
        pin_memory=pin_memory,  # ★ 关键: 锁页内存，允许 GPU 直接 DMA 访问
    )
```

**Pinned Memory 的优势**：
- cudaMemcpyAsync 可以直接 DMA，无需额外的 memcpy
- 避免 OS 页缓存污染
- 传输延迟更低，带宽更高

---

### 2.3 场景2: DRAM → HBM 加载路径 (本地)

与场景1对称，只是传输方向相反：
- `memcpy_type = cudaMemcpyHostToDevice`
- Handler 为 `SingleDirectionOffloadingHandler(gpu_to_cpu=False)`

---

### 2.4 场景3: HBM → SSD 卸载路径 (GDS)

#### 架构总览

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              KV Cache 卸载路径架构 (HBM → DRAM → SSD)                         │
│                                                                                              │
│   ┌──────────────────┐                                                                        │
│   │    GPU HBM       │  ← 热数据（当前请求正在使用的 KV Cache）                                │
│   │   ~80 GB         │    访问延迟: ~100 ns, 带宽: ~3 TB/s                                    │
│   └────────┬─────────┘                                                                        │
│            │                                                                                  │
│            │  阶段1: cudaMemcpyAsync (pinned memory)                                          │
│            │  实现组件: vLLM OffloadingConnector                                              │
│            │  传输技术: CUDA DMA (绕过 OS 页缓存)                                             │
│            ▼                                                                                  │
│   ┌──────────────────┐                                                                        │
│   │    CPU DRAM      │  ← 温数据（近期可能用到的 KV Cache）                                   │
│   │   ~512 GB        │    访问延迟: ~10 μs, 带宽: ~200 GB/s (PCIe 4.0)                       │
│   └────────┬─────────┘                                                                        │
│            │                                                                                  │
│            │  阶段2: 文件写入 / cuFile (可选)                                                  │
│            │  实现组件: Mooncake NVMeoFTransport                                              │
│            │  传输技术: DMA 写入 (CPU 参与，非 GDS 核心特性)                                    │
│            ▼                                                                                  │
│   ┌──────────────────┐                                                                        │
│   │    NVMe SSD      │  ← 冷数据（跨会话复用、持久化）                                        │
│   │   ~4 TB          │    访问延迟: ~100 μs, 带宽: ~12 GB/s                                  │
│   └──────────────────┘                                                                        │
│                                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 完整调用链

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 1: TransferEngine 提交传输请求                                                         │
│  文件: mooncake-transfer-engine/include/transfer_engine.h:84-85                             │
│                                                                                              │
│  TransferRequest 结构:                                                                       │
│  ├── source: void* (源地址，可以是 GPU 或 CPU 地址)                                          │
│  ├── target_id: SegmentID (目标 segment ID)                                                 │
│  ├── target_offset: uint64_t (目标偏移)                                                      │
│  ├── length: uint64_t (传输长度)                                                             │
│  └── opcode: OpCode (READ 或 WRITE)                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ submitTransfer()
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 2: NVMeoFTransport 分发传输任务                                                        │
│  文件: mooncake-transfer-engine/src/transport/nvmeof_transport/nvmeof_transport.cpp:115-206 │
│                                                                                              │
│  Status NVMeoFTransport::submitTransfer(BatchID batch_id, ...) {                             │
│      ├── 1. 遍历所有请求，解析 segment descriptor                                            │
│      ├── 2. 构建 slice 任务                                                                  │
│      ├── 3. 获取或创建 cuFile handle                                                         │
│      └── 4. 提交到 cuFile batch ★                                                           │
│  }                                                                                           │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ addSliceToCUFileBatch()
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 3: CUFileDescPool 构建 cuFile 参数                                                     │
│  文件: mooncake-transfer-engine/src/transport/nvmeof_transport/cufile_desc_pool.cpp:128-163 │
│                                                                                              │
│  CUfileIOParams_t params;                                                                    │
│  params.mode = CUFILE_BATCH;                                                                 │
│  params.opcode = CUFILE_WRITE;  // ★ WRITE for offload                                      │
│  params.u.batch.devPtr_base = source_addr;    // GPU 或 CPU 地址                            │
│  params.u.batch.file_offset = file_offset;                                                  │
│  params.u.batch.size = slice_len;                                                           │
│  params.fh = fh;  // cuFile handle                                                          │
│                                                                                              │
│  CUFILE_CHECK(cuFileBatchIOSubmit(desc->batch_handle->handle,                               │
│                                   desc->io_params.size(),                                   │
│                                   desc->io_params.data(), 0));  ★                           │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ cuFileBatchIOSubmit()
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 4: NVIDIA cuFile API (GPUDirect Storage)                                               │
│                                                                                              │
│  cuFile API 调用栈:                                                                          │
│  ├── cuFileDriverOpen() - 初始化 cuFile driver                                              │
│  ├── cuFileBufRegister() - 注册 GPU/CPU buffer                                              │
│  ├── cuFileHandleRegister() - 注册文件 handle                                               │
│  ├── cuFileBatchIOSetUp() - 设置 batch IO                                                   │
│  ├── cuFileBatchIOSubmit() - 提交 batch IO 请求 ★                                           │
│  └── cuFileBatchIOGetStatus() - 获取完成状态                                                 │
│                                                                                              │
│  注意: DRAM→SSD 场景源地址是 CPU DRAM，GDS 核心特性是 HBM↔SSD 直连                           │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ DMA 写入 (cuFile 或 write)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 5: 硬件层 (DRAM → SSD 文件写入)                                                        │
│                                                                                              │
│  ┌──────────────────────┐                         ┌──────────────────────┐                   │
│  │      CPU DRAM        │   DMA (可选 cuFile)    │      NVMe SSD        │                   │
│  │  ┌────────────────┐  │   ~12-24 GB/s          │  ┌────────────────┐  │                   │
│  │  │ KV Cache       │  │ ────────────────────► │  │ File System    │  │                   │
│  │  │ (Pinned)       │  │   DMA 写入            │  │ /data/...      │  │                   │
│  │  │ 0x7f1234560000 │  │   CPU 参与            │  │                │  │                   │
│  │  └────────────────┘  │                       │  └────────────────┘  │                   │
│  └──────────────────────┘                       └──────────────────────┘                   │
│                                                                                              │
│  注意: 这是 DRAM→SSD 的常规写入，CPU 参与 DMA 设置                                           │
│        (真正 GDS 的 HBM↔SSD 直连，见场景4加载路径)                                           │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### GDS Transport (Tent 版本)

```cpp
// 文件: mooncake-transfer-engine/tent/src/transport/gds/gds_transport.cpp:258-302

Status GdsTransport::submitTransferTasks(SubBatchRef batch,
                                          const std::vector<Request>& request_list) {
    auto gds_batch = dynamic_cast<GdsSubBatch*>(batch);
    const static size_t kMaxSliceSize = 16ull << 20;  // 16 MB max slice

    for (auto& request : request_list) {
        GdsFileContext* context = findFileContext(request.target_id);

        // 将大请求分割成 16MB 的 slice
        for (size_t offset = 0; offset < request.length; offset += kMaxSliceSize) {
            size_t length = std::min(kMaxSliceSize, request.length - offset);

            CUfileIOParams_t params;
            params.mode = CUFILE_BATCH;
            params.opcode = (request.opcode == Request::READ) ? CUFILE_READ : CUFILE_WRITE;
            params.u.batch.devPtr_base = request.source;
            params.u.batch.devPtr_offset = offset;
            params.u.batch.file_offset = request.target_offset + offset;
            params.u.batch.size = length;
            params.fh = context->getHandle();

            gds_batch->io_params.push_back(params);
        }
    }

    // 批量提交
    auto result = cuFileBatchIOSubmit(gds_batch->batch_handle->handle,
                                      num_params,
                                      &gds_batch->io_params[first_param_index], 0);
    return Status::OK();
}
```

**关键优化**：
- BatchHandle 复用池：避免频繁的 cuFileBatchIOSetUp/Destroy
- 16MB slice 分割：优化大文件传输
- 异步状态查询：非阻塞的完成检查

---

### 2.5 场景4: SSD → HBM 加载路径 (GDS)

#### 架构总览

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              KV Cache 加载路径架构 (SSD → HBM)                                 │
│                                                                                              │
│   ┌──────────────────┐                                                                        │
│   │    NVMe SSD      │  ← 冷数据存储位置                                                      │
│   │   ~4 TB          │    访问延迟: ~100 μs, 带宽: ~12 GB/s                                  │
│   └────────┬─────────┘                                                                        │
│            │                                                                                  │
│            │  GDS / cuFile (CUFILE_READ)                                                      │
│            │  实现组件: Mooncake NVMeoFTransport                                              │
│            │  传输技术: GPUDirect Storage (绕过 CPU)                                          │
│            ▼                                                                                  │
│   ┌──────────────────┐                                                                        │
│   │    GPU HBM       │  ← 热数据加载目标                                                      │
│   │   ~80 GB         │    访问延迟: ~100 ns, 带宽: ~3 TB/s                                    │
│   └──────────────────┘                                                                        │
│                                                                                              │
│  ★ GDS 加载路径直接从 SSD 到 GPU HBM，完全绕过 CPU DRAM ★                                    │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 完整调用链

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 1: 应用层请求加载 KV Cache                                                             │
│                                                                                              │
│  场景: 跨会话 KV Cache 复用、Checkpoint 恢复、模型权重加载                                    │
│                                                                                              │
│  请求示例:                                                                                    │
│  TransferRequest request;                                                                    │
│  request.source = gpu_buffer;      // 目标 GPU buffer                                       │
│  request.target_id = file_segment_id;                                                       │
│  request.target_offset = 0;                                                                  │
│  request.length = cache_size;                                                                │
│  request.opcode = Transport::TransferRequest::READ;  // ★ READ operation                    │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ submitTransfer(batch_id, {request})
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 2: NVMeoFTransport 处理读取请求                                                        │
│  文件: mooncake-transfer-engine/src/transport/nvmeof_transport/nvmeof_transport.cpp:115-206 │
│                                                                                              │
│  处理逻辑:                                                                                    │
│  ├── 1. 解析 segment descriptor                                                              │
│  ├── 2. 计算文件偏移和范围                                                                    │
│  ├── 3. 获取 cuFile handle                                                                   │
│  └── 4. 添加到 cuFile batch (opcode=READ) ★                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ addSliceToCUFileBatch() with opcode=READ
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 3: CUFileDescPool 构建 READ 参数                                                       │
│                                                                                              │
│  CUfileIOParams_t params;                                                                    │
│  params.mode = CUFILE_BATCH;                                                                 │
│  params.opcode = CUFILE_READ;  // ★ READ from file to GPU                                   │
│  params.u.batch.devPtr_base = source_addr;    // GPU buffer address                         │
│  params.u.batch.file_offset = file_offset;    // File offset                                │
│  params.u.batch.size = slice_len;             // Read size                                  │
│  params.fh = fh;                              // cuFile handle                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ cuFileBatchIOSubmit(opcode=CUFILE_READ)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 4: NVIDIA cuFile API 执行 GDS 读取                                                     │
│                                                                                              │
│  cuFile 读取流程:                                                                             │
│  ├── 1. cuFileBatchIOSubmit() - 提交批量读取请求                                             │
│  │   ├── 参数验证                                                                            │
│  │   ├── DMA 引擎配置                                                                        │
│  │   └── 提交到 NVMe 驱动                                                                    │
│  │                                                                                           │
│  ├── 2. NVMe 驱动执行                                                                        │
│  │   ├── NVMe 命令构建                                                                       │
│  │   ├── PCIe DMA 设置                                                                       │
│  │   └── 直接传输到 GPU HBM                                                                  │
│  │                                                                                           │
│  └── 3. cuFileBatchIOGetStatus() - 轮询完成状态                                              │
│      ├── 检查传输状态                                                                        │
│      └── 返回传输字节数                                                                      │
│                                                                                              │
│  GDS 读取数据流: NVMe SSD ──PCIe──► GPU HBM (绕过 CPU)                                       │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ GDS DMA (SSD → GPU)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 5: 硬件层 (GDS DMA)                                                                    │
│                                                                                              │
│  ┌──────────────────────┐                         ┌──────────────────────┐                   │
│  │      NVMe SSD        │   PCIe + NVMe          │      GPU HBM         │                   │
│  │  ┌────────────────┐  │   ~12-24 GB/s          │  ┌────────────────┐  │                   │
│  │  │ File System    │  │ ─────────────────────► │  │ KV Cache       │  │                   │
│  │  │ /data/kv_cache │  │   GDS DMA             │  │ Block 0,1,2... │  │                   │
│  │  │ checkpoint.bin │  │   绕过 CPU            │  │ Ready for use  │  │                   │
│  │  └────────────────┘  │                       │  └────────────────┘  │                   │
│  └──────────────────────┘                       └──────────────────────┘                   │
│                                                                                              │
│  ★ GDS 加载路径: SSD 直接 DMA 到 GPU，无需 CPU 参与，实现零拷贝加载 ★                        │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 状态查询与同步

```cpp
// 文件: mooncake-transfer-engine/src/transport/nvmeof_transport/cufile_desc_pool.cpp:165-190

CUfileIOEvents_t CUFileDescPool::getTransferStatus(int idx, int slice_id) {
    auto* desc = descs_[idx];

    // 查询所有 slice 的状态
    unsigned nr = desc->io_params.size();
    CUFILE_CHECK(cuFileBatchIOGetStatus(desc->batch_handle->handle,
                                        0, &nr,
                                        desc->io_events.data(),
                                        nullptr));

    return desc->io_events[slice_id];  // 返回特定 slice 的状态
}

// 状态枚举
enum CUfileStatus_t {
    CUFILE_WAITING = 0,    // 等待中
    CUFILE_PENDING = 1,    // 处理中
    CUFILE_COMPLETE = 2,   // 已完成
    CUFILE_CANCELED = 3,   // 已取消
    CUFILE_FAILED = 4,     // 失败
    CUFILE_TIMEOUT = 5,    // 超时
    CUFILE_INVALID = 6,    // 无效
};
```

---

### 2.6 场景5: HBM → HBM 跨节点传输 (GDRDMA)

#### 架构总览

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                         KV Cache 跨节点传输架构 (GDRDMA)                                       │
│                                                                                              │
│   ┌──────────────────┐                                                                        │
│   │  Prefill Node    │                                                                        │
│   │  ┌────────────┐  │                                                                        │
│   │  │  GPU HBM   │  │  ← KV Cache 源节点                                      │
│   │  │  ~80 GB    │  │    存储 prefill 产生的完整 KV Cache                     │
│   │  └─────┬──────┘  │                                                        │
│   └────────┼─────────┘                                                        │
│            │                                                                                  │
│            │  GDRDMA (GPUDirect RDMA)                                         │
│            │  实现组件: Mooncake RdmaTransport                                 │
│            │  传输技术: RDMA (绕过 CPU，直接 GPU-to-GPU)                       │
│            │  典型带宽: ~25 GB/s (200Gbps IB)                                  │
│            │                                                                                  │
│            ▼                                                                                  │
│   ┌──────────────────┐                                                                        │
│   │  Decode Node     │                                                                        │
│   │  ┌────────────┐  │                                                                        │
│   │  │  GPU HBM   │  │  ← KV Cache 目标节点                                    │
│   │  │  ~80 GB    │  │    接收 KV Cache 用于 decode                            │
│   │  └────────────┘  │                                                        │
│   └──────────────────┘                                                                        │
│                                                                                              │
│  ★ GDRDMA: Prefill GPU 直接 RDMA 写入 Decode GPU，完全绕过两端 CPU ★                         │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 完整调用链

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  Step 1: Decode 节点发起请求 (vLLM)                                                          │
│  ├── get_num_new_matched_tokens() 返回 (num_tokens, True)                                   │
│  └── build_connector_meta() 构建请求元数据                                                   │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  Step 2: MooncakeConnector 发送请求 (Mooncake)                                               │
│  ├── 通过 ZMQ 发送 MooncakeXferMetadata                                                     │
│  │   ├── remote_hostname: Decode 节点 IP                                                   │
│  │   ├── remote_port: Mooncake RPC 端口                                                    │
│  │   ├── req_blocks: {req_id: (transfer_id, block_ids)}                                    │
│  │   └── kv_caches_base_addr: GPU 内存基地址                                                │
│  └── sock.send(encoded_metadata)                                                            │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  Step 3: Prefill 节点处理请求 (Mooncake)                                                      │
│  ├── _mooncake_sender_listener 接收 ZMQ 请求                                                │
│  ├── send_kv_to_decode() 执行发送                                                           │
│  └── engine.batch_transfer_sync_write()                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  Step 4: RdmaTransport 执行 RDMA 传输 (Mooncake → CUDA)                                      │
│  ├── 获取 RDMA Queue Pair                                                                   │
│  ├── ibv_reg_mr() 注册 GPU 内存                                                             │
│  ├── 构建 RDMA Work Request                                                                 │
│  │   ├── wr.wr.rdma.remote_addr = Decode GPU 地址                                           │
│  │   ├── wr.wr.rdma.rkey = 远程内存 key                                                     │
│  │   └── wr.sg_list->addr = Prefill GPU 地址                                                │
│  └── ibv_post_send() 执行 RDMA Send                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  Step 5: 硬件层 RDMA 传输                                                                    │
│  └── Prefill GPU ──InfiniBand/RoCE──► Decode GPU                                            │
│       (绕过 CPU，直接 GPU-to-GPU)                                                            │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、组件间接口契约

### 3.1 vLLM → PyTorch 接口

| 接口 | 类型 | 说明 |
|------|------|------|
| `torch.Tensor` | 数据结构 | KV Cache 的物理存储载体 |
| `tensor.device` | 属性 | "cuda" (GPU) 或 "cpu" (CPU) |
| `tensor.data_ptr()` | 方法 | 获取底层内存指针 |
| `torch.cuda.Stream()` | 构造函数 | 创建 CUDA 流 |
| `torch.Event()` | 构造函数 | 创建 CUDA 事件 |

### 3.2 PyTorch → vLLM CUDA Extension 接口

| 接口 | ATen C++ API | 说明 |
|------|--------------|------|
| 设备信息 | `tensor.device()` | 判断是 GPU 还是 CPU |
| 数据指针 | `tensor.data_ptr()` | 获取底层内存地址 |
| CUDA Stream | `at::cuda::getCurrentCUDAStream()` | 获取当前 CUDA 流 |
| 设备切换 | `at::cuda::OptionalCUDAGuard` | 自动切换 GPU 设备 |

### 3.3 vLLM CUDA Extension → CUDA Runtime 接口

| 接口 | CUDA API | 说明 |
|------|----------|------|
| 内存传输 | `cudaMemcpyAsync()` | GPU↔CPU DMA 传输 |
| 流同步 | `cudaStreamSynchronize()` | 等待流完成 |
| 事件记录 | `cudaEventRecord()` | 记录时间点 |
| 事件查询 | `cudaEventQuery()` | 非阻塞检查完成状态 |

### 3.4 Mooncake → CUDA/cuFile 接口

| 接口 | API | 说明 |
|------|-----|------|
| GDS 初始化 | `cuFileDriverOpen()` | 初始化 cuFile 驱动 |
| Buffer 注册 | `cuFileBufRegister()` | 注册 GPU/CPU buffer |
| 文件注册 | `cuFileHandleRegister()` | 注册文件句柄 |
| 批量 IO | `cuFileBatchIOSubmit()` | 提交批量 GDS 请求 |
| 状态查询 | `cuFileBatchIOGetStatus()` | 获取完成状态 |

### 3.5 Mooncake → RDMA verbs 接口

| 接口 | API | 说明 |
|------|-----|------|
| 内存注册 | `ibv_reg_mr()` | 注册 RDMA 内存区域 |
| QP 创建 | `ibv_create_qp()` | 创建 Queue Pair |
| 发送请求 | `ibv_post_send()` | 提交 RDMA Send |
| 完成轮询 | `ibv_poll_cq()` | 轮询完成队列 |

---

## 四、数据流与控制流分离

### 4.1 控制流 (Control Flow)

```
vLLM Scheduler (决策)
    │
    ├──► OffloadingConnectorScheduler (构建 TransferSpec)
    │        │
    │        └──► OffloadingConnectorWorker (分配 job_id)
    │                 │
    │                 └──► OffloadingWorker (路由到 Handler)
    │                          │
    │                          └──► SingleDirectionOffloadingHandler (准备资源)
    │                                   │
    │                                   └──► PyTorch Stream/Event 管理
```

### 4.2 数据流 (Data Flow)

```
GPU HBM (torch.Tensor, device="cuda")
    │
    │  cudaMemcpyAsync()
    │  ├── src_ptr = tensor.data_ptr()  (GPU 地址)
    │  ├── dst_ptr = tensor.data_ptr()  (CPU pinned 地址)
    │  └── stream = at::cuda::getCurrentCUDAStream()
    │
    ▼
CPU DRAM (torch.Tensor, device="cpu", pin_memory=True)
```

### 4.3 关键分离点

| 分离点 | 位置 | 说明 |
|--------|------|------|
| **决策与执行分离** | Scheduler vs Worker | Scheduler 决定"传什么"，Worker 执行"怎么传" |
| **数据与传输分离** | Tensor vs cudaMemcpy | Tensor 是数据载体，cudaMemcpy 是传输执行 |
| **同步与异步分离** | submit vs get_finished | 异步提交，轮询检查完成 |
| **流与计算分离** | transfer_stream vs compute_stream | 传输流与计算流并行 |

---

## 五、关键技术对比

### 5.1 传输技术对比

| 特性 | cudaMemcpyAsync (HBM↔DRAM) | GDS (HBM↔SSD) | RDMA (HBM↔HBM) |
|------|---------------------------|---------------|----------------|
| **实现组件** | vLLM OffloadingConnector | Mooncake NVMeoFTransport | Mooncake RdmaTransport |
| **底层 API** | cudaMemcpyAsync | cuFileBatchIOSubmit | ibv_post_send |
| **数据路径** | GPU ←PCIe→ CPU | GPU ←PCIe→ SSD | GPU ←IB/RoCE→ GPU |
| **是否绕过 CPU** | 是 (pinned memory) | 是 (GDS) | 是 (RDMA) |
| **典型带宽** | ~32 GB/s (PCIe 4.0 x16) | ~12-24 GB/s (NVMe) | ~25 GB/s (200Gbps IB) |
| **延迟** | ~10 μs | ~100 μs | ~2-5 μs |
| **使用场景** | 本地 KV Cache offload | KV Cache 持久化 | 跨节点 KV Cache 传输 |
| **内存要求** | CPU pinned memory | GPU/CPU buffer | GPU RDMA registered MR |

### 5.2 核心概念辨析

| 技术 | 全称 | 数据路径 | 典型带宽 | 使用场景 |
|------|------|----------|----------|----------|
| **GDS** | GPUDirect Storage | GPU HBM ↔ NVMe SSD | ~12-24 GB/s | KV Cache 持久化、Checkpoint |
| **GDRDMA** | GPUDirect RDMA | GPU HBM ↔ GPU HBM (跨节点) | ~25 GB/s (200Gbps IB) | Prefill-Decode 分离架构 |
| **cudaMemcpyAsync** | CUDA DMA | GPU HBM ↔ CPU DRAM | ~32 GB/s (PCIe 4.0 x16) | 本地 KV Cache Offload |

**关键区分**：
- **GDS ≠ GDRDMA**：GDS 是 GPU 直连存储，GDRDMA 是 GPU 间 RDMA 通信
- **典型 Offload 路径**：HBM → DRAM (cudaMemcpyAsync)，而非 HBM → SSD (GDS)
- **跨节点传输**：使用 RDMA (GDRDMA)，而非 GDS

### 5.3 PyTorch GDS 接口 vs Mooncake NVMeoFTransport

| 特性 | PyTorch torch.cuda.gds | Mooncake NVMeoFTransport |
|------|------------------------|--------------------------|
| **接口层级** | Python 高级接口 | C++ 底层实现 |
| **Batch 支持** | 单个 IO | cuFileBatchIOSubmit (批量) |
| **集成方式** | 需要显式调用 | 与 TransferEngine 统一接口 |
| **vLLM 使用** | **未使用** | **使用** |

**结论**：vLLM 选择 Mooncake 的传输引擎而非 PyTorch 原生 GDS 接口，主要原因是：
- Mooncake 提供统一的传输抽象（RDMA + GDS + TCP）
- 支持批量 IO 操作
- 与 vLLM 的分布式架构更好集成

---

## 六、性能优化建议

### 6.1 HBM → DRAM 优化

1. **使用 pinned memory 分配 CPU tensor**
   ```python
   cpu_tensor = torch.zeros(shape, dtype=dtype, device="cpu", pin_memory=True)
   ```

2. **使用独立 CUDA stream 进行异步传输**
   ```python
   stream = torch.cuda.Stream()
   with torch.cuda.stream(stream):
       ops.swap_blocks(src_tensor, dst_tensor, ...)
   ```

3. **合并小 block 为大 batch 减少调用开销**

### 6.2 GDS 使用优化

1. **复用 CUfileBatchHandle_t，避免频繁 SetUp/Destroy**
2. **使用 16MB slice 分割大文件传输**
3. **批量提交多个 IO 请求**

### 6.3 内存管理

1. **及时 deregister 不再使用的 buffer**
2. **监控 GPU 内存使用，避免 OOM**
3. **使用 LRU/ARC 策略管理 offloaded blocks**

---

## 七、完整代码位置索引

### 7.1 本地传输 (HBM ↔ DRAM)

| 组件 | 文件 | 关键函数 |
|------|------|----------|
| vLLM Scheduler | vllm/v1/kv_offload/lru_manager.py | LRUManager.touch() |
| vLLM Scheduler | vllm/v1/kv_offload/arc_manager.py | ARCManager.touch() |
| vLLM Connector Scheduler | vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:358-475 | update_state_after_alloc(), _get_reqs_to_store() |
| vLLM Connector Worker | vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:629-651 | start_kv_transfers(), prepare_store_kv() |
| vLLM Offloading Worker | vllm/v1/kv_offload/worker/worker.py:114-146 | transfer_async() |
| PyTorch Handler | vllm/v1/kv_offload/worker/cpu_gpu.py:119-192 | SingleDirectionOffloadingHandler.transfer_async() |
| CUDA Ops | vllm/_custom_ops.py, csrc/cache_kernels.cu, csrc/attention/swap_blocks.cu | swap_blocks() → cudaMemcpyAsync() |

### 7.2 跨节点传输 (RDMA)

| 组件 | 文件 | 关键函数 |
|------|------|----------|
| vLLM Mooncake Connector | vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py | send_kv_to_decode() |
| Mooncake TransferEngine | mooncake-transfer-engine/src/transfer_engine.cpp | submitTransfer() |
| Mooncake RdmaTransport | mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp | submitTransfer() |
| RDMA verbs | libibverbs | ibv_post_send() |

### 7.3 GDS 传输 (HBM ↔ SSD)

| 组件 | 文件 | 关键函数 |
|------|------|----------|
| Mooncake TransferEngine | mooncake-transfer-engine/src/transfer_engine.cpp | submitTransfer() |
| Mooncake NVMeoFTransport | mooncake-transfer-engine/src/transport/nvmeof_transport/nvmeof_transport.cpp:115-206 | submitTransfer() |
| Mooncake CUFileDescPool | mooncake-transfer-engine/src/transport/nvmeof_transport/cufile_desc_pool.cpp:128-163, 165-190 | submitBatch(), getTransferStatus() |
| NVIDIA cuFile | libcufile.so | cuFileBatchIOSubmit(), cuFileBatchIOGetStatus() |

### 7.4 GDS Transport (Tent 版本)

| 组件 | 文件 | 关键函数 |
|------|------|----------|
| GDS Transport | mooncake-transfer-engine/tent/src/transport/gds/gds_transport.cpp:258-302 | submitTransferTasks() |
| GDS File Context | mooncake-transfer-engine/tent/src/transport/gds/gds_transport.cpp:32-68 | GdsFileContext (cuFileHandleRegister) |
| Memory Buffer | mooncake-transfer-engine/tent/src/transport/gds/gds_transport.cpp:333-344 | addMemoryBuffer() (cuFileBufRegister) |

### 7.5 PyTorch GDS 接口 (参考)

| 组件 | 文件 | 关键函数 |
|------|------|----------|
| PyTorch GDS | pytorch/torch/cuda/gds.py | GdsFile, gds_register_buffer() |

---

## 八、总结

### 8.1 关键结论

1. **GDS 主要用于 GPU ↔ NVMe SSD 的直接传输**
2. **典型 KV Cache offload 路径是 HBM → DRAM，使用 cudaMemcpyAsync**
3. **跨节点传输使用 RDMA (GDRDMA)，而非 GDS**
4. **pinned memory 是实现 CPU 绕过的关键技术**
5. **PyTorch 提供数据载体和 CUDA 抽象，但不直接执行传输**

### 8.2 架构设计原则

1. **决策与执行分离**：Scheduler 决策，Worker 执行
2. **数据与传输分离**：Tensor 是数据，cudaMemcpy 是传输
3. **同步与异步分离**：异步提交，轮询完成
4. **流与计算分离**：传输流与计算流并行
