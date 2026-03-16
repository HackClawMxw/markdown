# vLLM + Mooncake + GDS KV Cache 完整调用链分析

> 本文档深入分析 vLLM 通过 Mooncake Transfer Engine 利用 NVIDIA GPUDirect Storage (GDS/cuFile) 进行 KV Cache 传输的完整技术栈。

---

## 目录

1. [技术背景与架构概览](#1-技术背景与架构概览)
2. [完整技术栈架构](#2-完整技术栈架构)
3. [Mooncake GDS 传输实现](#3-mooncake-gds-传输实现)
4. [PyTorch 层：GPU 内存地址获取与传递](#4-pytorch-层gpu-内存地址获取与传递)
5. [vLLM + Mooncake 集成](#5-vllm--mooncake-集成)
6. [cuFile API 调用详解](#6-cufile-api-调用详解)
7. [完整调用链图](#7-完整调用链图)
8. [三种传输模式对比](#8-三种传输模式对比)
9. [性能优化关键点](#9-性能优化关键点)
10. [关键文件索引](#10-关键文件索引)

---

## 1. 技术背景与架构概览

### 1.1 为什么需要 GDS？

传统的 KV Cache 存储层次：

```
┌─────────────────────────────────────────────────────────────────┐
│                    传统存储层次 (无 GDS)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GPU HBM ◄──────────────► CPU Memory ◄──────────────► SSD      │
│  (80GB)     cudaMemcpyAsync    (256GB)    read/write   (TB级)   │
│             ~32 GB/s            任意       ~5 GB/s              │
│                                                                 │
│  问题：                                                          │
│  1. CPU Memory 容量有限，无法存储大量 KV Cache                   │
│  2. 数据需要经过 CPU，增加延迟和内存占用                          │
│  3. SSD 访问需要 CPU 中转，效率低下                               │
└─────────────────────────────────────────────────────────────────┘
```

**GDS (GPUDirect Storage) 的解决方案**：

```
┌─────────────────────────────────────────────────────────────────┐
│                    GDS 存储层次                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GPU HBM ◄──────────────────────────────► NVMe SSD              │
│  (80GB)          cuFile API                 (TB级)               │
│                  直接 DMA 传输               ~12-24 GB/s         │
│                                                                 │
│  优势：                                                          │
│  1. 绕过 CPU，减少内存拷贝                                       │
│  2. 更高带宽（接近 NVMe 原生性能）                               │
│  3. 降低 CPU 负载和延迟                                          │
│  4. 支持 TB 级存储容量                                           │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 核心组件关系

| 组件 | 职责 | 技术栈 |
|------|------|--------|
| **vLLM** | LLM 推理引擎，管理 KV Cache | Python |
| **PyTorch** | GPU 内存管理，tensor.data_ptr() 获取 GPU 地址 | Python/C++ |
| **pybind11** | Python ↔ C++ 绑定，类型转换 (int → void*) | C++ |
| **Mooncake** | 分布式传输引擎，支持多种传输协议 | C++/Python |
| **cuFile** | NVIDIA GDS SDK，GPU 直接访问存储 | C API |
| **CUDA** | GPU 计算和内存管理 | Driver/Runtime API |
| **NVMe** | 高性能存储设备 | 硬件 |

---

## 2. 完整技术栈架构

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              vLLM 应用层                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│  │                         MooncakeConnectorScheduler                               │  │
│  │   • get_num_new_matched_tokens() - 判断是否需要远程 KV                           │  │
│  │   • update_state_after_alloc() - 准备传输元数据                                  │  │
│  │   • build_connector_meta() - 构建传输请求                                        │  │
│  └─────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ IPC (SchedulerOutput / MooncakeConnectorMetadata)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              vLLM Worker 层                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│  │                         MooncakeConnectorWorker                                   │  │
│  │   • register_kv_caches() - 注册 GPU KV Cache 到 Mooncake                         │  │
│  │   • start_load_kv() - 启动 KV 加载                                               │  │
│  │   • send_kv_to_decode() - 发送 KV 到 Decode 节点                                 │  │
│  │   • receive_kv() - 接收 KV 数据                                                  │  │
│  └─────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ kv_caches: dict[str, torch.Tensor]
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           ★ PyTorch 层 (GPU 内存管理) ★                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              torch.Tensor                                         │  │
│  │   • kv_cache.data_ptr() → GPU 内存地址 (int)    例如: 0x7f1234567890             │  │
│  │   • kv_cache.nbytes     → 字节大小 (int)        例如: 1073741824 (1GB)           │  │
│  │   • kv_cache.device     → torch.device('cuda:0')                                 │  │
│  │                                                                                  │  │
│  │   关键：data_ptr() 返回的是 CUDA 驱动分配的 GPU 虚拟地址                          │  │
│  │         这个地址可以直接传给 cuFileBufRegister 进行 GDS DMA 传输                  │  │
│  └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                              │
│                                        │ kv_data_ptrs: List[int] = [0x7f..., 0x7f...] │
│                                        │ kv_data_lens: List[int] = [1GB, 1GB, ...]     │
│                                        ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│  │                     engine.register_local_memory_batch()                         │  │
│  │   Python 调用：self.engine.register_local_memory_batch(addr_list, size_list)     │  │
│  └─────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ Python List[int] → C++ std::vector<uint64_t>
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           ★ pybind11 层 (类型转换) ★                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│  │                     mooncake-transfer-engine/tent/src/python/pybind.cpp          │  │
│  │                                                                                  │  │
│  │   // 关键转换函数：Python int → C++ void*                                        │  │
│  │   static inline void* U64ToPtr(uint64_t a) {                                     │  │
│  │       return reinterpret_cast<void*>(static_cast<std::uintptr_t>(a));            │  │
│  │   }                                                                              │  │
│  │                                                                                  │  │
│  │   // 批量转换：List[int] → std::vector<void*>                                    │  │
│  │   std::vector<void*> U64VectorToPtrVector(const std::vector<uint64_t>& list) {   │  │
│  │       for (auto a : list) ptrs.push_back(U64ToPtr(a));                           │  │
│  │       return ptrs;  // {(void*)0x7f1234567890, ...}                              │  │
│  │   }                                                                              │  │
│  │                                                                                  │  │
│  │   转换结果：Python int 0x7f1234567890 → C++ void* (void*)0x7f1234567890          │  │
│  └─────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ std::vector<void*> gpu_addrs
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              Mooncake Transfer Engine 层                                │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              TransferEngine                                       │  │
│  │   • registerLocalMemory(void* ptr, size_t size) - 单个内存注册                   │  │
│  │   • registerLocalMemory(vector<void*> ptrs, ...) - 批量内存注册                  │  │
│  │   • submitTransfer() - 提交传输任务                                              │  │
│  └─────────────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                              │
│            ┌───────────────────────────┼───────────────────────────┐                  │
│            ▼                           ▼                           ▼                  │
│  ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐           │
│  │  RDMA Transport │        │ NVMeoF Transport│        │  GDS Transport  │           │
│  │  (IB/RoCE)      │        │  (NVMe-over-    │        │  (cuFile)       │           │
│  │                 │        │   Fabric)       │        │                 │           │
│  │  IB Verbs       │        │  cuFile API     │        │  cuFile API     │           │
│  └─────────────────┘        └─────────────────┘        └─────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ cuFileBufRegister(void* devPtr, size_t size, int flags)
                                        │ cuFileBatchIOSubmit(handle, nr, io_params, flags)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              NVIDIA cuFile / GDS 层                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│  │                           cuFile Driver                                           │  │
│  │   • cuFileDriverOpen() - 初始化 GDS 驱动                                         │  │
│  │   • cuFileHandleRegister() - 注册文件句柄                                        │  │
│  │   • cuFileBufRegister() - 注册 GPU 内存缓冲区 ★ 接收 void* GPU 地址              │  │
│  │   • cuFileBatchIOSetUp() - 设置批量 I/O                                          │  │
│  │   • cuFileBatchIOSubmit() - 提交批量 I/O                                         │  │
│  │   • cuFileBatchIOGetStatus() - 获取 I/O 状态                                     │  │
│  └─────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ GDS Kernel Module / DMA Engine
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              硬件层                                                     │
│  ┌──────────────────┐                              ┌──────────────────┐                │
│  │    GPU HBM       │◄────────────────────────────►│   NVMe SSD       │                │
│  │   (~80 GB)       │      PCIe 4.0/5.0 + NVMe     │   (TB 级)        │                │
│  │                  │      ~12-24 GB/s 带宽         │                  │                │
│  └──────────────────┘                              └──────────────────┘                │
│                                                                                         │
│  关键：DMA 引擎直接在 GPU HBM 和 NVMe SSD 之间传输数据，完全绕过 CPU 内存               │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 数据流转关键点

| 层级 | 输入类型 | 输出类型 | 关键操作 |
|------|----------|----------|----------|
| **PyTorch** | `torch.Tensor` | `int` (0x7f...) | `tensor.data_ptr()` |
| **pybind11** | `List[int]` | `std::vector<void*>` | `U64VectorToPtrVector()` |
| **C++ Engine** | `void*` | `void*` | 直接传递指针 |
| **cuFile** | `void*` | DMA 地址 | `cuFileBufRegister()` |

---

## 3. Mooncake GDS 传输实现

### 3.1 两种 GDS 实现

Mooncake 提供了两种使用 cuFile 的传输实现：

| 实现 | 路径 | 用途 |
|------|------|------|
| **NVMeoFTransport** | `mooncake-transfer-engine/src/transport/nvmeof_transport/` | NVMe-over-Fabric 远程存储 |
| **GdsTransport (TENT)** | `mooncake-transfer-engine/tent/src/transport/gds/` | 本地 NVMe SSD 直连 |

### 3.2 NVMeoFTransport 核心实现

**文件**: `mooncake-transfer-engine/src/transport/nvmeof_transport/nvmeof_transport.cpp`

```cpp
namespace mooncake {

// 初始化：打开 cuFile 驱动
NVMeoFTransport::NVMeoFTransport() {
    CUFILE_CHECK(cuFileDriverOpen());  // 初始化 GDS 驱动
    desc_pool_ = std::make_shared<CUFileDescPool>();
}

// 注册 GPU 内存到 cuFile
int NVMeoFTransport::registerLocalMemory(void *addr, size_t length,
                                         const std::string &location,
                                         bool remote_accessible,
                                         bool update_metadata) {
    // 关键：将 GPU 内存注册到 cuFile，允许直接 DMA 访问
    CUFILE_CHECK(cuFileBufRegister(addr, length, 0));
    return 0;
}

// 注销 GPU 内存
int NVMeoFTransport::unregisterLocalMemory(void *addr, bool update_metadata) {
    CUFILE_CHECK(cuFileBufDeregister(addr));
    return 0;
}

// 提交传输任务
Status NVMeoFTransport::submitTransfer(
    BatchID batch_id, const std::vector<TransferRequest> &entries) {

    for (auto &request : entries) {
        // 1. 获取文件路径
        const char *file_path = buffer_desc.local_path_map[local_server_name_].c_str();

        // 2. 获取或创建 CuFileContext（包含 CUfileHandle_t）
        auto buf_key = std::make_pair(target_id, buffer_id);
        if (!segment_to_context_.count(buf_key)) {
            segment_to_context_[buf_key] = std::make_shared<CuFileContext>(file_path);
        }
        CUfileHandle_t fh = segment_to_context_.at(buf_key)->getHandle();

        // 3. 添加到批量 I/O
        addSliceToCUFileBatch(source_addr, file_offset, slice_len,
                              nvmeof_desc.desc_idx_, request.opcode, fh);
    }

    // 4. 提交批量 I/O
    desc_pool_->submitBatch(nvmeof_desc.desc_idx_);
    return Status::OK();
}
```

### 3.3 CuFileContext 文件句柄管理

**文件**: `mooncake-transfer-engine/include/transport/nvmeof_transport/cufile_context.h`

```cpp
class CuFileContext {
    CUfileHandle_t handle = NULL;
    CUfileDescr_t desc;

   public:
    // 创建 GDS 文件句柄
    explicit CuFileContext(const char *filename) {
        // 1. 以 O_DIRECT 方式打开文件（必须！）
        int fd = open(filename, O_RDWR | O_DIRECT);

        // 2. 配置 cuFile 描述符
        memset(&desc, 0, sizeof(desc));
        desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;  // 使用文件描述符
        desc.handle.fd = fd;

        // 3. 注册文件句柄到 cuFile
        CUFILE_CHECK(cuFileHandleRegister(&handle, &desc));
    }

    ~CuFileContext() {
        if (handle) cuFileHandleDeregister(handle);
        if (desc.handle.fd) close(desc.handle.fd);
    }

    CUfileHandle_t getHandle() const { return handle; }
};
```

### 3.4 批量 I/O 描述符池

**文件**: `mooncake-transfer-engine/src/transport/nvmeof_transport/cufile_desc_pool.cpp`

```cpp
class CUFileDescPool {
    // 批量 I/O 句柄池（复用昂贵的 cuFileBatchIOSetUp）
    std::vector<BatchHandle*> handle_pool_;

    // 分配批量描述符
    int allocCUfileDesc(size_t batch_size) {
        auto* desc = new CUFileBatchDesc();

        BatchHandle* batch_handle = nullptr;
        {
            std::lock_guard<std::mutex> lock(handle_pool_lock_);
            if (!handle_pool_.empty()) {
                // 从池中复用句柄（避免昂贵的 cuFileBatchIOSetUp）
                batch_handle = handle_pool_.back();
                handle_pool_.pop_back();
            }
        }

        if (!batch_handle) {
            // 首次创建句柄（耗时操作）
            batch_handle = new BatchHandle();
            batch_handle->max_nr = max_batch_size_;
            CUFILE_CHECK(cuFileBatchIOSetUp(&batch_handle->handle, max_batch_size_));
        }

        desc->batch_handle = batch_handle;
        descs_[idx] = desc;
        return idx;
    }

    // 提交批量 I/O
    int submitBatch(int idx) {
        auto* desc = descs_[idx];
        // 调用 cuFile 批量提交
        CUFILE_CHECK(cuFileBatchIOSubmit(desc->batch_handle->handle,
                                         desc->io_params.size(),
                                         desc->io_params.data(), 0));
        return 0;
    }

    // 获取传输状态
    CUfileIOEvents_t getTransferStatus(int idx, int slice_id) {
        auto* desc = descs_[idx];
        unsigned nr = desc->io_params.size();
        CUFILE_CHECK(cuFileBatchIOGetStatus(desc->batch_handle->handle, 0, &nr,
                                            desc->io_events.data(), nullptr));
        return desc->io_events[slice_id];
    }
};
```

### 3.5 TENT GdsTransport（新架构）

**文件**: `mooncake-transfer-engine/tent/src/transport/gds/gds_transport.cpp`

```cpp
namespace mooncake {
namespace tent {

class GdsTransport : public Transport {
    // 提交传输任务
    Status GdsTransport::submitTransferTasks(
        SubBatchRef batch, const std::vector<Request>& request_list) {

        auto gds_batch = dynamic_cast<GdsSubBatch*>(batch);

        for (auto& request : request_list) {
            // 1. 获取文件上下文
            GdsFileContext* context = findFileContext(request.target_id);

            // 2. 构建 I/O 参数（支持大文件分片）
            for (size_t offset = 0; offset < request.length; offset += kMaxSliceSize) {
                size_t length = std::min(kMaxSliceSize, request.length - offset);

                CUfileIOParams_t params;
                params.mode = CUFILE_BATCH;
                params.opcode = (request.opcode == Request::READ)
                              ? CUFILE_READ : CUFILE_WRITE;
                params.u.batch.devPtr_base = request.source;    // GPU 内存地址
                params.u.batch.devPtr_offset = offset;           // GPU 偏移
                params.u.batch.file_offset = request.target_offset + offset;
                params.u.batch.size = length;
                params.fh = context->getHandle();

                gds_batch->io_params.push_back(params);
            }
        }

        // 3. 批量提交
        auto result = cuFileBatchIOSubmit(gds_batch->batch_handle->handle,
                                          num_params,
                                          &gds_batch->io_params[first_param_index], 0);
        return Status::OK();
    }

    // 注册 GPU 内存缓冲区
    Status GdsTransport::addMemoryBuffer(BufferDesc& desc, const MemoryOptions& options) {
        if (location.type() != "cuda") return Status::OK();

        // 关键：将 GPU 内存注册到 cuFile
        auto result = cuFileBufRegister((void*)desc.addr, desc.length, 0);
        if (result.err != CU_FILE_SUCCESS)
            return Status::InternalError("Failed to register GDS buffer");

        desc.transports.push_back(GDS);
        return Status::OK();
    }
};

}  // namespace tent
}  // namespace mooncake
```

---

## 4. PyTorch 层：GPU 内存地址获取与传递

### 4.1 PyTorch Tensor 内存模型

vLLM 的 KV Cache 在 PyTorch 层以 `torch.Tensor` 形式存在，存储在 GPU HBM 中：

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           PyTorch GPU 内存模型                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Python 层:  torch.Tensor                                                    │
│              ├── .data_ptr()  →  GPU 内存地址 (uint64)                       │
│              ├── .nbytes      →  字节大小                                    │
│              ├── .device      →  torch.device('cuda:0')                     │
│              └── .storage     →  底层 Storage 对象                           │
│                                                                              │
│  PyTorch C++ 层 (libtorch):                                                  │
│              at::Tensor                                                      │
│              └── storage().data()  →  void* (GPU 指针)                       │
│                                                                              │
│  CUDA 驱动层:                                                                │
│              cudaMalloc 分配的 GPU 内存                                       │
│              地址空间: 0x7f0000000000+ (GPU 虚拟地址)                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 torch.Tensor.data_ptr() 获取 GPU 地址

**文件**: `mooncake/mooncake_connector_v1.py`

```python
class MooncakeConnectorWorker:
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        将 vLLM 的 KV Cache 注册到 Mooncake Transfer Engine

        kv_caches 格式:
        {
            "layer.0.key_cache":    torch.Tensor(shape=[num_blocks, block_size, num_heads, head_dim], device='cuda'),
            "layer.0.value_cache":  torch.Tensor(shape=[num_blocks, block_size, num_heads, head_dim], device='cuda'),
            "layer.1.key_cache":    torch.Tensor(...),
            ...
        }
        """
        kv_data_ptrs = []  # GPU 内存地址列表
        kv_data_lens = []  # 字节大小列表

        for layer_name, cache in kv_caches.items():
            # 关键：data_ptr() 返回 GPU 内存的起始地址
            # 类型：int (Python 整数，实际是 uint64)
            base_addr = cache.data_ptr()

            # 关键：nbytes 返回 tensor 的总字节数
            # = num_elements * element_size
            tensor_size = cache.nbytes

            kv_data_ptrs.append(base_addr)
            kv_data_lens.append(tensor_size)

        # 批量注册到 Mooncake（地址是 Python int，会被 pybind11 转换）
        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed")
```

### 4.3 PyTorch 内存地址的本质

```python
import torch

# vLLM 中 KV Cache 的创建方式
kv_cache = torch.zeros(
    num_blocks,
    block_size,
    num_heads,
    head_dim,
    dtype=torch.float16,
    device='cuda:0'
)

# 获取 GPU 内存信息
addr = kv_cache.data_ptr()   # 例如: 0x7f1234567890 (GPU 虚拟地址)
size = kv_cache.nbytes       # 例如: 1073741824 (1GB)

print(f"GPU Memory Address: {hex(addr)}")  # 0x7f1234567890
print(f"Memory Size: {size / 1024**3:.2f} GB")  # 1.00 GB

# 这个地址是 CUDA 驱动分配的 GPU 虚拟地址
# 可以直接传给 cuFileBufRegister 进行 GDS 注册
```

### 4.4 pybind11 层：Python int → C++ void* 转换

**文件**: `mooncake-transfer-engine/tent/src/python/pybind.cpp`

```cpp
// =============================================================================
// 关键函数：将 Python uint64 转换为 C++ void* 指针
// =============================================================================

// 单个地址转换
static inline void* U64ToPtr(uint64_t a) {
    // reinterpret_cast 将整数转换为指针
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(a));
}

// 批量地址转换
static inline std::vector<void*> U64VectorToPtrVector(
    const std::vector<uint64_t>& addr_list) {
    std::vector<void*> ptrs;
    ptrs.reserve(addr_list.size());
    for (auto a : addr_list) {
        ptrs.push_back(U64ToPtr(a));  // 逐个转换
    }
    return ptrs;
}

// =============================================================================
// Python 绑定：register_local_memory_batch
// =============================================================================

py::class_<TransferEngine>(m, "TransferEngine")
    // ...

    .def(
        "register_local_memory_batch",  // Python 调用的方法名
        [](TransferEngine& self,
           const std::vector<uint64_t>& addr_list,  // Python List[int] → C++ vector<uint64_t>
           const std::vector<size_t>& size_list,    // Python List[int] → C++ vector<size_t>
           Permission permission) {

            py::gil_scoped_release release;  // 释放 GIL，允许并行

            // 关键步骤：将 uint64 地址转换为 void* 指针
            auto ptrs = U64VectorToPtrVector(addr_list);

            // 调用 C++ 层的批量注册方法
            auto s = self.registerLocalMemory(ptrs, size_list, permission);

            ThrowStatus(s, "register_local_memory_batch");
        },
        py::arg("addr_list"),
        py::arg("size_list"),
        py::arg("permission") = Permission::kGlobalReadWrite)
```

### 4.5 完整的地址传递链

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    PyTorch → pybind11 → C++ 地址传递链                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [Python - vLLM/Mooncake]                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ kv_cache: torch.Tensor         # GPU 上的 KV Cache                   │   │
│  │     │                                                                │   │
│  │     ├── .data_ptr() → 0x7f1234567890  # Python int (uint64)         │   │
│  │     └── .nbytes    → 1073741824       # Python int (size_t)         │   │
│  │                                                                      │   │
│  │ kv_data_ptrs = [0x7f1234567890, 0x7f2345678901, ...]                │   │
│  │ kv_data_lens = [1073741824, 1073741824, ...]                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                         │                                                   │
│                         │ engine.register_local_memory_batch(addr_list,     │
│                         │                              size_list)           │
│                         ▼                                                   │
│  [pybind11 - C++ 绑定层]                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 输入: const std::vector<uint64_t>& addr_list                        │   │
│  │       {0x7f1234567890, 0x7f2345678901, ...}                         │   │
│  │                                                                      │   │
│  │ 转换: U64VectorToPtrVector(addr_list)                               │   │
│  │       │                                                              │   │
│  │       ├── for (auto a : addr_list)                                  │   │
│  │       │       ptrs.push_back(reinterpret_cast<void*>(a))            │   │
│  │       │                                                              │   │
│  │       └── 输出: std::vector<void*>                                  │   │
│  │              {(void*)0x7f1234567890, (void*)0x7f2345678901, ...}    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                         │                                                   │
│                         │ self.registerLocalMemory(ptrs, size_list, perm)  │
│                         ▼                                                   │
│  [C++ - TransferEngine]                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ TransferEngine::registerLocalMemory(                                │   │
│  │     const std::vector<void*>& ptrs,                                 │   │
│  │     const std::vector<size_t>& sizes,                               │   │
│  │     Permission perm)                                                │   │
│  │                                                                      │   │
│  │ 对于每个 GPU 内存区域:                                               │   │
│  │     └── GdsTransport::addMemoryBuffer(desc, options)                │   │
│  │              └── cuFileBufRegister(desc.addr, desc.length, 0)       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4.6 数据类型对应关系

| Python 层 | pybind11 层 | C++ 层 | 说明 |
|-----------|-------------|--------|------|
| `int` (0x7f...) | `uint64_t` | `void*` | GPU 内存地址 |
| `torch.Tensor` | - | - | Python tensor 对象 |
| `tensor.data_ptr()` | - | `uint64_t` | 获取地址的方法 |
| `List[int]` | `std::vector<uint64_t>` | `std::vector<void*>` | 地址列表 |
| `List[int]` | `std::vector<size_t>` | `std::vector<size_t>` | 大小列表 |

---

## 5. vLLM + Mooncake 集成

### 5.1 MooncakeConnector 架构

**文件**: `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`

```python
class MooncakeConnector(KVConnectorBase_V1):
    """vLLM 与 Mooncake Transfer Engine 的集成点"""

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MooncakeConnectorScheduler(vllm_config)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = MooncakeConnectorWorker(vllm_config)
```

### 5.2 Worker 端核心实现

**文件**: `mooncake/mooncake_connector_v1.py`

```python
class MooncakeConnectorWorker:
    """Worker 端：实际执行 KV Cache 传输"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        # 1. 初始化 Mooncake Transfer Engine
        from mooncake.engine import TransferEngine
        self.engine = TransferEngine()

        # 2. 初始化传输引擎（支持 RDMA/NVMeoF/GDS）
        ret_value = self.engine.initialize(
            self.hostname,           # 本地主机名
            "P2PHANDSHAKE",          # 握手协议
            VLLM_MOONCAKE_PROTOCOL,  # "rdma" 或其他
            ""
        )

        # 3. 获取 RPC 端口
        self.rpc_port = self.engine.get_rpc_port()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """注册 GPU KV Cache 到 Mooncake"""

        kv_data_ptrs = []
        kv_data_lens = []

        for layer_name, cache in kv_caches.items():
            base_addr = cache.data_ptr()  # GPU 内存地址
            tensor_size = cache.nbytes    # 字节大小

            kv_data_ptrs.append(base_addr)
            kv_data_lens.append(tensor_size)

        # 批量注册内存（内部调用 cuFileBufRegister）
        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed")

    async def send_kv_to_decode(self, meta: MooncakeAgentMetadata):
        """发送 KV Cache 到 Decode 节点"""

        # 1. 构建传输参数
        src_ptrs, dst_ptrs, lengths = await self._build_transfer_params(send_reqs, meta)

        # 2. 执行同步批量写入
        remote_session = f"{meta.remote_hostname}:{meta.remote_port}"
        ret_value = await self.sender_loop.run_in_executor(
            self._sender_executor,
            self._send_blocks,
            remote_session,
            src_ptrs,   # GPU 源地址
            dst_ptrs,   # 远程目标地址
            lengths,    # 传输长度
        )

    def _send_blocks(self, remote_session: str,
                     src_ptrs: list[int], dst_ptrs: list[int], lengths: list[int]) -> int:
        """底层批量传输"""
        return self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )
```

### 5.3 远程 Prefill 流程

```python
# 在 Decode 节点：请求远程 Prefill
async def receive_kv(self, path: str, req_blocks: list[tuple[str, list[int]]]):
    """从 Prefill 节点接收 KV Cache"""

    req_ids, block_ids = map(list, zip(*req_blocks))

    # 1. 构建元数据
    metadata = MooncakeAgentMetadata(
        remote_hostname=self.hostname,
        remote_port=self.rpc_port,
        request_ids=req_ids,
        kv_caches_base_addr=self.kv_caches_base_addr,  # GPU 基地址
        block_ids=block_ids,
    )

    # 2. 发送请求到 Prefill 节点
    sock = make_zmq_socket(self.async_zmq_ctx, path, zmq.REQ)
    await sock.send(self._encoder.encode(metadata))

    # 3. 等待传输完成
    ret_msg = await sock.recv()
    if ret_msg == TRANS_DONE:
        self.finished_recving_reqs.update(req_ids)
```

---

## 6. cuFile API 调用详解

### 6.1 核心 API 列表

| API | 功能 | 调用时机 |
|-----|------|---------|
| `cuFileDriverOpen()` | 初始化 GDS 驱动 | TransferEngine 初始化 |
| `cuFileDriverClose()` | 关闭 GDS 驱动 | TransferEngine 销毁 |
| `cuFileHandleRegister()` | 注册文件句柄 | 打开 SSD 文件时 |
| `cuFileHandleDeregister()` | 注销文件句柄 | 关闭文件时 |
| `cuFileBufRegister()` | 注册 GPU 内存 | 注册 KV Cache 时 |
| `cuFileBufDeregister()` | 注销 GPU 内存 | 注销 KV Cache 时 |
| `cuFileBatchIOSetUp()` | 创建批量 I/O 句柄 | 首次分配批量描述符 |
| `cuFileBatchIODestroy()` | 销毁批量 I/O 句柄 | 释放资源时 |
| `cuFileBatchIOSubmit()` | 提交批量 I/O 请求 | 执行传输时 |
| `cuFileBatchIOGetStatus()` | 获取 I/O 完成状态 | 检查传输完成 |

### 6.2 GPU 内存注册流程

```cpp
// 注册 GPU 内存到 cuFile（允许 DMA 直接访问）
CUfileError_t cuFileBufRegister(void *devPtr_base, size_t length, int flags);

// 在 Mooncake 中的调用
Status GdsTransport::addMemoryBuffer(BufferDesc& desc, const MemoryOptions& options) {
    // GPU 内存地址（由 PyTorch tensor.data_ptr() 获取）
    void* gpu_addr = (void*)desc.addr;
    size_t length = desc.length;

    // 注册到 cuFile
    auto result = cuFileBufRegister(gpu_addr, length, 0);
    if (result.err != CU_FILE_SUCCESS) {
        return Status::InternalError("Failed to register GDS buffer");
    }

    desc.transports.push_back(GDS);
    return Status::OK();
}
```

### 6.3 批量 I/O 提交流程

```cpp
// 1. 创建批量 I/O 句柄
CUfileBatchHandle_t batch_handle;
CUfileError_t result = cuFileBatchIOSetUp(&batch_handle, max_nr_requests);

// 2. 构建 I/O 参数数组
std::vector<CUfileIOParams_t> io_params;
for (auto& request : request_list) {
    CUfileIOParams_t params;
    params.mode = CUFILE_BATCH;
    params.opcode = CUFILE_WRITE;  // 或 CUFILE_READ
    params.fh = file_handle;
    params.u.batch.devPtr_base = gpu_memory_addr;  // GPU 地址
    params.u.batch.devPtr_offset = 0;
    params.u.batch.file_offset = file_offset;
    params.u.batch.size = transfer_size;
    io_params.push_back(params);
}

// 3. 提交批量 I/O
result = cuFileBatchIOSubmit(batch_handle, io_params.size(), io_params.data(), 0);

// 4. 轮询完成状态
std::vector<CUfileIOEvents_t> io_events(io_params.size());
unsigned nr_completed;
result = cuFileBatchIOGetStatus(batch_handle, 0, &nr_completed, io_events.data(), nullptr);

for (auto& event : io_events) {
    if (event.status == CUFILE_COMPLETE) {
        // 传输完成，event.ret 包含实际传输字节数
    }
}
```

---

## 7. 完整调用链图（含 PyTorch 层）

### 7.1 KV Cache 写入 SSD (GPU → NVMe) - 完整调用链

```
┌──────────────────────────────────────────────────────────────────────────────┐
│            KV Cache 卸载到 SSD (GPU → NVMe) - 含 PyTorch 层                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  [PyTorch / Python 层] - GPU 内存管理与地址获取                         ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                        ║  │
│  ║  vLLM Worker: kv_caches: dict[str, torch.Tensor]                       ║  │
│  ║  ┌─────────────────────────────────────────────────────────────────┐  ║  │
│  ║  │ kv_caches = {                                                    │  ║  │
│  ║  │   "layer.0.key_cache":   torch.Tensor([N, B, H, D], device='cuda'),║  │
│  ║  │   "layer.0.value_cache": torch.Tensor([N, B, H, D], device='cuda'),║  ║  │
│  ║  │   ...                                                            │  ║  │
│  ║  │ }                                                                │  ║  │
│  ║  │                                                                  │  ║  │
│  ║  │ # PyTorch 内部结构:                                               │  ║  │
│  ║  │ tensor.data_ptr()  ──►  GPU 虚拟地址 (e.g., 0x7f1234567890)      │  ║  │
│  ║  │ tensor.nbytes      ──►  字节大小 (e.g., 1073741824)              │  ║  │
│  ║  │ tensor.device      ──►  torch.device('cuda:0')                   │  ║  │
│  ║  └─────────────────────────────────────────────────────────────────┘  ║  │
│  ║                                                                        ║  │
│  ║  MooncakeConnectorWorker.register_kv_caches():                         ║  │
│  ║  ┌─────────────────────────────────────────────────────────────────┐  ║  │
│  ║  │ kv_data_ptrs = []                                                │  ║  │
│  ║  │ kv_data_lens = []                                                │  ║  │
│  ║  │                                                                  │  ║  │
│  ║  │ for layer_name, cache in kv_caches.items():                      │  ║  │
│  ║  │     base_addr = cache.data_ptr()  # int (uint64)                 │  ║  │
│  ║  │     tensor_size = cache.nbytes    # int (size_t)                 │  ║  │
│  ║  │     kv_data_ptrs.append(base_addr)                               │  ║  │
│  ║  │     kv_data_lens.append(tensor_size)                             │  ║  │
│  ║  │                                                                  │  ║  │
│  ║  │ # 结果示例:                                                       │  ║  │
│  ║  │ kv_data_ptrs = [0x7f1234567890, 0x7f2345678901, ...]             │  ║  │
│  ║  │ kv_data_lens = [1073741824, 1073741824, ...]                     │  ║  │
│  ║  └─────────────────────────────────────────────────────────────────┘  ║  │
│  ║                               │                                        ║  │
│  ║                               │ self.engine.register_local_memory_batch(║  │
│  ║                               │     kv_data_ptrs, kv_data_lens)        ║  │
│  ║                               ▼                                        ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  [pybind11 层] - Python int → C++ void* 转换                            ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                        ║  │
│  ║  文件: mooncake-transfer-engine/tent/src/python/pybind.cpp             ║  │
│  ║  ┌─────────────────────────────────────────────────────────────────┐  ║  │
│  ║  │ // 辅助函数: uint64 → void*                                      │  ║  │
│  ║  │ static inline void* U64ToPtr(uint64_t a) {                       │  ║  │
│  ║  │     return reinterpret_cast<void*>(                              │  ║  │
│  ║  │         static_cast<std::uintptr_t>(a));                         │  ║  │
│  ║  │ }                                                                │  ║  │
│  ║  │                                                                  │  ║  │
│  ║  │ // 批量转换                                                      │  ║  │
│  ║  │ static inline std::vector<void*> U64VectorToPtrVector(           │  ║  │
│  ║  │     const std::vector<uint64_t>& addr_list) {                    │  ║  │
│  ║  │     std::vector<void*> ptrs;                                     │  ║  │
│  ║  │     for (auto a : addr_list)                                     │  ║  │
│  ║  │         ptrs.push_back(U64ToPtr(a));                             │  ║  │
│  ║  │     return ptrs;  // {(void*)0x7f1234567890, ...}                │  ║  │
│  ║  │ }                                                                │  ║  │
│  ║  │                                                                  │  ║  │
│  ║  │ // Python 绑定                                                   │  ║  │
│  ║  │ .def("register_local_memory_batch",                              │  ║  │
│  ║  │     [](TransferEngine& self,                                     │  ║  │
│  ║  │        const std::vector<uint64_t>& addr_list,  // Python List   │  ║  │
│  ║  │        const std::vector<size_t>& size_list) {                   │  ║  │
│  ║  │         auto ptrs = U64VectorToPtrVector(addr_list);             │  ║  │
│  ║  │         return self.registerLocalMemory(ptrs, size_list, perm);  │  ║  │
│  ║  │     })                                                           │  ║  │
│  ║  └─────────────────────────────────────────────────────────────────┘  ║  │
│  ║                               │                                        ║  │
│  ║                               ▼                                        ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  [C++ Transfer Engine 层] - 内存注册与传输调度                           ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                        ║  │
│  ║  TransferEngine::registerLocalMemory(ptrs, sizes, perm):               ║  │
│  ║  ┌─────────────────────────────────────────────────────────────────┐  ║  │
│  ║  │ for (size_t i = 0; i < ptrs.size(); ++i) {                       │  ║  │
│  ║  │     void* gpu_addr = ptrs[i];      // (void*)0x7f1234567890      │  ║  │
│  ║  │     size_t length = sizes[i];      // 1073741824                 │  ║  │
│  ║  │                                                                  │  ║  │
│  ║  │     // 调用 GDS Transport 注册 GPU 内存                           │  ║  │
│  ║  │     GdsTransport::addMemoryBuffer(desc, options) {               │  ║  │
│  ║  │         cuFileBufRegister(gpu_addr, length, 0);                  │  ║  │
│  ║  │     }                                                            │  ║  │
│  ║  │ }                                                                │  ║  │
│  ║  └─────────────────────────────────────────────────────────────────┘  ║  │
│  ║                                                                        ║  │
│  ║  GdsTransport::submitTransferTasks(batch, requests):                   ║  │
│  ║  ┌─────────────────────────────────────────────────────────────────┐  ║  │
│  ║  │ for (auto& request : request_list) {                             │  ║  │
│  ║  │     CUfileIOParams_t params;                                     │  ║  │
│  ║  │     params.mode = CUFILE_BATCH;                                  │  ║  │
│  ║  │     params.opcode = CUFILE_WRITE;                                │  ║  │
│  ║  │     params.u.batch.devPtr_base = request.source;  // GPU 地址    │  ║  │
│  ║  │     params.u.batch.file_offset = request.target_offset;          │  ║  │
│  ║  │     params.u.batch.size = request.length;                        │  ║  │
│  ║  │     params.fh = context->getHandle();  // cuFile 文件句柄         │  ║  │
│  ║  │     io_params.push_back(params);                                 │  ║  │
│  ║  │ }                                                                │  ║  │
│  ║  │                                                                  │  ║  │
│  ║  │ cuFileBatchIOSubmit(batch_handle, num_params, io_params, 0);     │  ║  │
│  ║  └─────────────────────────────────────────────────────────────────┘  ║  │
│  ║                               │                                        ║  │
│  ║                               ▼                                        ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  [NVIDIA cuFile / GDS 层] - DMA 直接传输                                 ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                        ║  │
│  ║  cuFileBatchIOSubmit() 内部执行:                                        ║  │
│  ║  ┌─────────────────────────────────────────────────────────────────┐  ║  │
│  ║  │ 1. 验证 GPU 内存已注册 (cuFileBufRegister)                        │  ║  │
│  ║  │ 2. 验证文件句柄有效 (cuFileHandleRegister)                        │  ║  │
│  ║  │ 3. GDS Kernel Module 设置 DMA 描述符                              │  ║  │
│  ║  │ 4. DMA Engine: GPU HBM ──────直接──────► NVMe SSD                │  ║  │
│  ║  │                   0x7f1234567890           /data/kv_cache.bin    │  ║  │
│  ║  │                   (完全绕过 CPU 内存，零拷贝)                      │  ║  │
│  ║  └─────────────────────────────────────────────────────────────────┘  ║  │
│  ║                                                                        ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║  [硬件层] - PCIe 4.0/5.0 + NVMe                                         ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                        ║  │
│  ║  ┌──────────────────┐      PCIe 4.0/5.0       ┌──────────────────┐    ║  │
│  ║  │    GPU HBM       │◄═══════════════════════►│   NVMe SSD       │    ║  │
│  ║  │   (~80 GB)       │   DMA ~12-24 GB/s       │   (TB 级)        │    ║  │
│  ║  │                  │   延迟 ~5-10 μs         │                  │    ║  │
│  ║  └──────────────────┘                         └──────────────────┘    ║  │
│  ║                                                                        ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 数据类型流转详解

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          数据类型流转 (从 PyTorch 到 cuFile)                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 1. PyTorch 层                                                        │    │
│  │    torch.Tensor (GPU)                                                │    │
│  │         │                                                            │    │
│  │         ├── .data_ptr() → Python int (例如: 139734566219408)        │    │
│  │         │                   (实际上是 0x7F1234567890)                │    │
│  │         │                                                            │    │
│  │         └── .nbytes → Python int (例如: 1073741824)                  │    │
│  │                            (1GB)                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 2. pybind11 层                                                       │    │
│  │    Python List[int] → C++ std::vector<uint64_t>                      │    │
│  │                                                                      │    │
│  │    [0x7f1234567890, 0x7f2345678901, ...]                            │    │
│  │              │                                                       │    │
│  │              ▼ U64VectorToPtrVector()                                │    │
│  │    std::vector<void*> {(void*)0x7f1234567890, ...}                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 3. C++ Transfer Engine 层                                            │    │
│  │    void* gpu_addr → CUfileIOParams_t.devPtr_base                     │    │
│  │                                                                      │    │
│  │    CUfileIOParams_t params;                                          │    │
│  │    params.u.batch.devPtr_base = (void*)0x7f1234567890;              │    │
│  │    params.u.batch.size = 1073741824;                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 4. cuFile API 层                                                     │    │
│  │    cuFileBufRegister(void* devPtr_base, size_t length, int flags)    │    │
│  │    cuFileBatchIOSubmit(CUfileBatchHandle_t, ...)                     │    │
│  │                                                                      │    │
│  │    GPU 虚拟地址被 pin 到物理页面，允许 DMA 直接访问                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 5. 硬件层                                                            │    │
│  │    GPU 物理地址 (通过 IOMMU 映射) ←→ NVMe SSD 物理 LBA               │    │
│  │                                                                      │    │
│  │    DMA Engine 直接读写:                                              │    │
│  │    - 源: GPU HBM 物理页面                                            │    │
│  │    - 目标: NVMe SSD 物理扇区                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 KV Cache 从 SSD 加载 (NVMe → GPU)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    KV Cache 从 SSD 加载 (NVMe → GPU)                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [vLLM Decode Node]                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. MooncakeConnectorWorker.receive_kv()                              │   │
│  │    └──► 构建 MooncakeAgentMetadata                                   │   │
│  │         {                                                             │   │
│  │           request_ids,                                                │   │
│  │           kv_caches_base_addr,  // GPU 基地址                        │   │
│  │           block_ids,                                                  │   │
│  │         }                                                             │   │
│  │    └──► ZMQ send to Prefill Node                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  [vLLM Prefill Node]                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2. MooncakeConnectorWorker._mooncake_sender_listener()               │   │
│  │    └──► 接收 ZMQ 请求                                                │   │
│  │    └──► dispatch to _sender_worker()                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 3. send_kv_to_decode(metadata)                                       │   │
│  │    └──► await self._build_transfer_params(send_reqs, metadata)       │   │
│  │         └──► src_ptrs = local_base_addr + block_id * block_len      │   │
│  │         └──► dst_ptrs = remote_base_addr + block_id * block_len     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  [Mooncake Transfer Engine]                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 4. TransferEngine::batch_transfer_sync_write()                       │   │
│  │    └──► GdsTransport::submitTransferTasks()                          │   │
│  │         params.opcode = CUFILE_READ  // 从 SSD 读取                  │   │
│  │         └──► cuFileBatchIOSubmit()                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  [NVIDIA cuFile Driver]                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 5. DMA Transfer: NVMe SSD → GPU HBM                                  │   │
│  │    └──► 完成后返回 TRANS_DONE                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  [vLLM Decode Node]                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 6. finished_recving_reqs.update(req_ids)                             │   │
│  │    └──► KV Cache 已加载到 GPU，可以开始推理                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. 三种传输模式对比

### 8.1 模式对比表

| 传输模式 | 协议 | 带宽 | 延迟 | 适用场景 |
|---------|------|------|------|---------|
| **RDMA Transport** | IB/RoCE | 100-400 Gbps | ~1-2 μs | 跨节点 P/D 分离 |
| **NVMeoF Transport** | NVMe-over-Fabric | ~25 GB/s | ~10-20 μs | 远程 NVMe 存储 |
| **GDS Transport** | 本地 NVMe | ~12-24 GB/s | ~5-10 μs | 本地 SSD 卸载 |

### 8.2 数据路径对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         三种传输模式数据路径                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [1] RDMA Transport (跨节点)                                                │
│  GPU HBM ──IB/RDMA──► Network ──IB/RDMA──► GPU HBM                         │
│           ~100 Gbps            ~100 Gbps                                    │
│  特点：低延迟，高带宽，适合跨节点 KV 传输                                     │
│                                                                             │
│  [2] NVMeoF Transport (远程存储)                                            │
│  GPU HBM ──PCIe──► Network ──TCP──► Remote NVMe SSD                        │
│           ~32 GB/s    ~25 Gbps                                              │
│  特点：访问远程 NVMe 存储，支持分布式存储集群                                  │
│                                                                             │
│  [3] GDS Transport (本地存储)                                               │
│  GPU HBM ══════════════════════════════► Local NVMe SSD                    │
│           Direct DMA (~12-24 GB/s)                                          │
│  特点：零拷贝，绕过 CPU，最高效的本地存储访问                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.3 性能参考

| 场景 | 传统方式 (GPU→CPU→SSD) | GDS 方式 (GPU→SSD) | 提升 |
|------|----------------------|-------------------|------|
| 1GB KV Cache 写入 | ~50 ms | ~40 ms | 1.25x |
| 10GB KV Cache 写入 | ~500 ms | ~400 ms | 1.25x |
| CPU 负载 | 高（数据拷贝） | 低（仅元数据） | ~90% 降低 |
| 内存占用 | 需要额外 CPU 缓冲 | 无额外占用 | ~50% 降低 |

---

## 9. 性能优化关键点

### 9.1 批量 I/O 句柄复用

```cpp
// cuFileBatchIOSetUp 是耗时操作（~100ms）
// Mooncake 使用句柄池来复用已创建的句柄

class CUFileDescPool {
    std::vector<BatchHandle*> handle_pool_;  // 句柄池

    int allocCUfileDesc(size_t batch_size) {
        BatchHandle* batch_handle = nullptr;

        // 优先从池中获取
        if (!handle_pool_.empty()) {
            batch_handle = handle_pool_.back();
            handle_pool_.pop_back();
        }

        // 池空才创建新句柄
        if (!batch_handle) {
            batch_handle = new BatchHandle();
            cuFileBatchIOSetUp(&batch_handle->handle, max_batch_size_);
        }

        return batch_handle;
    }

    int freeCUfileDesc(int idx) {
        // 返回池中复用，而不是销毁
        handle_pool_.push_back(desc->batch_handle);
        return 0;
    }
};
```

### 9.2 GPU 内存预注册

```python
# 在 Worker 初始化时一次性注册所有 KV Cache
def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
    kv_data_ptrs = []
    kv_data_lens = []

    for layer_name, cache in kv_caches.items():
        kv_data_ptrs.append(cache.data_ptr())
        kv_data_lens.append(cache.nbytes)

    # 批量注册（比逐个注册更高效）
    self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
```

### 9.3 异步传输与计算重叠

```python
# 在后台线程中执行传输
async def _sender_worker(self, sock: zmq.asyncio.Socket):
    while True:
        identity, metadata_bytes = await self.sender_worker_queue.get()
        try:
            metadata = self._decoder.decode(metadata_bytes)
            # 在线程池中执行阻塞的传输操作
            await self.send_kv_to_decode(metadata)
            await sock.send_multipart((identity, b"", TRANS_DONE))
        except Exception as e:
            await sock.send_multipart((identity, b"", TRANS_ERROR))
```

### 9.4 关键配置参数

| 参数 | 默认值 | 说明 | 调优建议 |
|------|-------|------|---------|
| `io_batch_depth` | 32 | 批量 I/O 深度 | 增大可提高吞吐 |
| `kMaxSliceSize` | 16 MB | 单次传输最大分片 | 根据 SSD 性能调整 |
| `VLLM_MOONCAKE_SENDER_WORKERS` | 10 | 发送线程数 | 根据并发量调整 |
| `VLLM_MOONCAKE_PROTOCOL` | "rdma" | 传输协议 | "rdma" 或 "tcp" |

---

## 10. 关键文件索引

### 10.1 Mooncake Transfer Engine

| 功能 | 文件路径 | 关键类/函数 |
|------|----------|-------------|
| NVMeoF 传输 | `mooncake-transfer-engine/src/transport/nvmeof_transport/nvmeof_transport.cpp` | `NVMeoFTransport` |
| cuFile 上下文 | `mooncake-transfer-engine/include/transport/nvmeof_transport/cufile_context.h` | `CuFileContext` |
| 批量描述符池 | `mooncake-transfer-engine/src/transport/nvmeof_transport/cufile_desc_pool.cpp` | `CUFileDescPool` |
| TENT GDS 传输 | `mooncake-transfer-engine/tent/src/transport/gds/gds_transport.cpp` | `GdsTransport` |
| Python 绑定 | `mooncake-transfer-engine/tent/src/python/pybind.cpp` | PYBIND11 模块 |

### 10.2 vLLM 集成

| 功能 | 文件路径 | 关键类/函数 |
|------|----------|-------------|
| Mooncake Connector | `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py` | `MooncakeConnector` |
| OOT Connector | `mooncake/mooncake_connector_v1.py` | `MooncakeConnectorWorker` |

### 10.3 cuFile API

| API | 头文件 | 功能 |
|-----|-------|------|
| `cuFileDriverOpen/Close` | `<cufile.h>` | 驱动初始化/关闭 |
| `cuFileHandleRegister/Deregister` | `<cufile.h>` | 文件句柄注册 |
| `cuFileBufRegister/Deregister` | `<cufile.h>` | GPU 内存注册 |
| `cuFileBatchIOSetUp/Destroy` | `<cufile.h>` | 批量 I/O 句柄管理 |
| `cuFileBatchIOSubmit` | `<cufile.h>` | 提交批量 I/O |
| `cuFileBatchIOGetStatus` | `<cufile.h>` | 获取 I/O 状态 |

---

## 参考文献

- [NVIDIA GPUDirect Storage Documentation](https://docs.nvidia.com/gpudirect-storage/)
- [cuFile API Reference](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/)
- [Mooncake GitHub Repository](https://github.com/kvcache-ai/Mooncake)
- [vLLM Documentation](https://docs.vllm.ai/)
- [NVMe Specification](https://nvmexpress.org/specifications/)
