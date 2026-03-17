# KV Cache GDS 调用链详解

本文档详细分析 vLLM + Mooncake 架构下，KV Cache 与 NVMe SSD 之间通过 GDS 进行数据传输的两条核心调用链：

1. **KV Cache 写入 SSD** (Offload/Write Path)
2. **SSD 加载到 KV Cache** (Load/Read Path)

---

## 调用链概览

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           KV Cache 写入 SSD (Write Path)                         │
│  torch.Tensor → data_ptr() → batch_transfer_sync_write() → cuFileBatchIOSubmit  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SSD 加载到 KV Cache (Read Path)                         │
│  torch.Tensor → data_ptr() → transfer_sync_read() → cuFileBatchIOSubmit         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. KV Cache 写入 SSD (Write Path)

### 1.1 完整调用链图

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 1: vLLM Scheduler                                                                      │
│                                                                                              │
│  MooncakeConnectorScheduler.request_finished()                                               │
│  ├── 判断请求是否需要远程传输 (do_remote_decode)                                              │
│  ├── 收集 block_ids: list[int]                                                               │
│  └── 返回: (delay_free_blocks, kv_transfer_params)                                           │
│       kv_transfer_params = {                                                                 │
│           "remote_host": "192.168.1.100",                                                    │
│           "remote_port": 6557                                                                │
│       }                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ IPC: SchedulerOutput
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 2: vLLM Worker (MooncakeConnectorWorker)                                               │
│                                                                                              │
│  文件: mooncake_connector_v1.py:543-571                                                      │
│                                                                                              │
│  async def send_kv_to_decode(self, meta: MooncakeAgentMetadata):                            │
│      │                                                                                       │
│      ├── 1. 获取发送请求列表: send_reqs = [(req_id, SendBlockMeta), ...]                     │
│      │                                                                                       │
│      ├── 2. 构建传输参数 ────────────────────────────────────────────────────────┐          │
│      │   src_ptrs, dst_ptrs, lengths = await self._build_transfer_params()    │          │
│      │   │                                                                      │          │
│      │   │  关键数据处理:                                                        │          │
│      │   │  ┌─────────────────────────────────────────────────────────────────┐│          │
│      │   │  │ local_base_addr = self.kv_caches_base_addr  # GPU 地址列表      ││          │
│      │   │  │ remote_base_addr = agent_meta.kv_caches_base_addr               ││          │
│      │   │  │ block_len = self.block_len  # 每个 block 的字节数                ││          │
│      │   │  │                                                                 ││          │
│      │   │  │ for local_block_id, remote_block_id in zip(...):                ││          │
│      │   │  │     src_ptr = local_layer_addr + block_id * block_len            ││          │
│      │   │  │     dst_ptr = remote_layer_addr + block_id * block_len           ││          │
│      │   │  │     length = block_len * num_blocks                              ││          │
│      │   │  └─────────────────────────────────────────────────────────────────┘│          │
│      │   └─────────────────────────────────────────────────────────────────────┘          │
│      │                                                                                       │
│      └── 3. 调用同步写入:                                                                     │
│          ret_value = self.engine.batch_transfer_sync_write(                                 │
│              remote_session,  # "192.168.1.100:6557"                                        │
│              src_ptrs,        # [0x7f1234560000, 0x7f1234570000, ...]                       │
│              dst_ptrs,        # [0x7f8765430000, 0x7f8765440000, ...]                       │
│              lengths          # [16777216, 16777216, ...]  (16MB each)                      │
│          )                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ Python List[int] → C++ std::vector<uintptr_t>
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 3: pybind11 绑定层                                                                      │
│                                                                                              │
│  文件: mooncake-integration/transfer_engine/transfer_engine_py.cpp:1014-1015                │
│                                                                                              │
│  .def("batch_transfer_sync_write", &TransferEnginePy::batchTransferSyncWrite)               │
│                                                                                              │
│  关键类型转换:                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ Python 输入:                     C++ 接收:                                               ││
│  │ src_ptrs: List[int]         →    std::vector<uintptr_t> buffers                         ││
│  │ dst_ptrs: List[int]         →    std::vector<uintptr_t> peer_buffer_addresses           ││
│  │ lengths: List[int]          →    std::vector<size_t> lengths                            ││
│  │                                                                                          ││
│  │ 示例:                                                                                     ││
│  │ [0x7f1234560000]         →    (uintptr_t)0x7f1234560000 = (void*)0x7f1234560000         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 4: TransferEnginePy::batchTransferSyncWrite()                                          │
│                                                                                              │
│  文件: mooncake-integration/transfer_engine/transfer_engine_py.cpp:450-539                  │
│                                                                                              │
│  int batchTransferSyncWrite(                                                                 │
│      const char *target_hostname,                                                            │
│      const std::vector<uintptr_t> &buffers,          // GPU 源地址                           │
│      const std::vector<uintptr_t> &peer_buffer_addresses,  // 目标偏移                       │
│      const std::vector<size_t> &lengths             // 传输长度                              │
│  ) {                                                                                         │
│      │                                                                                       │
│      ├── 1. 获取目标 Segment Handle                                                          │
│      │   handle = engine_->openSegment(target_hostname);                                    │
│      │                                                                                       │
│      ├── 2. 构建 TransferRequest 列表 ───────────────────────────────────────────┐          │
│      │   for (size_t i = 0; i < batch_size; ++i) {                               │          │
│      │       TransferRequest entry;                                              │          │
│      │       entry.opcode = TransferRequest::WRITE;  ★ 写入操作                   │          │
│      │       entry.length = lengths[i];                                          │          │
│      │       entry.source = (void *)buffers[i];      ★ GPU 地址                  │          │
│      │       entry.target_id = handle;               ★ 目标 Segment              │          │
│      │       entry.target_offset = peer_buffer_addresses[i]; ★ 文件偏移          │          │
│      │       entries.push_back(entry);                                           │          │
│      │   }                                                                       │          │
│      │   └───────────────────────────────────────────────────────────────────────┘          │
│      │                                                                                       │
│      ├── 3. 分配 Batch ID                                                                    │
│      │   batch_id = engine_->allocateBatchID(batch_size);                                   │
│      │                                                                                       │
│      ├── 4. 提交传输请求 ★ 核心调用                                                           │
│      │   Status s = engine_->submitTransfer(batch_id, entries);                             │
│      │                                                                                       │
│      └── 5. 等待完成                                                                         │
│          while (!completed) {                                                                │
│              engine_->getBatchTransferStatus(batch_id, status);                             │
│              if (status.s == TransferStatusEnum::COMPLETED) completed = true;               │
│          }                                                                                   │
│  }                                                                                           │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ submitTransfer(batch_id, entries)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 5: TransferEngine::submitTransfer()                                                    │
│                                                                                              │
│  文件: mooncake-transfer-engine/src/transfer_engine.cpp                                      │
│                                                                                              │
│  Status submitTransfer(BatchID batch_id, const std::vector<TransferRequest> &entries) {     │
│      │                                                                                       │
│      ├── 1. 根据 target_id 选择 Transport                                                    │
│      │   Transport* transport = getTransportForSegment(entries[i].target_id);              │
│      │                                                                                       │
│      └── 2. 调用 Transport 的 submitTransfer ★                                               │
│          return transport->submitTransfer(batch_id, entries);                               │
│  }                                                                                           │
│                                                                                              │
│  当使用 NVMeoF/GDS 传输时，transport = NVMeoFTransport                                       │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 6: NVMeoFTransport::submitTransfer()                                                   │
│                                                                                              │
│  文件: mooncake-transfer-engine/src/transport/nvmeof_transport/nvmeof_transport.cpp:115-206│
│                                                                                              │
│  Status submitTransfer(BatchID batch_id, const std::vector<TransferRequest> &entries) {     │
│      │                                                                                       │
│      ├── 1. 遍历每个 TransferRequest                                                         │
│      │   for (auto &request : entries) { ... }                                              │
│      │                                                                                       │
│      ├── 2. 获取目标 Segment 描述                                                            │
│      │   segment_desc = metadata_->getSegmentDescByID(target_id);                           │
│      │   assert(desc->protocol == "nvmeof");                                                │
│      │                                                                                       │
│      ├── 3. 计算文件偏移和切片 ─────────────────────────────────────────────────┐           │
│      │   for (auto &buffer_desc : desc->nvmeof_buffers) {                      │           │
│      │       uint64_t slice_start = std::max(segment_start, current_offset);   │           │
│      │       uint64_t slice_end = std::min(segment_end, ...);                  │           │
│      │       uint64_t slice_len = slice_end - slice_start;                     │           │
│      │                                                                         │           │
│      │       void *source_addr = (char *)request.source + slice_start - ...;   │           │
│      │       uint64_t file_offset = slice_start - current_offset;              │           │
│      │   }                                                                     │           │
│      │   └─────────────────────────────────────────────────────────────────────┘           │
│      │                                                                                       │
│      ├── 4. 获取 cuFile Handle ──────────────────────────────────────────────────┐         │
│      │   auto buf_key = std::make_pair(target_id, buffer_id);                   │         │
│      │   if (!segment_to_context_.count(buf_key)) {                              │         │
│      │       segment_to_context_[buf_key] =                                      │         │
│      │           std::make_shared<CuFileContext>(file_path);  ★ 打开文件         │         │
│      │   }                                                                      │         │
│      │   CUfileHandle_t fh = segment_to_context_.at(buf_key)->getHandle();      │         │
│      │   └───────────────────────────────────────────────────────────────────────┘         │
│      │                                                                                       │
│      ├── 5. 添加到 cuFile Batch ─────────────────────────────────────────────────┐         │
│      │   addSliceToCUFileBatch(source_addr, file_offset, slice_len,            │         │
│      │                         desc_idx, request.opcode, fh);                   │         │
│      │   └───────────────────────────────────────────────────────────────────────┘         │
│      │                                                                                       │
│      └── 6. 提交 Batch ★                                                                     │
│          desc_pool_->submitBatch(nvmeof_desc.desc_idx_);                                    │
│  }                                                                                           │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ addSliceToCUFileBatch() → desc_pool_->submitBatch()
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 7: NVMeoFTransport::addSliceToCUFileBatch()                                            │
│                                                                                              │
│  文件: nvmeof_transport.cpp:265-281                                                          │
│                                                                                              │
│  void addSliceToCUFileBatch(void *source_addr, uint64_t file_offset,                        │
│                             uint64_t slice_len, uint64_t desc_id,                            │
│                             TransferRequest::OpCode op, CUfileHandle_t fh) {                │
│      │                                                                                       │
│      CUfileIOParams_t params;                                                                │
│      params.mode = CUFILE_BATCH;                                                             │
│      params.opcode = (op == READ) ? CUFILE_READ : CUFILE_WRITE;  ★ 设置写入模式             │
│      params.cookie = (void *)0;                                                              │
│      params.u.batch.devPtr_base = source_addr;    ★ GPU 内存地址                            │
│      params.u.batch.devPtr_offset = 0;                                                       │
│      params.u.batch.file_offset = file_offset;    ★ 文件偏移                                │
│      params.u.batch.size = slice_len;             ★ 传输大小                                │
│      params.fh = fh;                              ★ cuFile 文件句柄                         │
│      │                                                                                       │
│      desc_pool_->pushParams(desc_id, params);                                               │
│  }                                                                                           │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ desc_pool_->submitBatch()
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 8: CUFileDescPool::submitBatch()                                                       │
│                                                                                              │
│  文件: mooncake-transfer-engine/src/transport/nvmeof_transport/cufile_desc_pool.cpp:145-163│
│                                                                                              │
│  int submitBatch(int idx) {                                                                  │
│      auto* desc = descs_[idx];                                                              │
│      │                                                                                       │
│      │   // desc->io_params 是 CUfileIOParams_t 数组                                         │
│      │   // 每个 params 包含: GPU地址、文件偏移、大小、操作类型                               │
│      │                                                                                       │
│      └── ★ cuFile 核心 API 调用 ★                                                           │
│          CUFILE_CHECK(cuFileBatchIOSubmit(                                                  │
│              desc->batch_handle->handle,  // Batch 句柄                                     │
│              desc->io_params.size(),      // 请求数量                                       │
│              desc->io_params.data(),      // CUfileIOParams_t 数组                          │
│              0                             // flags                                          │
│          ));                                                                                 │
│  }                                                                                           │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ cuFileBatchIOSubmit() → GDS Driver
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 9: NVIDIA cuFile / GDS Driver                                                          │
│                                                                                              │
│  cuFileBatchIOSubmit() 内部流程:                                                             │
│                                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ 1. 验证 GPU 内存地址已注册 (cuFileBufRegister)                                           ││
│  │                                                                                          ││
│  │ 2. 锁定 GPU 内存页 (pinning)                                                             ││
│  │                                                                                          ││
│  │ 3. 设置 DMA 描述符                                                                       ││
│  │    ┌────────────────────────────────────────────────────────────────────────┐           ││
│  │    │ DMA Descriptor:                                                         │           ││
│  │    │   src_addr: 0x7f1234560000 (GPU HBM)                                   │           ││
│  │    │   dst_addr: NVMe LBA 0x12345678 (SSD)                                  │           ││
│  │    │   length:   16777216 bytes (16MB)                                      │           ││
│  │    │   direction: GPU → SSD                                                 │           ││
│  │    └────────────────────────────────────────────────────────────────────────┘           ││
│  │                                                                                          ││
│  │ 4. 提交到 NVMe Submission Queue (SQ)                                                      ││
│  │                                                                                          ││
│  │ 5. 触发 PCIe DMA 传输                                                                    ││
│  └─────────────────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ PCIe DMA
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 10: 硬件层                                                                             │
│                                                                                              │
│  ┌──────────────────────┐                         ┌──────────────────────┐                   │
│  │      GPU HBM         │                         │      NVMe SSD        │                   │
│  │  ┌────────────────┐  │   PCIe 4.0/5.0 DMA    │  ┌────────────────┐  │                   │
│  │  │ KV Cache Block │──┼────────────────────────┼─►│   File Data    │  │                   │
│  │  │ 0x7f1234560000 │  │   ~12-24 GB/s         │  │  LBA 0x12345678│  │                   │
│  │  │ 16 MB          │  │                       │  │  16 MB         │  │                   │
│  │  └────────────────┘  │                       │  └────────────────┘  │                   │
│  │                      │   ★ 绕过 CPU 内存 ★   │                      │                   │
│  └──────────────────────┘                       └──────────────────────┘                   │
│                                                                                              │
│  关键: DMA 引擎直接在 GPU HBM 和 NVMe SSD 之间传输，CPU 不参与数据搬运                       │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. SSD 加载到 KV Cache (Read Path)

### 2.1 完整调用链图

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 1: vLLM Scheduler                                                                      │
│                                                                                              │
│  MooncakeConnectorScheduler.get_num_new_matched_tokens()                                     │
│  ├── 检查 request.kv_transfer_params.get("do_remote_prefill")                               │
│  ├── 返回: (num_tokens, True) 表示需要异步加载                                               │
│  │                                                                                           │
│  MooncakeConnectorScheduler.update_state_after_alloc()                                       │
│  ├── 收集 local_block_ids                                                                    │
│  ├── 添加到 self._reqs_need_recv[request_id] = (request, block_ids)                         │
│  └── 通过 build_connector_meta() 传递给 Worker                                               │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ IPC: MooncakeConnectorMetadata
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 2: vLLM Worker (MooncakeConnectorWorker)                                               │
│                                                                                              │
│  文件: mooncake_connector_v1.py:847-858                                                      │
│                                                                                              │
│  def start_load_kv(self, metadata: MooncakeConnectorMetadata):                              │
│      │                                                                                       │
│      ├── 1. 分组拉取请求                                                                     │
│      │   kv_pulls = self.group_kv_pull(metadata)                                            │
│      │   # kv_pulls = {path: [(req_id, block_ids), ...]}                                    │
│      │                                                                                       │
│      └── 2. 异步触发接收                                                                     │
│          for path, req_blocks in kv_pulls.items():                                          │
│              asyncio.run_coroutine_threadsafe(                                              │
│                  self.receive_kv(path, req_blocks),                                         │
│                  self.receiver_loop                                                          │
│              )                                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ receive_kv(path, req_blocks)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 3: MooncakeConnectorWorker.receive_kv()                                                │
│                                                                                              │
│  文件: mooncake_connector_v1.py:775-815                                                      │
│                                                                                              │
│  async def receive_kv(self, path: str, req_blocks: list[tuple[str, list[int]]]):           │
│      │                                                                                       │
│      ├── 1. 构建请求元数据 ─────────────────────────────────────────────────────┐           │
│      │   metadata = MooncakeAgentMetadata(                                      │           │
│      │       remote_hostname=self.hostname,                                      │           │
│      │       remote_port=self.rpc_port,                                          │           │
│      │       request_ids=req_ids,                                                │           │
│      │       kv_caches_base_addr=self.kv_caches_base_addr,  ★ GPU 目标地址       │           │
│      │       block_ids=block_ids                                                 │           │
│      │   )                                                                       │           │
│      │   └───────────────────────────────────────────────────────────────────────┘           │
│      │                                                                                       │
│      ├── 2. 通过 ZMQ 发送请求到 Prefill 节点                                                  │
│      │   sock = make_zmq_socket(ctx, path, zmq.REQ)                                         │
│      │   await sock.send(encoded_data)                                                      │
│      │                                                                                       │
│      ├── 3. 等待 Prefill 节点完成发送                                                        │
│      │   ret_msg = await sock.recv()                                                        │
│      │                                                                                       │
│      └── 4. 标记完成                                                                         │
│          self.finished_recving_reqs.update(req_ids)                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ ZMQ 消息触发 Prefill 节点的 send_kv_to_decode()
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 4: Prefill 节点处理 (send_kv_to_decode)                                                │
│                                                                                              │
│  文件: mooncake_connector_v1.py:543-571                                                      │
│                                                                                              │
│  async def send_kv_to_decode(self, meta: MooncakeAgentMetadata):                            │
│      │                                                                                       │
│      ├── 1. 构建传输参数                                                                     │
│      │   src_ptrs, dst_ptrs, lengths = await self._build_transfer_params(...)              │
│      │                                                                                       │
│      │   ★ 关键: 此时 opcode = READ (从本地 SSD 读取)                                        │
│      │   ★ src_ptrs = 本地文件偏移 (转为地址)                                                │
│      │   ★ dst_ptrs = 远程 GPU 地址                                                          │
│      │                                                                                       │
│      └── 2. 调用传输引擎                                                                     │
│          ret_value = self.engine.batch_transfer_sync_write(                                 │
│              remote_session, src_ptrs, dst_ptrs, lengths                                    │
│          )                                                                                  │
│          # 注: 虽然叫 write，但对于 NVMeoF transport 来说                                    │
│          # 实际是从本地 NVMe 读取，写入远程 GPU                                               │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ 与 Write Path 相同的底层调用链
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 5-9: TransferEngine → NVMeoFTransport → cuFileDescPool → cuFileBatchIOSubmit          │
│                                                                                              │
│  ★ 与 Write Path 的区别仅在于 opcode = CUFILE_READ ★                                        │
│                                                                                              │
│  CUfileIOParams_t params;                                                                    │
│  params.opcode = CUFILE_READ;  ★ 读取操作                                                   │
│  params.u.batch.devPtr_base = dst_gpu_addr;  ★ 目标 GPU 地址                                │
│  params.u.batch.file_offset = src_file_offset;  ★ 源文件偏移                                │
│  params.u.batch.size = slice_len;                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ cuFileBatchIOSubmit() → GDS Driver
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 10: 硬件层 (Read Path)                                                                 │
│                                                                                              │
│  ┌──────────────────────┐                         ┌──────────────────────┐                   │
│  │      NVMe SSD        │                         │      GPU HBM         │                   │
│  │  ┌────────────────┐  │   PCIe 4.0/5.0 DMA    │  ┌────────────────┐  │                   │
│  │  │   File Data    │──┼────────────────────────┼─►│ KV Cache Block │  │                   │
│  │  │  LBA 0x12345678│  │   ~12-24 GB/s         │  │ 0x7f1234560000 │  │                   │
│  │  │  16 MB         │  │                       │  │ 16 MB          │  │                   │
│  │  └────────────────┘  │                       │  └────────────────┘  │                   │
│  │                      │   ★ SSD → GPU ★       │                      │                   │
│  └──────────────────────┘                       └──────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 关键数据处理节点详解

### 3.1 GPU 内存地址获取 (torch.Tensor → void*)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 文件: mooncake_connector_v1.py:650-699                                                       │
│                                                                                              │
│ def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):                          │
│     │                                                                                        │
│     │  输入: kv_caches = {                                                                   │
│     │      "layer.0.k_cache": torch.Tensor(shape=[num_blocks, num_heads, head_dim]),        │
│     │      "layer.0.v_cache": torch.Tensor(shape=[num_blocks, num_heads, head_dim]),        │
│     │      ...                                                                               │
│     │  }                                                                                     │
│     │                                                                                        │
│     ├── 1. 遍历每个层的 KV Cache                                                             │
│     │   for layer_name, cache_or_caches in kv_caches.items():                              │
│     │                                                                                        │
│     ├── 2. 提取 GPU 内存地址 ★ 关键操作                                                      │
│     │   base_addr = cache.data_ptr()      # 例如: 0x7f1234567890                           │
│     │   tensor_size = cache.nbytes        # 例如: 1073741824 (1GB)                         │
│     │                                                                                        │
│     │   ★ data_ptr() 返回的是 CUDA 驱动分配的设备内存虚拟地址                                │
│     │   ★ 这个地址可以直接用于 cuFileBufRegister                                            │
│     │                                                                                        │
│     ├── 3. 收集地址和大小                                                                    │
│     │   kv_data_ptrs.append(base_addr)                                                      │
│     │   kv_data_lens.append(tensor_size)                                                    │
│     │                                                                                        │
│     └── 4. 批量注册到 Mooncake                                                               │
│         ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)          │
│                                                                                              │
│ 数据流:                                                                                       │
│ torch.Tensor → .data_ptr() → int (0x7f...) → pybind11 → void* → cuFileBufRegister          │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Block 地址计算

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 文件: mooncake_connector_v1.py:573-629                                                       │
│                                                                                              │
│ async def _build_transfer_params(...):                                                       │
│     │                                                                                        │
│     │  已知:                                                                                  │
│     │  - local_base_addr: list[int] = [0x7f1000000000, 0x7f2000000000, ...]  # 每层基址     │
│     │  - remote_base_addr: list[int] = [0x7f8000000000, 0x7f9000000000, ...]                │
│     │  - block_len: int = 16 * 1024 * 1024  # 16MB per block                                │
│     │  - block_ids: list[int] = [0, 1, 2, 5, 6, 10]  # 非连续 block                         │
│     │                                                                                        │
│     ├── 1. 分组连续 block                                                                    │
│     │   group_concurrent_contiguous(block_ids, remote_block_ids)                            │
│     │   # 输出: [[0,1,2], [5,6], [10]]                                                       │
│     │                                                                                        │
│     ├── 2. 计算每层的传输参数                                                                │
│     │   for local_layer_addr, remote_layer_addr in zip(local_base_addr, remote_base_addr):  │
│     │       for group_local_block_id, group_remote_block_id in zip(...):                    │
│     │                                                                                        │
│     │           ★ 地址计算公式:                                                               │
│     │           src_ptr = local_layer_addr + group_block_id[0] * block_len                  │
│     │           dst_ptr = remote_layer_addr + group_block_id[0] * block_len                 │
│     │           length = block_len * len(group_block_id)                                     │
│     │                                                                                        │
│     │           示例:                                                                         │
│     │           src_ptr = 0x7f1000000000 + 0 * 16MB = 0x7f1000000000                        │
│     │           dst_ptr = 0x7f8000000000 + 0 * 16MB = 0x7f8000000000                        │
│     │           length = 16MB * 3 = 48MB  (blocks 0,1,2)                                     │
│     │                                                                                        │
│     │           src_ptrs.append(src_ptr)                                                     │
│     │           dst_ptrs.append(dst_ptr)                                                     │
│     │           lengths.append(length)                                                       │
│     │                                                                                        │
│     └── 3. 返回批量传输参数                                                                  │
│         return src_ptrs, dst_ptrs, lengths                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 cuFile 批量 I/O 参数构建

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 文件: nvmeof_transport.cpp:265-281                                                           │
│                                                                                              │
│ void addSliceToCUFileBatch(void *source_addr, uint64_t file_offset,                        │
│                            uint64_t slice_len, TransferRequest::OpCode op, ...) {           │
│     │                                                                                        │
│     CUfileIOParams_t params;                                                                 │
│     │                                                                                        │
│     params.mode = CUFILE_BATCH;                          // 批量模式                        │
│     params.opcode = (op == READ) ? CUFILE_READ            // 操作类型                       │
│                                   : CUFILE_WRITE;                                           │
│     params.cookie = (void *)0;                              // 用户 cookie                   │
│     params.fh = fh;                                         // cuFile 文件句柄               │
│     │                                                                                        │
│     // 批量参数                                                                              │
│     params.u.batch.devPtr_base = source_addr;              // GPU 内存基地址                │
│     params.u.batch.devPtr_offset = 0;                      // GPU 内存偏移                   │
│     params.u.batch.file_offset = file_offset;              // 文件偏移                       │
│     params.u.batch.size = slice_len;                       // 传输大小                       │
│     │                                                                                        │
│     desc_pool_->pushParams(desc_id, params);                                               │
│ }                                                                                            │
│                                                                                              │
│ 示例 params:                                                                                 │
│ ┌────────────────────────────────────────────────────────────────────────────────────────┐  │
│ │ params = {                                                                             │  │
│ │     mode: CUFILE_BATCH,                                                                │  │
│ │     opcode: CUFILE_WRITE,                                                              │  │
│ │     fh: 0x55a1b2c3d4e5,  // cuFileHandle                                               │  │
│ │     u.batch = {                                                                        │  │
│ │         devPtr_base: 0x7f1234560000,  // GPU KV Cache 地址                             │  │
│ │         devPtr_offset: 0,                                                              │  │
│ │         file_offset: 0x100000,  // 1MB into file                                       │  │
│ │         size: 0x1000000        // 16MB                                                 │  │
│ │     }                                                                                  │  │
│ │ }                                                                                      │  │
│ └────────────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 内存注册流程 (prerequisite)

在传输之前，GPU 内存必须先注册到 cuFile：

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ GPU 内存注册流程                                                                              │
│                                                                                              │
│ 1. vLLM 初始化时调用:                                                                         │
│    MooncakeConnectorWorker.register_kv_caches(kv_caches)                                    │
│    │                                                                                         │
│    ├── for cache in kv_caches:                                                              │
│    │     base_addr = cache.data_ptr()  # 0x7f1234567890                                     │
│    │     tensor_size = cache.nbytes    # 1073741824                                         │
│    │     kv_data_ptrs.append(base_addr)                                                     │
│    │     kv_data_lens.append(tensor_size)                                                   │
│    │                                                                                         │
│    └── engine.batch_register_memory(kv_data_ptrs, kv_data_lens)                             │
│                                                                                              │
│ 2. pybind11 类型转换:                                                                         │
│    List[int] → std::vector<uint64_t> → U64VectorToPtrVector() → std::vector<void*>          │
│                                                                                              │
│ 3. TransferEngine::registerLocalMemoryBatch():                                               │
│    for (addr, size) in buffer_list:                                                         │
│        transport->registerLocalMemory(addr, size, location)                                 │
│                                                                                              │
│ 4. NVMeoFTransport::registerLocalMemory():                                                   │
│    文件: nvmeof_transport.cpp:226-234                                                        │
│                                                                                              │
│    int registerLocalMemory(void *addr, size_t length, ...) {                                │
│        CUFILE_CHECK(cuFileBufRegister(addr, length, 0));  ★ GPU 内存注册到 cuFile            │
│        return 0;                                                                            │
│    }                                                                                         │
│                                                                                              │
│ 5. cuFileBufRegister() 内部:                                                                  │
│    ├── 锁定 GPU 内存页 (pin pages)                                                           │
│    ├── 建立 GPU 虚拟地址到物理地址的映射                                                      │
│    ├── 注册到 GDS 驱动                                                                       │
│    └── 允许后续 DMA 操作直接访问这块 GPU 内存                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 完整数据流总结

### 5.1 Write Path 数据流

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                              KV Cache → SSD 数据流                                            │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  vLLM KV Cache                           Mooncake Engine                  NVIDIA cuFile       │
│  ┌─────────────────┐                    ┌─────────────────┐            ┌─────────────────┐  │
│  │ torch.Tensor    │                    │ TransferRequest │            │ CUfileIOParams  │  │
│  │ .data_ptr()     │ ─────────────────► │ .source = void* │ ────────► │ .devPtr_base    │  │
│  │ 0x7f1234560000  │   pybind11          │ .length = 16MB  │  Transport  │ .file_offset   │  │
│  │ .nbytes = 1GB   │   U64ToPtr()        │ .opcode = WRITE │            │ .opcode = WRITE │  │
│  └─────────────────┘                    └─────────────────┘            └─────────────────┘  │
│                                                                                              │
│  GPU Memory                              C++ Layer                   cuFile API            │
│                                                                                              │
│                                              │                         │                    │
│                                              ▼                         ▼                    │
│                                      ┌─────────────────┐            ┌─────────────────┐    │
│                                      │ NVMeoFTransport │            │ GDS Driver      │    │
│                                      │ .submitTransfer │ ────────► │ .DMA Engine     │    │
│                                      └─────────────────┘            └─────────────────┘    │
│                                                                         │                    │
│                                                                         ▼                    │
│                                                                   ┌─────────────────┐      │
│                                                                   │   NVMe SSD      │      │
│                                                                   │   File Write    │      │
│                                                                   └─────────────────┘      │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Read Path 数据流

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                              SSD → KV Cache 数据流                                            │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                              │
│  NVMe SSD                               Mooncake Engine                  NVIDIA cuFile       │
│  ┌─────────────────┐                    ┌─────────────────┐            ┌─────────────────┐  │
│  │ File Data       │                    │ TransferRequest │            │ CUfileIOParams  │  │
│  │ LBA 0x12345678  │ ◄───────────────── │ .target_offset  │ ◄──────── │ .file_offset    │  │
│  │ 16 MB           │   DMA Read         │ .length = 16MB  │  Transport │ .opcode = READ  │  │
│  └─────────────────┘                    │ .opcode = READ  │            │ .devPtr_base    │  │
│         ▲                               └─────────────────┘            └─────────────────┘  │
│         │                                     │                         │                    │
│         │                                     ▼                         ▼                    │
│  GDS Driver                             NVMeoFTransport            GPU Memory             │
│  ┌─────────────────┐                    ┌─────────────────┐            ┌─────────────────┐  │
│  │ .DMA Engine     │ ────────────────► │ .submitTransfer │ ────────► │ torch.Tensor    │  │
│  │ PCIe DMA Read   │   Data Transfer    │ opcode = READ   │   Write   │ 0x7f1234560000  │  │
│  └─────────────────┘                    └─────────────────┘            └─────────────────┘  │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 关键代码位置索引

| 层级 | 文件 | 函数/方法 | 行号 |
|------|------|----------|------|
| **PyTorch** | `mooncake_connector_v1.py` | `register_kv_caches()` | 650-699 |
| **PyTorch** | `mooncake_connector_v1.py` | `cache.data_ptr()` | 670 |
| **vLLM Worker** | `mooncake_connector_v1.py` | `send_kv_to_decode()` | 543-571 |
| **vLLM Worker** | `mooncake_connector_v1.py` | `_build_transfer_params()` | 573-629 |
| **vLLM Worker** | `mooncake_connector_v1.py` | `_send_blocks()` | 631-648 |
| **vLLM Worker** | `mooncake_connector_v1.py` | `start_load_kv()` | 847-858 |
| **vLLM Worker** | `mooncake_connector_v1.py` | `receive_kv()` | 775-815 |
| **pybind11** | `transfer_engine_py.cpp` | `batch_transfer_sync_write` | 1014 |
| **pybind11** | `pybind.cpp` | `U64ToPtr()` | 100-102 |
| **pybind11** | `pybind.cpp` | `U64VectorToPtrVector()` | 108-114 |
| **TransferEngine** | `transfer_engine_py.cpp` | `batchTransferSyncWrite()` | 450-539 |
| **TransferEngine** | `transfer_engine.cpp` | `submitTransfer()` | - |
| **NVMeoFTransport** | `nvmeof_transport.cpp` | `submitTransfer()` | 115-206 |
| **NVMeoFTransport** | `nvmeof_transport.cpp` | `addSliceToCUFileBatch()` | 265-281 |
| **NVMeoFTransport** | `nvmeof_transport.cpp` | `registerLocalMemory()` | 226-234 |
| **CUFileDescPool** | `cufile_desc_pool.cpp` | `submitBatch()` | 145-163 |
| **CUFileDescPool** | `cufile_desc_pool.cpp` | `pushParams()` | 128-143 |
| **cuFile Context** | `cufile_context.h` | `CuFileContext()` | constructor |

---

## 7. 性能关键点

### 7.1 批量操作优化

```python
# 使用批量 API 而非单次调用
# ✗ 低效: 多次单次传输
for block_id in block_ids:
    engine.transfer_sync_write(session, src_ptr, dst_ptr, length)

# ✓ 高效: 批量传输
engine.batch_transfer_sync_write(
    session,
    src_ptrs,   # 所有源地址
    dst_ptrs,   # 所有目标地址
    lengths     # 所有长度
)
```

### 7.2 内存预注册

```python
# GPU 内存只需注册一次，后续传输可重复使用
engine.batch_register_memory(kv_data_ptrs, kv_data_lens)

# 后续所有传输都可以直接使用已注册的地址
engine.batch_transfer_sync_write(...)
engine.batch_transfer_sync_read(...)
```

### 7.3 cuFile Batch Handle 复用

```cpp
// cufile_desc_pool.cpp:82-108
// cuFileBatchIOSetUp 是耗时操作，使用 handle pool 复用
if (!handle_pool_.empty()) {
    batch_handle = handle_pool_.back();  // 复用
    handle_pool_.pop_back();
} else {
    // 仅在 pool 为空时创建新 handle
    cuFileBatchIOSetUp(&new_batch_handle->handle, max_batch_size_);
}
```

---

## 8. 错误处理流程

```
传输失败时的状态检查:

CUFILE_WAITING   → 传输等待中
CUFILE_PENDING   → 传输进行中
CUFILE_COMPLETE  → 传输成功
CUFILE_FAILED    → 传输失败
CUFILE_TIMEOUT   → 传输超时
CUFILE_CANCELED  → 传输取消

错误恢复策略:
1. 检查 cuFileBufRegister 是否成功
2. 验证 GPU 内存地址有效性
3. 确认 NVMe 设备可用性
4. 检查 PCIe 带宽和 NUMA 配置
```
