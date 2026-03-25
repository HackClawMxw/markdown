# Mooncake Transfer Engine 源代码解析文档

## 目录
1. [概述](#1-概述)
2. [整体架构](#2-整体架构)
3. [核心类详解](#3-核心类详解)
4. [传输层实现](#4-传输层实现)
5. [元数据管理](#5-元数据管理)
6. [Tent 新架构](#6-tent-新架构)
7. [类图与调用关系](#7-类图与调用关系)

---

## 1. 概述

Mooncake Transfer Engine 是一个高性能、多协议的数据传输引擎，主要用于分布式系统中跨节点、跨设备的数据传输。它支持多种传输协议：

- **RDMA** - 远程直接内存访问，支持 RoCE/iWARP
- **TCP** - 传统 TCP socket 传输
- **NVLink** - NVIDIA GPU 间高速互联
- **NVMeoF (GDS)** - GPU Direct Storage
- **CXL** - Compute Express Link
- **EFA** - AWS Elastic Fabric Adapter
- **Ascend** - 华为昇腾 NPU 传输
- **UBShmem** - 共享内存传输

该引擎支持 CPU 内存、GPU 显存、NPU 内存等多种存储介质之间的高效数据传输。

---

## 2. 整体架构

### 2.1 架构层次图

```
┌─────────────────────────────────────────────────────────────────┐
│                    TransferEngine (用户接口)                      │
├─────────────────────────────────────────────────────────────────┤
│  TransferEngineImpl / tent::TransferEngineImpl (核心实现)        │
├─────────────────────────────────────────────────────────────────┤
│                     MultiTransport (传输调度)                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │   RDMA  │ │   TCP   │ │ NVLink  │ │ NVMeoF  │ │  ...    │   │
│  │Transport│ │Transport│ │Transport│ │Transport│ │Transport│   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
├───────┴──────────┴──────────┴──────────┴──────────┴───────────┤
│                    TransferMetadata (元数据管理)                 │
├─────────────────────────────────────────────────────────────────┤
│                      Topology (拓扑发现)                         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 目录结构

```
mooncake-transfer-engine/
├── include/                    # 头文件目录
│   ├── transfer_engine.h       # 主入口接口
│   ├── transfer_engine_impl.h  # 核心实现
│   ├── multi_transport.h       # 多传输协议管理
│   ├── transfer_metadata.h     # 元数据管理
│   ├── topology.h              # 网络拓扑发现
│   ├── config.h                # 全局配置
│   ├── memory_location.h       # 内存位置检测
│   └── transport/              # 传输层实现
│       ├── transport.h         # 传输层基类
│       ├── rdma_transport/     # RDMA 传输
│       ├── tcp_transport/      # TCP 传输
│       ├── nvlink_transport/   # NVLink 传输
│       └── ...                 # 其他传输层
├── src/                        # 源文件目录
├── tent/                       # Tent 新架构 (下一代实现)
└── tests/                      # 测试用例
```

---

## 3. 核心类详解

### 3.1 TransferEngine

**文件位置**: [transfer_engine.h](Mooncake/mooncake-transfer-engine/include/transfer_engine.h)

**职责**: 提供用户级别的 API 接口，是整个传输引擎的入口点。

```cpp
class TransferEngine {
public:
    // 初始化引擎
    int init(const std::string& metadata_conn_string,
             const std::string& local_server_name,
             const std::string& ip_or_host_name = "",
             uint64_t rpc_port = 12345);

    // 内存注册
    int registerLocalMemory(void* addr, size_t length,
                            const std::string& location = kWildcardLocation,
                            bool remote_accessible = true,
                            bool update_metadata = true);

    // 批量管理
    BatchID allocateBatchID(size_t batch_size);
    Status freeBatchID(BatchID batch_id);

    // 传输操作
    Status submitTransfer(BatchID batch_id,
                          const std::vector<TransferRequest>& entries);
    Status getTransferStatus(BatchID batch_id, size_t task_id,
                             TransferStatus& status);

    // 段管理
    SegmentHandle openSegment(const std::string& segment_name);
    int closeSegment(SegmentHandle handle);

private:
    std::shared_ptr<TransferEngineImpl> impl_;        // 原始实现
    std::shared_ptr<mooncake::tent::TransferEngine> impl_tent_;  // Tent 实现
    bool use_tent_{false};
};
```

**关键特性**:
- 封装了两套实现：原始 `TransferEngineImpl` 和新的 `tent::TransferEngine`
- 使用 PIMPL 模式隐藏实现细节
- 提供统一的 API 供上层调用

---

### 3.2 TransferEngineImpl

**文件位置**: [transfer_engine_impl.h](Mooncake/mooncake-transfer-engine/include/transfer_engine_impl.h)

**职责**: 传输引擎的核心实现，管理所有资源和协调各组件工作。

```cpp
class TransferEngineImpl {
public:
    // 初始化和资源管理
    int init(const std::string& metadata_conn_string,
             const std::string& local_server_name,
             const std::string& ip_or_host_name,
             uint64_t rpc_port);

    // 传输控制
    Status submitTransfer(BatchID batch_id,
                          const std::vector<TransferRequest>& entries);
    Status getTransferStatus(BatchID batch_id, size_t task_id,
                             TransferStatus& status);
    Status getBatchTransferStatus(BatchID batch_id, TransferStatus& status);

    // 内存管理
    int registerLocalMemory(void* addr, size_t length,
                            const std::string& location,
                            bool remote_accessible,
                            bool update_metadata);

    // 通知机制
    int getNotifies(std::vector<TransferMetadata::NotifyDesc>& notifies);
    int sendNotifyByID(SegmentID target_id, TransferMetadata::NotifyDesc msg);

private:
    std::shared_ptr<TransferMetadata> metadata_;          // 元数据管理
    std::shared_ptr<MultiTransport> multi_transports_;    // 多传输协议
    std::shared_ptr<Topology> local_topology_;            // 本地拓扑
    std::vector<MemoryRegion> local_memory_regions_;      // 本地内存区域
};
```

**关键数据结构**:

```cpp
struct MemoryRegion {
    void* addr;              // 内存地址
    uint64_t length;         // 长度
    std::string location;    // 位置 (cpu:0, cuda:0 等)
    bool remote_accessible;  // 是否可远程访问
};
```

---

### 3.3 Transport (传输层基类)

**文件位置**: [transport.h](Mooncake/mooncake-transfer-engine/include/transport/transport.h)

**职责**: 定义所有传输层的通用接口和数据结构。

```cpp
class Transport {
public:
    using SegmentID = uint64_t;
    using BatchID = uint64_t;

    // 传输请求结构
    struct TransferRequest {
        enum OpCode { READ, WRITE };
        OpCode opcode;
        void* source;
        SegmentID target_id;
        uint64_t target_offset;
        size_t length;
        int advise_retry_cnt = 0;
    };

    // 传输状态枚举
    enum TransferStatusEnum {
        WAITING, PENDING, INVALID,
        CANCELED, COMPLETED, TIMEOUT, FAILED
    };

    // 核心虚函数
    virtual BatchID allocateBatchID(size_t batch_size);
    virtual Status freeBatchID(BatchID batch_id);
    virtual Status submitTransfer(BatchID batch_id,
                                  const std::vector<TransferRequest>& entries) = 0;
    virtual Status getTransferStatus(BatchID batch_id, size_t task_id,
                                     TransferStatus& status) = 0;

protected:
    virtual int registerLocalMemory(void* addr, size_t length,
                                    const std::string& location,
                                    bool remote_accessible,
                                    bool update_metadata = true) = 0;
    virtual int unregisterLocalMemory(void* addr,
                                      bool update_metadata = true) = 0;
};
```

**核心数据结构**:

```cpp
// 传输切片 - 最小的传输单元
struct Slice {
    enum SliceStatus { PENDING, POSTED, SUCCESS, TIMEOUT, FAILED };

    void* source_addr;
    size_t length;
    TransferRequest::OpCode opcode;
    SegmentID target_id;
    SliceStatus status;
    TransferTask* task;

    // 协议特定字段 (union)
    union {
        struct { uint64_t dest_addr; uint32_t source_lkey; ... } rdma;
        struct { void* dest_addr; } local;
        struct { uint64_t dest_addr; } tcp;
        struct { uint64_t offset; int cufile_desc; ... } nvmeof;
        // ...
    };

    void markSuccess();  // 标记成功
    void markFailed();   // 标记失败
};

// 传输任务 - 包含多个切片
struct TransferTask {
    volatile uint64_t slice_count = 0;
    volatile uint64_t success_slice_count = 0;
    volatile uint64_t failed_slice_count = 0;
    volatile uint64_t transferred_bytes = 0;
    BatchID batch_id = 0;
    const TransferRequest* request = nullptr;
    std::vector<Slice*> slice_list;
};

// 批次描述符 - 包含多个任务
struct BatchDesc {
    BatchID id;
    size_t batch_size;
    std::vector<TransferTask> task_list;
    std::atomic<bool> is_finished{false};
    std::atomic<uint64_t> finished_transfer_bytes{0};
};
```

---

### 3.4 MultiTransport

**文件位置**: [multi_transport.h](Mooncake/mooncake-transfer-engine/include/multi_transport.h)

**职责**: 管理多种传输协议，根据目标段自动选择合适的传输层。

```cpp
class MultiTransport {
public:
    MultiTransport(std::shared_ptr<TransferMetadata> metadata,
                   std::string& local_server_name);

    // 批次管理 (代理到各个 Transport)
    BatchID allocateBatchID(size_t batch_size);
    Status freeBatchID(BatchID batch_id);

    // 传输操作
    Status submitTransfer(BatchID batch_id,
                          const std::vector<TransferRequest>& entries);
    Status getTransferStatus(BatchID batch_id, size_t task_id,
                             TransferStatus& status);

    // 传输层管理
    Transport* installTransport(const std::string& proto,
                                std::shared_ptr<Topology> topo);
    Transport* getTransport(const std::string& proto);

private:
    // 根据请求选择合适的传输层
    Status selectTransport(const TransferRequest& entry, Transport*& transport);

    std::shared_ptr<TransferMetadata> metadata_;
    std::map<std::string, std::shared_ptr<Transport>> transport_map_;
};
```

**传输层选择逻辑**:
```cpp
Status MultiTransport::selectTransport(const TransferRequest& entry,
                                       Transport*& transport) {
    // 1. 获取目标段的描述信息
    auto target_segment_desc = metadata_->getSegmentDescByID(entry.target_id);

    // 2. 从描述中获取协议类型
    auto proto = target_segment_desc->protocol;

    // 3. 查找对应的传输层
    transport = transport_map_[proto].get();
    return Status::OK();
}
```

---

### 3.5 TransferMetadata

**文件位置**: [transfer_metadata.h](Mooncake/mooncake-transfer-engine/include/transfer_metadata.h)

**职责**: 管理传输所需的元数据，包括段信息、RPC 通信、握手协议等。

```cpp
class TransferMetadata {
public:
    // 设备描述
    struct DeviceDesc {
        std::string name;    // 设备名称如 mlx5_0
        uint16_t lid;        // RDMA LID
        std::string gid;     // RDMA GID
    };

    // 缓冲区描述
    struct BufferDesc {
        std::string name;           // 位置名称
        uint64_t addr;              // 地址
        uint64_t length;            // 长度
        std::vector<uint32_t> lkey; // 本地密钥 (RDMA)
        std::vector<uint32_t> rkey; // 远程密钥 (RDMA)
        std::string shm_name;       // 共享内存名称
    };

    // 段描述
    struct SegmentDesc {
        std::string name;               // 段名称
        std::string protocol;           // 协议类型
        std::vector<DeviceDesc> devices;// 设备列表
        Topology topology;              // 拓扑信息
        std::vector<BufferDesc> buffers;// 缓冲区列表
    };

    // RPC 元数据
    struct RpcMetaDesc {
        std::string ip_or_host_name;
        uint16_t rpc_port;
        int sockfd;
    };

    // 握手描述
    struct HandShakeDesc {
        std::string local_nic_path;
        std::string peer_nic_path;
        std::vector<uint32_t> qp_num;   // QP 号列表
    };

    // 段管理
    std::shared_ptr<SegmentDesc> getSegmentDescByName(const std::string& name);
    SegmentID getSegmentID(const std::string& segment_name);
    int updateLocalSegmentDesc(SegmentID segment_id);

    // RPC 管理
    int addRpcMetaEntry(const std::string& server_name, RpcMetaDesc& desc);
    int getRpcMetaEntry(const std::string& server_name, RpcMetaDesc& desc);

    // 握手服务
    int startHandshakeDaemon(OnReceiveHandShake callback, uint16_t port, int sockfd);
    int sendHandshake(const std::string& peer_server_name,
                      const HandShakeDesc& local_desc,
                      HandShakeDesc& peer_desc);

private:
    std::unordered_map<uint64_t, std::shared_ptr<SegmentDesc>> segment_id_to_desc_map_;
    std::unordered_map<std::string, RpcMetaDesc> rpc_meta_map_;
    std::shared_ptr<MetadataStoragePlugin> storage_plugin_;  // etcd/redis 等
};
```

---

### 3.6 Topology

**文件位置**: [topology.h](Mooncake/mooncake-transfer-engine/include/topology.h)

**职责**: 发现和管理网络拓扑，支持 NUMA 感知和设备亲和性。

```cpp
struct TopologyEntry {
    std::string name;                   // 存储类型
    std::vector<std::string> preferred_hca;  // 首选 HCA
    std::vector<std::string> avail_hca;      // 可用 HCA
};

class Topology {
public:
    Topology();

    // 发现网络拓扑
    int discover();
    int discover(const std::vector<std::string>& filter);

    // 设备选择
    int selectDevice(const std::string storage_type, int retry_count = 0);
    int selectDevice(const std::string storage_type, std::string_view hint,
                     int retry_count = 0);

    // 配置
    int parse(const std::string& topology_json);
    int disableDevice(const std::string& device_name);

    // 访问器
    TopologyMatrix getMatrix() const;
    const std::vector<std::string>& getHcaList() const;

private:
    TopologyMatrix matrix_;              // 拓扑矩阵
    std::vector<std::string> hca_list_;  // HCA 列表

    struct ResolvedTopologyEntry {
        std::vector<int> preferred_hca;
        std::vector<int> avail_hca;
        std::unordered_map<std::string, int> preferred_hca_name_to_index_map_;
        std::unordered_map<std::string, int> avail_hca_name_to_index_map_;
    };
    std::unordered_map<std::string, ResolvedTopologyEntry> resolved_matrix_;
};
```

---

### 3.7 GlobalConfig

**文件位置**: [config.h](Mooncake/mooncake-transfer-engine/include/config.h)

**职责**: 全局配置管理，控制引擎行为。

```cpp
struct GlobalConfig {
    size_t num_cq_per_ctx = 1;           // 每个上下文的 CQ 数量
    size_t num_comp_channels_per_ctx = 1;// 完成通道数量
    uint8_t port = 1;                    // RDMA 端口
    int gid_index = -1;                  // GID 索引 (-1 自动选择)
    uint64_t max_mr_size = 0x10000000000;// 最大 MR 大小 (1TB)
    size_t max_cqe = 4096;               // 最大完成队列条目
    int max_ep_per_ctx = 65536;          // 每个上下文最大端点数
    size_t num_qp_per_ep = 2;            // 每个端点的 QP 数量
    size_t max_sge = 4;                  // 最大分散/聚集元素
    size_t max_wr = 256;                 // 最大工作请求数
    size_t max_inline = 64;              // 最大内联数据
    size_t slice_size = 65536;           // 切片大小 (64KB)
    int retry_cnt = 9;                   // 重试次数
    int workers_per_ctx = 2;             // 每个上下文的工作线程数
    bool metacache = true;               // 元数据缓存
    int log_level = google::INFO;        // 日志级别
    int64_t slice_timeout = -1;          // 切片超时
    size_t fragment_limit = 16384;       // 分片限制
    bool enable_dest_device_affinity = false; // 目标设备亲和性
    int parallel_reg_mr = -1;            // 并行 MR 注册
    EndpointStoreType endpoint_store_type = EndpointStoreType::SIEVE;
    int ib_traffic_class = -1;           // IB 流量类别
    int ib_pci_relaxed_ordering_mode = 0;// PCI 宽松排序模式
};
```

---

## 4. 传输层实现

### 4.1 RdmaTransport

**文件位置**:
- [rdma_transport.h](Mooncake/mooncake-transfer-engine/include/transport/rdma_transport/rdma_transport.h)
- [rdma_transport.cpp](Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp)

**职责**: RDMA 传输协议实现，支持 RoCE/iWARP。

```cpp
class RdmaTransport : public Transport {
public:
    int install(std::string& local_server_name,
                std::shared_ptr<TransferMetadata> meta,
                std::shared_ptr<Topology> topo) override;

    // 内存注册
    int registerLocalMemory(void* addr, size_t length,
                            const std::string& location,
                            bool remote_accessible,
                            bool update_metadata) override;

    // 传输操作
    Status submitTransfer(BatchID batch_id,
                          const std::vector<TransferRequest>& entries) override;
    Status getTransferStatus(BatchID batch_id, size_t task_id,
                             TransferStatus& status) override;

    // 设备选择
    static int selectDevice(SegmentDesc* desc, uint64_t offset,
                            size_t length, int& buffer_id, int& device_id);

    // 握手回调
    int onSetupRdmaConnections(const HandShakeDesc& peer_desc,
                               HandShakeDesc& local_desc);

private:
    std::vector<std::shared_ptr<RdmaContext>> context_list_;  // RDMA 上下文列表
    std::shared_ptr<Topology> local_topology_;
};
```

#### 4.1.1 RdmaContext

**职责**: 管理单个 RDMA 网卡的所有资源。

```cpp
class RdmaContext {
public:
    RdmaContext(RdmaTransport& engine, const std::string& device_name);

    int construct(size_t num_cq_list, size_t num_comp_channels,
                  uint8_t port, int gid_index, size_t max_cqe,
                  int max_endpoints);

    // 内存区域管理
    int registerMemoryRegion(void* addr, size_t length, int access);
    int unregisterMemoryRegion(void* addr);
    uint32_t rkey(void* addr);
    uint32_t lkey(void* addr);

    // 端点管理
    std::shared_ptr<RdmaEndPoint> endpoint(const std::string& peer_nic_path);
    int deleteEndpoint(const std::string& peer_nic_path);

    // 设备信息
    std::string deviceName() const;
    std::string nicPath() const;
    uint16_t lid() const;
    std::string gid() const;
    ibv_pd* pd() const;

    // 提交发送
    int submitPostSend(const std::vector<Transport::Slice*>& slice_list);

private:
    const std::string device_name_;
    RdmaTransport& engine_;
    ibv_context* context_;
    ibv_pd* pd_;
    uint16_t lid_;
    int gid_index_;
    ibv_gid gid_;

    std::vector<MemoryRegionMeta> memory_region_list_;
    std::vector<RdmaCq> cq_list_;
    std::shared_ptr<EndpointStore> endpoint_store_;
    std::shared_ptr<WorkerPool> worker_pool_;
};
```

#### 4.1.2 RdmaEndPoint

**职责**: 管理与远程 NIC 的 QP 连接。

```cpp
class RdmaEndPoint {
public:
    enum Status { INITIALIZING, UNCONNECTED, CONNECTED };

    RdmaEndPoint(RdmaContext& context);

    int construct(ibv_cq* cq, size_t num_qp_list, size_t max_sge,
                  size_t max_wr, size_t max_inline);

    // 连接建立
    int setupConnectionsByActive();
    int setupConnectionsByPassive(const HandShakeDesc& peer_desc,
                                  HandShakeDesc& local_desc);

    // 提交发送
    int submitPostSend(std::vector<Transport::Slice*>& slice_list,
                       std::vector<Transport::Slice*>& failed_slice_list);

    bool connected() const;
    void disconnect();

private:
    RdmaContext& context_;
    std::atomic<Status> status_;
    std::vector<ibv_qp*> qp_list_;
    std::string peer_nic_path_;
    volatile int* wr_depth_list_;
    int max_wr_depth_;
};
```

#### 4.1.3 WorkerPool

**职责**: 管理工作线程池，处理 RDMA 发送和完成轮询。

```cpp
class WorkerPool {
public:
    WorkerPool(RdmaContext& context, int numa_socket_id = 0);

    int submitPostSend(const std::vector<Transport::Slice*>& slice_list);

private:
    void transferWorker(int thread_id);
    void performPostSend(int thread_id);
    void performPollCq(int thread_id);
    void redispatch(std::vector<Transport::Slice*>& slice_list, int thread_id);

    RdmaContext& context_;
    std::vector<std::thread> worker_thread_;
    std::atomic<bool> workers_running_;

    // 分片队列 (按目标端点分组)
    const static int kShardCount = 8;
    std::unordered_map<std::string, SliceList> slice_queue_[kShardCount];
    TicketLock slice_queue_lock_[kShardCount];
};
```

#### 4.1.4 EndpointStore

**职责**: 端点缓存管理，支持 FIFO 和 SIEVE 淘汰策略。

```cpp
class EndpointStore {
public:
    virtual std::shared_ptr<RdmaEndPoint> getEndpoint(
        const std::string& peer_nic_path) = 0;
    virtual std::shared_ptr<RdmaEndPoint> insertEndpoint(
        const std::string& peer_nic_path, RdmaContext* context) = 0;
    virtual void evictEndpoint() = 0;
    virtual void reclaimEndpoint() = 0;
};

// FIFO 实现
class FIFOEndpointStore : public EndpointStore { ... };

// SIEVE 实现 (NSDI 24 论文算法)
class SIEVEEndpointStore : public EndpointStore { ... };
```

### 4.2 TcpTransport

**文件位置**: [tcp_transport.h](Mooncake/mooncake-transfer-engine/include/transport/tcp_transport/tcp_transport.h)

**职责**: TCP socket 传输，作为无 RDMA 时的回退方案。

```cpp
class TcpTransport : public Transport {
public:
    Status submitTransfer(BatchID batch_id,
                          const std::vector<TransferRequest>& entries) override;
    Status getTransferStatus(BatchID batch_id, size_t task_id,
                             TransferStatus& status) override;

private:
    int install(std::string& local_server_name,
                std::shared_ptr<TransferMetadata> meta,
                std::shared_ptr<Topology> topo);

    void startTransfer(Slice* slice);
    void worker();

    TcpContext* context_;
    std::atomic_bool running_;
    std::thread thread_;

    // 连接池 (可选)
    std::unordered_map<ConnectionKey, std::deque<std::shared_ptr<PooledConnection>>>
        connection_pool_;
    bool enable_connection_pool_ = false;
};
```

### 4.3 NvlinkTransport

**文件位置**: [nvlink_transport.h](Mooncake/mooncake-transfer-engine/include/transport/nvlink_transport/nvlink_transport.h)

**职责**: NVIDIA NVLink 跨节点 GPU 互联传输。

```cpp
class NvlinkTransport : public Transport {
public:
    Status submitTransfer(BatchID batch_id,
                          const std::vector<TransferRequest>& entries) override;
    Status getTransferStatus(BatchID batch_id, size_t task_id,
                             TransferStatus& status) override;

    // 固定内存管理
    static void* allocatePinnedLocalMemory(size_t length);
    static void freePinnedLocalMemory(void* addr);

protected:
    int relocateSharedMemoryAddress(uint64_t& dest_addr, uint64_t length,
                                    uint64_t target_id);

private:
    std::atomic_bool running_;

    struct OpenedShmEntry {
        void* shm_addr;
        uint64_t length;
    };
    std::unordered_map<std::pair<uint64_t, uint64_t>, OpenedShmEntry, PairHash>
        remap_entries_;
    bool use_fabric_mem_;
};
```

### 4.4 NVMeoFTransport

**文件位置**: [nvmeof_transport.h](Mooncake/mooncake-transfer-engine/include/transport/nvmeof_transport/nvmeof_transport.h)

**职责**: NVMe over Fabrics / GPU Direct Storage 传输。

```cpp
class NVMeoFTransport : public Transport {
public:
    BatchID allocateBatchID(size_t batch_size) override;
    Status submitTransferTask(const std::vector<TransferTask*>& task_list) override;
    Status submitTransfer(BatchID batch_id,
                          const std::vector<TransferRequest>& entries) override;
    Status getTransferStatus(BatchID batch_id, size_t task_id,
                             TransferStatus& status) override;
    Status freeBatchID(BatchID batch_id) override;

private:
    void startTransfer(Slice* slice);
    void addSliceToCUFileBatch(void* source_addr, uint64_t file_offset,
                               uint64_t slice_len, uint64_t desc_id,
                               TransferRequest::OpCode op, CUfileHandle_t fh);

    std::unordered_map<BatchID, int> batch_to_cufile_desc_;
    std::unordered_map<std::pair<SegmentHandle, uint64_t>,
                       std::shared_ptr<CuFileContext>, pair_hash>
        segment_to_context_;
    std::shared_ptr<CUFileDescPool> desc_pool_;
};
```

---

## 5. 元数据管理

### 5.1 MetadataStoragePlugin

元数据存储插件接口，支持 etcd、Redis、HTTP 等后端。

```cpp
struct MetadataStoragePlugin {
    virtual int get(const std::string& key, std::string& value) = 0;
    virtual int put(const std::string& key, const std::string& value) = 0;
    virtual int remove(const std::string& key) = 0;
    virtual ~MetadataStoragePlugin() = default;
};
```

### 5.2 握手协议

RDMA 连接建立流程:

```
┌─────────────┐                              ┌─────────────┐
│   本地端    │                              │   远程端    │
└──────┬──────┘                              └──────┬──────┘
       │                                            │
       │  1. sendHandshake(local_desc)              │
       │ ─────────────────────────────────────────> │
       │                                            │
       │                     2. setupConnectionsByPassive()
       │                        (创建 QP, 状态转换到 RTR/RTS)
       │                                            │
       │  3. return peer_desc (包含 QP 号等)        │
       │ <───────────────────────────────────────── │
       │                                            │
       │  4. setupConnectionsByActive()             │
       │     (使用 peer_desc 完成本地 QP 连接)       │
       │                                            │
       │  ========== RDMA 连接建立完成 ============ │
       │                                            │
```

---

## 6. Tent 新架构

Tent 是 Mooncake Transfer Engine 的下一代实现，提供了更现代化的架构设计。

### 6.1 tent::TransferEngine

**文件位置**: [tent/include/tent/transfer_engine.h](Mooncake/mooncake-transfer-engine/tent/include/tent/transfer_engine.h)

```cpp
namespace mooncake::tent {

class TransferEngine {
public:
    TransferEngine();
    TransferEngine(const std::string config_path);
    TransferEngine(std::shared_ptr<Config> config);

    bool available() const;

    // 段管理
    Status openSegment(SegmentID& handle, const std::string& segment_name);
    Status closeSegment(SegmentID handle);
    Status getSegmentInfo(SegmentID handle, SegmentInfo& info);

    // 内存管理
    Status allocateLocalMemory(void** addr, size_t size, Location location);
    Status freeLocalMemory(void* addr);
    Status registerLocalMemory(void* addr, size_t size, Permission permission);
    Status unregisterLocalMemory(void* addr, size_t size = 0);

    // 高级内存注册
    Status allocateLocalMemory(void** addr, size_t size, MemoryOptions& options);
    Status registerLocalMemory(void* addr, size_t size, MemoryOptions& options);

    // 传输操作
    BatchID allocateBatch(size_t batch_size);
    Status freeBatch(BatchID batch_id);
    Status submitTransfer(BatchID batch_id,
                          const std::vector<Request>& request_list);
    Status submitTransfer(BatchID batch_id,
                          const std::vector<Request>& request_list,
                          const Notification& notifi);

    // 状态查询
    Status getTransferStatus(BatchID batch_id, size_t task_id,
                             TransferStatus& status);
    Status getTransferStatus(BatchID batch_id, TransferStatus& overall_status);

    // 通知机制
    Status sendNotification(SegmentID target_id, const Notification& notifi);
    Status receiveNotification(std::vector<Notification>& notifi_list);

private:
    std::unique_ptr<TransferEngineImpl> impl_;
};

} // namespace mooncake::tent
```

### 6.2 tent::TransferEngineImpl

**文件位置**: [tent/include/tent/runtime/transfer_engine_impl.h](Mooncake/mooncake-transfer-engine/tent/include/tent/runtime/transfer_engine_impl.h)

```cpp
class TransferEngineImpl {
public:
    Status construct();
    Status deconstruct();
    Status setupLocalSegment();

    // 传输类型选择
    TransportType getTransportType(const Request& request, int priority = 0);
    std::vector<TransportType> getSupportedTransports(TransportType request_type);
    TransportType resolveTransport(const Request& req, int priority,
                                   bool invalidate_on_fail = true);

    // 请求合并优化
    void findStagingPolicy(const Request& request,
                           std::vector<std::string>& policy);

private:
    std::shared_ptr<Config> conf_;
    std::shared_ptr<ControlService> metadata_;
    std::shared_ptr<Topology> topology_;
    bool available_;

    // 传输层数组 (按 TransportType 索引)
    std::array<std::shared_ptr<Transport>, kSupportedTransportTypes>
        transport_list_;

    std::unique_ptr<SegmentTracker> local_segment_tracker_;
    ThreadLocalStorage<BatchSet> batch_set_;
    std::unique_ptr<ProxyManager> staging_proxy_;
    bool merge_requests_;
};
```

### 6.3 tent::Transport

**文件位置**: [tent/include/tent/runtime/transport.h](Mooncake/mooncake-transfer-engine/tent/include/tent/runtime/transport.h)

```cpp
class Transport {
public:
    struct SubBatch {
        virtual ~SubBatch() {}
        virtual size_t size() const = 0;
    };
    using SubBatchRef = SubBatch*;

    // 能力声明
    struct Capabilities {
        bool dram_to_dram = false;
        bool dram_to_gpu = false;
        bool gpu_to_dram = false;
        bool gpu_to_gpu = false;
        bool dram_to_file = false;
        bool gpu_to_file = false;
    };

    virtual Status install(std::string& local_segment_name,
                           std::shared_ptr<ControlService> metadata,
                           std::shared_ptr<Topology> local_topology,
                           std::shared_ptr<Config> conf = nullptr);

    // 子批次管理
    virtual Status allocateSubBatch(SubBatchRef& batch, size_t max_size);
    virtual Status freeSubBatch(SubBatchRef& batch);

    // 传输操作
    virtual Status submitTransferTasks(SubBatchRef batch,
                                       const std::vector<Request>& request_list);
    virtual Status getTransferStatus(SubBatchRef batch, int task_id,
                                     TransferStatus& status);

    // 内存管理
    virtual Status addMemoryBuffer(BufferDesc& desc, const MemoryOptions& options);
    virtual Status removeMemoryBuffer(BufferDesc& desc);

    // 通知支持
    virtual bool supportNotification() const;
    virtual Status sendNotification(SegmentID target_id, const Notification& notify);
    virtual Status receiveNotification(std::vector<Notification>& notify_list);

protected:
    Capabilities caps;
};
```

### 6.4 tent::SegmentManager

**文件位置**: [tent/include/tent/runtime/segment_manager.h](Mooncake/mooncake-transfer-engine/tent/include/tent/runtime/segment_manager.h)

```cpp
class SegmentManager {
public:
    SegmentManager(std::unique_ptr<SegmentRegistry> registry);

    // 远程段管理
    Status openRemote(SegmentID& handle, const std::string& segment_name);
    Status closeRemote(SegmentID handle);
    Status getRemoteCached(SegmentDesc*& desc, SegmentID handle);
    Status invalidateRemote(SegmentID handle);

    // 本地段管理
    SegmentDescRef getLocal();
    Status synchronizeLocal();
    Status deleteLocal();

private:
    struct RemoteSegmentCache {
        uint64_t last_refresh = 0;
        uint64_t version = 0;
        std::unordered_map<SegmentID, SegmentDescRef> id_to_desc_map;
    };

    RWSpinlock lock_;
    std::unordered_map<SegmentID, std::string> id_to_name_map_;
    std::unordered_map<std::string, SegmentID> name_to_id_map_;
    std::atomic<SegmentID> next_id_;

    SegmentDescRef local_desc_;
    ThreadLocalStorage<RemoteSegmentCache> tl_remote_cache_;
    std::unique_ptr<SegmentRegistry> registry_;
    uint64_t ttl_ms_ = 10 * 1000;
};
```

### 6.5 tent::ControlService

**文件位置**: [tent/include/tent/runtime/control_plane.h](Mooncake/mooncake-transfer-engine/tent/include/tent/runtime/control_plane.h)

```cpp
struct BootstrapDesc {
    std::string local_nic_path;
    std::string peer_nic_path;
    std::vector<uint32_t> qp_num;
    uint16_t local_lid = 0;
    std::string local_gid;
    std::string reply_msg;
    uint32_t notify_qp_num = 0;
};

class ControlClient {
public:
    static Status getSegmentDesc(const std::string& server_addr,
                                 std::string& response);
    static Status bootstrap(const std::string& server_addr,
                            const BootstrapDesc& request,
                            BootstrapDesc& response);
    static Status sendData(const std::string& server_addr,
                           uint64_t peer_mem_addr, void* local_mem_addr,
                           size_t length);
    static Status notify(const std::string& server_addr,
                         const Notification& message);
};

class ControlService {
public:
    ControlService(const std::string& type, const std::string& servers,
                   const std::string& password, uint8_t db_index,
                   TransferEngineImpl* impl);

    SegmentManager& segmentManager();
    Status start(uint16_t& port, bool ipv6_ = false);

    void setBootstrapRdmaCallback(const OnReceiveBootstrap& callback);
    void setNotifyCallback(const OnNotify& callback);

private:
    std::unique_ptr<SegmentManager> manager_;
    std::shared_ptr<CoroRpcAgent> rpc_server_;
    OnReceiveBootstrap bootstrap_callback_;
    OnNotify notify_callback_;
};
```

### 6.6 Tent 类型定义

**文件位置**: [tent/include/tent/common/types.h](Mooncake/mooncake-transfer-engine/tent/include/tent/common/types.h)

```cpp
namespace mooncake::tent {

using BatchID = uint64_t;
using SegmentID = uint64_t;
using Location = std::string;

struct Notification {
    std::string name;
    std::string msg;
};

struct Request {
    enum OpCode { READ, WRITE };
    OpCode opcode;
    void* source;
    SegmentID target_id;
    uint64_t target_offset;
    size_t length;
};

enum TransferStatusEnum {
    INITIAL, PENDING, INVALID,
    CANCELED, COMPLETED, TIMEOUT, FAILED
};

enum Permission {
    kLocalReadWrite,
    kGlobalReadOnly,
    kGlobalReadWrite,
};

enum TransportType {
    RDMA = 0,
    MNNVL,      // Multi-Node NVLink
    SHM,        // Shared Memory
    NVLINK,     // NVLink
    GDS,        // GPU Direct Storage
    IOURING,    // io_uring
    TCP,
    AscendDirect,
    UNSPEC
};
const static int kSupportedTransportTypes = 8;

struct MemoryOptions {
    Location location = kWildcardLocation;
    Permission perm = kGlobalReadWrite;
    TransportType type = UNSPEC;
    std::string shm_path = "";
    size_t shm_offset = 0;
    bool internal = false;
};

} // namespace mooncake::tent
```

---

## 7. 类图与调用关系

### 7.1 核心类关系图

```
                    ┌─────────────────┐
                    │ TransferEngine  │ (用户接口)
                    └────────┬────────┘
                             │ 组合
              ┌──────────────┴──────────────┐
              ▼                             ▼
    ┌──────────────────┐          ┌─────────────────────┐
    │TransferEngineImpl│          │tent::TransferEngine │
    │    (原始实现)    │          │   (Tent 新架构)     │
    └────────┬─────────┘          └──────────┬──────────┘
             │ 组合                           │ 组合
             ▼                                ▼
    ┌─────────────────┐            ┌──────────────────────┐
    │ MultiTransport  │            │tent::TransferEngineImpl│
    └────────┬────────┘            └──────────┬───────────┘
             │ 聚合                            │ 聚合
             ▼                                 ▼
    ┌─────────────────┐            ┌──────────────────────┐
    │    Transport    │◄───────────│  tent::Transport     │
    │   (传输层基类)  │            │   (Tent 传输层基类)  │
    └────────┬────────┘            └──────────┬───────────┘
             │ 继承                            │ 继承
    ┌────────┴────────┐              ┌────────┴───────────┐
    │                 │              │                    │
    ▼                 ▼              ▼                    ▼
┌───────────┐  ┌───────────┐  ┌───────────┐      ┌───────────┐
│RdmaTransport│ │TcpTransport│ │tent::RdmaTransport│ │tent::ShmTransport│
└─────┬─────┘  └───────────┘  └───────────┘      └───────────┘
      │ 组合
      ▼
┌─────────────────────────────────────────────────────────────┐
│                      RdmaContext                             │
│  (管理单个 RDMA 设备的所有资源: MR, CQ, Endpoint)           │
└─────────────────────────────┬───────────────────────────────┘
                              │ 组合
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      ┌───────────────┐ ┌───────────┐ ┌─────────────┐
      │ RdmaEndPoint  │ │ WorkerPool│ │EndpointStore│
      │  (QP 连接)    │ │ (工作线程)│ │ (端点缓存)  │
      └───────────────┘ └───────────┘ └─────────────┘
```

### 7.2 数据传输调用流程

```
用户代码
    │
    ▼
TransferEngine::submitTransfer(batch_id, requests)
    │
    ▼
TransferEngineImpl::submitTransfer(batch_id, requests)
    │
    ▼
MultiTransport::submitTransfer(batch_id, requests)
    │
    ├─── 对每个请求 ───┐
    │                  │
    │                  ▼
    │         selectTransport(request, transport)
    │                  │
    │                  ▼
    │         Transport::submitTransferTask(task_list)
    │                  │
    │                  ▼ (以 RDMA 为例)
    │         RdmaTransport::submitTransferTask(task_list)
    │                  │
    │                  ├── 1. 创建 Slices
    │                  ├── 2. 选择设备 (selectDevice)
    │                  └── 3. 提交到 Context
    │                          │
    └──────────────────────────┘
                               │
                               ▼
                    RdmaContext::submitPostSend(slice_list)
                               │
                               ▼
                    WorkerPool::submitPostSend(slice_list)
                               │
                               ▼
                    RdmaEndPoint::submitPostSend(slice_list)
                               │
                               ▼
                         ibv_post_send()
                               │
                               ▼
                        硬件处理传输
```

### 7.3 内存注册流程

```
用户代码
    │
    ▼
TransferEngine::registerLocalMemory(addr, length, location)
    │
    ▼
TransferEngineImpl::registerLocalMemory(addr, length, location)
    │
    ├── 1. 检查重叠
    │
    ├── 2. 对每个 Transport 调用 registerLocalMemory
    │       │
    │       ▼ (以 RDMA 为例)
    │   RdmaTransport::registerLocalMemory(addr, length, location)
    │       │
    │       ├── preTouchMemory() (可选，大内存优化)
    │       │
    │       └── 对每个 RdmaContext:
    │               │
    │               ▼
    │           RdmaContext::registerMemoryRegion(addr, length, access)
    │               │
    │               ▼
    │           ibv_reg_mr(pd, addr, length, access)
    │
    ├── 3. 获取内存位置 (getMemoryLocation)
    │
    ├── 4. 添加到元数据 (metadata_->addLocalMemoryBuffer)
    │
    └── 5. 记录到本地内存区域列表
```

### 7.4 握手连接流程

```
本地端                                              远程端
    │                                                  │
    │ 1. 获取远程段信息                                │
    │    getSegmentDescByID(target_id)                 │
    │                                                  │
    │ 2. 发起握手请求                                  │
    │ ──── sendHandshake(peer_name, local_desc) ────> │
    │                                                  │
    │                               3. 接收握手请求    │
    │                                  onBootstrapRdma │
    │                                                  │
    │                               4. 查找对应 Context│
    │                                  (根据 local_nic_path)
    │                                                  │
    │                               5. 获取/创建 Endpoint
    │                                                  │
    │                               6. setupConnectionsByPassive
    │                                  - 创建 QP
    │                                  - 状态转换: INIT → RTR → RTS
    │                                  - 返回 QP 号等信息
    │                                                  │
    │ <─── 返回 peer_desc (QP号, LID, GID) ────────── │
    │                                                  │
    │ 7. setupConnectionsByActive                     │
    │    - 使用 peer_info 连接本地 QP                  │
    │    - 状态转换: INIT → RTR → RTS                 │
    │                                                  │
    │ ============ RDMA 连接建立完成 ================== │
    │                                                  │
```

---

## 8. 总结

### 8.1 关键设计模式

1. **策略模式 (Strategy Pattern)**
   - `Transport` 基类定义统一接口，各传输层实现具体策略
   - `MultiTransport` 根据目标段自动选择合适的传输层

2. **工厂模式 (Factory Pattern)**
   - `MultiTransport::installTransport` 根据协议类型创建对应的传输层实例

3. **代理模式 (Proxy Pattern)**
   - `TransferEngine` 作为 `TransferEngineImpl` 的代理
   - 提供简洁的用户 API，隐藏实现细节

4. **对象池模式 (Object Pool Pattern)**
   - `ThreadLocalSliceCache` 管理 Slice 对象的复用
   - `EndpointStore` 管理端点连接的缓存和复用

5. **观察者模式 (Observer Pattern)**
   - 通知机制 (`Notification`) 允许传输完成后通知相关方

### 8.2 性能优化技术

1. **零拷贝传输**
   - RDMA 直接内存访问，避免 CPU 拷贝
   - GPU Direct Storage 绕过 CPU 直接访问存储

2. **并行处理**
   - 多工作线程并行处理发送请求
   - 并行内存注册 (`parallel_reg_mr`)

3. **内存预取**
   - 大内存注册时预取页面 (`preTouchMemory`)

4. **批处理**
   - 批量提交传输请求
   - 批量内存注册/注销

5. **连接复用**
   - 端点缓存避免重复建立连接
   - TCP 连接池可选支持

### 8.3 扩展性

1. **传输层扩展**
   - 继承 `Transport` 基类实现新的传输协议
   - 通过编译宏控制支持的传输层

2. **元数据存储扩展**
   - `MetadataStoragePlugin` 接口支持多种后端
   - 支持 etcd、Redis、HTTP 等

3. **配置扩展**
   - 环境变量控制运行时行为
   - JSON 配置文件支持

---

*文档版本: 1.0*
*生成日期: 2026-03-25*
*基于 Mooncake Transfer Engine 源代码分析*
