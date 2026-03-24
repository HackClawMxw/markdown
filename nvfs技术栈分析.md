# NVIDIA GPUDirect Storage (GDS) Kernel Driver - 架构与模块调用链分析

## 1. 概述

**项目名称**: nvidia-fs (GPUDirect Storage Kernel Driver)
**版本**: 2.28.2
**功能定位**: 实现从DMA/RDMA存储设备直接到GPU内存的零拷贝数据传输

### 1.1 核心特性

- 支持NVMe/NVMeOF/ScaleFlux CSD设备上的XFS/EXT4文件系统
- 支持NFS over RDMA (MOFED 5.1+)
- 支持RDMA分布式文件系统(DDN Exascaler, WekaFS, VAST, IBM GPFS)
- 实现GPU与存储设备之间的P2P (Peer-to-Peer) DMA传输

---

## 2. 源文件清单与职责

| 文件 | 职责 |
|------|------|
| `nvfs-mod.c` | **主模块入口** - 模块初始化/退出、字符设备注册、IOCTL路由 |
| `nvfs-core.c` | **核心逻辑** - IO操作管理、DMA地址获取、GPU页面管理 |
| `nvfs-core.h` | 核心头文件 - IOCTL结构定义、常量宏 |
| `nvfs-dma.c` | **DMA操作** - scatter/gather列表映射、块请求处理、ops注册 |
| `nvfs-dma.h` | DMA头文件 - ops结构定义、模块注册表 |
| `nvfs-mmap.c` | **内存映射** - VMA操作、mgroup管理、shadow buffer |
| `nvfs-mmap.h` | mmap头文件 - 状态机、mgroup结构定义 |
| `nvfs-pci.c` | **PCI拓扑** - GPU-Peer距离计算、NUMA亲和性 |
| `nvfs-pci.h` | PCI头文件 - pdevinfo编码/解码宏 |
| `nvfs-rdma.c` | **RDMA支持** - RDMA注册信息管理 (GPFS专用) |
| `nvfs-rdma.h` | RDMA头文件 |
| `nvfs-batch.c` | **批量IO** - 批量IO请求提交 |
| `nvfs-batch.h` | Batch头文件 |
| `nvfs-proc.c` | **Proc接口** - /proc/driver/nvidia-fs/* 文件 |
| `nvfs-stat.c` | **统计** - 性能计数器、吞吐量/延迟计算 |
| `nvfs-stat.h` | 统计头文件 |
| `nvfs-fault.c` | **故障注入** - debugfs故障注入点 (调试用) |
| `nvfs-fault.h` | Fault头文件 |
| `nvfs-kernel-interface.c` | **内核兼容** - 内核API版本适配 |
| `nvfs-kernel-interface.h` | 内核接口头文件 |
| `nvfs-p2p.h` | **NVIDIA P2P** - NVIDIA P2P API封装 |
| `nvfs-vers.h` | **版本定义** - 驱动版本号 |

---

## 3. 模块调用关系图

```
                            ┌─────────────────────────────────────────┐
                            │           User Space (cuFile)          │
                            │    ioctl(NVFS_IOCTL_MAP/READ/WRITE)    │
                            └─────────────────┬───────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              nvfs-mod.c                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ nvfs_init() → register_chrdev() + device_create() + nvfs_proc_init()│   │
│  │ nvfs_ioctl() ───────────────────────────────────────────────────────┼───┼──► nvfs_map()
│  │                    │                                                 │   └──► nvfs_io_init()
│  │                    ├── NVFS_IOCTL_READ/WRITE ────────────────────────┼───┼──► nvfs_io_start_op()
│  │                    │                                                 │   └──► nvfs_io_free()
│  │                    └── NVFS_IOCTL_BATCH_IO ──────────────────────────┼───┼──► nvfs_io_batch_submit()
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              │                               │                               │
              ▼                               ▼                               ▼
┌───────────────────────┐    ┌───────────────────────┐    ┌───────────────────────┐
│     nvfs-mmap.c       │    │     nvfs-core.c       │    │     nvfs-batch.c      │
│                       │    │                       │    │                       │
│ nvfs_mgroup_mmap()    │    │ nvfs_io_init()        │    │ nvfs_io_batch_init()  │
│ nvfs_mgroup_get()     │◄───│ nvfs_io_start_op()    │───►│ nvfs_io_batch_submit()│
│ nvfs_mgroup_put()     │    │ nvfs_get_dma()        │    │                       │
│ nvfs_vma_close()      │    │ nvfs_direct_io()      │    │                       │
│ nvfs_mgroup_from_page()    │ nvfs_pin_gpu_pages() │    │                       │
└───────────┬───────────┘    └───────────┬───────────┘    └───────────────────────┘
            │                            │
            │                            │
            ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              nvfs-p2p.h                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ nvidia_p2p_get_pages()      → 获取GPU物理页面                         │  │
│  │ nvidia_p2p_dma_map_pages()  → 建立DMA映射                             │  │
│  │ nvidia_p2p_put_pages()      → 释放GPU页面                             │  │
│  │ nvidia_p2p_dma_unmap_pages()→ 解除DMA映射                             │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              nvfs-dma.c                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        DMA Ops Registration                           │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐ │  │
│  │  │nvme_ops     │ │nvme_rdma_ops│ │sfxv_ops     │ │nvmesh_ops       │ │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────┘ │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐ │  │
│  │  │lustre_ops   │ │beegfs_ops   │ │gpfs_ops     │ │rpcrdma_ops      │ │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                          Key Functions                                │  │
│  │  nvfs_blk_rq_map_sg()     → 构建scatter/gather列表                   │  │
│  │  nvfs_dma_map_sg_attrs()  → DMA地址映射                              │  │
│  │  nvfs_dma_unmap_sg()      → DMA地址解除映射                          │  │
│  │  nvfs_is_gpu_page()       → 判断是否为GPU页面                        │  │
│  │  nvfs_device_priority()   → 设备优先级(PCI距离)                      │  │
│  │  nvfs_blk_rq_dma_map_iter_start/next() → 迭代器API (6.17+内核)      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              nvfs-pci.c                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ nvfs_fill_gpu2peer_distance_table_once() → 初始化PCI拓扑距离矩阵      │  │
│  │ nvfs_get_gpu2peer_distance()              → 获取GPU-Peer距离          │  │
│  │ nvfs_update_peer_usage()                  → 更新Peer使用统计          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 核心数据结构

### 4.1 nvfs_io_mgroup (内存组)

```c
struct nvfs_io_mgroup {
    atomic_t ref;                      // 引用计数
    atomic_t dma_ref;                  // DMA引用计数
    struct hlist_node hash_link;       // 哈希表链接
    u64 cpu_base_vaddr;                // CPU虚拟地址基址
    unsigned long base_index;          // 基础索引
    unsigned long nvfs_blocks_count;   // 4K块数量
    struct page **nvfs_ppages;         // Shadow buffer页面数组
    struct nvfs_io_metadata *nvfs_metadata; // 元数据数组
    struct nvfs_gpu_args gpu_info;     // GPU参数
    nvfs_io_t nvfsio;                  // IO结构
    struct nvfs_rdma_info rdma_info;   // RDMA信息 (可选)
    atomic_t next_segment;             // 下一段索引
};
```

### 4.2 nvfs_gpu_args (GPU参数)

```c
struct nvfs_gpu_args {
    nvidia_p2p_page_table_t *page_table;  // P2P页面表
    u64 gpuvaddr;                          // GPU虚拟地址
    u64 gpu_buf_len;                       // 缓冲区长度
    struct page *end_fence_page;           // Fence页面
    u32 offset_in_page;                    // 页内偏移
    atomic_t io_state;                     // IO状态机
    atomic_t dma_mapping_in_progress;      // DMA映射进行中标志
    wait_queue_head_t callback_wq;         // 等待队列
    bool is_bounce_buffer;                 // 是否为bounce buffer
    int n_phys_chunks;                     // 物理块数量
    u64 pdevinfo;                          // PCI设备信息
    DECLARE_HASHTABLE(buckets, 5);         // PCI设备映射哈希表
};
```

### 4.3 nvfs_io (IO请求)

```c
typedef struct nvfs_io {
    char __user *cpuvaddr;              // Shadow buffer地址
    u64 length;                         // IO长度
    ssize_t ret;                        // 返回值
    loff_t fd_offset;                   // 文件偏移
    loff_t gpu_page_offset;             // GPU页偏移
    u64 end_fence_value;                // Fence值
    struct fd fd;                       // 文件描述符
    int op;                             // READ/WRITE
    bool sync;                          // 同步标志
    bool hipri;                         // 高优先级
    bool check_sparse;                  // 检查稀疏文件
    unsigned long cur_gpu_base_index;   // 当前GPU基索引
    struct kiocb common;                // 内核IO控制块
    ktime_t start_io;                   // 开始时间
    ssize_t rdma_seg_offset;            // RDMA段偏移
    bool use_rkeys;                     // 使用RDMA rkey
} nvfs_io_t;
```

### 4.4 nvfs_dma_rw_ops (DMA操作)

```c
struct nvfs_dma_rw_ops {
    unsigned long long ft_bmap;         // 特性位图

    // Scatter/Gather操作
    int (*nvfs_blk_rq_map_sg)(struct request_queue *, struct request *, struct scatterlist *);
    int (*nvfs_dma_map_sg_attrs)(struct device *, struct scatterlist *, int, enum dma_data_direction, unsigned long);
    int (*nvfs_dma_unmap_sg)(struct device *, struct scatterlist *, int, enum dma_data_direction);

    // GPU页面识别
    bool (*nvfs_is_gpu_page)(struct page *);
    unsigned int (*nvfs_gpu_index)(struct page *);
    unsigned int (*nvfs_device_priority)(struct device *, unsigned int);

    // RDMA信息获取
    int (*nvfs_get_gpu_sglist_rdma_info)(struct scatterlist *, int, struct nvfs_rdma_info *);
};
```

---

## 5. IO状态机

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            IO State Machine                                  │
└──────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────┐
                    │   IO_FREE   │ ←─────────────────────────────┐
                    └──────┬──────┘                               │
                           │ mmap()                                │
                           ▼                                       │
                    ┌─────────────┐                                │
                    │   IO_ALLOC  │                                │
                    └──────┬──────┘                                │
                           │ nvfs_mgroup_fill_mpages()             │
                           ▼                                       │
                    ┌─────────────┐                                │
           ┌──────►│   IO_INIT   │◄──────┐                         │
           │       └──────┬──────┘       │                         │
           │              │ nvfs_map()   │                         │
           │              ▼              │ nvfs_io_free()          │
           │       ┌─────────────┐       │                         │
           │       │   IO_READY  │───────┘                         │
           │       └──────┬──────┘                                 │
           │              │ nvfs_io_start_op()                     │
           │              ▼                                        │
           │       ┌─────────────┐                                 │
           │       │IO_IN_PROGRESS│                                │
           │       └──────┬──────┘                                 │
           │              │                                        │
           │    ┌─────────┴─────────┐                              │
           │    │                   │                              │
           │    ▼                   ▼                              │
           │ ┌──────────────┐ ┌────────────────┐                   │
           │ │IO_TERMINATE_REQ│ │IO_CALLBACK_REQ │                  │
           │ └──────┬───────┘ └───────┬────────┘                  │
           │        │                 │                            │
           │        └────────┬────────┘                            │
           │                 ▼                                     │
           │          ┌──────────────┐                             │
           │          │ IO_TERMINATED│─────────────────────────────┤
           │          └──────────────┘                             │
           │                 │                                     │
           │                 ▼                                     │
           │          ┌──────────────┐                             │
           └──────────│IO_CALLBACK_END│                            │
           (reset)    └──────┬───────┘                             │
                           │ nvfs_free_gpu_info()                  │
                           ▼                                       │
                    ┌────────────────────────┐                     │
                    │IO_UNPIN_PAGES_ALREADY_INVOKED│────────────────┘
                    └────────────────────────┘
```

---

## 6. 核心调用链详解

### 6.1 模块初始化流程

```
nvfs_init()                                    [nvfs-mod.c:2503]
    │
    ├── get_nvidia_driver_version()            检查NVIDIA驱动版本
    │       └── 决定使用 legacy 或 persistent P2P API
    │
    ├── register_chrdev()                      注册字符设备
    │
    ├── class_create()                         创建设备类
    │
    ├── device_create() × N                    创建多个设备节点
    │
    ├── nvfs_mgroup_init()                     [nvfs-mmap.c:825] 初始化mgroup哈希表
    │
    ├── nvfs_proc_init()                       [nvfs-proc.c:241] 创建/proc接口
    │       ├── /proc/driver/nvidia-fs/version
    │       ├── /proc/driver/nvidia-fs/stats
    │       ├── /proc/driver/nvidia-fs/peer_distance
    │       └── /proc/driver/nvidia-fs/peer_affinity
    │
    ├── nvfs_stat_init()                       [nvfs-stat.c:602] 初始化统计
    │
    └── nvfs_fill_gpu2peer_distance_table_once() [nvfs-pci.c:645]
            │
            ├── __nvfs_find_all_device_paths(GPU)  扫描GPU设备
            ├── __nvfs_find_all_device_paths(IB)   扫描IB设备
            ├── __nvfs_find_all_device_paths(NVMe) 扫描NVMe设备
            └── nvfs_get_pci_gpu2peer_distance()   计算距离矩阵
```

### 6.2 内存映射流程 (NVFS_IOCTL_MAP)

```
nvfs_ioctl(NVFS_IOCTL_MAP)                     [nvfs-mod.c:2350]
    │
    └── nvfs_map()                             [nvfs-core.c:1505]
            │
            ├── nvfs_mgroup_pin_shadow_pages() [nvfs-mmap.c:347]
            │       │
            │       ├── pin_user_pages_fast()  固定用户页面
            │       │
            │       └── nvfs_mgroup_get()      获取/创建mgroup
            │
            ├── nvfs_map_gpu_info()            [nvfs-core.c:1481]
            │       │
            │       ├── nvfs_get_endfence_page() 固定fence页面
            │       │
            │       └── nvfs_pin_gpu_pages()   [nvfs-core.c:1241]
            │               │
            │               ├── nvfs_transit_state(IO_FREE → IO_INIT)
            │               │
            │               ├── nvidia_p2p_get_pages() /
            │               │   nvidia_p2p_get_pages_persistent()
            │               │       │
            │               │       └── 注册回调: nvfs_get_pages_free_callback()
            │               │
            │               └── nvfs_update_alloc_gpustat() 更新GPU统计
            │
            └── nvfs_transit_state(IO_INIT → IO_READY)
```

### 6.3 IO操作流程 (NVFS_IOCTL_READ/WRITE)

```
nvfs_ioctl(NVFS_IOCTL_READ/WRITE)              [nvfs-mod.c:2234]
    │
    ├── nvfs_io_init()                         [nvfs-core.c:1586]
    │       │
    │       ├── fdget()                        获取文件描述符
    │       │
    │       ├── nvfs_check_file_permissions()  检查文件权限
    │       │
    │       ├── nvfs_get_mgroup_from_vaddr()   [nvfs-mmap.c:303]
    │       │       │
    │       │       └── nvfs_mgroup_from_page()
    │       │
    │       ├── nvfs_transit_state(IO_READY → IO_IN_PROGRESS)
    │       │
    │       └── 初始化nvfsio结构
    │
    └── nvfs_io_start_op()                     [nvfs-core.c:1973]
            │
            ├── [WRITE] flush_dirty_pages()    刷新脏页
            │
            ├── nvfs_mgroup_fill_mpages()      填充元数据页
            │
            └── nvfs_direct_io()               [nvfs-core.c:994]
                    │
                    ├── nvfs_rw_verify_area()  验证IO区域
                    │
                    ├── init_sync_kiocb()      初始化kiocb
                    │
                    ├── [设置异步回调]
                    │       └── nvfsio->common.ki_complete = nvfs_io_complete
                    │
                    └── call_read_iter() / call_write_iter()
                            │
                            └── 触发块层IO ──────────────────┐
                                                              │
                                                              ▼
                                            ┌─────────────────────────────────┐
                                            │        Block Layer IO          │
                                            │                                │
                                            │ nvfs_blk_rq_map_sg()           │
                                            │     ↓                          │
                                            │ nvfs_dma_map_sg_attrs()        │
                                            │     ↓                          │
                                            │ nvfs_get_dma()                 │
                                            │     ↓                          │
                                            │ nvfs_get_p2p_dma_mapping()     │
                                            │     ↓                          │
                                            │ nvidia_p2p_dma_map_pages()     │
                                            └─────────────────────────────────┘
                                                              │
                                                              ▼
                                            ┌─────────────────────────────────┐
                                            │     nvfs_io_complete()         │
                                            │     [nvfs-core.c:812]         │
                                            │                                │
                                            │ ├── 更新统计                    │
                                            │ └── nvfs_io_free()             │
                                            │         ↓                      │
                                            │     nvfs_mgroup_put()          │
                                            │     nvfs_transit_state()       │
                                            │     [异步] 更新fence页面       │
                                            └─────────────────────────────────┘
```

### 6.4 DMA映射调用链

```
Block Layer (NVMe Driver)
    │
    └── nvfs_blk_rq_map_sg()                   [nvfs-dma.c:516]
            │
            ├── nvfs_blk_rq_check()            验证请求
            │
            ├── rq_for_each_segment()
            │       │
            │       ├── nvfs_mgroup_from_page() 获取mgroup
            │       │
            │       ├── nvfs_mgroup_metadata_set_dma_state() 设置DMA状态
            │       │
            │       ├── nvfs_mgroup_get_gpu_physical_address() 获取物理地址
            │       │
            │       └── 构建scatter/gather条目
            │
            └── 返回GPU页面数或0(CPU页面)

Block Layer DMA Mapping
    │
    └── nvfs_dma_map_sg_attrs()                [nvfs-dma.c:1469]
            │
            ├── for_each_sg()
            │       │
            │       ├── nvfs_get_dma()         [nvfs-core.c:561]
            │       │       │
            │       │       ├── nvfs_mgroup_from_page()
            │       │       │
            │       │       ├── nvfs_get_p2p_dma_mapping()
            │       │       │       │
            │       │       │       └── nvidia_p2p_dma_map_pages()
            │       │       │
            │       │       └── 设置sg_dma_address/sg_dma_len
            │       │
            │       └── nvfs_mgroup_metadata_set_dma_state()
            │
            └── 返回映射的条目数
```

### 6.5 设备优先级计算

```
nvfs_device_priority()                         [nvfs-mmap.c:1400]
    │
    └── nvfs_get_gpu2peer_distance()           [nvfs-pci.c:673]
            │
            ├── nvfs_get_peer_hash_index()     获取Peer索引
            │
            └── gpu_rank_matrix[gpu_index][peer_index].rank
                    │
                    └── 计算公式:
                        rank = (MAX_PCIE_BW_INDEX - bw) | (pci_dist << 16)
                        其中:
                        - bw = link_width × link_speed
                        - pci_dist = PCI拓扑距离
```

---

## 7. VMA操作与生命周期管理

### 7.1 VMA操作结构

```c
static const struct vm_operations_struct nvfs_mmap_ops = {
    .open        = nvfs_vma_open,      // 禁止: VMA复制
    .close       = nvfs_vma_close,     // 清理: munmap时释放资源
    .split       = nvfs_vma_split,     // 禁止: VMA分割
    .mremap      = nvfs_vma_mremap,    // 禁止: VMA重映射
    .fault       = nvfs_vma_fault,     // 禁止: 页面错误
    .pfn_mkwrite = nvfs_pfn_mkwrite,   // 禁止: 写保护
    .page_mkwrite= nvfs_page_mkwrite,  // 禁止: 写保护
};
```

### 7.2 VMA关闭流程

```
munmap() → nvfs_vma_close()                    [nvfs-mmap.c:500]
    │
    ├── 检查IO状态
    │       └── 如果 IO_CALLBACK_END: callback已执行
    │
    ├── nvfs_io_terminate_requested()          请求IO终止
    │
    ├── [如果IO_TERMINATED]
    │       │
    │       ├── nvfs_mgroup_unpin_shadow_pages()
    │       │
    │       └── nvfs_stat64(&nvfs_n_free)
    │
    └── nvfs_mgroup_put()                      释放mgroup引用
            │
            └── nvfs_mgroup_free()             [nvfs-mmap.c:125]
                    │
                    ├── nvfs_free_gpu_info()   [nvfs-core.c:1427]
                    │       │
                    │       ├── nvfs_unpin_gpu_pages()
                    │       │       │
                    │       │       ├── nvidia_p2p_dma_unmap_pages()
                    │       │       │
                    │       │       └── nvidia_p2p_put_pages()
                    │       │
                    │       └── nvfs_free_put_endfence_page()
                    │
                    ├── hash_del_rcu()        从哈希表删除
                    │
                    └── kfree(nvfs_mgroup)    释放内存
```

### 7.3 NVIDIA P2P回调

```
GPU驱动调用(cudaFree/进程退出)
    │
    └── nvfs_get_pages_free_callback()         [nvfs-core.c:291]
            │
            ├── nvfs_stat(&nvfs_n_callbacks)
            │
            ├── nvfs_io_terminate()            等待IO完成
            │
            ├── [如果 IO_CALLBACK_END]
            │       │
            │       ├── 清理DMA映射哈希表
            │       │       └── nvfs_nvidia_p2p_free_dma_mapping()
            │       │
            │       ├── nvidia_p2p_free_page_table()
            │       │
            │       ├── 设置metapage状态 = NVFS_IO_META_DIED
            │       │
            │       └── nvfs_mgroup_put()      释放引用
            │
            └── [否则] 等待put_pages调用
```

---

## 8. 支持的存储模块注册表

```c
// nvfs-dma.c: modules_list[]
┌─────────────────────────────────────────────────────────────────────────────┐
│ Module Name        │ Reg Symbol                        │ Ops Type         │
├────────────────────┼───────────────────────────────────┼──────────────────┤
│ nvme               │ nvme_v2_register_nvfs_dma_ops     │ blk_iter_ops     │
│                    │ nvme_v1_register_nvfs_dma_ops     │ rw_ops           │
├────────────────────┼───────────────────────────────────┼──────────────────┤
│ nvme_rdma          │ nvme_rdma_v1_register_nvfs_dma_ops│ rw_ops           │
├────────────────────┼───────────────────────────────────┼──────────────────┤
│ sfxvdriver         │ sfxv_v1_register_nvfs_dma_ops     │ rw_ops           │
├────────────────────┼───────────────────────────────────┼──────────────────┤
│ nvmeib_common      │ nvmesh_v1_register_nvfs_dma_ops   │ rw_ops           │
├────────────────────┼───────────────────────────────────┼──────────────────┤
│ lnet (Lustre)      │ lustre_v1_register_nvfs_dma_ops   │ dev_dma_rw_ops   │
├────────────────────┼───────────────────────────────────┼──────────────────┤
│ beegfs             │ beegfs_v1_register_nvfs_dma_ops   │ dev_dma_rw_ops   │
├────────────────────┼───────────────────────────────────┼──────────────────┤
│ wekafsio           │ (pseudo, no registration)         │ NULL             │
├────────────────────┼───────────────────────────────────┼──────────────────┤
│ mmfslinux (GPFS)   │ ibm_scale_v1_register_nvfs_dma_ops│ ibm_scale_rdma   │
├────────────────────┼───────────────────────────────────┼──────────────────┤
│ rpcrdma (NFS)      │ rpcrdma_register_nvfs_dma_ops     │ dev_dma_rw_ops   │
├────────────────────┼───────────────────────────────────┼──────────────────┤
│ scatefs            │ scatefs_register_nvfs_dma_ops     │ dev_dma_rw_ops   │
├────────────────────┼───────────────────────────────────┼──────────────────┤
│ scsi_mod           │ scsi_v1_register_dma_scsi_ops     │ dev_dma_rw_ops   │
└────────────────────┴───────────────────────────────────┴──────────────────┘
```

---

## 9. 特性位图 (Feature Bitmap)

```c
enum ft_bits {
    nvfs_ft_prep_sglist         = 1ULL << 0,  // 支持blk_rq_map_sg
    nvfs_ft_map_sglist          = 1ULL << 1,  // 支持dma_map_sg
    nvfs_ft_is_gpu_page         = 1ULL << 2,  // 支持GPU页面识别
    nvfs_ft_device_priority     = 1ULL << 3,  // 支持设备优先级
    nvfs_ft_get_gpu_sglist_rdma_info = 1ULL << 4,  // 支持RDMA信息获取
    nvfs_ft_blk_dma_map_iter_start = 1ULL << 5,    // 支持迭代器start
    nvfs_ft_blk_dma_map_iter_next  = 1ULL << 6,    // 支持迭代器next
};

// 默认特性集
#define NVIDIA_FS_SET_FT_ALL  (nvfs_ft_prep_sglist | nvfs_ft_map_sglist | \
                               nvfs_ft_is_gpu_page | nvfs_ft_device_priority | \
                               nvfs_ft_get_gpu_sglist_rdma_info)
```

---

## 10. IOCTL接口

| IOCTL | 代码 | 功能 |
|-------|------|------|
| `NVFS_IOCTL_MAP` | `_IOW('t', 3, int)` | 映射GPU缓冲区 |
| `NVFS_IOCTL_READ` | `_IOW('t', 2, int)` | 同步/异步读操作 |
| `NVFS_IOCTL_WRITE` | `_IOW('t', 4, int)` | 同步/异步写操作 |
| `NVFS_IOCTL_REMOVE` | `_IOW('t', 1, int)` | 移除映射 |
| `NVFS_IOCTL_SET_RDMA_REG_INFO` | `_IOW('t', 5, int)` | 设置RDMA注册信息 |
| `NVFS_IOCTL_GET_RDMA_REG_INFO` | `_IOW('t', 6, int)` | 获取RDMA注册信息 |
| `NVFS_IOCTL_CLEAR_RDMA_REG_INFO` | `_IOW('t', 7, int)` | 清除RDMA注册信息 |
| `NVFS_IOCTL_BATCH_IO` | `_IOW('t', 8, int)` | 批量IO操作 |

---

## 11. 关键常量

```c
#define GPU_PAGE_SIZE          (64KB)       // GPU页面大小
#define NVFS_BLOCK_SIZE        (4KB)        // 基本块大小
#define NVFS_MAX_SHADOW_PAGES  4096         // 最大shadow页面数
#define MAX_PCI_BUCKETS        32           // PCI映射哈希桶数
#define MAX_GPU_DEVS           64           // 最大GPU设备数
#define MAX_PEER_DEVS          64           // 最大Peer设备数
#define MAX_PCI_DEPTH          16           // 最大PCI深度
#define NVME_MAX_SEGS          127          // NVMe最大段数
#define NVFS_P2P_MAX_CONTIG_GPU_PAGES 65535 // 最大连续GPU页面
```

---

## 12. 内核版本兼容性

通过 `config-host.h` 和条件编译处理:

| 特性 | 检测宏 | 适配 |
|------|--------|------|
| `access_ok()` 参数 | `HAVE_ACCESS_OK_2_PARAMS` | 2参数 vs 3参数 |
| `pin_user_pages_fast` | `HAVE_PIN_USER_PAGES_FAST` | 新API vs `get_user_pages_fast` |
| `proc_ops` 结构 | `HAVE_STRUCT_PROC_OPS` | `proc_ops` vs `file_operations` |
| `vm_fault_t` 类型 | `HAVE_VM_FAULT` | 返回类型适配 |
| 迭代器DMA API | `HAVE_BLK_RQ_DMA_MAP_ITER_START` | 6.17+ 新API |
| `ki_complete` 参数 | `KI_COMPLETE_HAS_3_PARAMETERS` | 回调签名 |

---

## 13. 数据流总结

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              完整数据流                                      │
└─────────────────────────────────────────────────────────────────────────────┘

用户空间 (cuFile)
      │
      │ 1. mmap() → 获取shadow buffer
      ▼
┌─────────────┐
│ Shadow Buffer│ ←── CPU可访问的页面 (page->index编码mgroup信息)
│ (4K pages)  │
└──────┬──────┘
       │
       │ 2. ioctl(NVFS_IOCTL_MAP, gpu_vaddr)
       ▼
┌─────────────┐
│ GPU Memory  │ ←── nvidia_p2p_get_pages() 获取物理页面
│ (64K pages) │     page_table->pages[] 包含BAR物理地址
└──────┬──────┘
       │
       │ 3. ioctl(NVFS_IOCTL_READ/WRITE)
       ▼
┌─────────────┐
│  File I/O   │ ←── VFS read_iter/write_iter
└──────┬──────┘
       │
       │ 4. Block Layer
       ▼
┌─────────────┐
│ NVMe Driver │ ←── nvfs_blk_rq_map_sg() 构建SG列表
└──────┬──────┘     nvfs_dma_map_sg_attrs() 获取DMA地址
       │
       │ 5. DMA映射
       ▼
┌─────────────┐
│ nvidia_p2p  │ ←── nvidia_p2p_dma_map_pages()
│ DMA Mapping │     返回 peer → GPU 的DMA地址
└──────┬──────┘
       │
       │ 6. 硬件传输
       ▼
┌─────────────┐
│ NVMe Device │ ══════════════════════════► GPU Memory
└─────────────┘     PCIe P2P DMA传输
```

---

## 14. 文档版本

- **分析日期**: 2024
- **驱动版本**: 2.28.2
- **内核支持**: 4.15+ 至 6.17+
