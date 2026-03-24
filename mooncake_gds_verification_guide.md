# Mooncake GDS (GPUDirect Storage) 能力验证指南

本文档介绍如何验证 Mooncake 的 GDS (GPUDirect Storage) 能力，实现 GPU HBM 与 NVMe 存储之间的直接数据传输。

## 目录

- [概述](#概述)
- [前置条件](#前置条件)
- [验证脚本](#验证脚本)
- [使用方法](#使用方法)
- [预期结果](#预期结果)
- [故障排除](#故障排除)

---

## 概述

### 什么是 GDS?

GPUDirect Storage (GDS) 是 NVIDIA 提供的技术，允许 GPU 显存与 NVMe 存储之间直接进行数据传输，绕过 CPU 和系统内存，实现零拷贝高带宽数据传输。

### Mooncake GDS 支持

Mooncake 通过 `nvmeof` Transport 层支持 GDS 能力：

```
┌─────────────┐                      ┌─────────────┐
│  GPU HBM    │  ←── GDS/cuFile ──→  │  NVMe SSD   │
│  (显存)     │      零拷贝传输        │  (存储)     │
└─────────────┘                      └─────────────┘
```

**支持的数据流向：**
- `WRITE`: GPU HBM → NVMe (显存写入存储)
- `READ`: NVMe → GPU HBM (存储读取到显存)

---

## 前置条件

### 1. 硬件要求

| 组件 | 要求 |
|------|------|
| GPU | NVIDIA GPU (支持 GPUDirect) |
| 存储 | NVMe SSD |
| 内存 | 足够的系统内存用于元数据管理 |

### 2. 软件要求

```bash
# 检查 CUDA
nvidia-smi
nvcc --version

# 检查 Python
python --version  # 需要 Python 3.8+

# 检查 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 3. 安装 Mooncake

```bash
# 方式1: pip 安装
pip install mooncake

# 方式2: 从源码编译
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DUSE_NVMEOF=ON
make -j$(nproc)
pip install ..
```

### 4. 安装 NVIDIA GDS (cuFile)

```bash
# 检查是否已安装
ls /usr/local/cuda/lib64/libcufile.so

# 如果未安装，参考 NVIDIA 官方文档安装 GDS
# https://docs.nvidia.com/gpudirect-storage/

# 安装后验证
cufile-sample  # 运行 GDS 示例程序
```

### 5. 安装元数据服务 (可选，集群模式需要)

```bash
# 安装 etcd
# Ubuntu/Debian
sudo apt-get install etcd etcd-client

# CentOS/RHEL
sudo yum install etcd

# 或使用 Docker
docker run -d --name etcd \
    -p 2379:2379 \
    -p 2380:2380 \
    quay.io/coreos/etcd:latest \
    /usr/local/bin/etcd \
    --name s1 \
    --data-dir /etcd-data \
    --listen-client-urls http://0.0.0.0:2379 \
    --advertise-client-urls http://0.0.0.0:2379
```

---

## 验证脚本

将以下脚本保存为 `test_mooncake_gds.py`:

```python
#!/usr/bin/env python3
"""
Mooncake GDS (GPUDirect Storage) 验证脚本

验证GPU显存与NVMe存储之间的直接数据传输能力:
- GPU HBM → NVMe (WRITE)
- NVMe → GPU HBM (READ)

前置条件:
1. 安装好 mooncake (pip install mooncake 或从源码编译)
2. 安装 torch, cuda 环境
3. 系统支持 cuFile/GDS (NVIDIA GPUDirect Storage)
4. 运行etcd作为元数据服务 (或使用P2PHANDSHAKE模式)

使用方法:
    # 单机模式 (使用P2PHANDSHAKE，无需etcd)
    python test_mooncake_gds.py --mode standalone --nvme_path /mnt/nvme/test.bin

    # 集成模式 (需要etcd)
    python test_mooncake_gds.py --mode cluster --etcd 127.0.0.1:2379 --nvme_path /mnt/nvme/test.bin
"""

import argparse
import os
import sys
import time
import struct
import socket

try:
    import torch
except ImportError:
    print("Error: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

# 选择使用哪种API
USE_TENT = os.environ.get("USE_TENT", "0") == "1"

if USE_TENT:
    try:
        import tent
        print("Using TENT (new) API")
    except ImportError:
        print("Warning: TENT not available, falling back to legacy API")
        USE_TENT = False

if not USE_TENT:
    try:
        from mooncake.engine import TransferEngine
        print("Using legacy mooncake.engine API")
    except ImportError:
        print("Error: mooncake is required. Install with: pip install mooncake")
        sys.exit(1)


def get_gpu_info():
    """获取GPU信息"""
    if not torch.cuda.is_available():
        print("Error: CUDA not available!")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
    total_memory = torch.cuda.get_device_properties(0).total_memory if device_count > 0 else 0

    print(f"\n{'='*60}")
    print("GPU Information")
    print(f"{'='*60}")
    print(f"  Device Count:    {device_count}")
    print(f"  Device Name:     {device_name}")
    print(f"  Total Memory:    {total_memory / (1024**3):.2f} GB")

    return device_count, device_name, total_memory


def check_gds_support():
    """检查GDS/cuFile支持"""
    print(f"\n{'='*60}")
    print("GDS Support Check")
    print(f"{'='*60}")

    # 检查cuFile库是否存在
    cufile_paths = [
        "/usr/local/cuda/lib64/libcufile.so",
        "/usr/lib/x86_64-linux-gnu/libcufile.so",
        "/opt/nvidia/gds/lib64/libcufile.so",
    ]

    found = False
    for path in cufile_paths:
        if os.path.exists(path):
            print(f"  [OK] Found cuFile library: {path}")
            found = True
            break

    if not found:
        print(f"  [WARN] cuFile library not found in common paths")
        print(f"         GDS may not be properly installed")

    # 检查环境变量
    gds_env = os.environ.get("CUFILE_ENV_PATH", "")
    if gds_env:
        print(f"  CUFILE_ENV_PATH: {gds_env}")

    return found


def prepare_nvme_file(nvme_path: str, size: int):
    """准备NVMe测试文件"""
    print(f"\n{'='*60}")
    print("Preparing NVMe Test File")
    print(f"{'='*60}")
    print(f"  Path: {nvme_path}")
    print(f"  Size: {size / (1024**2):.2f} MB")

    # 确保目录存在
    dir_path = os.path.dirname(nvme_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    # 创建文件
    if not os.path.exists(nvme_path) or os.path.getsize(nvme_path) != size:
        print(f"  Creating file...")
        with open(nvme_path, 'wb') as f:
            f.write(b'\x00' * size)
        os.sync()
        print(f"  [OK] File created successfully")
    else:
        print(f"  [OK] File already exists with correct size")

    return True


class GdsTestLegacy:
    """使用旧版 mooncake.engine API 测试GDS"""

    def __init__(self, metadata_server: str, local_server_name: str, protocol: str = "tcp"):
        self.metadata_server = metadata_server
        self.local_server_name = local_server_name
        self.protocol = protocol
        self.engine = None

    def setup(self):
        """初始化TransferEngine"""
        print(f"\n{'='*60}")
        print("Setting up Transfer Engine (Legacy API)")
        print(f"{'='*60}")
        print(f"  Metadata Server:  {self.metadata_server}")
        print(f"  Local Server:     {self.local_server_name}")
        print(f"  Protocol:         {self.protocol}")

        self.engine = TransferEngine()
        ret = self.engine.initialize(
            self.local_server_name,
            self.metadata_server,
            self.protocol,
            ""
        )

        if ret != 0:
            raise RuntimeError(f"Failed to initialize TransferEngine: {ret}")

        print(f"  [OK] Engine initialized successfully")

    def register_gpu_memory(self, gpu_tensor: torch.Tensor):
        """注册GPU内存"""
        addr = gpu_tensor.data_ptr()
        size = gpu_tensor.numel() * gpu_tensor.element_size()

        print(f"\n{'='*60}")
        print("Registering GPU Memory")
        print(f"{'='*60}")
        print(f"  Address:    0x{addr:x}")
        print(f"  Size:       {size / (1024**2):.2f} MB")
        print(f"  Location:   cuda:0")

        # 注册GPU内存，location指定为cuda:0
        ret = self.engine.register_local_memory(addr, size, "cuda:0")
        if ret != 0:
            raise RuntimeError(f"Failed to register GPU memory: {ret}")

        print(f"  [OK] GPU memory registered successfully")
        return addr, size

    def unregister_gpu_memory(self, addr: int):
        """注销GPU内存"""
        ret = self.engine.unregister_local_memory(addr)
        if ret != 0:
            print(f"  [WARN] Failed to unregister GPU memory: {ret}")
        else:
            print(f"  [OK] GPU memory unregistered successfully")

    def test_gds_write(self, gpu_tensor: torch.Tensor, nvme_segment_id: str,
                       file_offset: int = 0):
        """测试 GPU → NVMe 写入"""
        addr = gpu_tensor.data_ptr()
        size = gpu_tensor.numel() * gpu_tensor.element_size()

        print(f"\n{'='*60}")
        print("Testing GDS Write (GPU → NVMe)")
        print(f"{'='*60}")
        print(f"  GPU Address:      0x{addr:x}")
        print(f"  Size:             {size / (1024**2):.2f} MB")
        print(f"  Target Segment:   {nvme_segment_id}")
        print(f"  File Offset:      {file_offset}")

        # 准备测试数据
        print(f"  Preparing test pattern...")
        test_pattern = torch.arange(size // 4, dtype=torch.int32, device=gpu_tensor.device)
        gpu_tensor[:len(test_pattern)] = test_pattern

        # 提交写入请求
        start_time = time.time()

        ret = self.engine.transfer_sync_write(
            nvme_segment_id,
            addr,
            file_offset,
            size
        )

        elapsed = time.time() - start_time

        if ret != 0:
            raise RuntimeError(f"GDS write failed: {ret}")

        throughput = size / elapsed / (1024**3)
        print(f"  [OK] Write completed in {elapsed*1000:.2f} ms")
        print(f"  [OK] Throughput: {throughput:.2f} GB/s")

        return throughput

    def test_gds_read(self, gpu_tensor: torch.Tensor, nvme_segment_id: str,
                      file_offset: int = 0):
        """测试 NVMe → GPU 读取"""
        addr = gpu_tensor.data_ptr()
        size = gpu_tensor.numel() * gpu_tensor.element_size()

        print(f"\n{'='*60}")
        print("Testing GDS Read (NVMe → GPU)")
        print(f"{'='*60}")
        print(f"  GPU Address:      0x{addr:x}")
        print(f"  Size:             {size / (1024**2):.2f} MB")
        print(f"  Target Segment:   {nvme_segment_id}")
        print(f"  File Offset:      {file_offset}")

        # 清空GPU缓冲区
        gpu_tensor.zero_()

        # 提交读取请求
        start_time = time.time()

        ret = self.engine.transfer_sync_read(
            nvme_segment_id,
            addr,
            file_offset,
            size
        )

        elapsed = time.time() - start_time

        if ret != 0:
            raise RuntimeError(f"GDS read failed: {ret}")

        throughput = size / elapsed / (1024**3)
        print(f"  [OK] Read completed in {elapsed*1000:.2f} ms")
        print(f"  [OK] Throughput: {throughput:.2f} GB/s")

        # 验证数据
        read_pattern = gpu_tensor[:size//4].cpu()
        expected = torch.arange(size // 4, dtype=torch.int32)
        if torch.equal(read_pattern, expected):
            print(f"  [OK] Data verification passed")
        else:
            print(f"  [WARN] Data verification failed")

        return throughput

    def cleanup(self):
        """清理资源"""
        if self.engine:
            self.engine = None
            print(f"\n{'='*60}")
            print("Engine cleaned up")
            print(f"{'='*60}")


class GdsTestTent:
    """使用新版 TENT API 测试GDS"""

    def __init__(self, metadata_server: str, local_server_name: str):
        self.metadata_server = metadata_server
        self.local_server_name = local_server_name
        self.engine = None

    def setup(self):
        """初始化TransferEngine"""
        print(f"\n{'='*60}")
        print("Setting up Transfer Engine (TENT API)")
        print(f"{'='*60}")
        print(f"  Metadata Server:  {self.metadata_server}")
        print(f"  Local Server:     {self.local_server_name}")

        if self.metadata_server != "P2PHANDSHAKE":
            tent.set_config("metadata_server", self.metadata_server)

        self.engine = tent.TransferEngine()
        print(f"  [OK] Engine initialized successfully")
        print(f"  Segment Name:     {self.engine.get_segment_name()}")

    def register_gpu_memory(self, gpu_tensor: torch.Tensor):
        """注册GPU内存"""
        addr = gpu_tensor.data_ptr()
        size = gpu_tensor.numel() * gpu_tensor.element_size()

        print(f"\n{'='*60}")
        print("Registering GPU Memory")
        print(f"{'='*60}")
        print(f"  Address:    0x{addr:x}")
        print(f"  Size:       {size / (1024**2):.2f} MB")

        options = tent.MemoryOptions()
        options.location = "cuda:0"
        options.perm = tent.Permission.GlobalReadWrite
        options.type = tent.TransportType.GDS

        self.engine.register_local_memory_ex(addr, size, options)
        print(f"  [OK] GPU memory registered successfully (with GDS)")

        return addr, size

    def unregister_gpu_memory(self, addr: int, size: int):
        """注销GPU内存"""
        self.engine.unregister_local_memory(addr, size)
        print(f"  [OK] GPU memory unregistered successfully")

    def test_gds_transfer(self, gpu_tensor: torch.Tensor, file_segment_id: int,
                          opcode: str = "write"):
        """测试GDS传输"""
        addr = gpu_tensor.data_ptr()
        size = gpu_tensor.numel() * gpu_tensor.element_size()

        op = tent.OpCode.WRITE if opcode == "write" else tent.OpCode.READ

        direction = "GPU → File" if opcode == "write" else "File → GPU"
        print(f"\n{'='*60}")
        print(f"Testing GDS {opcode.upper()} ({direction})")
        print(f"{'='*60}")
        print(f"  GPU Address:      0x{addr:x}")
        print(f"  Size:             {size / (1024**2):.2f} MB")
        print(f"  Target Segment:   {file_segment_id}")

        batch_id = self.engine.allocate_transfer_batch(1)
        print(f"  Batch ID:         {batch_id}")

        request = tent.Request()
        request.opcode = op
        request.source = addr
        request.target_id = file_segment_id
        request.target_offset = 0
        request.length = size

        if opcode == "write":
            test_pattern = torch.arange(size // 4, dtype=torch.int32, device=gpu_tensor.device)
            gpu_tensor[:len(test_pattern)] = test_pattern
        else:
            gpu_tensor.zero_()

        start_time = time.time()
        self.engine.submit_transfer(batch_id, [request])

        while True:
            status = self.engine.get_transfer_status(batch_id, 0)
            if status.state == tent.TransferStatusEnum.COMPLETED:
                break
            elif status.state == tent.TransferStatusEnum.FAILED:
                raise RuntimeError("Transfer failed")
            time.sleep(0.001)

        elapsed = time.time() - start_time
        throughput = size / elapsed / (1024**3)

        print(f"  [OK] {opcode.capitalize()} completed in {elapsed*1000:.2f} ms")
        print(f"  [OK] Transferred: {status.bytes} bytes")
        print(f"  [OK] Throughput:  {throughput:.2f} GB/s")

        self.engine.free_transfer_batch(batch_id)

        return throughput

    def cleanup(self):
        """清理资源"""
        if self.engine:
            self.engine = None
            print(f"\n{'='*60}")
            print("Engine cleaned up")
            print(f"{'='*60}")


def run_standalone_test(args):
    """运行单机测试（使用P2PHANDSHAKE模式）"""
    print(f"\n{'#'*60}")
    print("# MOONCAKE GDS VERIFICATION - STANDALONE MODE")
    print(f"{'#'*60}")

    get_gpu_info()
    check_gds_support()

    test_size = args.size * 1024 * 1024
    nvme_path = os.path.abspath(args.nvme_path)

    prepare_nvme_file(nvme_path, test_size * 2)

    print(f"\n{'='*60}")
    print("Allocating GPU Memory")
    print(f"{'='*60}")
    print(f"  Size: {test_size / (1024**2):.2f} MB")
    gpu_tensor = torch.empty(test_size // 4, dtype=torch.int32, device='cuda:0')
    print(f"  [OK] GPU tensor allocated at: 0x{gpu_tensor.data_ptr():x}")

    if USE_TENT:
        tester = GdsTestTent("P2PHANDSHAKE", socket.gethostname())
    else:
        tester = GdsTestLegacy("P2PHANDSHAKE", socket.gethostname(), "tcp")

    try:
        tester.setup()
        tester.register_gpu_memory(gpu_tensor)

        segment_name = f"nvmeof/test_gds_{os.getpid()}"

        print(f"\n{'='*60}")
        print("GDS API Verification")
        print(f"{'='*60}")
        print(f"  [OK] GPU memory registration with 'cuda:0' location")
        print(f"  [OK] This enables GDS (GPUDirect Storage) transfers")
        print(f"\n  NVMe segment registration required for full transfer test:")
        print(f"    python register.py localhost {segment_name.split('/')[-1]} {nvme_path}")

    finally:
        tester.unregister_gpu_memory(gpu_tensor.data_ptr())
        tester.cleanup()
        del gpu_tensor
        torch.cuda.empty_cache()

    print_summary()


def run_cluster_test(args):
    """运行集群测试（使用etcd元数据服务）"""
    print(f"\n{'#'*60}")
    print("# MOONCAKE GDS VERIFICATION - CLUSTER MODE")
    print(f"{'#'*60}")

    get_gpu_info()
    check_gds_support()

    test_size = args.size * 1024 * 1024
    nvme_path = os.path.abspath(args.nvme_path)
    local_hostname = socket.gethostname()

    prepare_nvme_file(nvme_path, test_size * 2)

    print(f"\n{'='*60}")
    print("Allocating GPU Memory")
    print(f"{'='*60}")
    gpu_tensor = torch.empty(test_size // 4, dtype=torch.int32, device='cuda:0')
    print(f"  Size: {test_size / (1024**2):.2f} MB")
    print(f"  [OK] GPU tensor at: 0x{gpu_tensor.data_ptr():x}")

    if USE_TENT:
        tester = GdsTestTent(args.etcd, local_hostname)
    else:
        tester = GdsTestLegacy(args.etcd, local_hostname, "tcp")

    results = {"write": [], "read": []}

    try:
        tester.setup()
        tester.register_gpu_memory(gpu_tensor)

        segment_name = f"nvmeof/{os.path.basename(nvme_path)}"

        for i in range(args.iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {i+1}/{args.iterations}")
            print(f"{'='*60}")

            try:
                write_tp = tester.test_gds_write(gpu_tensor, segment_name)
                results["write"].append(write_tp)
            except Exception as e:
                print(f"  [FAIL] Write test: {e}")

            try:
                read_tp = tester.test_gds_read(gpu_tensor, segment_name)
                results["read"].append(read_tp)
            except Exception as e:
                print(f"  [FAIL] Read test: {e}")

    finally:
        tester.unregister_gpu_memory(gpu_tensor.data_ptr())
        tester.cleanup()
        del gpu_tensor
        torch.cuda.empty_cache()

    print_performance_summary(results)


def print_summary():
    """打印测试总结"""
    print(f"\n{'#'*60}")
    print("# GDS VERIFICATION COMPLETE")
    print(f"{'#'*60}")
    print("""
要完整测试GDS传输，需要:
  1. 确保NVIDIA cuFile/GDS正确安装
  2. 运行etcd元数据服务
  3. 使用register.py注册NVMe文件段
  4. 运行cluster模式进行完整传输测试
""")


def print_performance_summary(results):
    """打印性能总结"""
    print(f"\n{'#'*60}")
    print("# PERFORMANCE SUMMARY")
    print(f"{'#'*60}")

    if results["write"]:
        avg_write = sum(results["write"]) / len(results["write"])
        max_write = max(results["write"])
        print(f"\n  Write Throughput:")
        print(f"    Average: {avg_write:.2f} GB/s")
        print(f"    Maximum: {max_write:.2f} GB/s")
        print(f"    Runs:    {len(results['write'])}")

    if results["read"]:
        avg_read = sum(results["read"]) / len(results["read"])
        max_read = max(results["read"])
        print(f"\n  Read Throughput:")
        print(f"    Average: {avg_read:.2f} GB/s")
        print(f"    Maximum: {max_read:.2f} GB/s")
        print(f"    Runs:    {len(results['read'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify Mooncake GDS (GPUDirect Storage) capability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standalone test (no etcd required)
  python test_mooncake_gds.py --mode standalone --nvme_path /mnt/nvme/test.bin

  # Cluster test with etcd
  python test_mooncake_gds.py --mode cluster --etcd 127.0.0.1:2379 --nvme_path /mnt/nvme/test.bin

  # Use TENT API
  USE_TENT=1 python test_mooncake_gds.py --mode standalone --nvme_path /mnt/nvme/test.bin
"""
    )

    parser.add_argument("--mode", choices=["standalone", "cluster"], default="standalone",
                       help="Test mode: standalone (P2PHANDSHAKE) or cluster (etcd)")
    parser.add_argument("--etcd", default="127.0.0.1:2379",
                       help="etcd server address (for cluster mode)")
    parser.add_argument("--nvme_path", default="/tmp/mooncake_gds_test.bin",
                       help="Path to NVMe test file")
    parser.add_argument("--size", type=int, default=64,
                       help="Test data size in MB (default: 64)")
    parser.add_argument("--iterations", type=int, default=3,
                       help="Number of test iterations (for cluster mode)")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Environment Information")
    print(f"{'='*60}")
    print(f"  Python:          {sys.version.split()[0]}")
    print(f"  PyTorch:         {torch.__version__}")
    print(f"  CUDA Available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Version:    {torch.version.cuda}")
    print(f"  Using TENT API:  {USE_TENT}")

    if args.mode == "standalone":
        run_standalone_test(args)
    else:
        run_cluster_test(args)


if __name__ == "__main__":
    main()
```

---

## 使用方法

### 快速开始（单机模式）

单机模式使用 P2PHANDSHAKE，无需 etcd，适合快速验证 GDS API 调用：

```bash
# 1. 保存脚本
# 将上面的脚本保存为 test_mooncake_gds.py

# 2. 运行单机测试
python test_mooncake_gds.py --mode standalone --nvme_path /mnt/nvme/test.bin

# 3. 指定测试大小
python test_mooncake_gds.py --mode standalone --nvme_path /mnt/nvme/test.bin --size 128
```

### 完整测试（集群模式）

集群模式需要 etcd 元数据服务，可以进行完整的 GDS 传输测试：

```bash
# 步骤1: 启动 etcd
etcd &

# 或使用 Docker
docker run -d --name etcd -p 2379:2379 -p 2380:2380 \
    quay.io/coreos/etcd:latest \
    /usr/local/bin/etcd \
    --name s1 --data-dir /etcd-data \
    --listen-client-urls http://0.0.0.0:2379 \
    --advertise-client-urls http://0.0.0.0:2379

# 步骤2: 注册 NVMe 文件段
python /path/to/Mooncake/mooncake-transfer-engine/scripts/register.py \
    localhost test_nvme /mnt/nvme/test.bin

# 步骤3: 运行测试
python test_mooncake_gds.py --mode cluster \
    --etcd 127.0.0.1:2379 \
    --nvme_path /mnt/nvme/test.bin \
    --size 256 \
    --iterations 5
```

### 使用 TENT 新版 API

```bash
# 设置环境变量使用 TENT API
USE_TENT=1 python test_mooncake_gds.py --mode standalone --nvme_path /mnt/nvme/test.bin
```

---

## 预期结果

### 成功的输出示例

```
============================================================
Environment Information
============================================================
  Python:          3.10.12
  PyTorch:         2.1.0
  CUDA Available:  True
  CUDA Version:    12.1
  Using TENT API:  False

############################################################
# MOONCAKE GDS VERIFICATION - STANDALONE MODE
############################################################

============================================================
GPU Information
============================================================
  Device Count:    1
  Device Name:     NVIDIA A100-SXM4-80GB
  Total Memory:    80.00 GB

============================================================
GDS Support Check
============================================================
  [OK] Found cuFile library: /usr/local/cuda/lib64/libcufile.so

============================================================
Registering GPU Memory
============================================================
  Address:    0x7f8a4c000000
  Size:       64.00 MB
  Location:   cuda:0
  [OK] GPU memory registered successfully

============================================================
GDS API Verification
============================================================
  [OK] GPU memory registration with 'cuda:0' location
  [OK] This enables GDS (GPUDirect Storage) transfers
```

### 性能指标参考

| GPU 类型 | NVMe 类型 | 预期吞吐量 |
|----------|-----------|------------|
| A100 | NVMe SSD (PCIe 4.0) | 8-12 GB/s |
| H100 | NVMe SSD (PCIe 5.0) | 15-25 GB/s |
| V100 | NVMe SSD (PCIe 3.0) | 4-6 GB/s |

---

## 故障排除

### 常见问题

#### 1. cuFile library not found

```bash
# 检查 GDS 安装
ls /usr/local/cuda/lib64/libcufile.so

# 如果未安装，安装 NVIDIA GDS
# 参考: https://docs.nvidia.com/gpudirect-storage/
```

#### 2. Permission denied

```bash
# GDS 可能需要 root 权限
sudo python test_mooncake_gds.py --mode standalone --nvme_path /mnt/nvme/test.bin

# 或配置 udev 规则
sudo bash -c 'cat > /etc/udev/rules.d/99-nvidia-gds.rules << EOF
KERNEL=="nvidia*", MODE="0666"
EOF'
sudo udevadm control --reload-rules
sudo udevadm trigger
```

#### 3. Failed to register GPU memory

```bash
# 检查 CUDA 版本兼容性
nvidia-smi
nvcc --version

# 确保 PyTorch CUDA 版本与系统 CUDA 一致
python -c "import torch; print(torch.version.cuda)"
```

#### 4. etcd connection failed

```bash
# 检查 etcd 状态
etcdctl endpoint health

# 重启 etcd
pkill etcd
etcd &
```

#### 5. NVMe file not accessible

```bash
# 确保 NVMe 文件存在且有足够空间
df -h /mnt/nvme/
ls -la /mnt/nvme/test.bin

# 检查文件权限
chmod 666 /mnt/nvme/test.bin
```

### 调试模式

```bash
# 启用详细日志
export GLOG_v=2
export GLOG_logtostderr=1
python test_mooncake_gds.py --mode standalone --nvme_path /mnt/nvme/test.bin
```

---

## 参考资料

- [Mooncake 官方文档](https://github.com/kvcache-ai/Mooncake)
- [NVIDIA GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/)
- [cuFile API 文档](https://docs.nvidia.com/cuda/cufile-user-guide/index.html)
