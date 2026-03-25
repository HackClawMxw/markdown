# vLLM 代码架构深度解析

## 概述

vLLM 是一个高性能的大语言模型推理和服务框架。本文档从代码层级深入分析 vLLM 的架构设计、核心组件及其与 PyTorch 的交互方式。

---

## 目录

1. [整体架构](#1-整体架构)
2. [核心组件详解](#2-核心组件详解)
3. [模型执行层](#3-模型执行层)
4. [与 PyTorch 的交互](#4-与-pytorch-的交互)
5. [CUDA 内核与自定义算子](#5-cuda-内核与自定义算子)
6. [调度系统](#6-调度系统)
7. [KV Cache 管理](#7-kv-cache-管理)
8. [分布式执行](#8-分布式执行)
9. [调用流程图](#9-调用流程图)

---

## 1. 整体架构

### 1.1 代码目录结构

```
vllm/
├── vllm/                     # 主包目录
│   ├── __init__.py           # 入口点，导出 LLM, AsyncLLM 等
│   ├── config/               # 配置管理
│   ├── engine/               # 引擎实现（V0 架构）
│   ├── v1/                   # V1 新架构
│   │   ├── engine/           # V1 引擎核心
│   │   ├── worker/           # Worker 实现
│   │   ├── executor/         # 执行器
│   │   ├── core/             # 调度器核心
│   │   ├── sample/           # 采样器
│   │   └── attention/        # 注意力机制
│   ├── model_executor/       # 模型执行器
│   │   ├── layers/           # 自定义层
│   │   ├── models/           # 模型实现
│   │   └── parameter/        # 参数管理
│   └── distributed/          # 分布式通信
├── csrc/                     # CUDA/C++ 扩展
│   ├── torch_bindings.cpp    # PyTorch 绑定
│   ├── cpu/                  # CPU 实现
│   └── rocm/                 # ROCm 实现
└── tests/                    # 测试代码
```

### 1.2 V1 架构核心组件

vLLM V1 采用分层架构：

```
┌─────────────────────────────────────────────────────────────┐
│                     API Layer                                │
│  ┌─────────────────┐  ┌─────────────────────────────────┐  │
│  │       LLM       │  │          AsyncLLM               │  │
│  │   (Sync API)    │  │        (Async API)              │  │
│  └────────┬────────┘  └────────────────┬────────────────┘  │
└───────────┼─────────────────────────────┼───────────────────┘
            │                             │
            ▼                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   Engine Core Layer                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    EngineCore                           ││
│  │  ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐ ││
│  │  │  Scheduler  │ │ InputProcessor│ │ OutputProcessor  │ ││
│  │  └─────────────┘ └──────────────┘ └──────────────────┘ ││
│  └─────────────────────────────────────────────────────────┘│
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   Executor Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ UniProc     │  │ Multiproc   │  │ RayDistributed      │ │
│  │ Executor    │  │ Executor    │  │ Executor            │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────┼────────────────┼────────────────────┼─────────────┘
          │                │                    │
          ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                     Worker Layer                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              GPUModelRunner                              ││
│  │  ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐ ││
│  │  │ Model Loader │ │   Sampler    │ │ Attention Backend│ ││
│  │  └──────────────┘ └──────────────┘ └─────────────────┘ ││
│  └─────────────────────────────────────────────────────────┘│
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   Model Layer (PyTorch)                      │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                 nn.Module Models                         ││
│  │  (Llama, Qwen, DeepSeek, etc.)                          ││
│  │  ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐ ││
│  │  │  Attention  │ │    Linear    │ │    RMSNorm       │ ││
│  │  └─────────────┘ └──────────────┘ └──────────────────┘ ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 核心组件详解

### 2.1 AsyncLLM (异步引擎入口)

**文件**: `vllm/v1/engine/async_llm.py`

AsyncLLM 是 vLLM V1 的主要异步接口，负责：
- 管理引擎核心的生命周期
- 处理异步生成请求
- 后台输出处理

```python
class AsyncLLM:
    def __init__(self, ...):
        # 核心组件
        self.input_processor = InputProcessor(...)
        self.output_processor = OutputProcessor(...)
        self.engine_core = EngineCore(...)

        # 启动后台输出处理任务
        self.output_handler = asyncio.create_task(
            self._run_output_handler()
        )

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        ...
    ) -> AsyncIterator[RequestOutput]:
        """异步生成接口"""
        # 1. 处理输入
        request = self.input_processor.preprocess(...)

        # 2. 提交到引擎核心
        self.engine_core.add_request(request)

        # 3. 异步等待输出
        async for output in self._wait_for_output(request):
            yield output
```

### 2.2 EngineCore (引擎核心)

**文件**: `vllm/v1/engine/core.py`

EngineCore 是推理引擎的核心，协调调度和执行：

```python
class EngineCore:
    def __init__(self, vllm_config: VllmConfig):
        self.scheduler = Scheduler(...)
        self.executor = Executor.get_class(vllm_config)(...)

    def step(self) -> list[EngineCoreOutput]:
        """执行一步推理"""
        # 1. 调度
        scheduler_output = self.scheduler.schedule()

        # 2. 执行模型
        model_output = self.executor.execute_model(scheduler_output)

        # 3. 更新调度器状态
        return self.scheduler.update_from_output(
            scheduler_output, model_output
        )
```

### 2.3 Scheduler (调度器)

**文件**: `vllm/v1/core/sched/scheduler.py`

调度器负责请求调度和资源管理，核心算法：

```python
class Scheduler:
    def __init__(self, ...):
        # 请求队列
        self.waiting = create_request_queue(self.policy)
        self.skipped_waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # KV Cache 管理器
        self.kv_cache_manager = KVCacheManager(...)

        # 编码器缓存
        self.encoder_cache_manager = EncoderCacheManager(...)

    def schedule(self) -> SchedulerOutput:
        """
        调度算法核心：
        - 没有 "prefill phase" 或 "decode phase" 的区分
        - 每个请求有 num_computed_tokens 和 num_tokens_with_spec
        - 调度器尝试让每个请求的 num_computed_tokens 追上 num_tokens_with_spec
        """
        token_budget = self.max_num_scheduled_tokens

        # 1. 调度 RUNNING 请求
        for request in self.running:
            if token_budget <= 0:
                break
            num_new_tokens = min(
                request.num_tokens_with_spec - request.num_computed_tokens,
                token_budget
            )
            # 分配 KV Cache
            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens
            )
            if new_blocks is not None:
                token_budget -= num_new_tokens

        # 2. 调度 WAITING 请求
        while self.waiting and token_budget > 0:
            request = self.waiting.peek_request()
            # 获取已缓存的 tokens (prefix caching)
            new_blocks, num_cached = self.kv_cache_manager.get_computed_blocks(request)
            # 分配新的 slots
            ...
```

---

## 3. 模型执行层

### 3.1 GPUModelRunner

**文件**: `vllm/v1/worker/gpu_model_runner.py`

GPUModelRunner 负责在 GPU 上执行模型：

```python
class GPUModelRunner:
    def __init__(self, vllm_config: VllmConfig, ...):
        # 模型加载器
        self.model_loader = get_model_loader(vllm_config.load_config)
        # 模型实例
        self.model: nn.Module = None
        # 采样器
        self.sampler = Sampler()
        # 注意力后端
        self.attn_backend = get_attn_backend(...)
        # CUDA Graph 调度器
        self.cudagraph_dispatcher = CudagraphDispatcher(...)

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput:
        """执行模型推理"""
        # 1. 准备输入
        input_batch = self._prepare_input_batch(scheduler_output)

        # 2. 设置 forward context
        with set_forward_context(...):
            # 3. 执行模型前向传播
            hidden_states = self.model(
                input_ids=input_batch.token_ids,
                positions=input_batch.positions,
            )

        # 4. 采样
        sampled_tokens = self.sampler(
            hidden_states,
            sampling_metadata,
        )

        return ModelRunnerOutput(
            sampled_token_ids=sampled_tokens,
            ...
        )
```

### 3.2 Executor (执行器)

**文件**: `vllm/v1/executor/abstract.py`

执行器抽象基类定义了分布式执行接口：

```python
class Executor(ABC):
    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:
        """根据配置选择执行器"""
        backend = vllm_config.parallel_config.distributed_executor_backend
        if backend == "ray":
            return RayDistributedExecutor
        elif backend == "mp":
            return MultiprocExecutor
        else:
            return UniProcExecutor

    @abstractmethod
    def collective_rpc(
        self,
        method: str,
        args: tuple = (),
        ...
    ) -> list[Any]:
        """在所有 Worker 上执行 RPC 调用"""
        pass

    @abstractmethod
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput:
        """执行模型"""
        pass
```

---

## 4. 与 PyTorch 的交互

### 4.1 模型定义：继承 nn.Module

vLLM 的模型实现完全基于 PyTorch 的 `nn.Module`：

**文件**: `vllm/model_executor/models/llama.py` (示例)

```python
from torch import nn
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.attention import Attention

class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        ...
    ):
        super().__init__()
        # 自注意力层
        self.self_attn = LlamaAttention(...)
        # MLP 层
        self.mlp = LlamaMLP(...)
        # 层归一化
        self.input_layernorm = RMSNorm(...)
        self.post_attention_layernorm = RMSNorm(...)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ...
    ) -> torch.Tensor:
        # Pre-norm 架构
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # 自注意力
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
```

### 4.2 自定义线性层

**文件**: `vllm/model_executor/layers/linear.py`

vLLM 提供了多种并行线性层：

```python
class ColumnParallelLinear(LinearBase):
    """列并行线性层 - 沿输出维度切分"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        ...
    ):
        super().__init__(...)
        # 张量并行切分
        self.tp_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = output_size // self.tp_size

        # 创建权重
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=input_size,
            output_partition_sizes=[self.output_size_per_partition],
            ...
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # 矩阵乘法
        output_parallel = self.quant_method.apply(self, input_, bias)

        # 如果需要，执行 all-gather
        if self.gather_output and self.tp_size > 1:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel

        return output


class RowParallelLinear(LinearBase):
    """行并行线性层 - 沿输入维度切分"""

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # 矩阵乘法
        output_parallel = self.quant_method.apply(self, input_parallel, bias)

        # 执行 all-reduce
        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        return output


class QKVParallelLinear(ColumnParallelLinear):
    """QKV 融合并行线性层"""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        ...
    ):
        # 计算 QKV 的输出尺寸
        self.output_sizes = [
            num_heads * head_size,           # Q
            num_kv_heads * head_size,        # K
            num_kv_heads * head_size,        # V
        ]
        ...
```

### 4.3 Attention 层

**文件**: `vllm/model_executor/layers/attention/attention.py`

vLLM 的 Attention 层与 PyTorch 深度集成：

```python
class Attention(nn.Module, AttentionLayerBase):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        ...
    ):
        super().__init__()
        # 选择注意力后端
        self.attn_backend = get_attn_backend(
            head_size, dtype, kv_cache_dtype, ...
        )

        # 获取后端实现类
        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            num_heads, head_size, scale, ...
        )

        # KV Cache (占位符，实际由 bind_kv_cache 填充)
        self.kv_cache = [torch.tensor([]) for _ in range(pp_size)]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播：
        1. 更新 KV Cache
        2. 执行注意力计算
        """
        # 注册为自定义 op 以支持 torch.compile
        if self.use_direct_call:
            return unified_attention(query, key, value, self.layer_name)
        else:
            return torch.ops.vllm.unified_attention(
                query, key, value, self.layer_name
            )
```

### 4.4 ForwardContext (前向上下文)

**文件**: `vllm/forward_context.py`

vLLM 使用上下文管理器传递注意力元数据：

```python
@contextmanager
def set_forward_context(
    context: ForwardContext,
):
    """设置当前的前向上下文"""
    _forward_context.set(context)
    try:
        yield
    finally:
        _forward_context.set(None)


def get_forward_context() -> ForwardContext:
    """获取当前的前向上下文"""
    return _forward_context.get()
```

在 `unified_attention` 函数中使用：

```python
def unified_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    # 从上下文获取注意力和 KV Cache
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    attn_layer = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]

    # 执行注意力
    output = attn_layer.impl.forward(
        attn_layer, query, key, value, kv_cache, attn_metadata
    )
    return output
```

---

## 5. CUDA 内核与自定义算子

### 5.1 PyTorch C++ 扩展绑定

**文件**: `csrc/torch_bindings.cpp`

vLLM 使用 PyTorch 的 `TORCH_LIBRARY` 机制注册自定义算子：

```cpp
#include <torch/library.h>

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // PagedAttention 算子
  ops.def(
      "paged_attention_v1("
      "    Tensor! out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"
      "    ...) -> ()");
  ops.impl("paged_attention_v1", torch::kCUDA, &paged_attention_v1);

  // 激活函数
  ops.def("silu_and_mul(Tensor! result, Tensor input) -> ()");
  ops.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  // RMSNorm
  ops.def(
      "rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> ()");
  ops.impl("rms_norm", torch::kCUDA, &rms_norm);

  // 量化算子
  ops.def(
      "static_scaled_fp8_quant(Tensor! result, Tensor input, Tensor scale) -> ()");
  ops.impl("static_scaled_fp8_quant", torch::kCUDA, &static_scaled_fp8_quant);

  // CUTLASS GEMM
  ops.def(
      "cutlass_scaled_mm(Tensor! out, Tensor a,"
      "                  Tensor b, Tensor a_scales,"
      "                  Tensor b_scales, Tensor? bias) -> ()");
  ops.impl("cutlass_scaled_mm", torch::kCUDA, &cutlass_scaled_mm);

  // ... 更多算子
}

// KV Cache 操作
TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cache_ops), cache_ops) {
  cache_ops.def(
      "reshape_and_cache(Tensor key, Tensor value,"
      "                  Tensor! key_cache, Tensor! value_cache,"
      "                  Tensor slot_mapping,"
      "                  str kv_cache_dtype,"
      "                  Tensor k_scale, Tensor v_scale) -> ()");
  cache_ops.impl("reshape_and_cache", torch::kCUDA, &reshape_and_cache);
}

// 自定义 All-Reduce
TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _custom_ar), custom_ar) {
  custom_ar.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
      "int rank, bool fully_connected) -> int");
  custom_ar.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);

  custom_ar.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, ...) -> ()");
  custom_ar.impl("all_reduce", torch::kCUDA, &all_reduce);
}
```

### 5.2 Python 端调用

在 Python 端通过 `torch.ops` 调用：

```python
import torch

# 调用 CUDA 算子
torch.ops.vllm.paged_attention_v1(
    output, query, key_cache, value_cache,
    num_kv_heads, scale, block_tables, seq_lens,
    block_size, max_seq_len, ...
)

# 调用 RMSNorm
torch.ops.vllm.rms_norm(
    output, input_tensor, weight, epsilon
)

# 调用量化
torch.ops.vllm.static_scaled_fp8_quant(
    output, input_tensor, scale
)
```

### 5.3 自定义 Op 注册

**文件**: `vllm/utils/torch_utils.py`

```python
def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: list[str],
    fake_impl: Callable | None = None,
):
    """
    直接注册自定义 op 到 torch
    """
    import torch.library

    lib = torch.library.Library("vllm", "FRAGMENT")
    lib.define(
        f"{op_name}({schema}) -> {return_type}",
        tags=torch.Tag.pt2_compliant_tag,
    )
    lib.impl(op_name, op_func, "CompositeExplicitAutograd")

    if fake_impl is not None:
        lib.impl(op_name, fake_impl, "Meta")
```

---

## 6. 调度系统

### 6.1 调度策略

vLLM V1 的调度器采用统一的调度策略：

```python
class SchedulingPolicy(Enum):
    FCFS = "fcfs"        # 先来先服务
    PRIORITY = "priority"  # 优先级调度
```

### 6.2 调度流程

```python
def schedule(self) -> SchedulerOutput:
    """
    调度核心逻辑
    """
    # 1. 处理 RUNNING 请求
    for request in self.running:
        num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
        new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
        if new_blocks is not None:
            scheduled_running_reqs.append(request)

    # 2. 处理 WAITING 请求
    while self.waiting and token_budget > 0:
        request = self.waiting.pop_request()

        # Prefix Caching: 检查已缓存的 tokens
        new_blocks, num_cached = self.kv_cache_manager.get_computed_blocks(request)

        # 分配新 slots
        new_blocks = self.kv_cache_manager.allocate_slots(
            request, num_new_tokens, new_computed_blocks=new_blocks
        )
        if new_blocks is not None:
            self.running.append(request)
            request.status = RequestStatus.RUNNING

    # 3. 处理 Preemption
    if kv_cache_full:
        preempted_req = self.running.pop()
        self._preempt_request(preempted_req)

    return SchedulerOutput(...)
```

### 6.3 请求状态机

```python
class RequestStatus(IntEnum):
    WAITING = 0
    RUNNING = 1
    PREEMPTED = 2
    FINISHED_STOPPED = 3
    FINISHED_LENGTH_CAPPED = 4
    FINISHED_ABORTED = 5
    FINISHED_IGNORED = 6
    WAITING_FOR_REMOTE_KVS = 7  # KV Transfer
    WAITING_FOR_FSM = 8         # 结构化输出
    WAITING_FOR_STREAMING_REQ = 9
```

---

## 7. KV Cache 管理

### 7.1 KVCacheManager

**文件**: `vllm/v1/core/kv_cache_manager.py`

```python
class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        ...
    ):
        # Block 池管理
        self.block_pool = BlockPool(...)

        # Prefix Caching (基于 hash)
        self.block_hashes: dict[BlockHashType, Block] = {}

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: list = None,
        ...
    ) -> KVCacheBlocks | None:
        """为请求分配 KV Cache slots"""
        # 计算需要的 blocks
        num_blocks_needed = self._get_num_blocks(num_tokens)

        # 尝试分配
        new_blocks = self.block_pool.allocate(num_blocks_needed)
        if new_blocks is None:
            return None  # 内存不足

        return KVCacheBlocks(blocks=new_blocks)

    def get_computed_blocks(
        self,
        request: Request,
    ) -> tuple[list[Block], int]:
        """获取已缓存的 blocks (prefix caching)"""
        computed_blocks = []
        num_cached_tokens = 0

        for i, block_hash in enumerate(request.block_hashes):
            if block_hash in self.block_hashes:
                cached_block = self.block_hashes[block_hash]
                computed_blocks.append(cached_block)
                num_cached_tokens += self.block_size
            else:
                break

        return computed_blocks, num_cached_tokens
```

### 7.2 Paged Attention Block 结构

```
KV Cache 内存布局 (Paged Attention):

┌─────────────────────────────────────────────────────────────┐
│                      KV Cache Tensor                         │
│  Shape: [num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]  │
│                                                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐     ┌─────────┐       │
│  │ Block 0 │ │ Block 1 │ │ Block 2 │ ... │ Block N │       │
│  └────┬────┘ └────┬────┘ └────┬────┘     └────┬────┘       │
│       │           │           │               │              │
│       ▼           ▼           ▼               ▼              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐     ┌─────────┐       │
│  │K: token │ │K: token │ │K: token │     │K: token │       │
│  │  0-15   │ │ 16-31   │ │ 32-47   │     │ ...     │       │
│  │V: token │ │V: token │ │V: token │     │V: token │       │
│  │  0-15   │ │ 16-31   │ │ 32-47   │     │ ...     │       │
│  └─────────┘ └─────────┘ └─────────┘     └─────────┘       │
└─────────────────────────────────────────────────────────────┘

Block Table (请求到 Block 的映射):
┌─────────────────────────────────────────────────────────────┐
│ Request 0: [Block 5, Block 12, Block 3, Block 8]           │
│ Request 1: [Block 2, Block 7]                              │
│ Request 2: [Block 1, Block 4, Block 9, Block 11, Block 15] │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. 分布式执行

### 8.1 Tensor Parallel (张量并行)

```python
# vllm/distributed/parallel_state.py

def get_tensor_model_parallel_group():
    """获取张量并行组"""
    return _TP_GROUP

def tensor_model_parallel_all_reduce(input_: torch.Tensor):
    """张量并行的 all-reduce"""
    return torch.distributed.all_reduce(
        input_, group=get_tensor_model_parallel_group()
    )

def tensor_model_parallel_all_gather(input_: torch.Tensor):
    """张量并行的 all-gather"""
    return torch.distributed.all_gather_into_tensor(
        input_, group=get_tensor_model_parallel_group()
    )
```

### 8.2 Pipeline Parallel (流水线并行)

```python
# vllm/distributed/parallel_state.py

class PipelineParallelState:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.is_first_stage = (rank == 0)
        self.is_last_stage = (rank == world_size - 1)

    def send_tensor(self, tensor: torch.Tensor):
        """发送张量到下一阶段"""
        torch.distributed.send(tensor, dst=self.rank + 1)

    def recv_tensor(self, tensor: torch.Tensor):
        """从上一阶段接收张量"""
        torch.distributed.recv(tensor, src=self.rank - 1)
```

### 8.3 分布式执行器类型

| 执行器 | 描述 | 适用场景 |
|--------|------|----------|
| UniProcExecutor | 单进程执行 | 单 GPU |
| MultiprocExecutor | 多进程执行 | 单机多 GPU |
| RayDistributedExecutor | Ray 分布式执行 | 多机多 GPU |
| ExecutorWithExternalLauncher | 外部启动器 | 自定义集群 |

---

## 9. 调用流程图

### 9.1 完整推理流程

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              用户请求                                     │
│                     AsyncLLM.generate(prompt)                            │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                           InputProcessor                                  │
│  1. Tokenize (使用 HuggingFace tokenizer)                                │
│  2. 创建 Request 对象                                                     │
│  3. 计算 block hashes (用于 prefix caching)                              │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                            EngineCore                                     │
│                     engine_core.add_request(request)                      │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              Scheduler                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ schedule() -> SchedulerOutput                                        │ │
│  │  1. 处理 RUNNING 请求 (继续生成)                                      │ │
│  │  2. 处理 WAITING 请求 (新请求)                                        │ │
│  │  3. 分配 KV Cache blocks                                             │ │
│  │  4. 检查 prefix caching                                              │ │
│  │  5. 处理 preemption                                                  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              Executor                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ execute_model(scheduler_output)                                      │ │
│  │  - collective_rpc("execute_model", args=(scheduler_output,))         │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                           GPUModelRunner                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ execute_model(scheduler_output)                                      │ │
│  │  1. 准备输入 batch (token_ids, positions, block_tables)             │ │
│  │  2. 构建 AttentionMetadata                                          │ │
│  │  3. 设置 ForwardContext                                             │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        Model Forward (PyTorch)                            │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ model(input_ids, positions, ...)                                     │ │
│  │  │                                                                    │ │
│  │  ├─> Embedding Layer (nn.Embedding)                                  │ │
│  │  │                                                                    │ │
│  │  ├─> For each DecoderLayer:                                          │ │
│  │  │   ├─> RMSNorm                                                     │ │
│  │  │   ├─> Attention (torch.ops.vllm.unified_attention)               │ │
│  │  │   │    ├─> QKV projection (ColumnParallelLinear)                  │ │
│  │  │   │    ├─> KV Cache update (torch.ops.vllm.reshape_and_cache)    │ │
│  │  │   │    ├─> PagedAttention (torch.ops.vllm.paged_attention_v1)    │ │
│  │  │   │    └─> Output projection (RowParallelLinear)                  │ │
│  │  │   ├─> RMSNorm                                                     │ │
│  │  │   └─> MLP (silu_and_mul + Linear)                                 │ │
│  │  │                                                                    │ │
│  │  └─> LM Head (Linear) -> logits                                      │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              Sampler                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ sampler(logits, sampling_metadata)                                   │ │
│  │  1. 应用 temperature                                                 │ │
│  │  2. 应用 top_p / top_k                                               │ │
│  │  3. 随机采样                                                         │ │
│  │  4. 返回 sampled_token_ids                                          │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        Scheduler.update_from_output()                     │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  1. 更新 request 状态                                                │ │
│  │  2. 检查是否需要停止 (EOS, max_tokens)                               │ │
│  │  3. 处理 speculative decoding 结果                                   │ │
│  │  4. 生成 EngineCoreOutput                                           │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          OutputProcessor                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  1. Detokenize (使用 HuggingFace tokenizer)                          │ │
│  │  2. 构建 RequestOutput                                              │ │
│  │  3. 处理 logprobs                                                   │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                            返回给用户                                     │
│                      yield RequestOutput                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 9.2 PyTorch 交互层次

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         vLLM Framework Layer                             │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ AsyncLLM / EngineCore / Scheduler / Executor                       │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        PyTorch Python API Layer                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ nn.Module / torch.Tensor / torch.distributed                       │ │
│  │ - 模型定义 (继承 nn.Module)                                         │ │
│  │ - 张量操作 (view, reshape, matmul)                                  │ │
│  │ - 分布式通信 (all_reduce, all_gather)                               │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      torch.ops Custom Ops Layer                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ torch.ops.vllm.* (通过 TORCH_LIBRARY 注册)                          │ │
│  │ - paged_attention_v1/v2                                            │ │
│  │ - reshape_and_cache                                                │ │
│  │ - rms_norm / fused_add_rms_norm                                    │ │
│  │ - silu_and_mul / gelu_and_mul                                      │ │
│  │ - static_scaled_fp8_quant / dynamic_scaled_fp8_quant               │ │
│  │ - cutlass_scaled_mm / marlin_gemm                                  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PyTorch C++ Extension                            │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ csrc/torch_bindings.cpp                                            │ │
│  │ - TORCH_LIBRARY 注册                                               │ │
│  │ - ops.def() 定义接口                                                │ │
│  │ - ops.impl() 绑定 CUDA 实现                                         │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            CUDA Kernels                                  │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ csrc/ *.cu / *.cuh                                                 │ │
│  │ - PagedAttention kernels                                           │ │
│  │ - CUTLASS GEMM kernels                                             │ │
│  │ - Quantization kernels                                             │ │
│  │ - Custom All-Reduce kernels                                        │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. 关键设计模式

### 10.1 PagedAttention

vLLM 的核心创新是 PagedAttention，将 KV Cache 组织成固定大小的 blocks：

- **内存效率**: 按需分配，无预分配浪费
- **Prefix Caching**: 基于 block hash 实现前缀缓存
- **内存共享**: 多个请求可共享相同的 KV blocks

### 10.2 Continuous Batching

vLLM 采用 iteration-level 调度：

- 每个 iteration 重新调度所有请求
- 新请求可随时加入 batch
- 完成的请求立即释放资源

### 10.3 CUDA Graph 优化

```python
# 在 warmup 阶段捕获 CUDA Graph
with torch.cuda.graph(cuda_graph):
    output = model(*inputs)

# 推理时直接执行捕获的 graph
cuda_graph.replay()
```

### 10.4 量化支持

vLLM 支持多种量化方法：

| 量化方法 | 权重位宽 | 激活位宽 | 实现方式 |
|----------|----------|----------|----------|
| FP8 | 8-bit | 8-bit | cutlass_scaled_mm |
| INT8 | 8-bit | 8-bit | cutlass_scaled_mm_azp |
| AWQ | 4-bit | 16-bit | awq_gemm / marlin |
| GPTQ | 4-bit | 16-bit | gptq_gemm / marlin |
| NVFP4 | 4-bit | 4-bit | cutlass_scaled_fp4_mm |

---

## 11. 总结

vLLM 是一个深度集成 PyTorch 的高性能推理框架：

1. **模型层**: 完全基于 `nn.Module`，复用 PyTorch 的模型定义范式
2. **自定义算子**: 通过 `TORCH_LIBRARY` 注册 CUDA 内核，与 PyTorch 无缝集成
3. **分布式**: 基于 `torch.distributed` 实现张量并行和流水线并行
4. **优化**: CUDA Graph、量化、PagedAttention 等底层优化

这种设计使得 vLLM 既保持了 PyTorch 的易用性和灵活性，又通过自定义 CUDA 内核实现了极致的性能优化。
