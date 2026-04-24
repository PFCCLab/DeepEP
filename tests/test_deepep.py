

import paddle
import fastdeploy
from fastdeploy import envs
from fastdeploy.config import MoEPhase
from fastdeploy.utils import singleton
from typing import Optional
import paddle.distributed as dist
from paddle.distributed import fleet
import traceback
from abc import abstractmethod
from fastdeploy.utils import singleton
import numpy as np

def uniform_int_tensor_no_repeat(M, K, E, dtype='int64'):
    """生成 [M, K] 整数 tensor，满足：
    1. M*K 个值绝对均匀分布在 [0, E)（每个值出现次数完全相同）
    2. 每行的 K 个数互不重复
    约束：K <= E 且 (M * K) % E == 0
    """
    assert K <= E, f'K={K} must be <= E={E}'
    assert (M * K) % E == 0, f'Total M*K={M*K} must be divisible by E={E}'

    per_value = (M * K) // E

    # 构造有序序列：[0,..,0, 1,..,1, ..., E-1,..,E-1]，每个值出现 per_value 次
    ordered = paddle.arange(E, dtype=dtype).reshape([-1, 1]).expand([E, per_value]).reshape([-1])

    # 全局随机打乱
    shuffled = paddle.index_select(ordered, paddle.randperm(ordered.shape[0])).numpy()
    matrix = shuffled.reshape([M, K])

    # 修复行内重复：贪心交换
    for _ in range(M * K):
        fixed_any = False
        for i in range(M):
            row = matrix[i]
            vals, cnts = np.unique(row, return_counts=True)
            if len(vals) == K:
                continue
            # 找重复值和缺失值
            dup_val = vals[cnts > 1][0]
            dup_pos = np.where(row == dup_val)[0][1]
            row_set = set(row.tolist())
            missing = [v for v in range(E) if v not in row_set]
            # 与其他行交换
            for replace_val in missing:
                for i2 in range(M):
                    if i2 == i:
                        continue
                    if replace_val in matrix[i2] and dup_val not in matrix[i2]:
                        pos2 = np.where(matrix[i2] == replace_val)[0][0]
                        matrix[i, dup_pos] = replace_val
                        matrix[i2, pos2] = dup_val
                        fixed_any = True
                        break
                else:
                    continue
                break
        if not fixed_any:
            break

    return paddle.to_tensor(matrix, dtype=dtype)

def load_deep_ep():
    """
    Load DeepEP module according to FastDeploy env switch.

    Returns:
        Imported deep_ep module object.
    """

    try:
        # Enable torch proxy before importing deep_ep (required by PFCC/PaddleFleet variants)
        paddle.compat.enable_torch_proxy(scope={"deep_ep"})

        import deep_ep  # type: ignore

        print("FD use PFCCLab/DeepEP now.")
        return deep_ep
    except Exception as e:
        print(
            f"import deep_ep failed! type={type(e).__name__}, err={e}"
        )
        print(f"Traceback:{traceback.format_exc()}")
        raise


deep_ep = load_deep_ep()


def init_distributed_environment(seed: int = 20):
    """Initialize Paddle Fleet and get rank of worker"""
    # Global rank
    ranks = dist.get_world_size()
    dist_strategy = fleet.DistributedStrategy()
    if ranks > 0:
        dist_strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": ranks,
            "pp_degree": 1,
            "sharding_degree": 1,
        }

        # Set control in tensor parallel
        dist_strategy.tensor_parallel_configs = {"tensor_init_seed": seed}
        fleet.init(is_collective=True, strategy=dist_strategy)
        # _log_mem("after_fleet_init")

        # Local rank
        local_rank = fleet.worker_index()
    else:
        local_rank = 0
    return ranks, local_rank

ranks, local_rank = init_distributed_environment()



class DeepEPBuffer:
    """
    Encapsulates DeepEP buffer creation, management and cleanup.
    """

    def __init__(
        self,
        group,
        hidden_size: int,
        num_experts: int,
        ep_size: int,
        num_max_dispatch_tokens_per_rank: int,
        moe_phase="decode",
        use_internode_ll_two_stage: bool = False,
        top_k: int = 8,
        quant_group_size: int = 128,
    ):
        self.group = group
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        self.moe_phase = moe_phase
        self.use_internode_ll_two_stage = use_internode_ll_two_stage
        self.top_k = top_k

        self.deepep_buffer = None
        self.num_nvl_bytes = 0
        self.num_rdma_bytes = 0

        # Precompute buffer sizes
        self._compute_buffer_sizes(quant_group_size=quant_group_size)

    def _compute_buffer_sizes(self, param_bytes: int = 2, quant_group_size=128):
        hidden_bytes = self.hidden_size * param_bytes  # bf16 or fp16

        for config in (
            deep_ep.Buffer.get_dispatch_config(self.group.world_size),
            deep_ep.Buffer.get_combine_config(self.group.world_size),
        ):
            self.num_nvl_bytes = max(
                config.get_nvl_buffer_size_hint(hidden_bytes, self.group.world_size), self.num_nvl_bytes
            )
            self.num_rdma_bytes = max(
                config.get_rdma_buffer_size_hint(hidden_bytes, self.group.world_size), self.num_rdma_bytes
            )

        if not self.use_internode_ll_two_stage:
            num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
                self.num_max_dispatch_tokens_per_rank,
                self.hidden_size,
                self.ep_size,
                self.num_experts,
                quant_group_size
            )
        else:
            num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint_two_stage(
                self.num_max_dispatch_tokens_per_rank, self.hidden_size, self.ep_size, self.num_experts, self.top_k
            )
            num_nvl_bytes = deep_ep.Buffer.get_low_latency_nvl_size_hint_two_stage(
                self.num_max_dispatch_tokens_per_rank,
                self.hidden_size,
                self.ep_size,
                self.num_experts,
                self.top_k,
                True,  # just supports dispatch_use_fp8 = True now!
            )
            self.num_nvl_bytes = max(self.num_nvl_bytes, num_nvl_bytes)
        self.num_rdma_bytes = max(self.num_rdma_bytes, num_rdma_bytes)

        print(f"DeepEP num nvl bytes : {self.num_nvl_bytes}, num rdma bytes : {self.num_rdma_bytes}")

    def create_buffer(self):
        """Create or recreate buffer based on role and phase."""
        if self.deepep_buffer is not None:
            self.clear_buffer()

        num_qps_per_rank = max(24, self.num_experts // self.ep_size)
        
        if self.moe_phase == "decode":
            self._create_low_latency_buffer()
        elif self.moe_phase == "prefill":
            print("Initializing High Throughput Buffer for prefill phase.")
            self.deepep_buffer = deep_ep.Buffer(
                self.group,
                self.num_nvl_bytes,
                self.num_rdma_bytes,
                low_latency_mode=True,
                num_qps_per_rank=num_qps_per_rank,
            )
        else:
            raise ValueError(f"Unknown generation phase: {self.moe_phase}")

        print("DeepEP buffer created successfully.")

    def _create_low_latency_buffer(self):
        if self.deepep_buffer is None:
            assert self.num_experts % self.ep_size == 0
            num_qps_per_rank_now = self.num_experts // self.ep_size
            
            self.deepep_buffer = deep_ep.Buffer(
                self.group,
                self.num_nvl_bytes,
                self.num_rdma_bytes,
                low_latency_mode=True,
                num_qps_per_rank=num_qps_per_rank_now,
            )

    def clear_buffer(self):
        """Clear buffer and free memory."""
        if self.deepep_buffer is not None:
            del self.deepep_buffer
            self.deepep_buffer = None
            print("DeepEP buffer cleared.")

    def get_buffer(self):
        return self.deepep_buffer

    def clean_low_latency_buffer(self):
        if self.deepep_buffer is not None:
            if not self.use_internode_ll_two_stage:
                self.deepep_buffer.clean_low_latency_buffer(
                    self.num_max_dispatch_tokens_per_rank,
                    self.hidden_size,
                    self.num_experts,
                )
            else:
                self.deepep_buffer.clean_low_latency_two_stage_buffer(
                    self.num_max_dispatch_tokens_per_rank,
                    self.hidden_size,
                    self.num_experts,
                    self.top_k,
                    self.ep_size,
                    True,  # just supports dispatch_use_fp8 = True now!
                )

    def barrier_all(self):
        if self.deepep_buffer is not None:
            self.deepep_buffer.barrier_all()


class DeepEPBufferManager:
    _engine: Optional["DeepEPEngine"] = None

    @classmethod
    def set_engine(cls, engine: "DeepEPEngine"):
        cls._engine = engine

    @classmethod
    def clear_buffer(cls):
        if cls._engine:
            cls._engine.clear_deep_ep_buffer()

    @classmethod
    def recreate_buffer(cls):
        if cls._engine:
            cls._engine.create_deep_ep_buffer()

@singleton
class DeepEPEngine:
    """
    A wrapper class for DeepEP engine.
    Manages buffer lifecycle based on role and phase.
    """

    def __init__(
        self,
        num_max_dispatch_tokens_per_rank: int,
        hidden_size: int,
        num_experts: int,
        ep_size: int,
        ep_rank: int,
        moe_phase="decode",
        async_finish: bool = True,
        group=None,
        use_internode_ll_two_stage: bool = False,
        top_k: int = 8,
        quant_group_size: int = 128,
    ):
        if group is None:
            group = paddle.distributed.new_group(range(ep_size))
        self.group = group
        self.ep_size = ep_size
        self.rank_id = ep_rank
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_local_experts = num_experts // ep_size
        self.top_k = top_k
        self.async_finish = async_finish

        self.ep_config = None

        # Store phase and role for buffer management
        self._moe_phase = moe_phase

        # Initialize buffer manager
        self.buffer = DeepEPBuffer(
            group=self.group,
            hidden_size=hidden_size,
            num_experts=num_experts,
            ep_size=ep_size,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            moe_phase=moe_phase,
            use_internode_ll_two_stage=use_internode_ll_two_stage,
            top_k=self.top_k,
            quant_group_size=quant_group_size
        )
        self.buffer.create_buffer()

        # Register for global buffer management
        DeepEPBufferManager.set_engine(self)

    @property
    def deepep_engine(self):
        """Backward compatibility alias."""
        return self.buffer.get_buffer()

    def clear_deep_ep_buffer(self):
        self.buffer.clear_buffer()

    def create_deep_ep_buffer(self):
        self.buffer.create_buffer()

    def low_latency_dispatch(
        self,
        hidden_states: paddle.Tensor,
        topk_idx: paddle.Tensor,
        expertwise_scale,
        use_fp8: bool = False,
        quant_group_size: int = 128,
        use_ue8m0: bool = False,
    ):
        if self.deepep_engine is None:
            raise RuntimeError("DeepEP buffer not initialized!")

        (
            packed_recv_x,
            recv_expert_count,
            handle,
            _,
            dispatch_hook,
        ) = self.deepep_engine.low_latency_dispatch(
            hidden_states,
            topk_idx,
            self.buffer.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            use_fp8=use_fp8,
            async_finish=False,
            return_recv_hook=True,
            round_scale=use_ue8m0,
            quant_group_size=quant_group_size,
            use_ue8m0=use_ue8m0,
        )
        

        return packed_recv_x, recv_expert_count, handle, dispatch_hook

    def low_latency_dispatch_two_stage(
        self,
        hidden_states: paddle.Tensor,
        topk_idx: paddle.Tensor,
        topk_weights: paddle.Tensor,
        expertwise_scale,
        use_fp8: bool = False,
        quant_group_size: int = 128,
    ):
        if self.deepep_engine is None:
            raise RuntimeError("DeepEP buffer not initialized!")

        (
            packed_recv_x,
            packed_recv_count,
            _,
            handle,
            _,
            dispatch_hook,
        ) = self.deepep_engine.low_latency_dispatch_two_stage(
            hidden_states,
            topk_idx,
            topk_weights,
            self.buffer.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            use_fp8=use_fp8,
            async_finish=False,
            return_recv_hook=True,
            num_per_channel=quant_group_size,
        )

        return packed_recv_x, packed_recv_count, handle, dispatch_hook

    def low_latency_combine(
        self,
        hidden_states: paddle.Tensor,
        topk_idx: paddle.Tensor,
        topk_weights: paddle.Tensor,
        handle,
    ):
        if paddle.__version__ != "0.0.0" and paddle.__version__ <= "3.1.0":
            # TODO(@wanglongzhi): Delete them when deepep in PaddlePaddle is fixed
            # and when the default recommended version of PaddlePaddle is greater than 3.1.0
            src_info, layout_range, num_max_dispatch_tokens_per_rank, num_experts = handle
            handle = (src_info, layout_range, num_max_dispatch_tokens_per_rank, None, num_experts)

        if self.deepep_engine is None:
            raise RuntimeError("DeepEP buffer not initialized!")

        combined_hidden_states, _, combine_hook = self.deepep_engine.low_latency_combine(
            hidden_states,
            topk_idx,
            topk_weights,
            handle,
            async_finish=False,
            return_recv_hook=True,
        )
        return combined_hidden_states, combine_hook

    def low_latency_combine_two_stage(
        self,
        hidden_states: paddle.Tensor,
        topk_idx: paddle.Tensor,
        topk_weights: paddle.Tensor,
        dispatch_use_fp8: bool,
        quant_group_size: int,
        handle,
    ):
        if self.deepep_engine is None:
            raise RuntimeError("DeepEP buffer not initialized!")

        combined_hidden_states, _, combine_hook = self.deepep_engine.low_latency_combine_two_stage(
            hidden_states,
            topk_idx,
            topk_weights,
            handle,
            async_finish=False,
            dispatch_use_fp8=dispatch_use_fp8,
            return_recv_hook=True,
            num_per_channel=quant_group_size,
        )
        return combined_hidden_states, combine_hook

    def clean_low_latency_buffer(self):
        self.buffer.clean_low_latency_buffer()

    def barrier_all(self):
        self.buffer.barrier_all()


class EPRunner:
    """
    EPRunnerBase
    """

    def __init__(
        self,
        top_k: int,
        hidden_size: int,
        num_experts: int,
        moe_phase="decode",
        num_max_dispatch_tokens_per_rank: int = 1,
        ep_size: int = 1,
        ep_rank: int = 0,
        redundant_experts_num: int = 0,
        ep_group=None,
        use_internode_ll_two_stage: bool = False,
        quant_group_size: int = 128,
    ):
        self.top_k = top_k
        self.num_experts = num_experts
        self.redundant_experts_num = redundant_experts_num
        self.use_internode_ll_two_stage = use_internode_ll_two_stage
        self.ep_engine = DeepEPEngine(
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            hidden_size=hidden_size,
            num_experts=num_experts + redundant_experts_num,
            ep_size=ep_size,
            ep_rank=ep_rank,
            moe_phase=moe_phase,
            group=ep_group,
            use_internode_ll_two_stage=self.use_internode_ll_two_stage,
            top_k=self.top_k,
            quant_group_size=quant_group_size,
        )

    @abstractmethod
    def dispatch(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def combine(self, *args, **kwargs):
        raise NotImplementedError

    def clean_low_latency_buffer(self):
        self.ep_engine.clean_low_latency_buffer()

    def clear_deep_ep_buffer(self):
        self.ep_engine.clear_deep_ep_buffer()

    def create_deep_ep_buffer(self):
        self.ep_engine.create_deep_ep_buffer()


class EPDecoderRunner(EPRunner):
    """
    EPDecoderRunner
    """

    def __init__(
        self,
        top_k: int,
        hidden_size: int,
        num_experts: int,
        num_max_dispatch_tokens_per_rank: int,
        ep_size: int = 1,
        ep_rank: int = 0,
        redundant_experts_num: int = 0,
        ep_group=None,
        moe_phase="decode",
        use_internode_ll_two_stage: bool = False,
        quant_group_size: int = 128,
    ):
        super().__init__(
            top_k,
            hidden_size,
            num_experts,
            moe_phase,
            num_max_dispatch_tokens_per_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            redundant_experts_num=redundant_experts_num,
            ep_group=ep_group,
            use_internode_ll_two_stage=use_internode_ll_two_stage,
            quant_group_size=quant_group_size,
        )

    def dispatch(
        self,
        x: paddle.Tensor,
        topk_idx: paddle.Tensor,
        topk_weights: paddle.Tensor,
        *args,
        **kwargs,
    ):
        expertwise_scale = kwargs.get("expertwise_scale", None)
        use_fp8 = kwargs.get("use_fp8", False)
        quant_group_size = kwargs.get("quant_group_size", 128)
        use_ue8m0 = kwargs.get("use_ue8m0", False)
        if not self.use_internode_ll_two_stage:
            recv_hidden_states, recv_expert_count, handle, dispatch_hook = self.ep_engine.low_latency_dispatch(
                x, topk_idx, expertwise_scale, use_fp8, quant_group_size, use_ue8m0
            )
        else:
            # just supports dispatch_use_fp8 = True now!
            assert use_fp8 is True
            recv_hidden_states, recv_expert_count, handle, dispatch_hook = (
                self.ep_engine.low_latency_dispatch_two_stage(
                    x, topk_idx, topk_weights, expertwise_scale, use_fp8, quant_group_size
                )
            )
        if dispatch_hook is not None:
            dispatch_hook()

        return recv_hidden_states, recv_expert_count, handle

    def combine(self, ffn_out, topk_idx, topk_weights, handle, **kwargs):
        quant_group_size = kwargs.get("quant_group_size", 128)
        if not self.use_internode_ll_two_stage:
            combined_hidden_states, combine_hook = self.ep_engine.low_latency_combine(
                ffn_out, topk_idx, topk_weights, handle
            )
        else:
            combined_hidden_states, combine_hook = self.ep_engine.low_latency_combine_two_stage(
                ffn_out,
                topk_idx,
                topk_weights,
                True,
                quant_group_size,
                handle,  # just supports dispatch_use_fp8 = True now!
            )
        if combine_hook is not None:
            combine_hook()

        return combined_hidden_states


class DeepEPMOE:
    def __init__(self, ):
        self.top_k  = 4
        self.token_num = 160
        self.hidden_size = 7168
        self.num_experts = 160
        self.num_max_dispatch_tokens_per_rank = 512
        self.ep_size = 8
        self.ep_rank = local_rank
        self.redundant_experts_num = 0
        self.moe_phase = "decode"
        self.use_internode_ll_two_stage = False
        self.quant_group_size = 32


    def init_ep(self) -> None:
        """
        Initialize EP (Expert Parallel) related modules.
        """
        if self.ep_size <= 1:
            return
        self.ep_group = dist.new_group(range(self.ep_size))


        # Common arguments for both runners
        common_args = {
            "top_k": self.top_k,
            "hidden_size": self.hidden_size,
            "num_experts": self.num_experts,
            "num_max_dispatch_tokens_per_rank": self.num_max_dispatch_tokens_per_rank,
            "ep_size": self.ep_size,
            "ep_rank": self.ep_rank,
            "redundant_experts_num": self.redundant_experts_num,
            "ep_group": self.ep_group,
        }


        # prefill_num_worst_tokens = 0
        # prefill_num_worst_tokens = (
        #     self.max_num_batched_tokens
        #     // self.tensor_parallel_size
        #     * self.ep_size
        #     * self.top_k
        # )

        # self.ep_prefill_runner = EPPrefillRunner(
        #     **common_args,
        #     use_internode_ll_two_stage=self.use_internode_ll_two_stage,
        #     prefill_num_worst_tokens=prefill_num_worst_tokens,
        # )
        self.ep_decoder_runner = EPDecoderRunner(
            **common_args,
            use_internode_ll_two_stage=self.use_internode_ll_two_stage,
            quant_group_size=self.quant_group_size,
        )

    def init_prefill_input(self, ):
        # x_fp8:paddle.Size([107, 7168]), paddle.float8_e4m3fn
        # topk_idx:paddle.Size([107, 4]), paddle.int64
        # topk_weights:paddle.Size([107, 4]), paddle.float32
        # x_scale_tensor:paddle.Size([107, 14]), paddle.int32
        x_fp8 = paddle.rand([self.token_num, self.hidden_size], dtype=paddle.float8_e4m3fn)
        topk_idx = paddle.randint(0, self.num_experts,[self.token_num, self.top_k], dtype=paddle.int64)
        topk_weights = paddle.rand([self.token_num, self.top_k], dtype=paddle.float32)
        x_scale_tensor = paddle.rand([self.token_num, self.hidden_size // self.group_size // 4], dtype=paddle.int32) if self.group_size == 128 else paddle.rand([self.token_num, self.hidden_size // self.group_size], dtype=paddle.uint8)
        return (x_fp8, topk_idx, topk_weights, x_scale_tensor)

    def apply_ep_prefill(
        self,
    ):
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_num_tokens_per_expert_list,
            handle,
            event,
        ) = self.ep_prefill_runner.dispatch(
            x_fp8, topk_idx, topk_weights, x_scale_tensor=x_scale_tensor, expert_alignment=128, previous_event=event
        )


    def init_decode_input(self):
        # x:paddle.Size([6, 7168]), paddle.bfloat16
        # topk_idx:paddle.Size([6, 4]), paddle.int64
        # topk_weights:paddle.Size([6, 4]), paddle.float32
        x = paddle.rand([self.token_num, self.hidden_size], dtype=paddle.bfloat16)
        # topk_idx = paddle.randint(0, self.num_experts, [self.token_num, self.top_k], dtype=paddle.int32).astype(paddle.int64)
        topk_idx = uniform_int_tensor_no_repeat(self.token_num, self.top_k, self.num_experts)
        topk_weights = paddle.rand([self.token_num, self.top_k], dtype=paddle.float32)
        return (x, topk_idx, topk_weights)

    def apply_ep_decode(
        self,
        x,
        topk_idx,
        topk_weights
    ):
        # 2. EP Dispatch
        print(f"dispatch start")
        permute_input, token_nums_per_expert, handle = self.ep_decoder_runner.dispatch(
            x, topk_idx, topk_weights, use_fp8=True, use_ue8m0=True, quant_group_size=self.quant_group_size
        )
        print(f"x_fp8:{permute_input[0]}")
        print(f"scale:{permute_input[1]}")
        print(f"token_nums_per_expert:{token_nums_per_expert}")

print(f"apply_ep_decode start")
ep = DeepEPMOE()
print(f"create DeepEPMOE end")
ep.init_ep()
print(f"init_ep end")
x, topk_idx, topk_weights = ep.init_decode_input()
print(f"init_decode_input end")
ep.apply_ep_decode(x, topk_idx, topk_weights)
print(f"apply_ep_decode end")
