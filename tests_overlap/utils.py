import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

import deep_ep


def initialize_fleet():
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    paddle.seed(rank)

    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "ep_degree": world_size,
        "pp_degree": 1,
        "sharding_degree": world_size,
        "moe_sharding_degree": 1,
        "dp_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    fleet.init(is_collective=True, strategy=strategy)

    hcg = fleet.get_hybrid_communicate_group()
    group = hcg.get_expert_parallel_group()
    return group


def configure_buffer(num_sms=None, dispatch_config=None, combine_config=None):
    """
    Configure the runtime parameters for deep_ep kernels.
    Must be called before calling get_buffer() to take effect.

    Args:
        num_sms (int): Number of SMs allocated to deep_ep kernels.
        dispatch_config (List[int]):
            Token capacity parameters for dispatch kernels, in the form
            [nvl_send_tokens, nvl_recv_tokens, rdma_send_tokens, rdma_recv_tokens].
            Trailing values may be omitted to use the defaults.
        combine_config (List[int]): Same as above, but for combine kernels.
    """
    if num_sms is not None:
        deep_ep.Buffer.set_num_sms(num_sms)
    if dispatch_config is not None:
        deep_ep.Buffer.get_dispatch_config = staticmethod(
            lambda _: deep_ep.Config(deep_ep.Buffer.num_sms, *dispatch_config)
        )
    if combine_config is not None:
        deep_ep.Buffer.get_combine_config = staticmethod(
            lambda _: deep_ep.Config(deep_ep.Buffer.num_sms, *combine_config)
        )


def get_buffer(group, hidden_bytes):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (paddle.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        deep_ep.Buffer.get_dispatch_config(group.world_size),
        deep_ep.Buffer.get_combine_config(group.world_size),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.world_size),
            num_nvl_bytes,
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.world_size),
            num_rdma_bytes,
        )

    buffer = deep_ep.Buffer(
        group,
        num_nvl_bytes,
        num_rdma_bytes,
        num_qps_per_rank=max(24, deep_ep.Buffer.num_sms),
    )
    return buffer


class AsyncLoad:
    def __init__(self):
        self._pin = None
        self._event = None

    def __call__(self, x, dtype=None) -> paddle.Tensor:
        """Copy x to GPU asynchronously."""
        assert self._pin is None, (
            "The previous copy is not finished, call wait() first before reuse for another copy.")

        cudart = paddle.cuda.cudart()
        self._pin = paddle.to_tensor(x, dtype=dtype, place=paddle.CUDAPinnedPlace())

        out = paddle.empty_like(self._pin)
        stream = paddle.cuda.current_stream()

        err = cudart.cudaMemcpyAsync(
            out.data_ptr(),
            self._pin.data_ptr(),
            out.size * out.itemsize,
            cudart.cudaMemcpyHostToDevice,
            stream.stream_base.cuda_stream,
        )
        assert err == cudart.cudaError.success, f"cudaMemcpyAsync failed: {err}"

        # the pinned tensor cannot be freed before the copy event is done
        if self._event is None:
            self._event = paddle.cuda.Event()
        self._event.record()

        return out

    def wait(self):
        """Wait the current copy to finish and free the pinned tensor."""
        if self._pin is not None:
            self._event.synchronize()
            self._pin = None

    def __del__(self):
        self.wait()


def grouped_launch(funcs, begin, end, calc_stream, comm_stream, event=None):
    """
    轮流在 calc/comm 两个 stream 上对 funcs 进行分组发射.

    使用两个 stream 可以让前后两个 kernel 重叠, 让下一个 kernel 充分利用上一个 kernel
    的尾部空出来的 SM, 达到类似 group_gemm 的效果.
    """
    if event is None:
        event = paddle.cuda.Event()

    # 对于每组 func，总是从 calc_stream 开始发射，这样同一个 task_idx 的前后 func
    # 必定在同一个 stream 上，可以天然保证同步
    stream_bases = [calc_stream.stream_base, comm_stream.stream_base]
    i = 0

    for n, func in enumerate(funcs):
        # 每组 func 开始前让 comm_stream 等待一次 calc_stream, 保持组的边界
        # 其实从正确性上没有必要, 只是让 timeline 更整齐, 对性能影响未知
        event.record()
        comm_stream.wait_event(event)

        paddle.base.core.nvprof_nvtx_push(f"G{n}")
        for i, task_idx in enumerate(range(begin, end)):
            # 直接调用 core 比用 stream_guard 开销更低, 这里的发射速度非常关键
            paddle.base.core._set_current_stream(stream_bases[i % 2])
            func(task_idx)
        paddle.base.core.nvprof_nvtx_pop()

    if i % 2 != 0:
        paddle.base.core._set_current_stream(stream_bases[0])
