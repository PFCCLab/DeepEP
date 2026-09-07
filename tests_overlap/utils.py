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
