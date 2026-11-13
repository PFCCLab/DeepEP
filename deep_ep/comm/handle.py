import os
import time
from typing import Any, Optional, Tuple

import paddle.distributed as dist
from paddle.distributed.communication.group import Group

# noinspection PyUnresolvedReferences
from .. import _C


class NCCLCommHandle:
    """
    A wrapper around a raw NCCL communicator. Manages the lifecycle of the communicator if created by DeepEP,
    or simply wraps an existing one if obtained from PyTorch.

    Attributes:
        nccl_comm: the raw NCCL communicator.
        managed: whether the communicator was created by DeepEP and should be destroyed when this handle is dropped.
    """

    def __init__(self, nccl_comm: int, managed: bool):
        self.nccl_comm = nccl_comm
        self.managed = managed
        self.destroy = _C.destroy_nccl_comm

    def __del__(self):
        if self.managed:
            self.destroy(self.nccl_comm)

    def get(self) -> int:
        """
        Get the raw NCCL communicator.

        Returns:
            nccl_comm: the raw NCCL communicator.
        """
        return self.nccl_comm


_storage = dict()


def get_nccl_comm_handle(group: Group, force_new_comm: bool = False) -> NCCLCommHandle:
    """
    Get or create an NCCL communicator handle for the given process group.
    Results are cached, so subsequent calls with the same group return the same handle.

    Arguments:
        group: the communication group (Paddle ``Group``).
        force_new_comm: if set, never reuse Paddle's communicator and never hit the cache; always
            create a fresh DeepEP-managed comm (needed for hybrid + CPU segment, `num_cpu_bytes>0`).

    Returns:
        handle: the NCCL communicator handle.
    """
    # Check cache hit
    global _storage
    if not force_new_comm and group in _storage:
        return _storage[group]

    # NOTE(paddle-compat): reuse Paddle's own NCCL communicator
    # (`group.id -> ProcessGroupNCCL -> GetNcclComm`), borrowed (managed=False): DeepEP must
    # NOT destroy it; Paddle owns its lifecycle. This avoids the extra self-built comm and its
    # dangling-handle risk.
    if not force_new_comm and int(os.getenv('EP_REUSE_NCCL_COMM', '1')):
        _storage[group] = NCCLCommHandle(_C.get_nccl_comm_from_group(group.id), managed=False)
        return _storage[group]

    # Fallback / hybrid path: build a fresh DeepEP-managed comm via all-gathered unique id.
    nccl_unique_ids = [None, ] * group.world_size
    dist.all_gather_object(nccl_unique_ids, _C.get_local_nccl_unique_id(), group)
    root_unique_id = nccl_unique_ids[0]
    key = time.time_ns() if force_new_comm else group
    _storage[key] = NCCLCommHandle(
        _C.create_nccl_comm(root_unique_id, group.world_size, group.rank), managed=True)
    return _storage[key]


def get_physical_domain_size(group_or_buffer: Any) -> Tuple[int, int]:
    """Return physical RDMA and NVLink domain sizes for a process group or buffer."""
    context = getattr(group_or_buffer, 'context', None)
    if context is not None:
        return context.get_physical_domain_size()
    return _C.get_physical_domain_size(get_nccl_comm_handle(group_or_buffer).get())


def get_logical_domain_size(group_or_buffer: Any,
                            allow_hybrid_mode: Optional[bool] = None) -> Tuple[int, int]:
    """Return logical scaleout and scaleup domain sizes for a process group or buffer."""
    context = getattr(group_or_buffer, 'context', None)
    if context is not None:
        return context.get_logical_domain_size()
    allow_hybrid_mode = True if allow_hybrid_mode is None else allow_hybrid_mode
    return _C.get_logical_domain_size(get_nccl_comm_handle(group_or_buffer).get(), allow_hybrid_mode)


def destroy_all_managed_nccl_comm() -> None:
    """
    Destroy all cached NCCL communicator handles and clear the cache.

    """
    _storage.clear()
