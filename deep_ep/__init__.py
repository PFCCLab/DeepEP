import filecmp
import glob
import os
import torch

from .utils.find_pkgs import find_nccl_root

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass


def check_nccl_so():
    """
    Verify that the NCCL library loaded at runtime matches the linked version.
    Aborts if duplicate NCCL libraries are found or if versions mismatch.
    """
    if int(os.environ.get('EP_SUPPRESS_NCCL_CHECK', 0)):
        return

    # PyTorch may load another NCCL library, which is different to the linked one
    with open('/proc/self/maps', 'r') as f:
        loaded_nccl_so = None
        for so in [line.strip().split(' ')[-1] for line in f if 'libnccl' in line]:
            loaded_nccl_so = so if loaded_nccl_so is None else loaded_nccl_so
            assert so == loaded_nccl_so, f'Duplicate NCCL runtime found in the current system: {so} and {loaded_nccl_so}'
    linked_nccl_so_candidates = sorted(glob.glob(f'{find_nccl_root()}/lib/libnccl.so*'))
    assert linked_nccl_so_candidates, f'No libnccl.so found in {find_nccl_root()}/lib/'
    linked_nccl_so = linked_nccl_so_candidates[0]

    # So checking binary-level equalness is necessary
    # noinspection PyTypeChecker
    assert filecmp.cmp(loaded_nccl_so, linked_nccl_so, shallow=False), \
        (f'Invalid NCCL versions: {loaded_nccl_so} (loaded) v.s. {linked_nccl_so} (expected), '
         f'please contact Chenggang or Shangyan to upgrade PyTorch NCCL version')


def align_paddle_nccl_dir():
    """
    Point Paddle's `FLAGS_nccl_dir` at the NCCL this extension was linked against.

    NOTE: Paddle-specific. A pip-installed Paddle wheel unconditionally sets that flag to its own
    bundled NCCL while `import paddle` runs, which is not necessarily the version DeepEP needs
    (DeepEP V2.5 requires >= 2.32.3 for the GIN API). Two NCCL runtimes in one process segfault as
    soon as Paddle creates a communicator -- and that happens after `check_nccl_so()` below, so the
    check cannot catch it. Realign the flag before anything builds a process group.
    """
    import paddle

    nccl_lib_dir = os.path.join(find_nccl_root(), 'lib')
    if not os.path.isdir(nccl_lib_dir):
        return
    if paddle.get_flags(['FLAGS_nccl_dir'])['FLAGS_nccl_dir'] != nccl_lib_dir:
        paddle.set_flags({'FLAGS_nccl_dir': nccl_lib_dir})


def init_jit():
    """
    Initialize the JIT compilation runtime.
    """
    # noinspection PyUnresolvedReferences
    from . import _C
    library_root_path = os.path.dirname(os.path.abspath(__file__))
    # NOTE: pass Paddle's include root so the runtime JIT (deep_jit) nvcc can
    # resolve the torch-compat headers (<ATen/cuda/CUDAContext.h> etc.) that the kernels include.
    import paddle
    paddle_include_root = paddle.sysconfig.get_include()
    _C.init_jit(library_root_path, find_nccl_root(), paddle_include_root)


# Run initialization
align_paddle_nccl_dir()
check_nccl_so()
init_jit()


# Import APIs after initialization
from . import comm
from .comm import destroy_all_managed_nccl_comm, get_physical_domain_size, get_logical_domain_size
from .buffers.allocator import BufferAllocator
from .buffers.base import BufferBase
from .buffers.ep import EPBuffer, EPHandle
from .buffers.engram import EngramBuffer
from .buffers.bucket import BucketBuffer, BucketSession
from .buffers.pp import PPBuffer
# noinspection PyUnresolvedReferences
from .utils.event import EventOverlap, EventHandle

# noinspection PyUnresolvedReferences
from ._C import (
    get_num_allocation_alignment,
    get_num_rdma_alignment,
    get_num_tma_alignment,
    topk_idx_t,
)

__version__ = '2.5.0'
