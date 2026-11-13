import torch
from typing import List, Tuple

# noinspection PyUnresolvedReferences
from .. import _C

from ..utils.math import align


class BufferAllocator:
    """Plan contiguous, aligned tensors before allocating their CUDA storage."""

    def __init__(self):
        self.alignment = _C.get_num_allocation_alignment()
        self.allocations: List[Tuple[torch.Tensor, int]] = []
        self.num_bytes = 0
        self.materialized = False

    def allocate(self, shape, dtype: torch.dtype) -> torch.Tensor:
        """Append an aligned tensor allocation and return its placeholder."""
        assert not self.materialized, 'A materialized allocation plan cannot be changed'
        # NOTE(paddle-compat): paddle has no 'meta' device, so keep a 0-size placeholder
        # and record the planned shape/dtype/size; real storage is bound in materialize().
        placeholder = torch.empty((0,), dtype=dtype, device='cuda')
        numel = 1
        for s in shape:
            numel *= int(s)
        nbytes = numel * placeholder.element_size()
        offset = align(self.num_bytes, self.alignment)
        self.allocations.append((placeholder, offset, tuple(int(s) for s in shape), dtype, nbytes))
        self.num_bytes = align(offset + nbytes, self.alignment)
        return placeholder

    def materialize(self, storage: torch.Tensor) -> None:
        assert not self.materialized, 'An allocation plan can only be materialized once'
        assert storage.is_contiguous() and storage.dtype == torch.uint8
        assert storage.numel() == self.num_bytes

        for placeholder, offset, shape, dtype, nbytes in self.allocations:
            # NOTE(paddle-compat): paddle's `.view(dtype)` on a byte slice with a non-zero
            # storage offset returns the wrong address ! (it rescales the offset by the new
            # itemsize). Reinterpret the whole storage first (offset 0 is correct), then
            # narrow in element units. The allocation offset/size are itemsize-aligned.
            #
            # Example: storage is a uint8 buffer, we want the bfloat16 (itemsize=2) tensor
            # at byte offset 4, 2 elements (bytes 4..7).
            #   storage bytes:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]
            #   want (bytes 4..7):                    [4, 5, 6, 7]
            #   BAD  storage.narrow(0, 4, 4).view(bf16):  [8, 9, 10, 11]  (offset rescaled 4*2)
            #   GOOD storage.view(bf16).narrow(0, 4//2, 4//2):  [4, 5, 6, 7]
            itemsize = placeholder.element_size()
            value = storage.view(dtype).narrow(0, offset // itemsize, nbytes // itemsize).view(shape)
            # NOTE(paddle-compat): torch.utils.swap_tensors does not exist; adopt the
            # real storage-backed view into the placeholder in place.
            value._share_underline_tensor_to(placeholder)
        self.materialized = True
