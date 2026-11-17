# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import os
import torch

from .runtime_paths import configure_runtime_paths


def _env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
    }


configure_runtime_paths()

from .hybrid_ep_buffer import HybridEPBuffer

from hybrid_ep_cpp import HybridEpConfigInstance

if not _env_flag_enabled("HYBRID_EP_SKIP_DEEP_EP"):
    from .utils import EventOverlap, get_event_from_comm_stream
    from .buffer import Buffer

    # noinspection PyUnresolvedReferences
    from deep_ep_cpp import Config
