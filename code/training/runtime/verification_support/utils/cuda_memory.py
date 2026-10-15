"""Project-local CUDA allocator defaults.

Entry points call this before importing torch.  Users therefore do not need to
prepend an allocator environment variable to the normal training command.
An allocator configuration explicitly supplied by the server environment is
left untouched.
"""
from __future__ import annotations

import os

CUDA_ALLOC_CONF_ENV = "PYTORCH_CUDA_ALLOC_CONF"
DEFAULT_CUDA_ALLOC_CONF = "expandable_segments:True"


def configure_cuda_allocator() -> str:
    """Set the project default allocator policy unless one already exists."""
    current = os.environ.get(CUDA_ALLOC_CONF_ENV)
    if current:
        return current
    os.environ[CUDA_ALLOC_CONF_ENV] = DEFAULT_CUDA_ALLOC_CONF
    return DEFAULT_CUDA_ALLOC_CONF
