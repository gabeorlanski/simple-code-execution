"""This file contains the configuration for the code execution module."""

from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Optional

import structlog
from pydantic import BaseModel

logger = structlog.get_logger()


class ExecutionConfig(BaseModel):
    """Base config for general execution."""

    concurrency: int
