"""Fail before an infeasible dense QuickQudits allocation kills the benchmark."""

from math import inf
from pathlib import Path

MAX_WORKING_SET_BYTES = 8 * 1024**3


class DenseMemoryError(MemoryError):
    def __init__(self, vector_bytes: int, estimated_working_set_bytes: int,
                 configured_limit_bytes: int, effective_limit_bytes: int):
        self.vector_bytes = vector_bytes
        self.estimated_working_set_bytes = estimated_working_set_bytes
        self.configured_limit_bytes = configured_limit_bytes
        self.effective_limit_bytes = effective_limit_bytes
        super().__init__(
            f"two dense grids need at least {estimated_working_set_bytes} bytes of statevector/workspace "
            f"({vector_bytes} bytes per player vector); budget is {effective_limit_bytes} bytes"
        )


def dense_memory_estimate(cells: int, dimension: int, qudits_per_cell: int = 1) -> tuple[int, int]:
    vector_bytes = 16 * dimension ** (cells * qudits_per_cell)
    return vector_bytes, 4 * vector_bytes


def set_max_working_set_gib(value: float) -> None:
    if value <= 0:
        raise ValueError("max statevector memory must be positive")
    global MAX_WORKING_SET_BYTES
    MAX_WORKING_SET_BYTES = int(value * 1024**3)


def _available_bytes() -> int:
    available = inf
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
            break
    cgroup = Path("/sys/fs/cgroup/memory.max")
    if cgroup.exists():
        limit = cgroup.read_text().strip()
        if limit != "max":
            current = int(Path("/sys/fs/cgroup/memory.current").read_text())
            available = min(available, int(limit) - current)
    return int(available)


def require_two_dense_grids(cells: int, dimension: int, qudits_per_cell: int = 1) -> None:
    """Budget two full vectors plus two temporary vectors during gate execution."""
    per_grid, minimum = dense_memory_estimate(cells, dimension, qudits_per_cell)
    budget = min(MAX_WORKING_SET_BYTES, _available_bytes() // 2)
    if minimum > budget:
        raise DenseMemoryError(per_grid, minimum, MAX_WORKING_SET_BYTES, budget)
