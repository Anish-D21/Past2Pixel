"""
Hardware and Compute Device Detection Module.
Detects CPU/CUDA execution environments, queries memory, and measures processing times honestly.
"""
import os
import platform
import time
from typing import Dict, Any, Optional
import torch


def get_device(preference: str = "auto") -> torch.device:
    """
    Returns the appropriate torch.device based on available hardware and user preference.

    Args:
        preference: 'auto', 'cuda', or 'cpu'

    Returns:
        torch.device
    """
    pref = preference.lower().strip()
    if pref == "cpu":
        return torch.device("cpu")

    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("[Past2Pixel Warning] CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")

    # 'auto' mode
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Gathers detailed hardware information for session reporting.

    Returns:
        Dict containing device name, processor, cores, memory, and CUDA properties.
    """
    if device is None:
        device = get_device()

    info: Dict[str, Any] = {
        "device_type": device.type,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    # Query CPU core counts
    try:
        info["cpu_count_logical"] = os.cpu_count()
    except Exception:
        info["cpu_count_logical"] = None

    # Query CUDA device details if applicable
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_idx = device.index if device.index is not None else torch.cuda.current_device()
        info["gpu_name"] = torch.cuda.get_device_name(cuda_idx)
        info["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated(cuda_idx) / (1024 ** 2), 2)
        info["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved(cuda_idx) / (1024 ** 2), 2)
        info["gpu_total_memory_mb"] = round(
            torch.cuda.get_device_properties(cuda_idx).total_memory / (1024 ** 2), 2
        )
    else:
        info["gpu_name"] = "None (CPU Execution)"
        info["gpu_total_memory_mb"] = 0

    return info


def log_system_info() -> None:
    """Prints a clean summary of the active compute hardware."""
    info = get_device_info()
    print("=" * 60)
    print("Past2Pixel — System & Compute Environment")
    print(f"Platform:        {info['platform']}")
    print(f"Processor:       {info['processor']}")
    print(f"Logical CPUs:    {info['cpu_count_logical']}")
    print(f"PyTorch Version: {info['torch_version']}")
    print(f"Active Device:   {info['device_type'].upper()} ({info['gpu_name']})")
    print("=" * 60)


class Timer:
    """Context manager for precise processing time measurement in milliseconds/seconds."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0

    @property
    def elapsed_seconds(self) -> float:
        return self.elapsed_ms / 1000.0
