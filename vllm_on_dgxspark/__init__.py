"""Runtime system inspection helpers for compatibility checks."""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
from importlib import metadata
from typing import List, Optional, TypedDict

import psutil
import torch

__version__: str = "0.1.0"
_COMMAND_TIMEOUT_SECONDS: float = 2.0
_OPTIONAL_LLM_DISTS: tuple[str, ...] = (
    "vllm",
    "transformers",
    "tokenizers",
    "accelerate",
    "triton",
    "xformers",
    "flash-attn",
    "bitsandbytes",
    "deepspeed",
    "peft",
    "safetensors",
)


def get_system_info(include_llm_build_details: bool = False) -> "SystemInfo":
    """Collect system, CUDA, and optional LLM build metadata."""
    virtual_memory: psutil._common.svmem = psutil.virtual_memory()
    cuda_available: bool = torch.cuda.is_available()
    gpu_devices: List[GpuInfo] = _get_gpu_infos(cuda_available=cuda_available)
    nvidia_smi_info: Optional[NvidiaSmiInfo] = _get_nvidia_smi_info()

    llm_libs: LlmLibVersions = _get_llm_lib_versions()
    llm_build: Optional[LlmBuildInfo] = (
        _get_llm_build_info(llm_libs=llm_libs) if include_llm_build_details else None
    )

    compatibility_notes: List[str] = _build_compatibility_notes(
        llm_libs=llm_libs,
        llm_build=llm_build,
        cuda_available=cuda_available,
        nvidia_smi_info=nvidia_smi_info,
    )

    return {
        "python": {
            "version": sys.version.split(" ", maxsplit=1)[0],
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "host": {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "machine": platform.machine(),
            "cpu_physical_cores": psutil.cpu_count(logical=False) or 0,
            "cpu_logical_cores": psutil.cpu_count(logical=True) or 0,
            "memory_total_bytes": int(virtual_memory.total),
            "memory_available_bytes": int(virtual_memory.available),
        },
        "pytorch": {
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": len(gpu_devices),
        },
        "nvidia_driver_runtime": {
            "nvidia_driver_version": (
                nvidia_smi_info["driver_version"] if nvidia_smi_info else None
            ),
            "nvidia_smi_version": (
                nvidia_smi_info["nvidia_smi_version"] if nvidia_smi_info else None
            ),
            "cuda_runtime_reported_by_driver": (
                nvidia_smi_info["cuda_runtime_version"] if nvidia_smi_info else None
            ),
        },
        "cuda_toolchain": {
            "nvcc_version": _get_nvcc_cuda_version(),
            "cuda_home": _get_cuda_home(),
            "ld_library_path_contains_cuda": _ld_library_path_contains_cuda(),
        },
        "gpu_devices": gpu_devices,
        "llm_libs": llm_libs,
        "llm_build": llm_build,
        "compatibility_notes": compatibility_notes,
    }


class PythonInfo(TypedDict):
    """Python runtime properties."""

    version: str
    implementation: str
    executable: str


class HostInfo(TypedDict):
    """Host compute and memory properties."""

    platform: str
    platform_release: str
    machine: str
    cpu_physical_cores: int
    cpu_logical_cores: int
    memory_total_bytes: int
    memory_available_bytes: int


class PytorchInfo(TypedDict):
    """PyTorch and CUDA runtime details."""

    torch_version: str
    cuda_available: bool
    torch_cuda_version: Optional[str]
    cudnn_version: Optional[int]
    device_count: int


class NvidiaDriverRuntimeInfo(TypedDict):
    """NVIDIA driver/runtime versions from nvidia-smi."""

    nvidia_driver_version: Optional[str]
    nvidia_smi_version: Optional[str]
    cuda_runtime_reported_by_driver: Optional[str]


class CudaToolchainInfo(TypedDict):
    """CUDA toolchain details for build checks."""

    nvcc_version: Optional[str]
    cuda_home: Optional[str]
    ld_library_path_contains_cuda: bool


class GpuInfo(TypedDict):
    """GPU hardware metadata."""

    index: int
    name: str
    total_memory_bytes: int
    capability: str


class NvidiaSmiInfo(TypedDict):
    """NVIDIA-SMI parsed information."""

    driver_version: Optional[str]
    nvidia_smi_version: Optional[str]
    cuda_runtime_version: Optional[str]


class LlmLibVersions(TypedDict):
    """Installed versions for LLM-adjacent libraries."""

    vllm_version: Optional[str]
    transformers_version: Optional[str]
    tokenizers_version: Optional[str]
    accelerate_version: Optional[str]
    triton_version: Optional[str]
    xformers_version: Optional[str]
    flash_attn_version: Optional[str]
    bitsandbytes_version: Optional[str]
    deepspeed_version: Optional[str]
    peft_version: Optional[str]
    safetensors_version: Optional[str]


class VllmBuildInfo(TypedDict):
    """vLLM package build metadata."""

    installed: bool
    version: Optional[str]
    module_file: Optional[str]
    commit: Optional[str]


class FlashAttnBuildInfo(TypedDict):
    """FlashAttention package build metadata."""

    installed: bool
    version: Optional[str]
    module_file: Optional[str]


class XformersBuildInfo(TypedDict):
    """xFormers package build metadata."""

    installed: bool
    version: Optional[str]
    module_file: Optional[str]


class BitsAndBytesBuildInfo(TypedDict):
    """bitsandbytes package build metadata."""

    installed: bool
    version: Optional[str]
    module_file: Optional[str]


class LlmBuildInfo(TypedDict):
    """Optional deep build information for selected libraries."""

    vllm: VllmBuildInfo
    flash_attn: FlashAttnBuildInfo
    xformers: XformersBuildInfo
    bitsandbytes: BitsAndBytesBuildInfo


class SystemInfo(TypedDict):
    """Top-level environment compatibility report."""

    python: PythonInfo
    host: HostInfo
    pytorch: PytorchInfo
    nvidia_driver_runtime: NvidiaDriverRuntimeInfo
    cuda_toolchain: CudaToolchainInfo
    gpu_devices: List[GpuInfo]
    llm_libs: LlmLibVersions
    llm_build: Optional[LlmBuildInfo]
    compatibility_notes: List[str]


def _run_command(command: List[str]) -> Optional[str]:
    """Run command safely and return stdout when successful."""
    try:
        completed: subprocess.CompletedProcess[str] = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if completed.returncode != 0:
        return None

    output: str = completed.stdout.strip()
    return output or None


def _get_gpu_infos(cuda_available: bool) -> List[GpuInfo]:
    """Collect GPU details from torch CUDA device properties."""
    if not cuda_available:
        return []

    gpus: List[GpuInfo] = []
    for gpu_index in range(torch.cuda.device_count()):
        properties: torch.cuda.device_properties._CudaDeviceProperties = (
            torch.cuda.get_device_properties(gpu_index)
        )
        gpus.append(
            {
                "index": gpu_index,
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "capability": f"{properties.major}.{properties.minor}",
            }
        )
    return gpus


def _get_nvidia_smi_info() -> Optional[NvidiaSmiInfo]:
    """Collect driver and CUDA runtime versions via nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return None

    output: Optional[str] = _run_command(["nvidia-smi"])
    if output is None:
        return None

    driver_match: Optional[re.Match[str]] = re.search(
        r"Driver Version:\s*([^\s]+)", output
    )
    smi_match: Optional[re.Match[str]] = re.search(r"NVIDIA-SMI\s+([^\s]+)", output)
    cuda_runtime_match: Optional[re.Match[str]] = re.search(
        r"CUDA Version:\s*([^\s|]+)", output
    )

    return {
        "driver_version": driver_match.group(1) if driver_match else None,
        "nvidia_smi_version": smi_match.group(1) if smi_match else None,
        "cuda_runtime_version": (
            cuda_runtime_match.group(1) if cuda_runtime_match else None
        ),
    }


def _get_nvcc_cuda_version() -> Optional[str]:
    """Collect CUDA toolkit version from nvcc if installed."""
    if shutil.which("nvcc") is None:
        return None

    output: Optional[str] = _run_command(["nvcc", "--version"])
    if output is None:
        return None

    version_match: Optional[re.Match[str]] = re.search(r"release\s+(\d+\.\d+)", output)
    return version_match.group(1) if version_match else None


def _get_cuda_home() -> Optional[str]:
    """Read CUDA install root from environment."""
    cuda_home_env: Optional[str] = __import__("os").environ.get(
        "CUDA_HOME"
    ) or __import__("os").environ.get("CUDA_PATH")
    return cuda_home_env


def _ld_library_path_contains_cuda() -> bool:
    """Check if LD_LIBRARY_PATH references CUDA paths."""
    ld_library_path: str = __import__("os").environ.get("LD_LIBRARY_PATH", "")
    if not ld_library_path:
        return False
    return any("cuda" in segment.lower() for segment in ld_library_path.split(":"))


def _get_distribution_version(distribution: str) -> Optional[str]:
    """Get installed distribution version without importing package."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _get_llm_lib_versions() -> LlmLibVersions:
    """Collect optional LLM-related library versions."""
    versions: dict[str, Optional[str]] = {}
    for distribution in _OPTIONAL_LLM_DISTS:
        normalized_key: str = distribution.replace("-", "_")
        versions[f"{normalized_key}_version"] = _get_distribution_version(distribution)

    return {
        "vllm_version": versions.get("vllm_version"),
        "transformers_version": versions.get("transformers_version"),
        "tokenizers_version": versions.get("tokenizers_version"),
        "accelerate_version": versions.get("accelerate_version"),
        "triton_version": versions.get("triton_version"),
        "xformers_version": versions.get("xformers_version"),
        "flash_attn_version": versions.get("flash_attn_version"),
        "bitsandbytes_version": versions.get("bitsandbytes_version"),
        "deepspeed_version": versions.get("deepspeed_version"),
        "peft_version": versions.get("peft_version"),
        "safetensors_version": versions.get("safetensors_version"),
    }


def _get_llm_build_info(llm_libs: LlmLibVersions) -> LlmBuildInfo:
    """Collect optional deep build metadata for selected libraries."""
    return {
        "vllm": _probe_vllm_build_info(llm_libs=llm_libs),
        "flash_attn": _probe_basic_build_info(
            module_name="flash_attn",
            version=llm_libs["flash_attn_version"],
        ),
        "xformers": _probe_basic_build_info(
            module_name="xformers",
            version=llm_libs["xformers_version"],
        ),
        "bitsandbytes": _probe_basic_build_info(
            module_name="bitsandbytes",
            version=llm_libs["bitsandbytes_version"],
        ),
    }


def _probe_vllm_build_info(llm_libs: LlmLibVersions) -> VllmBuildInfo:
    """Probe vLLM module metadata while keeping import optional."""
    vllm_version: Optional[str] = llm_libs["vllm_version"]
    if vllm_version is None:
        return {
            "installed": False,
            "version": None,
            "module_file": None,
            "commit": None,
        }

    module_file: Optional[str] = None
    commit: Optional[str] = None
    try:
        import vllm  # type: ignore[import-not-found]

        module_file = getattr(vllm, "__file__", None)
        commit = getattr(vllm, "__commit__", None)
    except Exception:
        pass

    return {
        "installed": True,
        "version": vllm_version,
        "module_file": module_file,
        "commit": commit,
    }


def _probe_basic_build_info(
    module_name: str, version: Optional[str]
) -> FlashAttnBuildInfo | XformersBuildInfo | BitsAndBytesBuildInfo:
    """Probe lightweight module metadata for optional libraries."""
    if version is None:
        return {"installed": False, "version": None, "module_file": None}

    module_file: Optional[str] = None
    try:
        module: object = __import__(module_name)
        module_file = getattr(module, "__file__", None)
    except Exception:
        pass

    return {"installed": True, "version": version, "module_file": module_file}


def _build_compatibility_notes(
    llm_libs: LlmLibVersions,
    llm_build: Optional[LlmBuildInfo],
    cuda_available: bool,
    nvidia_smi_info: Optional[NvidiaSmiInfo],
) -> List[str]:
    """Build actionable compatibility warnings for quick triage."""
    notes: List[str] = []

    if llm_libs["vllm_version"] is not None and not cuda_available:
        notes.append("vLLM is installed but torch.cuda.is_available() is false.")

    if llm_libs["flash_attn_version"] is not None and not cuda_available:
        notes.append("flash-attn is installed but CUDA runtime is unavailable.")

    if llm_libs["bitsandbytes_version"] is not None and not cuda_available:
        notes.append("bitsandbytes is installed without CUDA runtime visibility.")

    if cuda_available and nvidia_smi_info is None:
        notes.append(
            "CUDA is available in torch, but nvidia-smi is missing or failing."
        )

    if llm_build is not None and llm_build["vllm"]["installed"]:
        if llm_build["vllm"]["module_file"] is None:
            notes.append("vLLM is installed but module path could not be resolved.")

    return notes
