from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class GPUStatus:
    nvidia_smi: bool
    nvidia_summary: str
    cupy: bool
    cudf: bool
    cuml: bool
    torch_cuda: bool
    tensorflow_gpu: bool


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def detect_gpu() -> GPUStatus:
    nvidia_summary = ""
    nvidia_smi = shutil.which("nvidia-smi") is not None
    if nvidia_smi:
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            nvidia_summary = proc.stdout.strip()
        except Exception:
            nvidia_summary = ""
    torch_cuda = False
    if _module_available("torch"):
        try:
            import torch

            torch_cuda = bool(torch.cuda.is_available())
        except Exception:
            torch_cuda = False
    tensorflow_gpu = False
    if _module_available("tensorflow"):
        try:
            import tensorflow as tf

            tensorflow_gpu = bool(tf.config.list_physical_devices("GPU"))
        except Exception:
            tensorflow_gpu = False
    return GPUStatus(
        nvidia_smi=nvidia_smi,
        nvidia_summary=nvidia_summary,
        cupy=_module_available("cupy"),
        cudf=_module_available("cudf"),
        cuml=_module_available("cuml"),
        torch_cuda=torch_cuda,
        tensorflow_gpu=tensorflow_gpu,
    )


def choose_gpu_mode(use_gpu: Literal["auto", "yes", "no"] = "auto", *, workload: str = "unknown") -> str:
    status = detect_gpu()
    gpu_ready = status.cupy or status.cudf or status.torch_cuda
    suitable = workload in {"large_numeric_arrays", "large_feature_matrix", "large_dataframe_groupby", "batch_tensor"}
    if use_gpu == "no":
        return "cpu"
    if use_gpu == "yes" and not gpu_ready:
        raise RuntimeError("GPU requested, but no usable GPU dataframe/array backend is installed")
    if use_gpu == "yes" and suitable:
        return "gpu"
    if use_gpu == "auto" and gpu_ready and suitable:
        return "gpu"
    return "cpu"


def main() -> None:
    status = detect_gpu()
    print(json.dumps({"status": asdict(status), "recommended_default": "cpu_with_optional_gpu_for_large_batches"}, indent=2))


if __name__ == "__main__":
    main()
