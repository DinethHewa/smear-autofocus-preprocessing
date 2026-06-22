from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


LIBRARIES = (
    "pandas",
    "numpy",
    "polars",
    "pyarrow",
    "cudf",
    "cupy",
    "cuml",
    "dask",
    "dask_cuda",
    "numba",
    "joblib",
    "cv2",
    "torch",
    "tensorflow",
    "psutil",
    "memory_profiler",
    "line_profiler",
)


def _import_status(name: str) -> Dict[str, Any]:
    module_name = "cv2" if name == "opencv-python" else name
    spec = importlib.util.find_spec(module_name)
    out: Dict[str, Any] = {"installed": spec is not None, "version": ""}
    if spec is None:
        return out
    try:
        module = __import__(module_name)
        out["version"] = str(getattr(module, "__version__", "unknown"))
    except Exception as exc:
        out["installed"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _ram() -> Dict[str, Any]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {"total_bytes": int(vm.total), "available_bytes": int(vm.available)}
    except Exception:
        return {"total_bytes": None, "available_bytes": None}


def _gpu() -> Dict[str, Any]:
    out: Dict[str, Any] = {"nvidia_smi": False, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        out["nvidia_smi"] = True
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                check=False,
                text=True,
                capture_output=True,
                timeout=10,
            )
            out["nvidia_smi_output"] = result.stdout.strip()
        except Exception as exc:
            out["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import torch

        out["torch_cuda_available"] = bool(torch.cuda.is_available())
        out["torch_cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        out["torch_cuda_available"] = False
    try:
        import tensorflow as tf

        out["tensorflow_gpu_devices"] = [str(device) for device in tf.config.list_physical_devices("GPU")]
    except Exception:
        out["tensorflow_gpu_devices"] = []
    return out


def _disk() -> Dict[str, Any]:
    usage = shutil.disk_usage(Path.cwd())
    return {
        "cwd": str(Path.cwd()),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
    }


def classify(report: Dict[str, Any]) -> Dict[str, str]:
    libs = report["libraries"]
    gpu = report["gpu"]
    has_polars = bool(libs.get("polars", {}).get("installed"))
    has_joblib = bool(libs.get("joblib", {}).get("installed"))
    has_gpu_lib = any(bool(libs.get(name, {}).get("installed")) for name in ("cudf", "cupy", "cuml"))
    gpu_available = bool(gpu.get("nvidia_smi") or gpu.get("torch_cuda_available") or gpu.get("tensorflow_gpu_devices"))
    return {
        "cpu_only_path": "available",
        "cpu_polars_path": "available" if has_polars else "not_available_polars_missing",
        "cpu_multiprocessing_path": "available" if has_joblib or os.cpu_count() else "available_stdlib_only",
        "gpu_available_path": "available_with_optional_libraries" if gpu_available and has_gpu_lib else "not_selected",
        "gpu_not_useful_path": "use CPU for image-file loops and many tiny operations; GPU only helps large numeric batches/dataframes",
    }


def build_report() -> Dict[str, Any]:
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "ram": _ram(),
        "disk": _disk(),
        "gpu": _gpu(),
        "libraries": {name: _import_status(name) for name in LIBRARIES},
    }
    report["acceleration_classification"] = classify(report)
    return report


def main() -> None:
    out_dir = Path("performance_audit/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = out_dir / "environment_report.json"
    md_path = Path("performance_audit/optimization_report_environment.md")
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    lines = ["# Environment Report", ""]
    lines.append(f"- Python: `{report['python'].split()[0]}`")
    lines.append(f"- Platform: `{report['platform']}`")
    lines.append(f"- Logical CPU cores: `{report['cpu_count_logical']}`")
    lines.append(f"- RAM total bytes: `{report['ram'].get('total_bytes')}`")
    lines.append(f"- GPU classification: `{report['acceleration_classification']['gpu_available_path']}`")
    lines.append("")
    lines.append("## Libraries")
    for name, status in report["libraries"].items():
        lines.append(f"- `{name}`: installed={status.get('installed')} version={status.get('version', '')}")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report["acceleration_classification"], indent=2))
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
