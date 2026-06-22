from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


def _count_files(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _monitor(proc: subprocess.Popen) -> Dict[str, int]:
    peak = 0
    try:
        import psutil

        ps_proc = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                rss = ps_proc.memory_info().rss
                for child in ps_proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except Exception:
                        pass
                peak = max(peak, rss)
            except Exception:
                pass
            time.sleep(0.2)
    except Exception:
        status_path = Path(f"/proc/{proc.pid}/status")
        while proc.poll() is None:
            try:
                for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.startswith("VmRSS:"):
                        peak = max(peak, int(line.split()[1]) * 1024)
                        break
            except Exception:
                pass
            time.sleep(0.2)
    return {"peak_rss_bytes": int(peak)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a pipeline/script subprocess.")
    parser.add_argument("--target", required=True, choices=["original", "optimized"])
    parser.add_argument("--script", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    script = Path(args.script)
    if not script.exists():
        raise FileNotFoundError(script)
    output_dir = Path(args.output_dir) if args.output_dir else None
    before_files = _count_files(output_dir) if output_dir else 0
    script_args = list(args.script_args)
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]
    command = [sys.executable, str(script), *script_args]
    start = time.perf_counter()
    proc = subprocess.Popen(command, text=True)
    monitor = _monitor(proc)
    returncode = proc.wait()
    runtime = time.perf_counter() - start
    after_files = _count_files(output_dir) if output_dir else 0
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target": args.target,
        "script": str(script),
        "args": " ".join(script_args),
        "runtime_seconds": f"{runtime:.6f}",
        "peak_memory_bytes": monitor["peak_rss_bytes"],
        "input_file_count": "",
        "output_file_count": after_files,
        "output_file_count_delta": after_files - before_files,
        "rows_processed": "",
        "images_processed": "",
        "cpu_gpu_mode": "cpu_or_script_default",
        "returncode": returncode,
    }
    out_path = Path("performance_audit/speedup_summary.csv")
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2))
    if returncode != 0:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
