from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import runpy
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path


def _safe_name(script: str) -> str:
    return Path(script).as_posix().replace("/", "_").replace("\\", "_").replace(".", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a Python script with cProfile/tracemalloc.")
    parser.add_argument("--script", required=True)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    script = Path(args.script).resolve()
    if not script.exists():
        raise FileNotFoundError(script)
    out_dir = Path("performance_audit/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(str(script.relative_to(Path.cwd())) if script.is_relative_to(Path.cwd()) else script.name)

    old_argv = sys.argv[:]
    script_args = list(args.script_args)
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]
    sys.argv = [str(script), *script_args]
    profile = cProfile.Profile()
    tracemalloc.start()
    start = time.perf_counter()
    error = ""
    try:
        profile.enable()
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            error = f"SystemExit({exc.code})"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        profile.disable()
        runtime = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        sys.argv = old_argv

    stats_stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stats_stream).strip_dirs().sort_stats("cumulative")
    stats_stream.write("\nTop 20 by cumulative time\n")
    stats.print_stats(20)
    stats.sort_stats("tottime")
    stats_stream.write("\nTop 20 by total time\n")
    stats.print_stats(20)
    (out_dir / f"cprofile_{stem}.txt").write_text(stats_stream.getvalue(), encoding="utf-8")
    (out_dir / f"memory_{stem}.txt").write_text(
        f"current_tracemalloc_bytes={current}\npeak_tracemalloc_bytes={peak}\n",
        encoding="utf-8",
    )
    runtime_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(script),
        "args": script_args,
        "runtime_seconds": runtime,
        "peak_tracemalloc_bytes": peak,
        "error": error,
    }
    (out_dir / f"runtime_{stem}.json").write_text(json.dumps(runtime_payload, indent=2), encoding="utf-8")
    print(json.dumps(runtime_payload, indent=2))


if __name__ == "__main__":
    main()
