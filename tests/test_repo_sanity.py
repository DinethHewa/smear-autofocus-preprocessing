from pathlib import Path


def test_repo_sanity_no_placeholders():
    repo_root = Path(__file__).resolve().parents[1]
    targets = list((repo_root / "configs").glob("*.yaml"))
    targets.append(repo_root / "protocol.md")

    errors = []
    for path in targets:
        if not path.exists():
            errors.append(f"Missing required file: {path}")
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, start=1):
            if line.strip() == "...":
                errors.append(f"{path}:{idx} contains placeholder '...'")
        if path.name == "protocol.md":
            content = "\n".join(lines)
            if "TODO:" in content:
                errors.append(f"{path} contains TODO placeholders")

    assert not errors, " | ".join(errors)
