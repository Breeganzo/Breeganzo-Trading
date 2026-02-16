from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_no_demo_or_temp_artifacts_tracked():
    tracked = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files"],
        text=True,
    ).splitlines()

    disallowed = []
    for rel in tracked:
        p = Path(rel)
        lower = rel.lower()
        name = p.name.lower()
        stem = p.stem.lower()

        if name == ".ds_store":
            disallowed.append(rel)
            continue
        if name.endswith(".bak") or name.endswith(".tmp") or name.endswith(".swp"):
            disallowed.append(rel)
            continue
        if "__pycache__" in p.parts:
            disallowed.append(rel)
            continue
        if lower.startswith("tmp/") or "/tmp/" in lower:
            disallowed.append(rel)
            continue
        if lower.startswith("tests/fixtures/demo_"):
            disallowed.append(rel)
            continue
        if name.startswith("demo") or stem.startswith("demo"):
            disallowed.append(rel)
            continue
        if ("_demo" in stem or "demo_" in stem) and "no_demo" not in stem:
            disallowed.append(rel)
            continue
        if "webapp/static/demo_" in lower:
            disallowed.append(rel)
            continue

    assert not disallowed, f"Tracked demo/temp artifacts found: {disallowed}"
