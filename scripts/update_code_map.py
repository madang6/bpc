#!/usr/bin/env python3
"""Refresh the code map's flow diagram and source counts from the current source tree."""

import ast
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_SUFFIXES = (".py", ".sh", ".cpp", ".cu", ".h", ".hpp", ".cuh")


def run(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True)


def constant(path, name):
    tree = ast.parse((ROOT / path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"{name} not found in {path}")


def replace(text, name, body):
    start, stop = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    before, rest = text.split(start, 1)
    _, after = rest.split(stop, 1)
    return f"{before}{start}\n{body}\n{stop}{after}"


def flow():
    stages = constant("src/bpc/pipeline.py", "STAGES")
    steps = " --> ".join(f'{name}["{name}"]' for name in stages)
    return f'''```mermaid
flowchart LR
  config["resolved configuration"] --> {steps}
```'''


def source_counts():
    """Count physical lines, including staged/new files in the current worktree."""
    paths = run("git", "ls-files", "--cached", "--others", "--exclude-standard",
                "--", "src/bpc", "scripts", "tests").splitlines()
    return {path: len((ROOT / path).read_text().splitlines())
            for path in sorted(set(paths)) if path.endswith(CODE_SUFFIXES)
            and (ROOT / path).is_file()}


def size_table():
    groups = {
        "Python implementation": lambda p: p.startswith("src/") and p.endswith(".py"),
        "Python/shell scripts": lambda p: p.startswith("scripts/") and p.endswith((".py", ".sh")),
        "CUDA/C++ and headers": lambda p: p.endswith(CODE_SUFFIXES[2:]),
        "Python tests": lambda p: p.startswith("tests/") and p.endswith(".py"),
    }
    items = source_counts()
    counts = {label: sum(n for p, n in items.items() if include(p))
              for label, include in groups.items()}
    return "\n".join([
        "| Scope | Current worktree |", "|---|---:|",
        *(f"| {label} | {count} |" for label, count in counts.items()),
        f"| **Total** | **{sum(counts.values())}** |",
    ])


if __name__ == "__main__":
    document = ROOT / "docs/codebase.md"
    text = replace(document.read_text(), "flow", flow())
    document.write_text(replace(text, "size-table", size_table()))
