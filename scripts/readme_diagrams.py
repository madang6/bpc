#!/usr/bin/env python3
"""Refresh README's flow diagram and code-size table from code and Git."""

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
    stages = constant("src/dpc_nldl/pipeline.py", "STAGES")
    steps = " --> ".join(f'{name}["{name}"]' for name in stages)
    return f'''```mermaid
flowchart LR
  config["resolved configuration"] --> {steps}
```'''


def source_counts(revision=None):
    """Count physical lines, including staged/new files in the current worktree."""
    command = ("ls-tree", "-r", "--name-only", revision) if revision else (
        "ls-files", "--cached", "--others", "--exclude-standard")
    paths = run("git", *command, "--", "src/dpc_nldl", "scripts", "tests").splitlines()
    return {path: len((run("git", "show", f"{revision}:{path}") if revision else
                      (ROOT / path).read_text()).splitlines())
            for path in sorted(set(paths)) if path.endswith(CODE_SUFFIXES)
            and (revision or (ROOT / path).is_file())}


def size_table():
    revisions = ("97ebf4a", "48bc022", "2a00d4e", "6c8e70e", "2cb8291", "ab29f39", "fbb4641", None)
    groups = {
        "Python implementation": lambda p: p.startswith("src/") and p.endswith(".py"),
        "Python/shell scripts": lambda p: p.startswith("scripts/") and p.endswith((".py", ".sh")),
        "CUDA/C++ and headers": lambda p: p.endswith(CODE_SUFFIXES[2:]),
        "Python tests": lambda p: p.startswith("tests/") and p.endswith(".py"),
    }
    # Verified source-repository comparison outside this branch's ancestry.
    archived = {"48bc022": (10201, 5677, 1282, 10516)}
    counts = []
    for revision in revisions:
        if revision in archived:
            counts.append(archived[revision])
        else:
            items = source_counts(revision)
            counts.append(tuple(sum(n for p, n in items.items() if include(p))
                                for include in groups.values()))
    rows = []
    for index, label in enumerate(groups):
        values = " | ".join(str(items[index]) for items in counts)
        rows.append(f"| {label} | {values} |")
    totals = " | ".join(f"**{sum(items)}**" for items in counts)
    rows.append(f"| **Total** | {totals} |")
    return "\n".join([
        "| Scope | " + " | ".join(f"`{r}`" if r else "Current worktree" for r in revisions) + " |",
        "|---|" + "---:|" * len(revisions), *rows,
    ])


if __name__ == "__main__":
    readme = ROOT / "README.md"
    text = replace(readme.read_text(), "flow", flow())
    readme.write_text(replace(text, "size-table", size_table()))
