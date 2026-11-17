#!/usr/bin/env python3
"""Run component, workflow, or GPU checks with the standard unittest runner."""
import argparse
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

from tests.suites import cases, category


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "unit", "integration", "cuda"), default="all")
    parser.add_argument("--list", action="store_true", help="list selected tests without running them")
    args = parser.parse_args()
    discovered = list(cases(unittest.defaultTestLoader.discover(
        str(ROOT / "tests"), top_level_dir=str(ROOT))))
    if unittest.defaultTestLoader.errors:
        parser.exit(1, "\n".join(unittest.defaultTestLoader.errors))
    identities = [case.id() for case in discovered]
    if len(identities) != len(set(identities)):
        parser.error("duplicate test discovery: share fixtures instead of importing TestCase classes")
    selected = [case for case in discovered
                if args.suite == "all" or category(case) == args.suite]
    if args.list:
        for case in selected:
            print(f"{category(case):11} {case.id()}")
        return 0
    if args.suite == "cuda":
        import torch
        if not torch.cuda.is_available():
            parser.error("the CUDA suite requires a visible GPU and a CUDA-enabled PyTorch runtime")
    result = unittest.TextTestRunner().run(unittest.TestSuite(selected))
    return int(not result.wasSuccessful())


if __name__ == "__main__":
    raise SystemExit(main())
