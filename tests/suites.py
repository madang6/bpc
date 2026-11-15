"""Explicit test groups; undecorated tests exercise individual components."""
import unittest

import torch


def integration(test):
    """Exercise a workflow spanning components, persistence, or worker processes."""
    test.test_suite = "integration"
    return test


def cuda(test):
    """Require real GPU execution, whether testing a component or a workflow."""
    test.test_suite = "cuda"
    return unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")(test)


def cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from cases(item)
        else:
            yield item


def category(case):
    method = getattr(getattr(case, case._testMethodName), "test_suite", None)
    owner = getattr(type(case), "test_suite", "unit")
    return "cuda" if "cuda" in (method, owner) else method or owner
