"""Full-FP32 scopes preserve nested execution and historical backend restoration."""
import unittest

import torch

from bpc.precision import ieee_fp32


class MatmulPrecisionTest(unittest.TestCase):
    def test_nested_scopes_restore_backend_state_on_success_and_failure(self):
        precision = torch.get_float32_matmul_precision()
        tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            for requested in ('highest', 'high', 'medium'):
                for first, restore_first in ((False, False), (False, True), (True, True)):
                    for fail in (False, True):
                        with self.subTest(precision=requested, tf32_first=first,
                                          restore_tf32_first=restore_first, fail=fail):
                            torch.set_float32_matmul_precision(requested)
                            try:
                                with ieee_fp32(tf32_first=first, restore_tf32_first=restore_first):
                                    with ieee_fp32():
                                        self.assertEqual(torch.get_float32_matmul_precision(), 'highest')
                                        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                                    self.assertEqual(torch.get_float32_matmul_precision(), 'highest')
                                    if fail:
                                        raise RuntimeError('interrupted numerical operation')
                            except RuntimeError as error:
                                self.assertEqual(str(error), 'interrupted numerical operation')
                            # The TF32 setter maps medium to high when it runs last.
                            expected = ('high' if requested == 'medium' and not restore_first
                                        else requested)
                            self.assertEqual(torch.get_float32_matmul_precision(), expected)
                            self.assertEqual(torch.backends.cuda.matmul.allow_tf32, requested != 'highest')
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.set_float32_matmul_precision(precision)
