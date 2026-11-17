"""Native-history early cast preserves lookup inputs; temporal FP64 is unchanged."""
import unittest
from tests.suites import cuda
import torch
from tests.support.policies import toy_policy

class LookupGatherPrecisionTest(unittest.TestCase):
    def check_gather(self, device):
        p=toy_policy(5)
        for key,value in vars(p).copy().items():
            if isinstance(value,torch.Tensor):setattr(p,key,value.to(device))
        p.device=torch.device(device)
        for offset in (0,1):
            p.g_state_offset=offset;p._fit_sources=None
            rows=torch.tensor([[1,3,1],[4,2,5]],device=device)
            original=p._past_columns(rows).float()
            optimized=p._past_columns(rows,torch.float32)
            torch.testing.assert_close(optimized,original,rtol=0,atol=0)
            matrix=torch.randn(original.shape[-1],3,device=device,dtype=torch.float32)
            torch.testing.assert_close(optimized@matrix,original@matrix,rtol=0,atol=0)
    def test_cpu_native_gather(self):self.check_gather('cpu')
    @cuda
    def test_cuda_native_gather(self):self.check_gather('cuda')
