import unittest
import numpy as np
import scipy.sparse
import torch
from bpc.full_fisher import scatters, solve_full
from bpc.lda import fit_discriminant
from bpc.bpc import Policy
from bpc.pipeline import load_config, fit_groups
from tests.support.policies import manuscript_fixture, data_config

class FullFisherTest(unittest.TestCase):
    def test_sparse_full_scatter_matches_dense_metric(self):
        g=torch.Generator().manual_seed(12)
        x=torch.randn(41,13,generator=g,dtype=torch.float64)
        q=torch.rand(41,29,generator=g,dtype=torch.float64)
        q[q<.7]=0;q[:,0]=1;q[:,4]=0;q/=q.sum(1,keepdim=True)
        s=scatters(x.numpy(),scipy.sparse.csr_matrix(q.numpy()),'cpu',lambda *a:None,row_batch=7,class_batch=5)
        a,_=solve_full(s,5,threads=2)
        b=fit_discriminant(x,q,fisher_rank=5)['transform']
        torch.testing.assert_close(a@a.T,b@b.T,rtol=1e-8,atol=1e-9)

    def test_full_selector_and_default_config(self):
        torch.set_num_threads(2)
        method,bank=manuscript_fixture()
        method['g']['lda']['fit_all_histories']=True
        p=Policy(bank,data_config(),method,fit_residuals=False)
        result=p.support_selector.fit()
        self.assertEqual(result['anchors'],len(p.hankel_starts))
        self.assertEqual(result['fitting_histories'],len(p.hankel_starts))
        self.assertGreater(p.support_selector.payload['support_scale'],0)
        c=load_config('configs/robosuite_fisher_sparsemax_local.json')
        self.assertTrue(fit_groups(c)[0]['spec']['method']['g']['lda']['fit_all_histories'])
