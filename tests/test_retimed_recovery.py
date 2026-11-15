"""The recovery-selected future actions also feed the retimed correction."""
import unittest
import torch
from tests.support.policies import temporal_policy
from dpc_nldl.evaluate import _fixed_rows_direct

class RetimedRecoveryTest(unittest.TestCase):
    def test_recovery_uses_actual_temporal_actions_in_correction(self):
        p=temporal_policy();p.set_residual_action_source('retimed_first')
        p.g_supports=2;p.action[:,-1]=-1
        p.pf=p._g_lookup_projection(p._past_columns(torch.arange(len(p.hankel_starts))))
        p.pf_norm=p.pf.square().sum(1)
        p.recovery=True;p._open_future=None;p._recovery_open_rates={}
        p._raw_aperture=lambda states:torch.zeros(len(states),dtype=p.dtype)
        p._recovery_update=lambda states,actions,status:(status.clone(),torch.zeros(len(states),dtype=torch.bool))
        actions=p.action[10:20].reshape(2,5,2);states=p.state[10:20].reshape(2,5,4)
        status=torch.tensor([[0.,5.,1.],[0.,5.,1.]],dtype=p.dtype);tabu=torch.zeros(2,1,dtype=torch.bool)
        _,trace=_fixed_rows_direct(p,actions,states,4,recovery=(status,tabu))
        query=torch.cat([actions.flatten(1),states.flatten(1)],1)
        supports=trace['hankel_supports'];neighbors=p._support_endpoints(supports)
        futures=p._future_columns(supports,query,rates=trace['hankel_rates'])
        expected=p._query_features(states[:,-1],neighbors,futures[...,:p.action_dim])@p.W_gip
        torch.testing.assert_close(trace['residual_z'],expected)
        self.assertTrue((trace['recovery_active']>0).all())
        self.assertEqual(status[:,1].tolist(),[5.,5.])

if __name__=='__main__':unittest.main()
