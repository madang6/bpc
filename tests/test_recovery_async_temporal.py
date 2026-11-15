"""Recovery state belongs to episodes; temporal redirection shares the native search."""
import copy
from pathlib import Path
import tempfile
import unittest
from tests.suites import cuda, integration
from unittest.mock import patch
import numpy as np
import torch
from dpc_nldl import bpc
from dpc_nldl.evaluate import _fixed_rows_direct, _rollouts
from tests.support.rollouts import (
    BASE, INITIALIZERS, TRAJECTORIES, Encoder, Policy as FakePolicy, SchedulePool,
)
from tests.support.recovery import calibration_policy


class StatefulPolicy(FakePolicy):
    recovery=True
    demo_count=4
    def direct_actions(self,actions,states,recovery=None):
        output,trace=super().direct_actions(actions,states)
        if recovery is not None:
            status,tabu=recovery
            status[:,0]+=1
            status[:,2]+=1
            tabu[:,0]=True
            output=output+status[:,:1]*.01
            trace['recovery_ticks']=status[:,0].clone()
        return output,trace


class RecoveryStateTest(unittest.TestCase):
    def test_padding_updates_each_live_episode_once(self):
        p=StatefulPolicy();a=torch.ones(2,2,1,dtype=torch.float64);s=torch.ones(2,2,3,dtype=torch.float64)
        status=torch.tensor([[1.,0.,0.],[4.,0.,1.]],dtype=torch.float64)
        tabu=torch.zeros(2,4,dtype=torch.bool)
        _fixed_rows_direct(p,a,s,8,recovery=(status,tabu))
        self.assertEqual(status[:,0].tolist(),[2.,5.])
        self.assertEqual(status[:,2].tolist(),[1.,2.])
        self.assertEqual(tuple(tabu.shape),(2,4))
        self.assertTrue(tabu[:,0].all())

    @integration
    def test_async_slot_reuse_resets_state_and_preserves_trace_bytes(self):
        outputs=[]
        with tempfile.TemporaryDirectory() as temp:
            for async_reset in [False,True]:
                c=copy.deepcopy(BASE);c['evaluation'].update(readout_rows=2,async_reset=async_reset,
                    overlap_bookkeeping=True,fused_trace=True)
                root=Path(temp)/str(async_reset);pool=SchedulePool(c,2,delay=4,hazard=True)
                initializers=copy.deepcopy(INITIALIZERS)
                initializers[8]['success_at']=4;initializers[23]['success_at']=4
                with patch('dpc_nldl.episode_pool.EpisodePool',return_value=pool):
                    list(_rollouts(c,StatefulPolicy(),TRAJECTORIES,Encoder(),initializers,[8,23,5],root,'test'))
                traces={}
                for p in sorted((root/'traces/test').glob('*.npz')):
                    with np.load(p) as data:
                        ticks=data['recovery_ticks']
                        np.testing.assert_array_equal(ticks,np.arange(1,len(ticks)+1))
                        traces[p.name]={k:(data[k].dtype.str,data[k].shape,data[k].tobytes()) for k in data.files}
                outputs.append(traces)
        self.assertEqual(outputs[0],outputs[1])

    def test_custom_settle_is_used_before_auto_calibration(self):
        p=calibration_policy(-.009,-.024)
        seen=[]
        def calibrate():seen.append(p.recovery_settle);return -.012
        p._calibrate_aperture=calibrate
        p.configure_recovery({'enabled':True,'settle':17})
        self.assertEqual(seen,[17])
        for key in ['settle','hold','attempts']:
            with self.assertRaises(ValueError):p.configure_recovery({'enabled':True,key:0})


def retrieval_fixture():
    p=object.__new__(bpc.Policy)
    p.device=torch.device('cpu');p.dtype=torch.float64;p.g_lookup='past'
    p.history=2;p.samples=20;p.action_dim=7;p.g_align_future=True
    p.hankel_starts=torch.tensor([0,5,10]);p.hankel_trajectory=torch.tensor([0,1,2])
    p.action=torch.zeros(20,7,dtype=torch.float64);p.action[:,-1]=-1
    p.action[3,-1]=1 # column zero opens at rate one but closes at rate two
    p.action_scale=torch.ones(7,dtype=torch.float64);p.action_mean=torch.zeros(7,dtype=torch.float64)
    p._open_future=None;p._recovery_open_rates={}
    p.pf=torch.tensor([[0.],[2.],[4.]],dtype=torch.float64);p.pf_norm=p.pf.square().sum(1)
    warped=torch.tensor([[.1],[1.],[3.]],dtype=torch.float64)
    p._interpolated_keys=lambda:[(2.,warped,warped.square().sum(1),torch.zeros(3,dtype=torch.long),torch.ones(3,dtype=torch.bool))]
    return p


class TemporalRecoveryTest(unittest.TestCase):
    def test_shared_distance_pass_and_open_at_every_valid_rate(self):
        p=retrieval_fixture();q=torch.tensor([[0.],[4.]],dtype=torch.float64)
        status=torch.tensor([[0.,5.,0.],[0.,0.,0.]],dtype=torch.float64)
        with patch.object(bpc,'_squared_distance',wraps=bpc._squared_distance) as distance:
            supports,values,exhausted=p._recovery_topk(q,1,status,torch.zeros(2,3,dtype=torch.bool),torch.zeros(2,dtype=torch.bool))
        self.assertEqual(distance.call_count,2) # one pass per rate, not a second retrieval
        self.assertEqual(supports.tolist(),[[1],[2]])
        torch.testing.assert_close(values,torch.tensor([[1.],[0.]],dtype=torch.float64))
        self.assertFalse(exhausted.any())

    @cuda
    def test_cuda_temporal_recovery_matches_cpu(self):
        cpu=retrieval_fixture(); gpu=retrieval_fixture(); gpu.device=torch.device('cuda')
        entry=gpu._interpolated_keys()[0]
        for name,value in list(vars(gpu).items()):
            if isinstance(value,torch.Tensor):setattr(gpu,name,value.cuda())
        gpu._interpolated_keys=lambda:[tuple(v.cuda() if isinstance(v,torch.Tensor) else v for v in entry)]
        outputs=[]
        for p in [cpu,gpu]:
            q=torch.tensor([[0.],[4.]],dtype=p.dtype,device=p.device)
            status=torch.tensor([[0.,5.,0.],[0.,0.,0.]],dtype=p.dtype,device=p.device)
            outputs.append(p._recovery_topk(q,1,status,torch.zeros(2,3,dtype=torch.bool,device=p.device),torch.zeros(2,dtype=torch.bool,device=p.device)))
        for a,b in zip(*outputs):torch.testing.assert_close(a,b.cpu(),rtol=0,atol=0)

    def test_fired_support_demos_are_tabu_before_redirect(self):
        p=retrieval_fixture();p.action[:,-1]=-1
        tabu=torch.zeros(1,3,dtype=torch.bool)
        selected,_,_=p._recovery_topk(torch.zeros(1,1,dtype=torch.float64),1,
            torch.tensor([[3.,5.,1.]],dtype=torch.float64),tabu,torch.ones(1,dtype=torch.bool))
        self.assertTrue(tabu[0,0]);self.assertEqual(selected.item(),1)

    def test_exhaustion_returns_ordinary_supports_and_reports_all_rows(self):
        p=retrieval_fixture();q=torch.tensor([[0.],[4.]],dtype=torch.float64)
        selected,values,exhausted=p._recovery_topk(q,1,torch.ones(2,3,dtype=torch.float64),
            torch.ones(2,3,dtype=torch.bool),torch.zeros(2,dtype=torch.bool))
        self.assertEqual(selected.tolist(),[[0],[2]]);self.assertTrue(exhausted.all())
        torch.testing.assert_close(values,torch.zeros(2,1,dtype=torch.float64))

    def test_invalid_temporal_rates_do_not_exclude_native_opening(self):
        p=retrieval_fixture();entry=p._interpolated_keys()[0]
        p._interpolated_keys=lambda:[(*entry[:4],torch.tensor([False,True,True]))]
        selected,_,_=p._recovery_topk(torch.zeros(1,1,dtype=torch.float64),1,torch.ones(1,3,dtype=torch.float64),
            torch.zeros(1,3,dtype=torch.bool),torch.zeros(1,dtype=torch.bool))
        self.assertEqual(selected.item(),0)

if __name__=='__main__':unittest.main()
