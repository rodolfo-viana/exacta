from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from exacta.functional import classification as C, regression as R
from exacta.metrics.classification import ClassificationMetrics, ClassificationRanking
from exacta.metrics.regression import RegressionMetrics
from _process import destroy_process_group, spawn


def _worker(rank, init, empty):
    dist.init_process_group("gloo",init_method=init,rank=rank,world_size=2,timeout=timedelta(seconds=30))
    try:
        p=torch.tensor([0,1,2,1,2,0,1]);t=torch.tensor([0,2,2,1,0,0,0])
        x=torch.tensor([1.,4.,3.,7.,6.,5.,11.],dtype=torch.float64)+1e9
        y=torch.tensor([1.,3.,4.,8.,7.,6.,9.],dtype=torch.float64)+1e9
        weights=torch.tensor([1.,2.,1.,0.,3.,1.,2.],dtype=torch.float64)
        c=ClassificationMetrics(task="multiclass",num_classes=3)
        r=RegressionMetrics()
        ranking=ClassificationRanking(task="multiclass",num_classes=3,from_logits=True,bins=11)
        probs=torch.nn.functional.one_hot(p,3).double()*2
        if not empty or rank==0:
            selection=slice(None) if empty else slice(0,2) if rank==0 else slice(2,None)
            c.update(p[selection],t[selection],sample_weight=weights[selection])
            r.update(x[selection],y[selection],sample_weight=weights[selection])
            ranking.update(probs[selection],t[selection])
        expected_c=C.evaluate(p,t,task="multiclass",num_classes=3,sample_weight=weights)
        expected_r=R.evaluate(x,y,sample_weight=weights)
        for _ in range(2):
            for name,value in c.compute().items():torch.testing.assert_close(value,expected_c[name],equal_nan=True)
            for name,value in r.compute().items():torch.testing.assert_close(value,expected_r[name],rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(ranking.compute().auroc[0],C.auroc(probs,t,task="multiclass",num_classes=3,from_logits=True,bins=11))
        bad=RegressionMetrics(force_finite=rank==0)
        if rank==0:bad.update(x,y)
        with pytest.raises(ValueError,match="configuration"):bad.compute()
        bad_c=ClassificationMetrics(task="binary",threshold=.5 if rank==0 else .7)
        with pytest.raises(ValueError,match="configuration"):bad_c.compute()
        c.reset();r.reset()
        with pytest.raises(RuntimeError):c.compute()
        with pytest.raises(RuntimeError):r.compute()
    finally:
        destroy_process_group()


@pytest.mark.parametrize("empty",[False,True])
def test_distributed_counts_moments_and_empty_rank(tmp_path,empty):
    spawn(_worker,args=(f"file://{tmp_path / 'store'}",empty),directory=tmp_path / 'workers')
