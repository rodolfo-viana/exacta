"""Small-statistic metrics must include empty ranks and lesion-free cases."""
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from _process import destroy_process_group
import torch.multiprocessing as mp

from exacta import metrics
from exacta.detection import LesionDetection, lesion_detection
from exacta.functional.calibration import reliability_diagram


def worker(rank,path,owner):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=f'file://{path}',rank=rank,world_size=2,timeout=timedelta(seconds=60))
    try:
        pred=torch.tensor([[[[.2,.6],[.8,.9]]],[[[.1,.3],[.7,.8]]]],dtype=torch.float64)
        target=torch.tensor([[[0,1],[0,1]],[[0,0],[1,1]]])
        cal=metrics.Calibration(bins=3,mode='classwise')
        det=LesionDetection(connectivity=1,num_classes=2)
        selected=slice(rank,rank+1) if owner==-1 else slice(None) if rank==owner else slice(0,0)
        if owner==-1 or rank==owner:
            cal.update(pred[selected],target[selected])
            det.update(target[selected],target[selected],pred[selected,0])
        diagram=cal.compute(); curves=det.compute()
        # Pool all voxels in a synthetic single case for the independent bin reference.
        expected=reliability_diagram(pred.reshape(1,1,4,2),
                                     target.reshape(1,4,2),bins=3,mode='classwise')
        # Pred B,C,H,W flatten order above is unchanged since C=1.
        torch.testing.assert_close(diagram.counts,expected.counts)
        torch.testing.assert_close(diagram.ece,expected.ece)
        assert curves.curves[0].num_cases==2
        assert curves.curves[0].num_targets==2
        assert curves.average_precision.item()==1
    finally:
        destroy_process_group()


@pytest.mark.parametrize('owner',[-1,0,1])
def test_advanced_distributed_empty_rank(tmp_path,owner):
    mp.spawn(worker,args=(str(tmp_path/'init'),owner),nprocs=2,join=True)
