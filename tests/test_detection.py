import math

import pytest
import torch

from exacta.detection import LesionDetection, lesion_detection


def example(device):
    target=torch.zeros(2,7,9,dtype=torch.long,device=device)
    pred=target.clone(); confidence=torch.zeros_like(target,dtype=torch.float64)
    target[0,1,1]=target[0,3,3]=target[0,5,5]=1  # one target remains missed
    pred[0,1,1]=pred[0,3,3]=pred[1,2,2]=1
    confidence[0,1,1]=.9;confidence[1,2,2]=.8;confidence[0,3,3]=.7
    return pred,target,confidence


def test_detection_known_pr_froc(device):
    p,t,c=example(device)
    result=lesion_detection(p,t,c,connectivity=1,num_classes=2)
    assert result.class_ids==(1,)
    curve=result.curves[0]
    assert curve.num_targets==3 and curve.num_cases==2
    assert curve.true_positives.tolist()==[0,1,1,2]
    assert curve.false_positives.tolist()==[0,0,1,1]
    assert curve.average_precision.item()==pytest.approx((1+2/3)/3)
    assert curve.sensitivity_at([0,.5,1]).tolist()==pytest.approx([1/3,2/3,2/3])
    stream=LesionDetection(connectivity=1,num_classes=2)
    for i in range(2): stream.update(p[i:i+1],t[i:i+1],c[i:i+1])
    torch.testing.assert_close(stream.compute().average_precision,result.average_precision)
    stream.reset()
    with pytest.raises(RuntimeError):stream.compute()


def test_detection_confidence_ties_grouped(device):
    p,t,c=example(device);c[p>0]=.8
    curve=lesion_detection(p,t,c,connectivity=1,num_classes=2).curves[0]
    assert len(curve.thresholds)==2
    assert curve.average_precision.item()==pytest.approx((2/3)*(2/3))
    assert curve.sensitivity_at(0).item()==0
    assert curve.sensitivity_at(.5).item()==pytest.approx(2/3)


def test_detection_one_target_cannot_match_twice(device):
    t=torch.zeros(1,5,9,dtype=torch.long,device=device);t[0,2,1:8]=1
    p=t.clone();p[0,2,4]=0
    c=p.double();c[0,2,1:4]=.9;c[0,2,5:8]=.8
    curve=lesion_detection(p,t,c,connectivity=1,num_classes=2,iou_threshold=.4).curves[0]
    assert curve.true_positives.tolist()==[0,1,1]
    assert curve.false_positives.tolist()==[0,0,1]


@pytest.mark.parametrize('kind',['no_targets','no_predictions','both_empty'])
def test_detection_empty_cases(device,kind):
    p,t,c=example(device)
    if kind in ('no_targets','both_empty'):t.zero_()
    if kind in ('no_predictions','both_empty'):p.zero_()
    curve=lesion_detection(p,t,c,connectivity=1,num_classes=2).curves[0]
    if kind=='no_predictions':
        assert curve.average_precision==0 and curve.sensitivity_at(10)==0
    else:
        assert math.isnan(curve.average_precision) and math.isnan(curve.sensitivity_at(10))
    assert curve.false_positives_per_case[-1]==(1.5 if kind=='no_targets' else 0)


def test_detection_component_confidence_mean(device):
    t=torch.ones(1,1,2,3,dtype=torch.bool,device=device)
    c=torch.tensor([[[[.1,.2,.3],[.4,.5,.9]]]],device=device,dtype=torch.float64)
    for reduction,expected in [('max',.9),('mean',.4)]:
        curve=lesion_detection(t,t,c,input_format='one-hot',include_background=True,connectivity=1,
                               confidence_reduction=reduction).curves[0]
        assert curve.thresholds[1]==pytest.approx(expected)
        assert curve.average_precision==1
