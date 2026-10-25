import numpy as np
import pytest
import torch
import torch.nn.functional as TF

from exacta import functional as F, metrics


@pytest.mark.parametrize('classes',[1,3])
def test_probability_scores_reference(device, classes):
    torch.manual_seed(723)
    x = torch.randn(2,classes,4,5,dtype=torch.float64,device=device)
    y = torch.randint(0,max(2,classes),(2,4,5),device=device)
    if classes==1:
        p=x.sigmoid(); ref_b=(p[:,0]-y)**2
        ref_n=TF.binary_cross_entropy_with_logits(x[:,0],y.double(),reduction='none')
    else:
        p=x.softmax(1);ref_b=(p-TF.one_hot(y,classes).movedim(-1,1)).square().sum(1)
        ref_n=TF.cross_entropy(x,y,reduction='none')
    ref_b,ref_n=ref_b.flatten(1).mean(1)[:,None],ref_n.flatten(1).mean(1)[:,None]
    for from_logits,pred in [(True,x),(False,p)]:
        torch.testing.assert_close(F.brier_score(pred,y,from_logits=from_logits),ref_b)
        torch.testing.assert_close(F.negative_log_likelihood(pred,y,from_logits=from_logits),ref_n)


def test_reliability_exact_bins_and_ties(device):
    p=torch.tensor([[[[0.,.25,.5,.75,1.]]]],device=device,dtype=torch.float64)
    y=torch.tensor([[[0,1,0,1,1]]],device=device)
    result=F.reliability_diagram(p,y,bins=4,mode='classwise')
    torch.testing.assert_close(result.counts.cpu(),torch.tensor([[[1,1,1,2]]]))
    torch.testing.assert_close(result.confidence.cpu(),torch.tensor([[[0,.25,.5,.875]]],dtype=torch.float64))
    torch.testing.assert_close(result.accuracy.cpu(),torch.tensor([[[0,1,0,1]]],dtype=torch.float64))
    assert result.ece.item()==pytest.approx((.75+.5+.25)/5)
    top=F.reliability_diagram(p,y,bins=2)
    # Confidence .5 predicts 0 in the binary representation, matching two-channel argmax.
    assert top.accuracy[0,0,1].item()==pytest.approx(4/5)
    two=torch.cat((1-p,p),dim=1)
    torch.testing.assert_close(F.expected_calibration_error(two,y,bins=2),top.ece)


def test_calibration_mask_empty_inf_numpy(device):
    p=torch.tensor([[[[0.,1.]]],[[[float('nan'),float('nan')]]]],device=device)
    y=torch.tensor([[[1,1]],[[999,999]]],device=device)
    mask=torch.tensor([[[True,True]],[[False,False]]],device=device)
    b=F.brier_score(p,y,mask=mask)
    n=F.negative_log_likelihood(p,y,mask=mask)
    e=F.expected_calibration_error(p,y,mask=mask)
    assert b[0]==.5 and n[0].isinf() and b[1].isnan() and n[1].isnan() and e[1].isnan()
    assert isinstance(F.brier_score(p.cpu().numpy(),y.cpu().numpy(),mask=mask.cpu().numpy()),np.ndarray)
    perfect=torch.tensor([[[[1.,0.]],[[0.,1.]]]],device=device)
    labels=torch.tensor([[[0,1]]],device=device)
    assert F.brier_score(perfect,labels).item()==0
    assert F.negative_log_likelihood(perfect,labels).item()==0


def test_calibration_pooled_vs_case_average(device):
    # Opposing calibration errors cancel in pooled bins, but not per-case ECE.
    p=torch.full((2,1,2,2),.6,device=device,dtype=torch.float64)
    y=torch.tensor([[[0,0],[0,0]],[[1,1],[1,1]]],device=device)
    pooled=metrics.Calibration(bins=2,mode='classwise')
    per_case=metrics.ExpectedCalibrationError(bins=2,mode='classwise')
    for i in range(2):
        pooled.update(p[i:i+1],y[i:i+1]);per_case.update(p[i:i+1],y[i:i+1])
    assert pooled.compute().ece.item()==pytest.approx(.1)
    assert per_case.compute().mean.item()==pytest.approx(.5)
    assert pooled.compute().counts.sum()==8
    pooled.reset()
    with pytest.raises(RuntimeError): pooled.compute()
    for cls,fn in [(metrics.BrierScore,F.brier_score),(metrics.NegativeLogLikelihood,F.negative_log_likelihood)]:
        m=cls();m.update(p,y)
        torch.testing.assert_close(m.compute().per_case,fn(p,y))


@pytest.mark.parametrize('value',[-.01,1.01,float('nan')])
def test_probability_validation(value):
    with pytest.raises(ValueError):
        F.brier_score(torch.full((1,1,2,2),value),torch.zeros((1,2,2),dtype=torch.long))
    with pytest.raises(ValueError):
        F.brier_score(torch.full((1,3,2,2),.5),torch.zeros((1,2,2),dtype=torch.long))
