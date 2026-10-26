import math

import pytest
import torch

from exacta.losses import classification as C, regression as R


@pytest.mark.parametrize("name",["L1Loss","MSELoss","HuberLoss","SmoothL1Loss","GaussianNLLLoss","PoissonNLLLoss"])
def test_native_regression_identity(name):
    assert getattr(R,name) is getattr(torch.nn,name)


@pytest.mark.parametrize("name",["CrossEntropyLoss","BCEWithLogitsLoss","BCELoss","NLLLoss","KLDivLoss"])
def test_native_classification_identity(name):
    assert getattr(C,name) is getattr(torch.nn,name)


@pytest.mark.parametrize("q",[.1,.5,.9])
@pytest.mark.parametrize("reduction",["none","sum","mean"])
def test_pinball_value_and_derivatives(device,q,reduction):
    p=torch.tensor([-.7,.2,1.3],device=device,dtype=torch.float64,requires_grad=True)
    t=torch.tensor([.1,-.4,.3],device=device,dtype=torch.float64,requires_grad=True)
    loss=R.QuantileLoss(q,reduction=reduction)
    r=t-p;ref=torch.where(r>=0,q*r,(q-1)*r)
    if reduction=="sum":ref=ref.sum()
    elif reduction=="mean":ref=ref.mean()
    torch.testing.assert_close(loss(p,t),ref)
    assert torch.autograd.gradcheck(loss,(p,t))
    assert torch.autograd.gradgradcheck(loss,(p,t))


def test_logcosh_derivatives_including_zero(device):
    p=torch.tensor([-.7,0.,1.3],device=device,dtype=torch.float64,requires_grad=True)
    t=torch.zeros_like(p)
    loss=R.LogCoshLoss(reduction="none")
    torch.testing.assert_close(loss(p,t),p.cosh().log())
    assert torch.autograd.gradcheck(loss,(p,t))
    assert torch.autograd.gradgradcheck(loss,(p,t))
    grad=torch.autograd.grad(loss(p,t).sum(),p,create_graph=True)[0]
    second=torch.autograd.grad(grad.sum(),p)[0]
    torch.testing.assert_close(second,1-p.tanh().square())
    large=torch.tensor([-1000.,1000.],device=device,dtype=torch.float64,requires_grad=True)
    terms=loss(large,torch.zeros_like(large))
    torch.testing.assert_close(terms,large.abs()-math.log(2))
    gradient=torch.autograd.grad(terms.sum(),large,create_graph=True)[0]
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(torch.autograd.grad(gradient.sum(),large)[0]).all()
    tiny=torch.tensor([1e-5],device=device)
    torch.testing.assert_close(loss(tiny,torch.zeros_like(tiny)),tiny.square()/2,atol=1e-17,rtol=1e-6)


@pytest.mark.parametrize("cls",[R.LogCoshLoss,R.QuantileLoss])
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16,torch.float32,torch.float64])
def test_masks_mixed_precision_and_empty(device,cls,dtype):
    p=torch.tensor([float('nan'),1.,float('inf')],device=device,dtype=dtype,requires_grad=True)
    t=torch.zeros_like(p);mask=torch.tensor([False,True,False],device=device)
    loss=cls();actual=loss(p,t,mask=mask)
    torch.testing.assert_close(actual,loss(p[1:2],t[1:2]))
    actual.backward();assert p.grad[0]==0 and p.grad[2]==0
    assert actual.dtype==(torch.float64 if dtype==torch.float64 else torch.float32)
    p.grad=None;zero=loss(p,t,mask=mask&False);zero.backward();assert zero==0 and (p.grad==0).all()
    assert loss(p[:0],t[:0])==0


@pytest.mark.parametrize("cls",[R.LogCoshLoss,R.QuantileLoss])
def test_compile_and_validation(device,cls):
    p=torch.tensor([-.5,.2,.9],device=device,requires_grad=True);t=torch.zeros_like(p)
    mask=torch.tensor([True,False,True],device=device)
    loss=cls();compiled=torch.compile(loss,fullgraph=True)
    actual=compiled(p,t,mask=mask);expected=loss(p,t,mask=mask)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual,p)[0],torch.autograd.grad(expected,p)[0])
    with pytest.raises(ValueError):loss(p,t[:,None])
    with pytest.raises(ValueError):loss(p,t,mask=mask.float())
    with pytest.raises(ValueError):cls(reduction="bad")
    with pytest.raises(ValueError):loss(p.long(),t)


@pytest.mark.parametrize("q",[0,1,-1,float('nan')])
def test_bad_quantiles(q):
    with pytest.raises(ValueError):R.QuantileLoss(q)
