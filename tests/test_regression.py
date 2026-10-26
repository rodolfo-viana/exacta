import numpy as np
import pytest
import torch

from exacta.functional import regression as F
from exacta.metrics.regression import RegressionMetrics


@pytest.mark.parametrize("outputs",[1,4])
@pytest.mark.parametrize("weighted",[False,True])
def test_independent_weighted_moments(device,outputs,weighted):
    rng=np.random.default_rng(97)
    p,t=rng.normal(size=(2,71,outputs))
    m=rng.random((71,outputs))>.2
    w=rng.random(71) if weighted else np.ones(71)
    w[::9]=0 if weighted else 1
    expected={name:[] for name in F._NAMES}
    for j in range(outputs):
        valid=m[:,j] & (w>0); a,b,weights=p[valid,j],t[valid,j],w[valid]
        mean=lambda x: np.average(x,weights=weights)
        var=lambda x: mean((x-mean(x))**2)
        mse=mean((a-b)**2);cov=mean((a-mean(a))*(b-mean(b)))
        row=[mean(abs(a-b)),mse,np.sqrt(mse),1-mse/var(b),1-var(a-b)/var(b),cov/np.sqrt(var(a)*var(b)),2*cov/(var(a)+var(b)+(mean(a)-mean(b))**2)]
        for name,value in zip(F._NAMES,row):expected[name].append(value)
    p[~m]=np.nan;t[~m]=np.inf
    pa,ta,ma,wa=[torch.tensor(x,device=device) for x in (p,t,m,w)]
    kwargs=dict(mask=ma,sample_weight=wa if weighted else None,multioutput="raw_values")
    result=F.evaluate(pa,ta,**kwargs)
    for name,values in result.items():
        np.testing.assert_allclose(values.cpu(),expected[name],rtol=2e-13,atol=1e-14)
        torch.testing.assert_close(getattr(F,name)(pa,ta,**kwargs),values)
    stream=RegressionMetrics(multioutput="raw_values")
    for lo,hi in [(0,1),(1,22),(22,71)]:
        stream.update(pa[lo:hi],ta[lo:hi],mask=ma[lo:hi],sample_weight=wa[lo:hi] if weighted else None)
    pa.fill_(100);ta.fill_(100)
    for _ in range(2):
        for name,value in stream.compute().items():torch.testing.assert_close(value,result[name])


@pytest.mark.parametrize("multioutput",["raw_values","uniform_average","variance_weighted"])
def test_sklearn_reference(device,multioutput):
    sk=pytest.importorskip("sklearn.metrics")
    rng=np.random.default_rng(73)
    p,t=rng.normal(size=(2,47,3));w=rng.random(47)
    pa,ta,wa=[torch.tensor(x,device=device) for x in (p,t,w)]
    names=[("r2",sk.r2_score),("explained_variance",sk.explained_variance_score)]
    if multioutput!="variance_weighted":
        names +=[("mae",sk.mean_absolute_error),("mse",sk.mean_squared_error),("rmse",sk.root_mean_squared_error)]
    for name,fn in names:
        np.testing.assert_allclose(getattr(F,name)(pa,ta,sample_weight=wa,multioutput=multioutput).cpu(),fn(t,p,sample_weight=w,multioutput=multioutput,**({"force_finite":False} if name in ["r2","explained_variance"] else {})))


def test_large_offset_centering_and_batch_pooling(device):
    p=torch.tensor([1.,4.,3.,6.,9.,8.,4.,5.],device=device,dtype=torch.float64)
    t=torch.tensor([2.,3.,4.,5.,6.,7.,8.,9.],device=device,dtype=torch.float64)
    reference=F.evaluate(p,t)
    p+=1e12;t+=1e12
    stream=RegressionMetrics()
    stream.update(p[:4],t[:4]);stream.update(p[4:],t[4:])
    for result in [F.evaluate(p,t),stream.compute()]:
        for name,value in result.items():torch.testing.assert_close(value,reference[name],rtol=1e-12,atol=1e-12)
    # Averaging the two batch R² scores is not the dataset R².
    assert not torch.isclose((F.r2(p[:4],t[:4])+F.r2(p[4:],t[4:]))/2,reference["r2"])


def test_constant_empty_and_multioutput_conventions(device):
    y=torch.ones(3,device=device);x=y+2
    assert F.r2(y,y).isnan();assert F.r2(x,y).isneginf()
    assert F.r2(y,y,force_finite=True)==1;assert F.r2(x,y,force_finite=True)==0
    assert F.explained_variance(x,y,force_finite=True)==1  # constant bias removed
    assert F.r2(y[:1],y[:1],force_finite=True).isnan()
    assert F.pearson(x,y).isnan();assert F.concordance_correlation(y,y).isnan()
    assert F.concordance_correlation(x,y)==0
    for value in F.evaluate(y[:0],y[:0]).values():assert value.isnan()
    for value in F.evaluate(y,y,mask=torch.zeros_like(y,dtype=torch.bool)).values():assert value.isnan()
    m=RegressionMetrics();m.update(y[:0],y[:0]);m.update(x,y)
    torch.testing.assert_close(m.compute()["mse"],F.mse(x,y))
    m.reset()
    with pytest.raises(RuntimeError):m.compute()
    p=torch.tensor([[1.,4.],[1.,4.]],device=device);t=torch.zeros_like(p)
    assert F.rmse(p,t)==2.5  # mean of roots, not root of the global mean
    with pytest.raises(ValueError):F.mae(p,t,multioutput="variance_weighted")


def test_validation_zero_weight_and_numpy(device):
    p=torch.tensor([2.,float('nan'),9.],device=device);t=torch.tensor([1.,float('inf'),5.],device=device)
    assert F.mae(p,t,sample_weight=torch.tensor([1.,0.,2.],device=device))==3
    with pytest.raises(ValueError):F.mae(p,t)
    with pytest.raises(ValueError):F.mae(p,t,mask=torch.ones(3,device=device))
    with pytest.raises(ValueError):F.mae(p,t[:,None])
    with pytest.raises(ValueError):F.mae(p.to(torch.complex64),t)
    with pytest.raises(ValueError):F.mae(p,t,sample_weight=torch.tensor([1.,-1.,2.],device=device))
    result=F.mse(np.array([1.,2.,3.])[::-1],np.array([2.,2.,2.]))
    assert isinstance(result,np.ndarray) and result==pytest.approx(2/3)


def test_gradient_free_metrics(device):
    p=torch.tensor([1.,2.,3.],device=device,requires_grad=True)
    for value in F.evaluate(p,p+1).values():assert not value.requires_grad


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64,torch.int64])
@pytest.mark.parametrize("masked",[False,True])
def test_native_fallback_parity_and_strides(monkeypatch,dtype,masked):
    torch.manual_seed(24)
    p=torch.randn(109,6).to(dtype)[:,::2];t=torch.randn(109,6).to(dtype)[:,::2]
    mask=torch.rand(109,3)>.3 if masked else None
    w=torch.rand(109,dtype=torch.float64);w[::5]=0
    native=F.evaluate(p,t,mask=mask,sample_weight=w,multioutput="raw_values")
    mae=F.mae(p,t,mask=mask,sample_weight=w)
    monkeypatch.setattr(F._native,"available",lambda:False)
    reference=F.evaluate(p,t,mask=mask,sample_weight=w,multioutput="raw_values")
    for name,value in native.items():torch.testing.assert_close(value,reference[name],rtol=1e-12,atol=1e-13)
    torch.testing.assert_close(mae,F.mae(p,t,mask=mask,sample_weight=w))


def test_native_binding_rejects_shapes():
    if not F._native.available():pytest.skip("native extension unavailable")
    c=F._native._C
    p=torch.ones(3,2);t=p.clone();m=torch.empty(0,0,dtype=torch.uint8)
    weights=torch.empty(0,dtype=torch.float64);out=torch.empty(11,2,dtype=torch.float64)
    for bad_t,bad_m,bad_w,bad_out in [(t[:2],m,weights,out),(t,torch.ones(3,3,dtype=torch.uint8),weights,out),(t,m,torch.ones(2,dtype=torch.float64),out),(t,m,weights,out[:10])]:
        with pytest.raises(ValueError):c.regression_stats(p,bad_t,bad_m,bad_w,True,bad_out,2)
