"""General classification against independent count/probability definitions."""
import numpy as np
import pytest
import torch

from exacta.functional import classification as F
from exacta.metrics.classification import ClassificationMetrics, ClassificationRanking


@pytest.mark.parametrize("classes", [2, 5, 65])
@pytest.mark.parametrize("weighted", [False, True])
def test_multiclass_numpy_counts(device, classes, weighted):
    rng = np.random.default_rng(718)
    p, t = rng.integers(classes, size=(2, 179))
    mask = rng.random(179) > .2
    weight = rng.random(179) if weighted else np.ones(179)
    weight[::7] = 0 if weighted else 1
    p[~mask], t[~mask] = -991, -999
    expected = np.zeros((classes, classes))
    for a, b, m, w in zip(p, t, mask, weight):
        if m and w:
            expected[b, a] += w
    tensors = [torch.as_tensor(x, device=device) for x in (p, t, mask, weight)]
    kwargs = dict(task="multiclass", num_classes=classes, mask=tensors[2], sample_weight=tensors[3] if weighted else None)
    actual = F.confusion_matrix(*tensors[:2], **kwargs)
    np.testing.assert_allclose(actual.cpu(), expected)
    assert actual.dtype == (torch.float64 if weighted else torch.int64)
    scores = F.evaluate(*tensors[:2], **kwargs, average="none", zero_division=0)
    tp, pc, tc, n = expected.diagonal(), expected.sum(0), expected.sum(1), expected.sum()
    div = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    for name, value in dict(precision=div(tp, pc), recall=div(tp, tc), specificity=div(n-pc-tc+tp, n-tc), f1=div(2*tp, pc+tc)).items():
        np.testing.assert_allclose(scores[name].cpu(), value)
    assert scores["accuracy"].item() == pytest.approx(tp.sum()/n)
    assert scores["mcc"].item() == pytest.approx((tp.sum()*n-(pc*tc).sum()) / np.sqrt((pc*(n-pc)).sum()*(tc*(n-tc)).sum()))


@pytest.mark.parametrize("task", ["binary", "multiclass", "multilabel"])
@pytest.mark.parametrize("average", ["none", "micro", "macro", "weighted"])
def test_weighted_sklearn(device, task, average):
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(18)
    shape = (113, 4) if task == "multilabel" else (113,)
    classes = 4 if task == "multiclass" else 2
    p, t = rng.integers(classes, size=(2, *shape))
    w = rng.uniform(.1, 2, 113)
    pa, ta, wa = [torch.as_tensor(x, device=device) for x in (p, t, w)]
    kwargs = dict(task=task, num_classes=classes if task == "multiclass" else None,
                  sample_weight=wa, average=average, zero_division=0)
    ref_avg = "binary" if task == "binary" else None if average == "none" else average
    for name, reference in [("precision",sk.precision_score),("recall",sk.recall_score),("f1",sk.f1_score)]:
        actual = getattr(F,name)(pa,ta,**kwargs).cpu().numpy()
        expected = reference(t,p,average=ref_avg,sample_weight=w,zero_division=0)
        np.testing.assert_allclose(actual, np.atleast_1d(expected) if task == "binary" and average == "none" else expected)
    if task != "multilabel":
        for name,reference in [("mcc",sk.matthews_corrcoef),("cohen_kappa",sk.cohen_kappa_score)]:
            np.testing.assert_allclose(getattr(F,name)(pa,ta,**kwargs).cpu(), np.atleast_1d(reference(t,p,sample_weight=w)) if task == "binary" and average == "none" and name=="mcc" else reference(t,p,sample_weight=w))


@pytest.mark.parametrize("task", ["binary", "multiclass", "multilabel"])
@pytest.mark.parametrize("weighted", [False, True])
def test_streaming_masks_and_mutation(device, task, weighted):
    torch.manual_seed(77)
    shape = (27, 3) if task == "multilabel" else (27,)
    classes = 3 if task == "multiclass" else 2
    p, t = (torch.randint(classes, shape, device=device) for _ in range(2))
    mask = torch.rand(shape, device=device) > .3
    p[~mask], t[~mask] = 9999, -3
    w = torch.rand(27, device=device) if weighted else None
    opts = dict(task=task, num_classes=3 if task=="multiclass" else None, average="none")
    expected = F.evaluate(p,t,mask=mask,sample_weight=w,**opts)
    m = ClassificationMetrics(**opts)
    for a,b in [(0,1),(1,7),(7,27)]:
        m.update(p[a:b],t[a:b],mask=mask[a:b],sample_weight=None if w is None else w[a:b])
    p.fill_(0); t.fill_(0)
    for _ in range(2):
        for name,value in m.compute().items():
            torch.testing.assert_close(value,expected[name],equal_nan=True)
    m.reset()
    with pytest.raises(RuntimeError): m.compute()


def test_threshold_ties_ignore_and_binary_validation(device):
    p=torch.tensor([0.,.5,1.,float("nan")],device=device)
    t=torch.tensor([0,1,1,-1],device=device)
    assert F.accuracy(p,t,task="binary",ignore_index=-1)==1
    assert F.accuracy(torch.tensor([0.],device=device),torch.tensor([1],device=device),task="binary",from_logits=True)==1
    for value in [-1.,1.1,float("nan"),float("inf")]:
        with pytest.raises(ValueError): F.f1(torch.tensor([value],device=device),t[:1],task="binary")
    with pytest.raises(ValueError): F.confusion_matrix(p[:1],torch.tensor([.5],device=device),task="binary")
    x=torch.tensor([[1.,1.,1.],[2.,2.,0.],[float('nan')]*3],device=device)
    y=torch.tensor([0,1,-1],device=device)
    assert F.top_k_accuracy(x,y,k=1,ignore_index=-1)==.5
    assert F.top_k_accuracy(x,y,k=2,ignore_index=-1)==1


@pytest.mark.parametrize("task",["binary","multiclass","multilabel"])
def test_empty_and_undefined(device,task):
    shape=(0,2) if task=="multilabel" else (0,)
    x=torch.empty(shape,dtype=torch.long,device=device)
    kwargs=dict(task=task,num_classes=3 if task=="multiclass" else None)
    for value in F.evaluate(x,x,**kwargs).values(): assert value.isnan().all()
    for value in F.evaluate(x,x,**kwargs,zero_division=0).values(): assert (value==0).all()
    mask=torch.zeros_like(x,dtype=torch.bool)
    m=ClassificationMetrics(**kwargs);m.update(x,x,mask=mask)
    for value in m.compute().values(): assert value.isnan().all()


def test_fallback_and_large_classes(device,monkeypatch):
    p=torch.tensor([[0,1],[1,1],[0,0]],device=device);t=1-p
    expected=F.confusion_matrix(p,t,task="multilabel")
    monkeypatch.setattr(F._native,"available",lambda:False)
    monkeypatch.setattr(F._triton_counts,"available",lambda:False)
    torch.testing.assert_close(F.confusion_matrix(p,t,task="multilabel"),expected)
    with pytest.raises(ValueError): F.f1(p,t+.5,task="multilabel")
    def forbidden(*a,**kw): raise AssertionError("large class rates allocated a matrix")
    monkeypatch.setattr(F,"_matrix",forbidden)
    assert F.accuracy(p[:,0],p[:,0],task="multiclass",num_classes=10000)==1


@pytest.mark.parametrize("task",["binary","multiclass","multilabel"])
@pytest.mark.parametrize("from_logits",[False,True])
def test_probability_scores_and_ranking(device,task,from_logits):
    rng=np.random.default_rng(28)
    x=rng.normal(size=(31,4) if task!="binary" else 31)
    t=rng.integers(4 if task=="multiclass" else 2,size=(31,4) if task=="multilabel" else 31)
    p=torch.tensor(x,device=device)
    p=p.softmax(1) if task=="multiclass" else p.sigmoid()
    scores=torch.tensor(x,device=device) if from_logits else p
    target=torch.tensor(t,device=device)
    kwargs=dict(task=task,num_classes=4 if task=="multiclass" else None,from_logits=from_logits)
    probs=p.cpu().numpy()
    if task=="multiclass":
        truth=np.eye(4)[t];ref_b=((probs-truth)**2).sum(1).mean();ref_n=-np.log(probs[np.arange(31),t]).mean()
    else:
        truth=t;ref_b=((probs-t)**2).mean(0);ref_n=-(t*np.log(probs)+(1-t)*np.log1p(-probs)).mean(0)
    np.testing.assert_allclose(F.brier_score(scores,target,**kwargs).cpu(),ref_b)
    np.testing.assert_allclose(F.negative_log_likelihood(scores,target,**kwargs).cpu(),ref_n)
    assert torch.isfinite(F.expected_calibration_error(scores,target,**kwargs)).all()
    for bins in [None,13]:
        stream=ClassificationRanking(**kwargs,bins=bins)
        stream.update(scores[:2],target[:2]);stream.update(scores[2:],target[2:])
        result=stream.compute()
        for name in ["average_precision","auroc"]:
            torch.testing.assert_close(getattr(result,name)[0],getattr(F,name)(scores,target,bins=bins,**kwargs))
        torch.testing.assert_close(stream.compute().auroc,result.auroc)
    sk=pytest.importorskip("sklearn.metrics")
    for name,reference in [("average_precision",sk.average_precision_score),("auroc",sk.roc_auc_score)]:
        np.testing.assert_allclose(getattr(F,name)(scores,target,**kwargs).cpu(),np.atleast_1d(reference(truth,probs,average=None)))


def test_numpy_types_and_options():
    p=np.array([0,1,1]);t=np.array([0,1,0])
    assert isinstance(F.f1(p,t,task="binary"),np.ndarray)
    for opts in [dict(task="unknown"),dict(task="multiclass"),dict(task="binary",threshold=2),dict(task="binary",beta=0),dict(task="binary",average="bad"),dict(task="binary",zero_division=2)]:
        with pytest.raises((ValueError,TypeError)): F.evaluate(p,t,**opts)
    for kwargs in [dict(mask=np.ones(3)),dict(mask=np.ones((3,1),dtype=bool)),dict(sample_weight=np.ones((3,1))),dict(sample_weight=np.array([-1.,1,1]))]:
        with pytest.raises(ValueError): F.f1(p,t,task="binary",**kwargs)


@pytest.mark.parametrize("task",["binary","multiclass","multilabel"])
def test_probability_exclusions_and_empty(device,task):
    if task=="binary":
        p=torch.tensor([.2,float('nan'),.8],device=device)
        t=torch.tensor([0.,float('nan'),1.],device=device)
    else:
        p=torch.tensor([[.2,.8],[float('nan'),float('nan')],[.8,.2]],device=device)
        t=torch.tensor([[0.,1.],[99.,99.],[1.,0.]],device=device) if task=="multilabel" else torch.tensor([1,99,0],device=device)
    mask=torch.tensor([True,False,True],device=device)
    kw=dict(task=task,num_classes=2 if task=="multiclass" else None)
    for name in ["brier_score","negative_log_likelihood","expected_calibration_error","average_precision","auroc"]:
        fn=getattr(F,name)
        torch.testing.assert_close(fn(p,t,mask=mask,**kw),fn(p[mask],t[mask],**kw),equal_nan=True)
        assert fn(p[:0],t[:0],**kw).isnan().all()
        assert fn(p,t,mask=mask&False,**kw).isnan().all()
    if task=="binary":
        with pytest.raises(ValueError):F.brier_score(p[:1],t[:1]+.5,**kw)


@pytest.mark.parametrize("library",["numpy","cupy","jax"])
def test_tabular_array_libraries(library):
    xp=pytest.importorskip(library)
    if library=="cupy" and not torch.cuda.is_available():pytest.skip("CuPy needs CUDA")
    array=xp.numpy.asarray if library=="jax" else xp.asarray
    from exacta.functional import regression as R
    p=array([0,1,1]);t=array([0,1,0])
    score=F.f1(p,t,task="binary")
    error=R.mae(p,t)
    assert type(score).__module__.split('.')[0] == type(p).__module__.split('.')[0]
    assert str(score.dtype)=="float64" and str(error.dtype)=="float64"
    assert float(score)==pytest.approx(2/3) and float(error)==pytest.approx(1/3)


def test_weighted_constant_multiclass_mcc(device):
    # This reduction order leaves n-predicted_count nonzero by a few float64 ulps.
    rng=torch.Generator().manual_seed(9)
    target=torch.randint(10,(10003,),generator=rng).to(device)
    pred=torch.zeros_like(target)
    weights=torch.rand(10003,generator=rng,dtype=torch.float64).to(device)
    options=dict(task="multiclass",num_classes=10,sample_weight=weights)
    assert F.mcc(pred,target,**options).isnan()
    assert F.mcc(pred,target,**options,zero_division=0)==0
    assert F.mcc(pred,target,**options,zero_division=1)==1
