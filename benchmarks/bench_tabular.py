"""End-to-end tabular metrics and regression-loss forward/backward latency.

Run serially; warmup/compilation and constructors are outside measured regions.
Raw reports are local and excluded from distributions. Synthetic timing does not
measure predictive quality. Native and fallback counts use identical validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import torch

import exacta
from exacta.functional import classification as C, regression as R
from exacta.losses.regression import LogCoshLoss, QuantileLoss
from bench_advanced import cpu_model


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--samples',type=int,default=65536)
    parser.add_argument('--classes',type=int,default=8)
    parser.add_argument('--outputs',type=int,default=8)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--repeat',type=int,default=10)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--seed',type=int,default=796)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if min(args.samples,args.outputs,args.threads,args.repeat,args.warmup)<1 or args.classes<2:
        parser.error('positive sizes/repeats and classes>=2 required')
    torch.set_num_threads(args.threads)
    rng=torch.Generator().manual_seed(args.seed)
    p,t=(torch.randint(args.classes,(args.samples,),generator=rng).to(args.device) for _ in range(2))
    x,y=(torch.randn(args.samples,args.outputs,generator=rng).to(args.device) for _ in range(2))
    records=[]

    def sync():
        if args.device=='cuda':torch.cuda.synchronize()

    def measure(name,fn):
        for _ in range(args.warmup):fn()
        sync()
        if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
        allocated=torch.cuda.memory_allocated() if args.device=='cuda' else 0
        elapsed=[]
        for _ in range(args.repeat):
            sync();start=time.perf_counter();fn();sync()
            elapsed.append((time.perf_counter()-start)*1000)
        result=dict(name=name,median_ms=statistics.median(elapsed),samples_ms=elapsed,
                    peak_extra_bytes=torch.cuda.max_memory_allocated()-allocated if args.device=='cuda' else None)
        records.append(result);print(name,round(result['median_ms'],4),flush=True)

    options=dict(task='multiclass',num_classes=args.classes)
    expected=C.evaluate(p,t,**options)
    measure('classification dispatched',lambda:C.evaluate(p,t,**options))
    with patch.object(C._native,'available',return_value=False),patch.object(C._triton_counts,'available',return_value=False):
        reference=C.evaluate(p,t,**options)
        for name,value in reference.items():torch.testing.assert_close(value,expected[name])
        measure('classification torch fallback',lambda:C.evaluate(p,t,**options))
    measure('regression shared',lambda:R.evaluate(x,y))
    measure('regression separate',lambda:[getattr(R,n)(x,y) for n in R._NAMES])
    measure('regression mae',lambda:R.mae(x,y))
    if args.device=='cpu':
        expected=R.evaluate(x,y)
        with patch.object(R._native,'available',return_value=False):
            reference=R.evaluate(x,y)
            for name,value in reference.items():torch.testing.assert_close(value,expected[name])
            measure('regression torch fallback',lambda:R.evaluate(x,y))
            measure('regression mae torch fallback',lambda:R.mae(x,y))
    x.requires_grad_()
    for cls in [LogCoshLoss,QuantileLoss]:
        eager=cls();compiled=torch.compile(eager,fullgraph=True)
        torch.testing.assert_close(compiled(x,y),eager(x,y))
        for label,loss in [('eager',eager),('compiled',compiled)]:
            def step():
                x.grad=None
                loss(x,y).backward()
            measure(f'{cls.__name__} {label} forward_backward',step)
    files=[*Path('src/exacta').rglob('*.py'), *Path('csrc').rglob('*.cpp'),
           *Path('csrc').rglob('*.h'), Path('CMakeLists.txt'), Path(__file__)]
    source={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report=dict(arguments={**vars(args),'output':str(args.output)},version=exacta.__version__,
                git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                source_sha256=source,python=platform.python_version(),torch=torch.__version__,
                cpu=cpu_model(),gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,results=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
