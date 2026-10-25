"""End-to-end latency/memory of advanced segmentation objectives and metrics.

Run from the repository root. JSON output includes the environment, seed and input
configuration. Compilation/warmup are excluded; cached geometry rows exclude map
construction explicitly. Loss rows include backward. Metric rows include validation
and any device transfer. CUDA uses synchronized wall-clock time, not launch latency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
import torch

from exacta import __version__, functional as F, losses as L
from exacta.detection import lesion_detection


def cpu_model() -> str:
    info = Path('/proc/cpuinfo')
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    return platform.processor()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=723)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--compiled', action='store_true')
    parser.add_argument('--only', nargs='+', default=None, help='optional row name prefixes')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.size,args.batch,args.classes,args.threads,args.warmup,args.repeat)<1 or args.classes<2 or args.iterations<0:
        parser.error('positive dimensions/threads/warmup/repeat, classes>=2 and iterations>=0 required')
    torch.set_num_threads(args.threads)
    generator=torch.Generator().manual_seed(args.seed)
    device=torch.device(args.device)
    shape=(args.batch,args.classes,args.size,args.size,args.size)
    x=torch.randn(shape,generator=generator).to(device).requires_grad_()
    # Smooth synthetic labels: avoids an unrealistic million-lesion detection workload.
    axis=torch.linspace(-1,1,args.size)
    z,y,v=torch.meshgrid(axis,axis,axis,indexing='ij')
    target=torch.zeros((args.size,)*3,dtype=torch.long)
    for cls in range(1,args.classes):
        centre=-.6+1.2*cls/args.classes
        target[((z-centre)**2+(y-centre)**2+(v-centre)**2)<(.35/args.classes**.25)**2]=cls
    target=target.expand(args.batch,*target.shape).clone().to(device)
    truth=torch.nn.functional.one_hot(target,args.classes).movedim(-1,1).float()
    spacing=(1.,1.,2.)
    def wanted(name):
        return not args.only or any(name.startswith(prefix) for prefix in args.only)
    signed=L.distance_maps(truth[:,1:],spacing=spacing,signed=True) if wanted('boundary_cached') else None
    unsigned=L.distance_maps(truth[:,1:],spacing=spacing) if wanted('hausdorff_dt_cached') else None
    skeleton=L.soft_skeletonize(truth[:,1:],iterations=args.iterations) if wanted('soft_cldice_cached') else None
    rows=[]

    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)

    def measure(name,fn,backward=False):
        if args.only and not any(name.startswith(prefix) for prefix in args.only):return
        def step():
            x.grad=None
            result=fn()
            if backward:result.backward()
        try:
            for _ in range(args.warmup):step()
            x.grad=None;sync()
            if device.type=='cuda':
                baseline=torch.cuda.memory_allocated(device)+(x.numel()*x.element_size() if backward else 0)
                torch.cuda.reset_peak_memory_stats(device)
            times=[]
            for _ in range(args.repeat):
                sync();start=time.perf_counter();step();sync();times.append(1000*(time.perf_counter()-start))
            extra=max(0,torch.cuda.max_memory_allocated(device)-baseline)/2**20 if device.type=='cuda' else None
            row=dict(name=name,median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times),extra_mib=extra)
            print(json.dumps(row),flush=True);rows.append(row)
        except Exception as exc:
            row=dict(name=name,error=f'{type(exc).__name__}: {exc}')
            print(json.dumps(row),flush=True);rows.append(row)
        x.grad=None

    losses=[('focal_tversky_torch',L.FocalTverskyLoss(backend='torch'),{}),
            ('focal_tversky_auto',L.FocalTverskyLoss(),{}),
            ('topk_ce',L.TopKCELoss(),{}),('dice_topk_ce',L.DiceTopKCELoss(),{}),
            ('lovasz',L.LovaszSoftmaxLoss(),{}),
            ('boundary',L.BoundaryLoss(),dict(spacing=spacing)),
            ('boundary_cached',L.BoundaryLoss(),dict(distance_map=signed)),
            ('hausdorff_dt',L.HausdorffDTLoss(),dict(spacing=spacing)),
            ('hausdorff_dt_cached',L.HausdorffDTLoss(),dict(spacing=spacing,distance_map=unsigned)),
            ('soft_cldice_torch',L.SoftClDiceLoss(iterations=args.iterations,backend='torch'),{}),
            ('soft_cldice',L.SoftClDiceLoss(iterations=args.iterations),{}),
            ('soft_cldice_cached',L.SoftClDiceLoss(iterations=args.iterations),dict(target_skeleton=skeleton))]
    for name,loss,kw in losses:
        measure(name,lambda loss=loss,kw=kw:loss(x,target,**kw),True)
        # EDT and validation require eager orchestration; compile only tensor training kernels.
        if args.compiled and name in ('focal_tversky_torch','focal_tversky_auto','topk_ce','lovasz','soft_cldice','soft_cldice_cached'):
            compiled=torch.compile(loss)
            measure(name+'_compiled',lambda loss=compiled,kw=kw:loss(x,target,**kw),True)
    p=x.detach().softmax(1)
    for name,fn in [('brier',F.brier_score),('nll',F.negative_log_likelihood),
                    ('ece',F.expected_calibration_error),('reliability',F.reliability_diagram)]:
        measure(name,lambda fn=fn:fn(p,target))
    hard=target.roll(1,dims=-1)
    measure('cldice',lambda:F.cldice(hard,target,num_classes=args.classes,include_background=False))
    conf=torch.full(target.shape,.8,device=device)
    measure('lesion_detection',lambda:lesion_detection(hard,target,conf,num_classes=args.classes,connectivity=1))
    mask=(target[0]==1)
    measure('skeleton_cpp',lambda:F.skeletonize(mask))
    if device.type=='cpu':
        try:
            from skimage.morphology import skeletonize
            numpy_mask=mask.numpy()
            measure('skeleton_skimage',lambda:skeletonize(numpy_mask,method='lee'))
        except ImportError:pass
    digest=hashlib.sha256()
    for path in sorted([*Path('src').rglob('*.py'),*Path('csrc').rglob('*.cpp'),*Path('csrc').rglob('*.h')]):
        digest.update(str(path).encode());digest.update(path.read_bytes())
    metadata=dict(exacta=__version__,torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,
                  python=platform.python_version(),platform=platform.platform(),
                  hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else cpu_model(),
                  git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
                  source_sha256=digest.hexdigest(),config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  dtype='float32',spacing=spacing,rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()
