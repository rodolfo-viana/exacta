"""Regenerate Lee reference masks with scikit-image 0.26.0; seed 6271.

Run: python tests/fixtures/make_skeleton_fixture.py
The fixture makes reference parity available without a scikit-image test dependency.
"""
from pathlib import Path
import numpy as np
import skimage
from skimage.morphology import skeletonize

rng=np.random.default_rng(6271)
arrays={}
for i,shape in enumerate([(11,13),(5,7,9),(1,9,11),(4,4,4),(9,9,9)]):
    for j,density in enumerate([0.,.2,.5,.9,1.]):
        mask=rng.random(shape)<density
        arrays[f'mask_{i}_{j}']=mask
        arrays[f'skeleton_{i}_{j}']=skeletonize(mask,method='lee')
arrays['reference_version']=np.asarray(skimage.__version__)
np.savez_compressed(Path(__file__).with_name('skeleton_lee.npz'),**arrays)
