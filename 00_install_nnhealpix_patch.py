"""00_install_nnhealpix_patch.py: add the masked average pooling layer to NNhealpix.

Run once after `pip install -r requirements.txt`. The tensionnet uses a masked
average pooling layer (nnhealpix.layers.MaskedAveragePooling), which ignores
masked (zero) pixels when downgrading a HEALPix map instead of averaging them in.
It is not part of NNhealpix, so this copies nnhealpix_patch/layers__init__.py
(NNhealpix's layers module plus the new layer) over nnhealpix/layers/__init__.py
in the active environment. The original file is kept as __init__.py.orig.

Usage:
    python 00_install_nnhealpix_patch.py
"""

import os
import shutil

import nnhealpix

ROOT = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(os.path.dirname(nnhealpix.__file__), 'layers', '__init__.py')
patch = os.path.join(ROOT, 'nnhealpix_patch', 'layers__init__.py')

if not os.path.exists(target + '.orig'):
    shutil.copy(target, target + '.orig')
shutil.copy(patch, target)
print(f'Patched {target}')

import importlib
import nnhealpix.layers
importlib.reload(nnhealpix.layers)
assert hasattr(nnhealpix.layers, 'MaskedAveragePooling')
print('nnhealpix.layers.MaskedAveragePooling is available.')
