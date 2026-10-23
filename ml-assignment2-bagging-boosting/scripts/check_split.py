"""Rebuild the 6/2/2 split from the seed alone and compare with split/split.npz (no dataset download needed:
Olivetti targets are sorted, 10 consecutive images per identity)."""
import numpy as np, sys
seed = int(''.join(c for c in 'M26CSA008' if c.isdigit()))
lab = np.repeat(np.arange(40), 10)
g = np.random.default_rng(seed)
tr, va, te = [], [], []
for c in range(40):
    p = g.permutation(np.where(lab == c)[0])
    tr += p[:6].tolist(); va += p[6:8].tolist(); te += p[8:].tolist()
saved = np.load('split/split.npz')
ok = all(np.array_equal(np.array(a), saved[k]) for a, k in [(tr, 'train'), (va, 'val'), (te, 'test')])
print('seed', seed, '| split matches saved file:', ok)
sys.exit(0 if ok else 1)
