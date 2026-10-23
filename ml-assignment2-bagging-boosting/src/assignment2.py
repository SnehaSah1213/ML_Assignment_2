# %% [markdown]
# # ML Assignment 2: Bagging, Boosting, Bias-Variance on Olivetti Faces
# Roll no: M26CSA008. Runtime -> Change runtime type -> T4 GPU, then Run all (about 15-25 min).

# %%
import os, time, json, copy, shutil, warnings
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
try:
    from IPython.display import display
except ImportError:
    display = print
warnings.filterwarnings('ignore')

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', dev, '|', torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'no GPU, enable it from Runtime menu')

ROLL = 'M26CSA008'
SEED = int(''.join(ch for ch in ROLL if ch.isdigit()))
K = 40
os.makedirs('out', exist_ok=True)
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': .3})
tables = {}
print('seed =', SEED)

# %% data + split
ds = fetch_olivetti_faces(shuffle=False)
imgs = ds.images.astype(np.float32)
lab = ds.target.astype(int)
Xall = imgs.reshape(len(imgs), -1)

g = np.random.default_rng(SEED)
tr_i, va_i, te_i = [], [], []
for c in range(K):
    p = g.permutation(np.where(lab == c)[0])
    tr_i += p[:6].tolist(); va_i += p[6:8].tolist(); te_i += p[8:].tolist()
tr_i, va_i, te_i = np.array(tr_i), np.array(va_i), np.array(te_i)
np.savez('out/split.npz', train=tr_i, val=va_i, test=te_i)

Xtr, ytr = Xall[tr_i], lab[tr_i]
Xva, yva = Xall[va_i], lab[va_i]
Xte, yte = Xall[te_i], lab[te_i]

def blank(a):
    m = a.reshape(-1, 64, 64).copy()
    m[:, 24:40, 24:40] = 0
    return m.reshape(len(a), -1)

Xte_m = blank(Xte)
blk = np.zeros((64, 64), bool); blk[24:40, 24:40] = True; blk = blk.ravel()
tests = {'clean': Xte, 'masked': Xte_m}
print(Xtr.shape, Xva.shape, Xte.shape)

# %% helpers
def T(a, dt=None):
    t = torch.as_tensor(np.asarray(a), device=dev)
    return t if dt is None else t.to(dt)

ytr_t, yva_t, yte_t = T(ytr, torch.long), T(yva, torch.long), T(yte, torch.long)

def hard(trees, X):
    return T(np.stack([t.predict(X) for t in trees]), torch.long)

def count_votes(p, k=K):
    return F.one_hot(p, k).sum(0).double()

def win_margin(v):
    top = v.topk(2, dim=1).values
    return (top[:, 0] - top[:, 1]) / v.sum(1)

def pair_disagree(v):
    b = v.sum(1)
    if b[0].item() < 2:
        return float('nan')
    return (1 - (v * (v - 1)).sum(1) / (b * (b - 1))).mean().item()

def accy(pred, y):
    return (pred == y).double().mean().item()

def leaf_info(t):
    tr = t.tree_
    cnt = tr.n_node_samples[tr.children_left == -1]
    return int(len(cnt)), float(cnt.mean()), float((cnt == 1).mean())

def touches(trees, x):
    out = []
    for t in trees:
        f = t.tree_.feature[t.decision_path(x[None]).indices]
        out.append(bool(blk[f[f >= 0]].any()))
    return np.array(out)

def save(name):
    plt.tight_layout()
    plt.savefig(f'out/{name}.png', dpi=150)
    plt.show()

def _fit_tree(X, y, ix, d, l, s):
    return DecisionTreeClassifier(max_depth=d, min_samples_leaf=l, random_state=s).fit(X[ix], y[ix])

class Bag:
    def __init__(self, B, depth, leaf, seed, jobs=-1):
        self.B, self.depth, self.leaf, self.seed, self.jobs = B, depth, leaf, seed, jobs

    def fit(self, X, y):
        r = np.random.default_rng(self.seed)
        n = len(X)
        self.boot = r.integers(0, n, size=(self.B, n))
        sd = r.integers(0, 2**31 - 1, size=self.B)
        self.trees = Parallel(n_jobs=self.jobs, prefer='threads')(
            delayed(_fit_tree)(X, y, self.boot[b], self.depth, self.leaf, int(sd[b])) for b in range(self.B))
        self.inbag = np.zeros((self.B, n), bool)
        for b in range(self.B):
            self.inbag[b, self.boot[b]] = True
        return self

    def head(self, B):
        c = copy.copy(self)
        c.B, c.trees, c.boot, c.inbag = B, self.trees[:B], self.boot[:B], self.inbag[:B]
        return c

    def votes(self, X):
        return count_votes(hard(self.trees, X))

class Samme:
    def __init__(self, depth, rounds, seed, log=False):
        self.depth, self.rounds, self.seed, self.log = depth, rounds, seed, log

    def fit(self, X, y):
        n, k = len(X), int(y.max()) + 1
        self.k = k
        w = np.full(n, 1.0 / n)
        self.trees, self.alpha, self.eps, self.wused, self.stop = [], [], [], [], None
        for t in range(self.rounds):
            tree = DecisionTreeClassifier(max_depth=self.depth, random_state=self.seed + t)
            tree.fit(X, y, sample_weight=w)
            p = tree.predict(X)
            bad = p != y
            e = float(w[bad].sum())
            if e >= 1 - 1.0 / k:
                self.stop = (t, e)
                print(f'round {t}: weighted error {e:.3f} >= 1-1/K, not better than random guessing, stopping')
                break
            e = max(e, 1e-12)
            a = np.log((1 - e) / e) + np.log(k - 1)
            raw = w * np.exp(a * bad)
            w_new = raw / raw.sum()
            if self.log and t == 0:
                d = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'y': y, 'w_init': w, 'pred': p,
                                  'wrong': bad.astype(int), 'w_unnorm': raw, 'w_new': w_new})
                display(d.round(4))
                print(f'weighted error = {e:.4f}, alpha = {a:.4f}, sum(w_new) = {w_new.sum():.4f}')
            self.trees.append(tree); self.alpha.append(a); self.eps.append(e); self.wused.append(w)
            w = w_new
            if not bad.any():
                break
        self.alpha, self.eps, self.wused = np.array(self.alpha), np.array(self.eps), np.array(self.wused)
        return self

    def head(self, R):
        R = min(R, len(self.trees))
        c = copy.copy(self)
        c.trees, c.alpha, c.eps, c.wused = self.trees[:R], self.alpha[:R], self.eps[:R], self.wused[:R]
        return c

    def scores(self, X):
        oh = F.one_hot(hard(self.trees, X), self.k).double()
        return torch.einsum('t,tnk->nk', T(self.alpha, torch.double), oh)

    def proba(self, X):
        s = self.scores(X)
        return s / s.sum(1, keepdim=True)

# %% [markdown]
# ## Q1(a) Single tree baseline

# %%
DEPTHS, LEAFS = [2, 4, 8, 16, None], [1, 2, 4]
rows = []
for d in DEPTHS:
    for l in LEAFS:
        t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=l, random_state=SEED).fit(Xtr, ytr)
        nl, per, one = leaf_info(t)
        rows.append(dict(_d=d, max_depth=str(d), min_leaf=l, grown_depth=t.get_depth(), train=t.score(Xtr, ytr),
                         val=t.score(Xva, yva), leaves=nl, per_leaf=per, single_frac=one))
best = min(rows, key=lambda r: (-r['val'], 10**6 if r['_d'] is None else r['_d'], -r['min_leaf']))
FD, FL = best['_d'], best['min_leaf']
grid = pd.DataFrame(rows).drop(columns='_d')
tables['q1a_grid'] = grid
display(grid.round(4))
print('frozen setting (picked on val only):', 'max_depth =', FD, ', min_samples_leaf =', FL)

tree0 = DecisionTreeClassifier(max_depth=FD, min_samples_leaf=FL, random_state=SEED).fit(Xtr, ytr)
nl, per, one = leaf_info(tree0)
frozen = pd.DataFrame([dict(train=tree0.score(Xtr, ytr), val=tree0.score(Xva, yva), test=tree0.score(Xte, yte),
                            leaves=nl, mean_samples_per_leaf=per, single_example_leaf_frac=one)])
tables['q1a_frozen'] = frozen
display(frozen.round(4))

dep = grid[grid.min_leaf == FL]
tables['q1a_depth_effect'] = dep
xs = np.arange(len(dep))
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(xs, dep.train, 'o-', label='train'); ax[0].plot(xs, dep.val, 's-', label='val')
ax[0].set_title('accuracy vs depth'); ax[0].legend()
ax[1].plot(xs, dep.leaves, 'o-'); ax[1].set_title('number of leaves')
ax[2].plot(xs, dep.per_leaf, 'o-', label='samples/leaf')
ax[2].plot(xs, dep.single_frac, 's-', label='frac single-example leaves'); ax[2].legend()
ax[2].set_title('leaf occupancy')
for a in ax:
    a.set_xticks(xs); a.set_xticklabels(dep.max_depth); a.set_xlabel(f'max_depth (min_leaf={FL})')
save('q1a_depth')

# %% [markdown]
# ## Q1(b) Bagging from scratch

# %%
Bs = [1, 5, 10, 25, 50, 100, 200]
bag200 = Bag(200, FD, FL, SEED + 1).fit(Xtr, ytr)
ptr_all, pva_all, pte_all = hard(bag200.trees, Xtr), hard(bag200.trees, Xva), hard(bag200.trees, Xte)
n = len(Xtr)
theory = n * (1 - (1 - 1 / n) ** n)
rows = []
for B in Bs:
    rows.append(dict(B=B,
                     train=accy(count_votes(ptr_all[:B]).argmax(1), ytr_t),
                     val=accy(count_votes(pva_all[:B]).argmax(1), yva_t),
                     test=accy(count_votes(pte_all[:B]).argmax(1), yte_t),
                     distinct_mean=np.mean([len(np.unique(bag200.boot[b])) for b in range(B)]),
                     distinct_theory=theory))
tab_b = pd.DataFrame(rows)
tables['q1b_bagging'] = tab_b
display(tab_b.round(4))
print(f'theory: n(1-(1-1/n)^n) = {theory:.2f}  (~{theory / n:.3f} of n)')

# %% [markdown]
# ## Q1(c) OOB, disagreement, vote margin

# %%
def oob_acc(B):
    m = T(~bag200.inbag[:B])
    v = (F.one_hot(ptr_all[:B], K).double() * m[:, :, None]).sum(0)
    has = v.sum(1) > 0
    return accy(v.argmax(1)[has], ytr_t[has]), has.double().mean().item()

rows = []
for B in Bs:
    o, cov = oob_acc(B)
    v = count_votes(pva_all[:B])
    rows.append(dict(B=B, oob_acc=o, oob_coverage=cov, disagreement=pair_disagree(v),
                     mean_margin=win_margin(v).mean().item(), val_acc=accy(v.argmax(1), yva_t)))
tab_c = pd.DataFrame(rows)
tables['q1c_oob_dis_margin'] = tab_c
display(tab_c.round(4))
Bstar = int(tab_c.sort_values(['val_acc', 'B'], ascending=[False, True]).iloc[0].B)
print('final ensemble size (val acc, smallest B on ties):', Bstar)
print('B with best OOB acc:', int(tab_c.loc[tab_c.oob_acc.idxmax()].B), '| B with highest margin:', int(tab_c.loc[tab_c.mean_margin.idxmax()].B))

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, col, ttl in zip(ax, ['oob_acc', 'disagreement', 'mean_margin'], ['OOB accuracy', 'avg pairwise disagreement (val)', 'mean vote margin (val)']):
    a.plot(tab_c.B, tab_c[col], 'o-'); a.set_xscale('log'); a.set_xlabel('B'); a.set_title(ttl)
    a.axvline(Bstar, color='r', ls='--', alpha=.5)
save('q1c_curves')
bag_final = bag200.head(Bstar)

# %% [markdown]
# ## Q1(d) Tree vs bagging, clean and masked

# %%
fz = {}
for nm, X in tests.items():
    tp = tree0.predict(X); tc = tree0.predict_proba(X).max(1)
    v = bag_final.votes(X)
    bp = v.argmax(1).cpu().numpy(); bc = (v.max(1).values / v.sum(1)).cpu().numpy()
    fz[nm] = dict(tp=tp, tc=tc, bp=bp, bc=bc, v=v)

tab_d = pd.DataFrame([dict(set=nm, tree_acc=(r['tp'] == yte).mean(), bag_acc=(r['bp'] == yte).mean(),
                           tree_conf=r['tc'].mean(), bag_conf=r['bc'].mean()) for nm, r in fz.items()])
tables['q1d_tree_vs_bag'] = tab_d
display(tab_d.round(4))

c = fz['clean']
fixed = np.where((c['tp'] != yte) & (c['bp'] == yte))[0][:2]
both = np.where((c['tp'] != yte) & (c['bp'] != yte))[0][:2]
print('bagging corrects tree on', len(np.where((c['tp'] != yte) & (c['bp'] == yte))[0]), 'faces; both wrong on', len(np.where((c['tp'] != yte) & (c['bp'] != yte))[0]))
cases = []
for kind, ids in [('bag fixes tree', fixed), ('both wrong', both)]:
    for i in ids:
        vi = c['v'][i].cpu().numpy().astype(int)
        top = np.argsort(-vi)[:3]
        cases.append(dict(case=kind, test_row=int(i), true=int(yte[i]), tree_pred=int(c['tp'][i]), bag_pred=int(c['bp'][i]),
                          top3_votes=[(int(k), int(vi[k])) for k in top], votes_for_true=int(vi[yte[i]]), B=Bstar))
tab_cases = pd.DataFrame(cases)
tables['q1d_cases'] = tab_cases
display(tab_cases)
if len(cases):
    fig, ax = plt.subplots(1, len(cases), figsize=(3.2 * len(cases), 3.6), squeeze=False)
    for a, cs in zip(ax[0], cases):
        a.imshow(imgs[te_i[cs['test_row']]], cmap='gray'); a.axis('off')
        a.set_title(f"{cs['case']}\ntrue {cs['true']} tree {cs['tree_pred']} bag {cs['bag_pred']}\ntrue-class votes {cs['votes_for_true']}/{Bstar}", fontsize=8)
    save('q1d_cases')

# %% [markdown]
# ## Q2(a) SAMME on a toy 3 class problem

# %%
Xt = np.array([[1, 1], [1.5, 2], [2, 1.5], [4, 4], [4.5, 3.5], [3, 3.2], [1, 4.5], [2, 5], [4.2, 4.4]])
yt = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
toy = Samme(1, 3, 0, log=True).fit(Xt, yt)
print('alpha = log((1-e)/e) + log(K-1), K =', toy.k)

# %% [markdown]
# ## Q2(b) SAMME on faces: depth and rounds

# %%
runs, rows = {}, []
for d in [1, 2, 3, 4]:
    m = Samme(d, 100, SEED + 2).fit(Xtr, ytr)
    runs[d] = m
    for R in [10, 25, 50, 100]:
        h = m.head(R)
        rows.append(dict(depth=d, rounds=R, used=len(h.trees),
                         train=accy(h.scores(Xtr).argmax(1), ytr_t), val=accy(h.scores(Xva).argmax(1), yva_t)))
tab_q2b = pd.DataFrame(rows)
tables['q2b_grid'] = tab_q2b
display(tab_q2b.round(4))
sel = tab_q2b.sort_values(['val', 'depth', 'rounds'], ascending=[False, True, True]).iloc[0]
DB, RB = int(sel.depth), int(sel.rounds)
print('selected boosting: depth', DB, ', rounds', RB)
boost_final = runs[DB].head(RB)

W = boost_final.wused
top10 = np.sort(W, axis=1)[:, -10:].sum(1)
fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
ax[0].plot(boost_final.eps); ax[0].set_title('weighted training error')
ax[1].plot(boost_final.alpha); ax[1].set_title('learner weight alpha')
ax[2].plot(W.max(1)); ax[2].set_title('max sample weight')
ax[3].plot(top10); ax[3].set_title('weight of 10 heaviest images')
for a in ax: a.set_xlabel('round')
save('q2b_selected')
tables['q2b_rounds'] = pd.DataFrame(dict(round=np.arange(len(W)), eps=boost_final.eps, alpha=boost_final.alpha, max_w=W.max(1), top10_w=top10))

# %% [markdown]
# ## Q2(c) Boosting vs bagging, and a heavy face

# %%
rows = []
for L in [25, 50, 100]:
    t0 = time.perf_counter(); bg = Bag(L, FD, FL, SEED + 3, jobs=1).fit(Xtr, ytr); tb = time.perf_counter() - t0
    t0 = time.perf_counter(); bs = Samme(DB, L, SEED + 2).fit(Xtr, ytr); ts = time.perf_counter() - t0
    rows.append(dict(model='bagging', learners=L, train=accy(bg.votes(Xtr).argmax(1), ytr_t),
                     val=accy(bg.votes(Xva).argmax(1), yva_t), test=accy(bg.votes(Xte).argmax(1), yte_t), fit_sec=tb))
    rows.append(dict(model='boosting', learners=L, train=accy(bs.scores(Xtr).argmax(1), ytr_t),
                     val=accy(bs.scores(Xva).argmax(1), yva_t), test=accy(bs.scores(Xte).argmax(1), yte_t), fit_sec=ts))
tab_q2c = pd.DataFrame(rows)
tables['q2c_bag_vs_boost'] = tab_q2c
display(tab_q2c.round(4))

Wm = boost_final.wused
hv = int(Wm.max(0).argmax())
ptr_b = hard(boost_final.trees, Xtr)
al = T(boost_final.alpha, torch.double)
cum = torch.cumsum(al[:, None, None] * F.one_hot(ptr_b, K).double(), 0).argmax(2)
learner_ok = (ptr_b[:, hv] == int(ytr[hv])).cpu().numpy()
ens_ok = (cum[:, hv] == int(ytr[hv])).cpu().numpy()
print(f'heaviest train face: row {hv}, class {ytr[hv]}, init weight {Wm[0, hv]:.5f}, peak weight {Wm[:, hv].max():.5f} '
      f'({Wm[:, hv].max() / Wm[0, hv]:.1f}x), learner wrong in {(~learner_ok).sum()}/{len(learner_ok)} rounds, '
      f'ensemble correct from round {int(np.argmax(ens_ok)) if ens_ok.any() else None}')
fig, ax = plt.subplots(1, 3, figsize=(14, 3.5), gridspec_kw={'width_ratios': [1, 2.4, 2.4]})
ax[0].imshow(imgs[tr_i[hv]], cmap='gray'); ax[0].axis('off'); ax[0].set_title(f'train row {hv}, class {ytr[hv]}')
ax[1].plot(Wm[:, hv]); ax[1].scatter(np.where(~learner_ok)[0], Wm[~learner_ok, hv], c='r', s=14, label='learner wrong')
ax[1].set_title('weight of this face over rounds'); ax[1].set_xlabel('round'); ax[1].legend()
ax[2].step(np.arange(len(ens_ok)), ens_ok.astype(int), where='mid', label='ensemble correct')
ax[2].step(np.arange(len(learner_ok)), learner_ok.astype(int) * 0.9, where='mid', label='this round\'s learner correct', alpha=.7)
ax[2].set_title('prediction history'); ax[2].set_xlabel('round'); ax[2].legend(fontsize=8)
save('q2c_heavy_face')

# %% [markdown]
# ## Q2(d) Frozen tree / bagging / boosting on clean and masked

# %%
res = {}
for nm, X in tests.items():
    q = boost_final.proba(X)
    res[nm] = dict(pred=q.argmax(1).cpu().numpy(), conf=q.max(1).values.cpu().numpy())
tab_q2d = pd.DataFrame([dict(set=nm, tree_acc=(fz[nm]['tp'] == yte).mean(), tree_conf=fz[nm]['tc'].mean(),
                             bag_acc=(fz[nm]['bp'] == yte).mean(), bag_conf=fz[nm]['bc'].mean(),
                             boost_acc=(res[nm]['pred'] == yte).mean(), boost_conf=res[nm]['conf'].mean()) for nm in tests])
tables['q2d_three_models'] = tab_q2d
display(tab_q2d.round(4))

bc_ok, bm_ok = fz['clean']['bp'] == yte, fz['masked']['bp'] == yte
sc_ok, sm_ok = res['clean']['pred'] == yte, res['masked']['pred'] == yte
gap = (res['clean']['conf'] - res['masked']['conf']) - (fz['clean']['bc'] - fz['masked']['bc'])
cand = np.where(bm_ok & sc_ok & ~sm_ok)[0]
if len(cand) == 0:
    cand = np.where(sm_ok & bc_ok & ~bm_ok)[0]
if len(cand) == 0:
    cand = np.argsort(-np.abs(gap))[:1]
f = int(cand[np.argmax(np.abs(gap[cand]))])
xc, xm = Xte[f], Xte_m[f]

def changed(trees):
    return np.array([t.predict(xc[None])[0] != t.predict(xm[None])[0] for t in trees])

ab = boost_final.alpha / boost_final.alpha.sum()
vb_c, vb_m = fz['clean']['v'][f:f + 1], fz['masked']['v'][f:f + 1]
face_tab = pd.DataFrame([
    dict(model='bagging', true=int(yte[f]), pred_clean=int(fz['clean']['bp'][f]), pred_masked=int(fz['masked']['bp'][f]),
         conf_clean=fz['clean']['bc'][f], conf_masked=fz['masked']['bc'][f],
         frac_learners_path_touches_block=touches(bag_final.trees, xc).mean(),
         frac_learners_changed_pred=changed(bag_final.trees).mean(),
         disagreement_clean=pair_disagree(vb_c), disagreement_masked=pair_disagree(vb_m)),
    dict(model='boosting', true=int(yte[f]), pred_clean=int(res['clean']['pred'][f]), pred_masked=int(res['masked']['pred'][f]),
         conf_clean=res['clean']['conf'][f], conf_masked=res['masked']['conf'][f],
         frac_learners_path_touches_block=(ab * touches(boost_final.trees, xc)).sum(),
         frac_learners_changed_pred=(ab * changed(boost_final.trees)).sum(),
         disagreement_clean=np.nan, disagreement_masked=np.nan)])
tables['q2d_face'] = face_tab
print('chosen test row', f, '(boosting rows are alpha weighted)')
display(face_tab.round(4))

def split_share(trees, w=None):
    s = np.array([blk[t.tree_.feature[t.tree_.feature >= 0]].mean() for t in trees])
    return s.mean() if w is None else (w * s).sum()
print('share of split nodes inside the masked block: bagging', round(split_share(bag_final.trees), 4),
      '| boosting (alpha weighted)', round(split_share(boost_final.trees, ab), 4), '| block is', round(blk.mean(), 4), 'of all pixels')

fig, ax = plt.subplots(1, 2, figsize=(5, 2.8))
ax[0].imshow(xc.reshape(64, 64), cmap='gray'); ax[0].set_title('clean'); ax[1].imshow(xm.reshape(64, 64), cmap='gray'); ax[1].set_title('masked')
for a in ax: a.axis('off')
save('q2d_face')

# %% [markdown]
# ## Q3 Bias and variance over 30 resampled training sets

# %%
R = 30
gr = np.random.default_rng(SEED + 7)
resets = []
for r in range(R):
    ix = []
    for c in range(K):
        own = np.where(ytr == c)[0]
        ix += gr.choice(own, size=6, replace=True).tolist()
    resets.append(np.array(ix))

def oh(pred):
    return F.one_hot(T(pred, torch.long), K).double()

if FD is None:
    D3 = [2, 8, None]
else:
    D3 = [1 if FD == 2 else 2, FD, None]
print('depths used in Q3(b):', D3, '(edit D3 here if you want others)')

P = {m: {s: [] for s in tests} for m in ['tree', 'bag', 'boost']}
P3 = {d: dict(tree=[], bag=[], leaf=[], dis=[]) for d in D3}
t0 = time.time()
for r, ix in enumerate(resets):
    Xb, yb = Xtr[ix], ytr[ix]
    for d in D3:
        td = DecisionTreeClassifier(max_depth=d, min_samples_leaf=FL, random_state=SEED).fit(Xb, yb)
        bd = Bag(max(100, Bstar) if d == FD else 100, d, FL, SEED + 3000 + r).fit(Xb, yb)
        b100 = bd.head(100)
        vd = b100.votes(Xte)
        P3[d]['tree'].append(oh(td.predict(Xte)))
        P3[d]['bag'].append(vd / 100)
        P3[d]['leaf'].append(leaf_info(td) + (td.get_depth(),))
        P3[d]['dis'].append(pair_disagree(vd))
        if d == FD:
            tsel, bsel = td, bd.head(Bstar)
    s = Samme(DB, RB, SEED + 2000 + r).fit(Xb, yb)
    for nm, X in tests.items():
        P['tree'][nm].append(oh(tsel.predict(X)))
        P['bag'][nm].append(bsel.votes(X) / Bstar)
        P['boost'][nm].append(s.proba(X))
    if (r + 1) % 3 == 0 or r == 0:
        el = time.time() - t0
        print(f'rep {r + 1}/{R} done, {el:.0f}s elapsed, about {el / (r + 1) * (R - r - 1):.0f}s left')

def bv(Pm):
    e = F.one_hot(yte_t, K).double()
    pbar = Pm.mean(0)
    b2 = ((pbar - e) ** 2).sum(1).mean().item()
    var = ((Pm - pbar) ** 2).sum(2).mean().item()
    mse = ((Pm - e) ** 2).sum(2).mean().item()
    return b2, var, mse

def per_rep_acc(Pm):
    return (Pm.argmax(2) == yte_t).double().mean(1).cpu().numpy()

# %% Q3(a)
PT = {m: {s: torch.stack(P[m][s]) for s in tests} for m in P}
rows = []
for m in ['tree', 'bag', 'boost']:
    b2, var, mse = bv(PT[m]['clean'])
    a = per_rep_acc(PT[m]['clean'])
    rows.append(dict(model=m, bias2=b2, variance=var, bias2_plus_var=b2 + var, expected_sq_err=mse,
                     acc_mean=a.mean(), acc_std=a.std(ddof=1)))
tab_q3a = pd.DataFrame(rows)
tables['q3a'] = tab_q3a
display(tab_q3a.round(5))

# %% Q3(b), Q3(c)
rows = []
for d in D3:
    st = torch.stack(P3[d]['tree']); sb = torch.stack(P3[d]['bag'])
    b2s, vs, _ = bv(st); b2b, vb, _ = bv(sb)
    li = np.array(P3[d]['leaf'])
    a_s, a_b = per_rep_acc(st).mean(), per_rep_acc(sb).mean()
    rows.append(dict(depth=str(d), grown_depth=li[:, 3].mean(), bias2_tree=b2s, var_tree=vs, bias2_bag=b2b, var_bag=vb,
                     var_removed=vs - vb, var_removed_pct=100 * (vs - vb) / vs, bias2_change=b2b - b2s,
                     acc_tree=a_s, acc_bag=a_b, acc_gain=a_b - a_s,
                     leaves=li[:, 0].mean(), samples_per_leaf=li[:, 1].mean(), single_leaf_frac=li[:, 2].mean(),
                     bag_disagreement=np.mean(P3[d]['dis'])))
tab_q3b = pd.DataFrame(rows)
tables['q3b_q3c'] = tab_q3b
display(tab_q3b.round(4))
bd_ = tab_q3b.loc[tab_q3b.acc_gain.idxmax()]
print('largest bagging gain in test accuracy at depth:', bd_.depth, '| deepest is', tab_q3b.depth.iloc[-1],
      '| largest variance removed at depth:', tab_q3b.loc[tab_q3b.var_removed.idxmax()].depth)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
x = np.arange(len(D3)); w = .38
ax[0].bar(x - w / 2, tab_q3b.var_tree, w, label='single tree'); ax[0].bar(x + w / 2, tab_q3b.var_bag, w, label='bagged 100'); ax[0].set_title('variance')
ax[1].bar(x - w / 2, tab_q3b.bias2_tree, w, label='single tree'); ax[1].bar(x + w / 2, tab_q3b.bias2_bag, w, label='bagged 100'); ax[1].set_title('bias squared')
ax[2].bar(x - w / 2, tab_q3b.acc_tree, w, label='single tree'); ax[2].bar(x + w / 2, tab_q3b.acc_bag, w, label='bagged 100'); ax[2].set_title('mean test accuracy')
for a in ax:
    a.set_xticks(x); a.set_xticklabels(tab_q3b.depth); a.set_xlabel('max_depth'); a.legend(fontsize=8)
save('q3b_depth')

# %% Q3(d) final table
rows = []
for m, pc, pm in [('single tree', fz['clean']['tp'], fz['masked']['tp']), ('bagging', fz['clean']['bp'], fz['masked']['bp']),
                  ('boosting', res['clean']['pred'], res['masked']['pred'])]:
    k = {'single tree': 'tree', 'bagging': 'bag', 'boosting': 'boost'}[m]
    b2, var, _ = bv(PT[k]['clean'])
    ac, am = per_rep_acc(PT[k]['clean']), per_rep_acc(PT[k]['masked'])
    rows.append(dict(model=m, clean_acc=(pc == yte).mean(), masked_acc=(pm == yte).mean(), drop=(pc == yte).mean() - (pm == yte).mean(),
                     bias2=b2, variance=var, clean_acc_30refits=f'{ac.mean():.4f} +- {ac.std(ddof=1):.4f}',
                     masked_acc_30refits=f'{am.mean():.4f} +- {am.std(ddof=1):.4f}'))
tab_q3d = pd.DataFrame(rows)
tables['q3d_final'] = tab_q3d
display(tab_q3d.round(4))
print('lowest variance:', tab_q3d.loc[tab_q3d.variance.idxmin()].model,
      '| best clean accuracy:', tab_q3d.loc[tab_q3d.clean_acc.idxmax()].model,
      '| biggest masked drop:', tab_q3d.loc[tab_q3d['drop'].idxmax()].model)

# %% save everything
for k, v in tables.items():
    v.to_csv(f'out/{k}.csv', index=False)
json.dump(dict(seed=SEED, frozen_depth=str(FD), frozen_min_leaf=FL, B_star=Bstar, boost_depth=DB, boost_rounds=RB,
               q3b_depths=[str(d) for d in D3]), open('out/choices.json', 'w'), indent=1)
shutil.make_archive('assignment2_outputs', 'zip', 'out')
print('saved to out/ and assignment2_outputs.zip')
try:
    from google.colab import files
    files.download('assignment2_outputs.zip')
except ImportError:
    pass
