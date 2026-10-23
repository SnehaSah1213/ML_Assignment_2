"""Redraw the table-based figures from results/tables/*.csv into results/figures_regen/.
q1d_cases, q2c_heavy_face and q2d_face need images / per-round sample weights, so they are drawn inside src/assignment2.py."""
import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
T, O = 'results/tables/', 'results/figures_regen/'
os.makedirs(O, exist_ok=True)
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': .3})
def out(n): plt.tight_layout(); plt.savefig(O + n + '.png', dpi=150); plt.close()

g = pd.read_csv(T + 'q1a_grid.csv'); fl = json.load(open('configs/selected.json'))['frozen_min_leaf']
dep = g[g.min_leaf == fl]; xs = np.arange(len(dep))
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(xs, dep.train, 'o-', label='train'); ax[0].plot(xs, dep.val, 's-', label='val'); ax[0].set_title('accuracy vs depth'); ax[0].legend()
ax[1].plot(xs, dep.leaves, 'o-'); ax[1].set_title('number of leaves')
ax[2].plot(xs, dep.per_leaf, 'o-', label='samples/leaf'); ax[2].plot(xs, dep.single_frac, 's-', label='frac single-example leaves'); ax[2].legend(); ax[2].set_title('leaf occupancy')
for a in ax: a.set_xticks(xs); a.set_xticklabels(dep.max_depth.astype(str).str.replace('.0', '', regex=False)); a.set_xlabel(f'max_depth (min_leaf={fl})')
out('q1a_depth')

c = pd.read_csv(T + 'q1c_oob_dis_margin.csv'); bs = int(c.sort_values(['val_acc', 'B'], ascending=[False, True]).iloc[0].B)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, col, t in zip(ax, ['oob_acc', 'disagreement', 'mean_margin'], ['OOB accuracy', 'avg pairwise disagreement (val)', 'mean vote margin (val)']):
    a.plot(c.B, c[col], 'o-'); a.set_xscale('log'); a.set_xlabel('B'); a.set_title(t); a.axvline(bs, color='r', ls='--', alpha=.5)
out('q1c_curves')

r = pd.read_csv(T + 'q2b_rounds.csv')
fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
for a, col, t in zip(ax, ['eps', 'alpha', 'max_w', 'top10_w'], ['weighted training error', 'learner weight alpha', 'max sample weight', 'weight of 10 heaviest images']):
    a.plot(r[col]); a.set_title(t); a.set_xlabel('round')
out('q2b_selected')

q = pd.read_csv(T + 'q3b_q3c.csv'); x = np.arange(len(q)); w = .38
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, (s, b), t in zip(ax, [('var_tree', 'var_bag'), ('bias2_tree', 'bias2_bag'), ('acc_tree', 'acc_bag')], ['variance', 'bias squared', 'mean test accuracy']):
    a.bar(x - w / 2, q[s], w, label='single tree'); a.bar(x + w / 2, q[b], w, label='bagged 100'); a.set_title(t)
    a.set_xticks(x); a.set_xticklabels(['2', '8', 'None']); a.set_xlabel('max_depth'); a.legend(fontsize=8)
out('q3b_depth')
print('figures written to', O)
