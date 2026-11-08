"""Final-step Gini over Extractors and Producers only (Traders and Governor slots excluded),
from the per-agent wealth logs (<run>_wealth.csv) of the deposited batches.
Usage: python scripts/ex_trader_gini.py <batches_root>   (folders A_economic, C_g1_matched, D_..., as in the deposit)
"""
import glob, sys, numpy as np
root = sys.argv[1] if len(sys.argv) > 1 else '.'
def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x); s = x.sum()
    return float('nan') if s <= 0 else 2*np.sum(np.arange(1, n+1)*x)/(n*s) - (n+1)/n
def final(f):
    rows = []
    with open(f) as fh:
        next(fh)
        for l in fh:
            p = l.rstrip().split(','); rows.append((int(p[0]), p[2], float(p[3]), p[4] == 'True'))
    t = max(r[0] for r in rows)
    return gini([r[2] for r in rows if r[0] == t and r[3] and r[1] in ('extractor', 'producer')])
def welch(x, y):
    d = np.nanmean(x)-np.nanmean(y); se = np.sqrt(np.nanvar(x, ddof=1)/len(x)+np.nanvar(y, ddof=1)/len(y))
    return d, d-1.97*se, d+1.97*se
spec = {'G1 matched': 'C_g1_matched/runs/G1', 'G2': 'A_economic/runs/G2', 'G3': 'A_economic/runs/G3',
        'G2 Traders exempt (D)': 'D_G2_traders_exempt/runs/G2', 'G3 Traders exempt (E)': 'E_G3_traders_exempt/runs/G3',
        'G2 no levy (F)': 'F_G2_no_levy/runs/G2', 'G3 no levy (G)': 'G_G3_no_levy/runs/G3'}
res = {}
for lab, pat in spec.items():
    res[lab] = np.array([final(f) for f in sorted(glob.glob(f'{root}/{pat}/*_wealth.csv'))])
    print(f'{lab:24s} n={len(res[lab])} mean final Gini (Extractors+Producers) {np.nanmean(res[lab]):.3f}')
for k in ('G2', 'G3'):
    d, lo, hi = welch(res[k], res['G1 matched']); print(f'{k} minus G1 matched: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]')
