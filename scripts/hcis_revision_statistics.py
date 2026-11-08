"""Reproduces the statistics reported in the HCIS revision from per-run summary tables.

Inputs (docs/revision_summaries/): abl_<batch>.csv (final-state subsistence, Trader wealth),
mtscx_<batch>.csv (first MTSC event step), sum_<batch>.csv (final Gini per run).
Run: python scripts/hcis_revision_statistics.py docs/revision_summaries
"""
import sys, itertools, numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

d = sys.argv[1] if len(sys.argv) > 1 else '.'
def rd(n): return pd.read_csv(f'{d}/{n}.csv')
BURN = 50

def wilson(k, n):
    lo, hi = proportion_confint(k, n, method='wilson'); return 100*lo, 100*hi

def games_howell(groups):
    names = list(groups); out = []
    for a, b in itertools.combinations(names, 2):
        x, y = np.asarray(groups[a]), np.asarray(groups[b])
        va, vb = x.var(ddof=1)/len(x), y.var(ddof=1)/len(y)
        se = np.sqrt(va+vb); diff = y.mean()-x.mean()
        df = (va+vb)**2/(va**2/(len(x)-1)+vb**2/(len(y)-1))
        q = abs(diff)/se*np.sqrt(2)
        k = len(names)
        p = stats.studentized_range.sf(q, k, df)
        crit = stats.studentized_range.ppf(0.95, k, df)/np.sqrt(2)
        out.append((a, b, diff, diff-crit*se, diff+crit*se, p))
    return pd.DataFrame(out, columns=['a','b','diff','lo','hi','p'])

print('== Final-state subsistence collapse (Wilson 95% CI) ==')
A, C = rd('abl_A'), rd('abl_C')
rows = {'G1 matched (C)': C[C.gov=='G1'], 'G2 (A)': A[A.gov=='G2'], 'G3 (A)': A[A.gov=='G3']}
for b, g, lab in (('D','G2','D G2 Traders exempt'),('E','G3','E G3 Traders exempt'),('F','G2','F G2 no levy'),('G','G3','G G3 no levy'),('H','G2','H G2 matched frequency')):
    rows[lab] = rd(f'abl_{b}')
for lab, x in rows.items():
    k, n = int(x.final_collapsed.sum()), len(x); lo, hi = wilson(k, n)
    print(f'{lab:28s} {k:3d}/{n} = {100*k/n:5.1f}%  [{lo:.1f}, {hi:.1f}]  median Trader wealth {x.trader_mean_final.median():.1f}')
g1m = int(rows['G1 matched (C)'].final_collapsed.sum())
print('\n== Fisher exact vs matched G1 (uncorrected) ==')
for lab in ('D G2 Traders exempt','E G3 Traders exempt','F G2 no levy','G G3 no levy'):
    x = rows[lab]; k = int(x.final_collapsed.sum())
    print(lab, 'p = %.3f' % stats.fisher_exact([[k, len(x)-k],[g1m, 120-g1m]])[1])

print('\n== MTSC events (with / without 50-step burn-in), Wilson CI ==')
mA, mC = rd('mtscx_A'), rd('mtscx_C')
def cnt(m, g):
    x = m[m.gov==g]; return int((x.mtsc_any>BURN).sum()), int(x.mtsc_any.notna().sum()), len(x)
for lab, m, g in (('G1 unmatched (A)', mA,'G1'), ('G1 matched (C)', mC,'G1'), ('G2 (A)', mA,'G2'), ('G3 (A)', mA,'G3')):
    a, b, n = cnt(m, g); lo, hi = wilson(a, n)
    print(f'{lab:18s} with burn-in {a}/{n} [{lo:.1f}, {hi:.1f}]   any event incl. initialisation {b}/{n}')

print('\n== Final Gini, economic scope (batch A + matched G1 from C), Games-Howell ==')
sA, sC = rd('sum_A_economic'), rd('sum_C_g1_matched')
grp = {'G1': sA[sA.gov=='G1'].gini_final, 'G1m': sC[sC.gov=='G1'].gini_final, 'G2': sA[sA.gov=='G2'].gini_final, 'G3': sA[sA.gov=='G3'].gini_final}
print({k: round(float(v.mean()), 4) for k, v in grp.items()})
print(games_howell(grp).round(4).to_string(index=False))
print('Levene p = %.2e' % stats.levene(*grp.values())[1])

print('\n== Subsistence-flag events stored in the run summaries (collapse_ts): with burn-in / without burn-in (batch A) ==')
for g in ('G1','G2','G3'):
    x = sA[sA.gov==g]; print(g, int(x.collapse_ts.notna().sum()), '/', int(x.collapse_ts_noburn.notna().sum()), 'of', len(x))

print('\n== Final-step indicators: matched-frequency G2 (batch H) vs original G2 (batch A), and G2 vs G3 (Welch 95% CI) ==')
F = rd('final_step_H_A_G2_G3')
H_ = F[F.batch.str.startswith('H')]; G2_ = F[(F.batch=='A_economic') & (F.gov=='G2')]; G3_ = F[(F.batch=='A_economic') & (F.gov=='G3')]
def welch(x, y):
    x, y = np.asarray(x), np.asarray(y); d = x.mean()-y.mean()
    vx, vy = x.var(ddof=1)/len(x), y.var(ddof=1)/len(y); se = np.sqrt(vx+vy)
    df = (vx+vy)**2/(vx**2/(len(x)-1)+vy**2/(len(y)-1)); t = stats.t.ppf(0.975, df)
    return d, d-t*se, d+t*se
for lab, (x, y) in {'H minus G2': (H_, G2_), 'G2 minus G3': (G2_, G3_)}.items():
    for col in ('gini_final', 'hdi_final', 'gdp_final', 'stock_final'):
        d, lo, hi = welch(x[col], y[col])
        print(f'{lab:12s} {col:12s} {x[col].mean():9.4f} vs {y[col].mean():9.4f}  diff {d:+.4f} [{lo:+.4f}, {hi:+.4f}]')
