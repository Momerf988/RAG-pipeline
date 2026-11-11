"""Regenerates the statistics reported in the article from the files in results/ and
benchmark_datasets/. It makes no API calls and costs nothing.

Run from the top folder of the archive (the folder that contains src/ and results/):

    python src/15_reproduce_article_statistics.py

Requires pandas, numpy, scipy and openpyxl. Output is grouped by article section.

Not recomputed here: token totals and charges (Section 4.5), which were read from the
provider dashboards, and the retrieval-depth sweep in Section 3.3, which is produced by
src/07_tune_retrieval_depth.py.
"""
import ast
import re

import numpy as np
import pandas as pd
from scipy import stats

M = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
BLOOM = {'Remember': 1, 'Understand': 2, 'Apply': 3, 'Analyse': 4, 'Evaluate': 5}
DIFFICULTY = {'Easy': 1, 'Medium': 2, 'Hard': 3}
SPAN = {'R1': 1, 'R2': 2, 'R3': 3, 'R4': 4}
REFUSAL = "outside the scope of our current syllabus"  # part of System B's fixed refusal


# ---------------------------------------------------------------- helpers
def fp(p):
    """Format a p-value as in the article: three decimals, or one significant figure."""
    return f"{p:.0e}" if p < 0.001 else f"{p:.3f}"


def boot_ci(a, b, n=10_000, seed=42):
    """Percentile bootstrap 95% CI of the mean paired difference b - a."""
    rng = np.random.default_rng(seed)
    dd = (b - a).to_numpy()
    means = [rng.choice(dd, len(dd), replace=True).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def rank_biserial_paired(a, b):
    """Matched-pairs rank-biserial correlation for b - a (zero differences dropped)."""
    dd = (b - a).to_numpy()
    nz = dd[dd != 0]
    if len(nz) == 0:
        return 0.0
    r = stats.rankdata(np.abs(nz))
    return (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()


def partial_spearman(x, y, z):
    """Spearman correlation of x and y controlling for z, with its p-value (df = n - 3)."""
    rxy = stats.spearmanr(x, y)[0]
    rxz = stats.spearmanr(x, z)[0]
    ryz = stats.spearmanr(y, z)[0]
    r = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    df = len(x) - 3
    t = r * np.sqrt(df / (1 - r ** 2))
    return r, 2 * stats.t.sf(abs(t), df)


def holm_thresholds(pvals, alpha=0.05):
    """Holm step-down thresholds and decisions, in the original order."""
    order = np.argsort(pvals)
    m = len(pvals)
    thr = np.empty(m)
    sig = np.zeros(m, dtype=bool)
    still = True
    for rank, i in enumerate(order):
        thr[i] = alpha / (m - rank)
        still = still and pvals[i] < thr[i]
        sig[i] = still
    return thr, sig


def classifier(pred, truth):
    """Precision, recall, specificity, balanced accuracy and Fisher's exact p."""
    tp = int((pred & truth).sum())
    fp_ = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    prec, rec, spec = tp / (tp + fp_), tp / (tp + fn), tn / (tn + fp_)
    p = stats.fisher_exact([[tp, fp_], [fn, tn]])[1]
    return prec, rec, spec, (rec + spec) / 2, p, (tp, fp_, fn, tn)


def head(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------- data
A = pd.read_csv('results/System_A_final_scorecard.csv')
B = pd.read_csv('results/System_B_3way_scorecard.csv')
EA = pd.read_csv('results/System_A_final_eval_results.csv')
EB = pd.read_csv('results/System_B_3way_eval_results.csv')
S = (pd.read_excel('benchmark_datasets/benchmark_specification_matrix.xlsx',
                   'Benchmark Spec Matrix')
     .rename(columns={'Spec_ID': 'spec_id'}))

d = (A[['spec_id', 'response'] + M]
     .merge(B[['spec_id', 'response'] + M], on='spec_id', suffixes=('_A', '_B'))
     .merge(EA[['spec_id', 'contexts']], on='spec_id')
     .merge(EB[['spec_id', 'contexts', 'crag_decision_1', 'path_taken']], on='spec_id',
            suffixes=('_A', '_B'))
     .merge(S[['spec_id', 'Bloom_Level', 'Difficulty', 'Retrieval_Level', 'Question_Type',
               'Reference_Answer']], on='spec_id'))
d['bloom'] = d.Bloom_Level.map(BLOOM)
d['difficulty'] = d.Difficulty.map(DIFFICULTY)
d['span'] = d.Retrieval_Level.map(SPAN)
d['withheld'] = d.path_taken == 'fallback'

# ---------------------------------------------------------------- Section 3
head('Section 3.3 - context-size ablation (24-item sample)')
print(pd.read_csv('results/keep_top_ablation_summary.csv').round(3).to_string(index=False))

head('Section 3.5 - benchmark')
print(f"Items: {len(S)}; items with an audit note: {S.Validation_Notes.notna().sum()}")
print(pd.crosstab(S.Bloom_Level, S.Retrieval_Level).loc[list(BLOOM)].to_string())

# ---------------------------------------------------------------- Section 4.1
head('Section 4.1 - data integrity and the resolution of the instrument')
print(f"Items: A {len(A)}, B {len(B)}; paired {len(d)}")
print(f"Missing scores: A {A[M].isna().sum().to_dict()}, B {B[M].isna().sum().to_dict()}")
print(f"First-pass verdicts: {d.crag_decision_1.value_counts().to_dict()}")
print(f"Routing: {d.path_taken.value_counts().to_dict()}")
direct = d[d.path_taken == 'direct']
same = (direct.contexts_A.astype(str) == direct.contexts_B.astype(str)).sum()
print(f"Direct-path items with byte-identical retrieved context: {same}/{len(direct)}")

print("\nTable 1 - disagreement on the items with identical retrieved context")
print(f"{'Metric':18s} {'Identical':>10s} {'Mean|d|':>8s} {'Max|d|':>7s} {'Signed':>7s} {'p':>6s}")
for m in M:
    a, b = direct[m + '_A'], direct[m + '_B']
    ok = a.notna() & b.notna()
    diff = (b - a)[ok]
    print(f"{m:18s} {f'{(diff == 0).sum()}/{ok.sum()}':>10s} {diff.abs().mean():8.3f} "
          f"{diff.abs().max():7.3f} {diff.mean():+7.3f} {stats.wilcoxon(a[ok], b[ok]).pvalue:6.3f}")

# ---------------------------------------------------------------- Section 4.2
head('Section 4.2 - where the baseline fails (System A)')
print("Table 2 - Spearman rho (p) of each System A metric with the stratification dimensions")
for dim, label in [('bloom', 'Bloom level'), ('difficulty', 'Difficulty'),
                   ('span', 'Evidence span')]:
    cells = []
    for m in M:
        ok = d[m + '_A'].notna()
        rho, p = stats.spearmanr(d.loc[ok, dim], d.loc[ok, m + '_A'])
        cells.append(f"{rho:+.3f} ({fp(p)})")
    print(f"  {label:14s} " + '  '.join(f"{m}: {c}" for m, c in zip(M, cells)))

f = d[d.faithfulness_A.notna()]
print(f"\nPartial rank correlations with faithfulness (n = {len(f)})")
for x, z, label in [('bloom', 'span', 'Bloom | span controlled'),
                    ('span', 'bloom', 'Span | Bloom controlled'),
                    ('difficulty', 'span', 'Difficulty | span controlled'),
                    ('span', 'difficulty', 'Span | difficulty controlled')]:
    r, p = partial_spearman(f[x], f.faithfulness_A, f[z])
    print(f"  {label:30s} rho = {r:+.3f}  p = {fp(p)}")

print("\nWithin-stratum tests (Spearman rho with faithfulness)")
for lvl in ['Understand', 'Evaluate']:
    g = f[f.Bloom_Level == lvl]
    rho, p = stats.spearmanr(g.span, g.faithfulness_A)
    print(f"  span, within {lvl:10s} (n={len(g):2d}): rho = {rho:+.3f}  p = {fp(p)}")
g = f[f.Difficulty == 'Hard']
rho, p = stats.spearmanr(g.span, g.faithfulness_A)
print(f"  span, within Hard items      (n={len(g):2d}): rho = {rho:+.3f}  p = {fp(p)}")
for lvl in SPAN:
    g = f[f.Retrieval_Level == lvl]
    rho, p = stats.spearmanr(g.bloom, g.faithfulness_A)
    print(f"  Bloom, within {lvl}             (n={len(g):2d}): rho = {rho:+.3f}  p = {fp(p)}")

print("\nSystem A means by Bloom level (Fig. 2a)")
print(d.groupby('Bloom_Level')[['faithfulness_A', 'context_recall_A']].agg(['mean', 'size'])
      .loc[list(BLOOM)].round(3).to_string())
print("\nSystem A means by evidence span (Fig. 2b)")
print(d.groupby('Retrieval_Level')[['faithfulness_A', 'context_recall_A']].mean()
      .round(3).to_string())

# ---------------------------------------------------------------- Section 4.3
head('Section 4.3 - effect of the corrective interlock')
rows = []
for m in M:
    for view, g in [('all', d), ('substantive', d[~d.withheld])]:
        g = g[g[m + '_A'].notna() & g[m + '_B'].notna()]
        a, b = g[m + '_A'], g[m + '_B']
        lo, hi = boot_ci(a, b)
        rows.append(dict(metric=m, view=view, n=len(g), A=f"{a.mean():.3f} ({a.std():.2f})",
                         B=f"{b.mean():.3f} ({b.std():.2f})", delta=(b - a).mean(),
                         ci=f"[{lo:+.3f}, {hi:+.3f}]", r=rank_biserial_paired(a, b),
                         p=stats.wilcoxon(a, b).pvalue))
t3 = pd.DataFrame(rows)
t3['holm_alpha'], t3['sig'] = holm_thresholds(t3.p.to_numpy())
print("Table 3 - aggregate comparison (Wilcoxon signed-rank, Holm across the eight tests)")
print(t3.assign(delta=t3.delta.map('{:+.3f}'.format), r=t3.r.map('{:+.2f}'.format),
                p=t3.p.map('{:.3f}'.format), holm_alpha=t3.holm_alpha.map('{:.3f}'.format))
      .to_string(index=False))
fs = t3[(t3.metric == 'faithfulness') & (t3.view == 'substantive')].iloc[0]
print(f"Substantive faithfulness in more detail: delta = {fs.delta:+.4f}, "
      f"rank-biserial r = {fs.r:+.3f}")

rw = d[d.path_taken == 'rewritten']
print(f"\nRewritten and re-retrieved items (n = {len(rw)})")
for m in ['faithfulness', 'context_recall']:
    g = rw[rw[m + '_A'].notna() & rw[m + '_B'].notna()]
    print(f"  {m:15s} change = {(g[m + '_B'] - g[m + '_A']).mean():+.3f}  "
          f"p = {stats.wilcoxon(g[m + '_A'], g[m + '_B']).pvalue:.3f}  (n = {len(g)})")


def vocab(text):
    """Lower-cased alphabetic tokens of three or more characters."""
    return {w for w in re.findall(r'[a-zA-Z_]+', text.lower()) if len(w) >= 3}


def joined_contexts(cell):
    try:
        return ' '.join(ast.literal_eval(cell))
    except (ValueError, SyntaxError, TypeError):
        return None


cov = []
for _, row in d[d.path_taken != 'direct'].iterrows():
    ref = vocab(str(row.Reference_Answer))
    ca, cb = joined_contexts(row.contexts_A), joined_contexts(row.contexts_B)
    if not ref or ca is None or cb is None:
        continue
    cov.append((len(ref & vocab(ca)) / len(ref), len(ref & vocab(cb)) / len(ref)))
cov = np.array(cov)
print(f"Judge-free check: reference-vocabulary coverage of the retrieved context, first "
      f"retrieval vs after rewriting, {len(cov)} of {(d.path_taken != 'direct').sum()} "
      f"corrective-path items: change = {(cov[:, 1] - cov[:, 0]).mean():+.3f}, "
      f"p = {stats.wilcoxon(cov[:, 0], cov[:, 1]).pvalue:.3f}")

# ---------------------------------------------------------------- Section 4.4
head('Section 4.4 - the abstention policy')
w = d[d.withheld]
ans = d[~d.withheld]
wf, af = w.faithfulness_A.dropna(), ans.faithfulness_A.dropna()
u = stats.mannwhitneyu(wf, af, alternative='two-sided')
print(f"System A faithfulness on withheld items: mean {wf.mean():.3f}, median {wf.median():.3f} "
      f"(n = {len(wf)})")
print(f"System A faithfulness on answered items: mean {af.mean():.3f}, median {af.median():.3f} "
      f"(n = {len(af)})")
print(f"Mann-Whitney U = {u.statistic:.1f}, p = {u.pvalue:.2e}, rank-biserial r = "
      f"{1 - 2 * u.statistic / (len(wf) * len(af)):+.3f} (withheld group first)")
print(f"Withheld items where System A also refused: "
      f"{w.response_A.str.contains(REFUSAL, regex=False).sum()} of {len(w)}")
print(f"Withheld items with System A faithfulness of exactly 0: {(wf == 0).sum()}; "
      f"above .70: {(wf > 0.70).sum()}; maximum {wf.max():.3f}")

print("\nSystem A means by System B's first-pass verdict (Fig. 3b)")
print(d.groupby('crag_decision_1')[['faithfulness_A', 'context_recall_A']].mean()
      .loc[['CORRECT', 'AMBIGUOUS', 'INCORRECT']].round(3).to_string())

print(f"\nCoverage: {len(ans)}/{len(d)} = {len(ans) / len(d):.1%}; mean System A faithfulness "
      f"{d.faithfulness_A.mean():.3f} over all items, {af.mean():.3f} over the answered items")
print("Share of items withheld, by Bloom level")
print(d.groupby('Bloom_Level').withheld.agg(['sum', 'size', 'mean']).loc[list(BLOOM)]
      .round(3).to_string())
print("Withheld items and System A faithfulness, by question type")
print(d.groupby('Question_Type').agg(withheld=('withheld', 'sum'), n=('withheld', 'size'),
                                     faithfulness_A=('faithfulness_A', 'mean'))
      .sort_values('faithfulness_A').round(3).to_string())

g = d[d.faithfulness_A.notna()]
print("\nThe gate as a classifier (positive = withheld)")
for label, pred, truth in [
        ('retrieval insufficiency (A recall < 1.000)', d.withheld, d.context_recall_A < 1.0),
        ('generation risk (A faithfulness < .50)', g.withheld, g.faithfulness_A < 0.50),
        ('generation risk (A faithfulness < .70)', g.withheld, g.faithfulness_A < 0.70)]:
    prec, rec, spec, bal, p, cells = classifier(pred.to_numpy(), truth.to_numpy())
    print(f"  {label:44s} precision {prec:.3f}  recall {rec:.3f}  specificity {spec:.3f}  "
          f"balanced accuracy {bal:.3f}  Fisher p = {fp(p)}  (TP, FP, FN, TN) = {cells}")
print(f"Withheld items with System A context recall of 1.000: "
      f"{(w.context_recall_A == 1.0).sum()} of {len(w)}")

# ---------------------------------------------------------------- Section 4.5
head('Section 4.5 - latency (run with a 70-second pause between items)')
L = pd.read_csv('results/Latency_Results_t70.csv')
L['overhead_ms'] = (L.B_total_ms - L.B_retrieval_ms) - (L.A_total_ms - L.A_retrieval_ms)
is_direct = L.B_path == 'direct'
med_direct = L.overhead_ms[is_direct].median()
med_corrective = L.overhead_ms[~is_direct].median()
share = (d.path_taken == 'direct').mean()
print(f"Items: {len(L)} ({is_direct.sum()} direct, {(~is_direct).sum()} corrective), "
      f"{L.retrieval_level.value_counts().sort_index().to_dict()}")
print(f"Median retrieval time: A {L.A_retrieval_ms.median():.0f} ms (timed first), "
      f"B {L.B_retrieval_ms.median():.0f} ms (timed second); retrieval is excluded below")
print(f"Median within-item overhead: direct {med_direct:.0f} ms, corrective "
      f"{med_corrective:.0f} ms")
print(f"System A median total on the direct-path items: {L.A_total_ms[is_direct].median():.0f} ms")
print(f"Weighted by the benchmark routing split ({share:.3f} direct): "
      f"{share * med_direct + (1 - share) * med_corrective:.0f} ms per query")

# ---------------------------------------------------------------- Section 5.4
head('Section 5.4 - routing consistency across runs')
routes = {'main': d.set_index('spec_id').path_taken}
for t in ['t1', 't30', 't70']:
    routes[t] = pd.read_csv(f'results/Latency_Results_{t}.csv').set_index('spec_id').B_path
routes = pd.DataFrame({k: v.map(lambda p: 'direct' if p == 'direct' else 'corrective')
                       for k, v in routes.items()}).dropna()
changed = routes.nunique(axis=1) > 1
print(f"Profiled items routed differently in at least one of the four runs: "
      f"{changed.sum()} of {len(routes)}")
print(routes[changed].to_string())
