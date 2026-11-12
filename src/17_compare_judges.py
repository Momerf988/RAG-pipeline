"""Compares the two faithfulness judges and checks whether each of the article's
faithfulness findings still holds when the second judge's scores are used instead.

Inputs: the gpt-4o-mini scores from 11_run_ragas_evaluation.py and the second judge's scores
from 16_second_judge.py, plus the same files 15_reproduce_article_statistics.py
reads. It makes no API calls. The statistics are computed exactly as in script 15, once for
each judge, so the two columns can be compared line by line.

Run from the top folder of the archive (the folder that contains src/ and results/):

    python src/17_compare_judges.py

Prints the comparison and saves it to results/judge_agreement_summary.csv.
Requires pandas, numpy, scipy and openpyxl.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

BLOOM = {'Remember': 1, 'Understand': 2, 'Apply': 3, 'Analyse': 4, 'Evaluate': 5}
DIFFICULTY = {'Easy': 1, 'Medium': 2, 'Hard': 3}
SPAN = {'R1': 1, 'R2': 2, 'R3': 3, 'R4': 4}
JUDGES = {'judge1': 'gpt-4o-mini', 'judge2': None}   # judge2 name is read from its files
OUT_FILE = 'results/judge_agreement_summary.csv'
THRESHOLD = 0.50   # for the agreement on "faithful or not" (Cohen's kappa)


# ---------------------------------------------------------------- helpers (as in script 15)
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


def cohen_kappa(a, b):
    """Cohen's kappa for two boolean ratings of the same items."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def head(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------- data
# pandas 3.1 changed how it reads decimal numbers from CSV files, which moves a few rank
# statistics in the third decimal. The article's numbers were produced with earlier versions.
if tuple(int(x) for x in pd.__version__.split('.')[:2]) >= (3, 1):
    print(f"WARNING: pandas {pd.__version__} is installed. Use pandas older than 3.1 "
          f"(pip install \"pandas<3.1\") to reproduce the article's numbers exactly.")

for f in ['results/System_A_judge2_faithfulness.csv', 'results/System_B_judge2_faithfulness.csv']:
    if not os.path.exists(f):
        raise SystemExit(f"{f} not found. Run src/16_second_judge.py first, from "
                         f"the top folder of the archive.")

A = pd.read_csv('results/System_A_final_scorecard.csv')
B = pd.read_csv('results/System_B_3way_scorecard.csv')
EB = pd.read_csv('results/System_B_3way_eval_results.csv')
A2 = pd.read_csv('results/System_A_judge2_faithfulness.csv')
B2 = pd.read_csv('results/System_B_judge2_faithfulness.csv')
S = (pd.read_excel('benchmark_datasets/benchmark_specification_matrix.xlsx',
                   'Benchmark Spec Matrix')
     .rename(columns={'Spec_ID': 'spec_id'}))
JUDGES['judge2'] = str(A2.judge2_model.iloc[0])

d = (A[['spec_id', 'faithfulness']].rename(columns={'faithfulness': 'judge1_A'})
     .merge(B[['spec_id', 'faithfulness']].rename(columns={'faithfulness': 'judge1_B'}),
            on='spec_id')
     .merge(A2[['spec_id', 'faithfulness_judge2']].rename(
         columns={'faithfulness_judge2': 'judge2_A'}), on='spec_id', how='left')
     .merge(B2[['spec_id', 'faithfulness_judge2']].rename(
         columns={'faithfulness_judge2': 'judge2_B'}), on='spec_id', how='left')
     .merge(EB[['spec_id', 'path_taken', 'crag_decision_1']], on='spec_id')
     .merge(S[['spec_id', 'Bloom_Level', 'Difficulty', 'Retrieval_Level']], on='spec_id'))
d['bloom'] = d.Bloom_Level.map(BLOOM)
d['difficulty'] = d.Difficulty.map(DIFFICULTY)
d['span'] = d.Retrieval_Level.map(SPAN)
d['withheld'] = d.path_taken == 'fallback'

summary = []


def add(section, statistic, v1, v2):
    """A statistic computed once per judge."""
    summary.append(dict(section=section, statistic=statistic, judge1=v1, judge2=v2,
                        between_judges=''))


def add_agreement(statistic, value):
    """A statistic that compares the two judges."""
    summary.append(dict(section='1 agreement', statistic=statistic, judge1='', judge2='',
                        between_judges=value))


head(f"Judges: judge1 = {JUDGES['judge1']}, judge2 = {JUDGES['judge2']}")
for s, df2 in [('A', A2), ('B', B2)]:
    print(f"System {s}: second-judge rows {len(df2)} of 150, missing (NaN) "
          f"{df2.faithfulness_judge2.isna().sum()}, first-judge missing "
          f"{d['judge1_' + s].isna().sum()}")
if len(A2) < 150 or len(B2) < 150:
    print("WARNING: the second judge has not scored every row yet. Finish script 16 first; "
          "the numbers below use only the rows scored so far.")

# ---------------------------------------------------------------- 1. agreement
head("1. Agreement between the two judges (faithfulness)")
long = pd.concat([d[['judge1_A', 'judge2_A']].set_axis(['j1', 'j2'], axis=1).assign(sys='A'),
                  d[['judge1_B', 'judge2_B']].set_axis(['j1', 'j2'], axis=1).assign(sys='B')])
print(f"{'Set':8s} {'n':>4s} {'Spearman':>9s} {'Pearson':>8s} {'Mean j1':>8s} {'Mean j2':>8s} "
      f"{'j2-j1':>7s} {'Mean|d|':>8s} {'|d|<=.10':>9s} {'kappa@.50':>9s}")
for label, g in [('System A', long[long.sys == 'A']), ('System B', long[long.sys == 'B']),
                 ('Pooled', long)]:
    g = g.dropna(subset=['j1', 'j2'])
    rho, p_rho = stats.spearmanr(g.j1, g.j2)
    r, _ = stats.pearsonr(g.j1, g.j2)
    diff = g.j2 - g.j1
    within = (diff.abs() <= 0.10).mean()
    kappa = cohen_kappa(g.j1 >= THRESHOLD, g.j2 >= THRESHOLD)
    print(f"{label:8s} {len(g):4d} {rho:9.3f} {r:8.3f} {g.j1.mean():8.3f} {g.j2.mean():8.3f} "
          f"{diff.mean():+7.3f} {diff.abs().mean():8.3f} {within:9.1%} {kappa:9.3f}")
    add('1 agreement', f'{label}: mean faithfulness', f"{g.j1.mean():.3f}", f"{g.j2.mean():.3f}")
    add_agreement(f'{label}: n paired', len(g))
    add_agreement(f'{label}: Spearman rho (p)', f"{rho:.3f} ({fp(p_rho)})")
    add_agreement(f'{label}: Pearson r', f"{r:.3f}")
    add_agreement(f'{label}: mean difference judge2 - judge1', f"{diff.mean():+.3f}")
    add_agreement(f'{label}: mean absolute difference', f"{diff.abs().mean():.3f}")
    add_agreement(f'{label}: share of items within 0.10', f"{within:.3f}")
    add_agreement(f"{label}: Cohen's kappa, faithful at {THRESHOLD:.2f} or above", f"{kappa:.3f}")

# ---------------------------------------------------------------- 2. findings under each judge
head("2. The article's faithfulness findings under each judge")
holds = {}

print("\nSection 4.1 - judge noise on the 77 direct-path items (identical context in A and B)")
direct = d[d.path_taken == 'direct']
for j in ['judge1', 'judge2']:
    ok = direct[j + '_A'].notna() & direct[j + '_B'].notna()
    a, b = direct.loc[ok, j + '_A'], direct.loc[ok, j + '_B']
    diff = b - a
    print(f"  {JUDGES[j]:24s} identical {(diff == 0).sum()}/{ok.sum()}, mean |d| "
          f"{diff.abs().mean():.3f}, signed {diff.mean():+.3f}, p = {stats.wilcoxon(a, b).pvalue:.3f}")
add('4.1 noise floor', 'mean |B - A| on direct-path items',
    *[f"{(direct[j + '_B'] - direct[j + '_A']).abs().mean():.3f}" for j in ['judge1', 'judge2']])

print("\nSection 4.2 - Spearman rho (p) of System A faithfulness with each dimension")
for dim, label in [('bloom', 'Bloom level'), ('difficulty', 'Difficulty'),
                   ('span', 'Evidence span')]:
    cells = []
    for j in ['judge1', 'judge2']:
        ok = d[j + '_A'].notna()
        rho, p = stats.spearmanr(d.loc[ok, dim], d.loc[ok, j + '_A'])
        cells.append(f"{rho:+.3f} ({fp(p)})")
    print(f"  {label:14s} {JUDGES['judge1']}: {cells[0]:18s} {JUDGES['judge2']}: {cells[1]}")
    add('4.2 Table 2', f'faithfulness vs {label}: rho (p)', *cells)

print("\nSection 4.2 - partial rank correlations with System A faithfulness")
partials = {}
for x, z, label in [('bloom', 'span', 'Bloom | span controlled'),
                    ('span', 'bloom', 'Span | Bloom controlled'),
                    ('difficulty', 'span', 'Difficulty | span controlled'),
                    ('span', 'difficulty', 'Span | difficulty controlled')]:
    cells = []
    for j in ['judge1', 'judge2']:
        f = d[d[j + '_A'].notna()]
        r, p = partial_spearman(f[x], f[j + '_A'], f[z])
        partials[(label, j)] = (r, p)
        cells.append(f"{r:+.3f} ({fp(p)})")
    print(f"  {label:30s} {JUDGES['judge1']}: {cells[0]:18s} {JUDGES['judge2']}: {cells[1]}")
    add('4.2 partial correlations', f'{label}: rho (p)', *cells)
for j in ['judge1', 'judge2']:
    rb, pb = partials[('Bloom | span controlled', j)]
    rs, ps = partials[('Span | Bloom controlled', j)]
    holds[('Faithfulness tracks cognitive demand, not evidence span', j)] = \
        rb < 0 and pb < 0.05 and ps >= 0.05

print("\nSection 4.3 - System B vs System A faithfulness (Wilcoxon signed-rank)")
for view, g0 in [('all', d), ('substantive', d[~d.withheld])]:
    cells = []
    for j in ['judge1', 'judge2']:
        g = g0[g0[j + '_A'].notna() & g0[j + '_B'].notna()]
        a, b = g[j + '_A'], g[j + '_B']
        lo, hi = boot_ci(a, b)
        p = stats.wilcoxon(a, b).pvalue
        cells.append(f"n={len(g)} A={a.mean():.3f} B={b.mean():.3f} delta={(b - a).mean():+.3f} "
                     f"CI=[{lo:+.3f}, {hi:+.3f}] r={rank_biserial_paired(a, b):+.2f} p={p:.3f}")
        if view == 'substantive':
            holds[('Interlock does not measurably improve substantive answers', j)] = p >= 0.05
    print(f"  {view:12s} {JUDGES['judge1']}: {cells[0]}\n  {'':12s} {JUDGES['judge2']}: {cells[1]}")
    add('4.3 Table 3', f'faithfulness, {view} items', *cells)

print("\nSection 4.4 - System A faithfulness on items System B withheld vs answered")
cells = {}
for j in ['judge1', 'judge2']:
    wf = d.loc[d.withheld, j + '_A'].dropna()
    af = d.loc[~d.withheld, j + '_A'].dropna()
    u = stats.mannwhitneyu(wf, af, alternative='two-sided')
    rbs = 1 - 2 * u.statistic / (len(wf) * len(af))
    print(f"  {JUDGES[j]:24s} withheld {wf.mean():.3f} (n={len(wf)}), answered {af.mean():.3f} "
          f"(n={len(af)}), Mann-Whitney p = {u.pvalue:.1e}, r = {rbs:+.3f}, withheld above .70: "
          f"{(wf > 0.70).sum()}")
    cells[j] = (f"{wf.mean():.3f} / {af.mean():.3f}", f"{u.pvalue:.1e}", f"{rbs:+.3f}",
                int((wf > 0.70).sum()))
    holds[('Abstentions are selective', j)] = wf.mean() < af.mean() and u.pvalue < 0.05
for k, label in enumerate(['withheld mean / answered mean', 'Mann-Whitney p',
                           'rank-biserial r (withheld first)', 'withheld items above .70']):
    add('4.4 abstention audit', label, cells['judge1'][k], cells['judge2'][k])

# ---------------------------------------------------------------- 3. other quoted figures
head("3. Other faithfulness figures quoted in the article, under each judge")
J1, J2 = JUDGES['judge1'], JUDGES['judge2']

print("\nFig. 2a - mean System A faithfulness by Bloom level")
print(f"  {'Level':12s} {'n':>3s} {J1:>14s} {J2:>24s} {'judge2 - judge1':>16s}")
for level in BLOOM:
    g = d[d.Bloom_Level == level]
    m1, m2 = g.judge1_A.mean(), g.judge2_A.mean()
    print(f"  {level:12s} {len(g):3d} {m1:14.3f} {m2:24.3f} {m2 - m1:+16.3f}")
    add('Fig. 2a', f'System A mean faithfulness, {level} (n={len(g)})', f"{m1:.3f}", f"{m2:.3f}")

print("\nSection 4.3 - the corrective loop alone: B - A faithfulness on the rewritten items")
rw = d[d.path_taken == 'rewritten']
cells = []
for j in ['judge1', 'judge2']:
    g = rw[rw[j + '_A'].notna() & rw[j + '_B'].notna()]
    p = stats.wilcoxon(g[j + '_A'], g[j + '_B']).pvalue
    cells.append(f"n={len(g)} delta={(g[j + '_B'] - g[j + '_A']).mean():+.3f} p={p:.3f}")
    print(f"  {JUDGES[j]:24s} {cells[-1]}")
add('4.3 rewrite path', 'faithfulness change on rewritten items', *cells)

print("\nSection 4.4 - mean System A faithfulness by System B's first-pass verdict (Fig. 3b)")
for verdict in ['CORRECT', 'AMBIGUOUS', 'INCORRECT']:
    g = d[d.crag_decision_1 == verdict]
    m1, m2 = g.judge1_A.mean(), g.judge2_A.mean()
    print(f"  {verdict:10s} n={len(g):3d}   {J1}: {m1:.3f}   {J2}: {m2:.3f}")
    add('4.4 Fig. 3b', f'System A mean faithfulness, first-pass {verdict} (n={len(g)})',
        f"{m1:.3f}", f"{m2:.3f}")

print("\nSection 4.4 - selective prediction: mean System A faithfulness, all items vs answered")
cells = []
for j in ['judge1', 'judge2']:
    cells.append(f"{d[j + '_A'].mean():.3f} -> {d.loc[~d.withheld, j + '_A'].mean():.3f}")
    print(f"  {JUDGES[j]:24s} {cells[-1]}")
add('4.4 selective prediction', 'System A mean faithfulness, all items -> answered items', *cells)

print("\nSection 4.4 - the gate as a classifier of low-faithfulness answers (positive = withheld)")
for cut in [0.50, 0.70]:
    cells = []
    for j in ['judge1', 'judge2']:
        g = d[d[j + '_A'].notna()]
        pred, truth = g.withheld.to_numpy(), (g[j + '_A'] < cut).to_numpy()
        tp, fp_ = int((pred & truth).sum()), int((pred & ~truth).sum())
        fn, tn = int((~pred & truth).sum()), int((~pred & ~truth).sum())
        p = stats.fisher_exact([[tp, fp_], [fn, tn]])[1]
        cells.append(f"precision {tp / (tp + fp_):.3f} recall {tp / (tp + fn):.3f} "
                     f"Fisher p = {fp(p)} (TP, FP, FN, TN) = {(tp, fp_, fn, tn)}")
        print(f"  faithfulness < {cut:.2f}  {JUDGES[j]:24s} {cells[-1]}")
    add('4.4 gate as classifier', f'System A faithfulness < {cut:.2f}', *cells)

print("\nWhere the two judges disagree (both systems pooled, judge2 - judge1)")
long2 = pd.concat([d[['judge1_A', 'judge2_A', 'Bloom_Level']].set_axis(['j1', 'j2', 'level'], axis=1),
                   d[['judge1_B', 'judge2_B', 'Bloom_Level']].set_axis(['j1', 'j2', 'level'], axis=1)]
                  ).dropna(subset=['j1', 'j2'])
long2['diff'] = long2.j2 - long2.j1
for level in BLOOM:
    g = long2[long2.level == level]
    print(f"  {level:12s} n={len(g):3d}  mean difference {g['diff'].mean():+.3f}")
    add_agreement(f'Mean difference judge2 - judge1, {level} (n={len(g)})', f"{g['diff'].mean():+.3f}")
lower, higher = int((long2['diff'] < -0.5).sum()), int((long2['diff'] > 0.5).sum())
p_shift = stats.wilcoxon(long2.j1, long2.j2).pvalue
print(f"  Answers scored more than .50 apart: {lower + higher} of {len(long2)} "
      f"({JUDGES['judge2']} lower in {lower}, higher in {higher})")
print(f"  Shift between judges, Wilcoxon signed-rank p = {fp(p_shift)}")
add_agreement('Answers scored more than .50 apart (judge2 lower / higher)',
              f"{lower + higher} of {len(long2)} ({lower} / {higher})")
add_agreement('Shift between judges, Wilcoxon signed-rank p', fp(p_shift))
ok = direct.judge1_A.notna() & direct.judge2_A.notna()
between = (direct.loc[ok, 'judge2_A'] - direct.loc[ok, 'judge1_A']).abs().mean()
print(f"  Mean |judge2 - judge1| on the {ok.sum()} direct-path System A answers: {between:.3f} "
      f"(compare the within-judge figures in Section 4.1 above)")
add_agreement(f'Mean |judge2 - judge1|, direct-path System A answers (n={ok.sum()})',
              f"{between:.3f}")

# ---------------------------------------------------------------- 4. verdict
head("4. Does each finding hold under both judges?")
for finding in ['Faithfulness tracks cognitive demand, not evidence span',
                'Interlock does not measurably improve substantive answers',
                'Abstentions are selective']:
    v1, v2 = holds[(finding, 'judge1')], holds[(finding, 'judge2')]
    print(f"  {finding:58s} {JUDGES['judge1']}: {'holds' if v1 else 'DOES NOT HOLD':14s} "
          f"{JUDGES['judge2']}: {'holds' if v2 else 'DOES NOT HOLD'}")
    add('3 verdict', finding, 'holds' if v1 else 'does not hold', 'holds' if v2 else 'does not hold')

out = pd.DataFrame(summary).rename(columns={'judge1': JUDGES['judge1'], 'judge2': JUDGES['judge2']})
out = out.astype(str).replace({'nan': ''})
out.to_csv(OUT_FILE, index=False)
print(f"\nSaved to {OUT_FILE}")
