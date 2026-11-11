"""Reproduces every statistic reported in the results and discussion sections from the raw
scorecards. Run: python 14_verify_statistics.py
Requires: pandas, scipy, openpyxl"""
import pandas as pd
from scipy import stats

M = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']

A = pd.read_csv('results/System_A_final_scorecard.csv')
B = pd.read_csv('results/System_B_3way_scorecard.csv')
E = pd.read_csv('results/System_B_3way_eval_results.csv')[['spec_id', 'crag_decision_1', 'path_taken']]
S = pd.read_excel('benchmark_datasets/benchmark_specification_matrix.xlsx', 'Benchmark Spec Matrix')
S = S[S.Generation_Status == 'Generated'][
    ['Spec_ID', 'Bloom_Level', 'Difficulty', 'Retrieval_Level', 'Question_Type']
].rename(columns={'Spec_ID': 'spec_id'})

d = (A[['spec_id'] + M]
     .merge(B[['spec_id'] + M], on='spec_id', suffixes=('_A', '_B'))
     .merge(E, on='spec_id').merge(S, on='spec_id'))

print(f"Paired items: n = {len(d)}\n")

# ---- aggregate comparison, all items and substantive-only ----
for label, df in [("All items", d), ("Substantive answers", d[d.path_taken != 'fallback'])]:
    print(f"--- {label} (n={len(df)}) ---")
    for m in M:
        a, b = df[m + '_A'], df[m + '_B']
        w = stats.wilcoxon(a, b)
        print(f"  {m:18s} A={a.mean():.3f}({a.std():.2f})  B={b.mean():.3f}({b.std():.2f})  "
              f"d={(b-a).mean():+.3f}  W={w.statistic:.1f}  p={w.pvalue:.4f}")
    print()

# ---- complexity gradient by retrieval level ----
print("--- Complexity gradient by retrieval level ---")
for lvl, g in d.groupby('Retrieval_Level'):
    print(f"  {lvl} n={len(g):3d}  A_faith={g.faithfulness_A.mean():.3f}  "
          f"d_faith={(g.faithfulness_B-g.faithfulness_A).mean():+.3f}  "
          f"A_recall={g.context_recall_A.mean():.3f}  "
          f"d_recall={(g.context_recall_B-g.context_recall_A).mean():+.3f}")

# ---- R4 alone + Spearman rho, gradient vs effect size ----
print("\n--- R4 alone ---")
r4 = d[d.Retrieval_Level == 'R4']
for m in ['faithfulness', 'context_recall']:
    print(f"  {m:18s} d={(r4[m+'_B']-r4[m+'_A']).mean():+.3f}  "
          f"p={stats.wilcoxon(r4[m+'_A'], r4[m+'_B']).pvalue:.3f}")

print("\n--- Spearman rho, gradient vs effect size ---")
rank = d.Retrieval_Level.map({'R1': 1, 'R2': 2, 'R3': 3, 'R4': 4})
for m in ['context_recall', 'faithfulness']:
    rho, p = stats.spearmanr(rank, d[m + '_B'] - d[m + '_A'])
    print(f"  {m:18s} rho={rho:+.3f}  p={p:.4f}")

# ---- the abstained items ----
print("\n--- The abstained items ---")
fb = d[d.path_taken == 'fallback'][['spec_id', 'Retrieval_Level', 'Question_Type',
                                    'faithfulness_A', 'context_recall_A']]
print(fb.sort_values('faithfulness_A', ascending=False).round(3).to_string(index=False))
print(f"  A faithfulness >= 0.70 : {(fb.faithfulness_A >= 0.70).sum()}")
print(f"  A faithfulness == 0.00 : {(fb.faithfulness_A == 0).sum()}")
print(f"  A context recall == 1.000 : {(fb.context_recall_A == 1.0).sum()} of {len(fb)}")

print("\n--- By question type ---")
for qt, g in d.groupby('Question_Type'):
    print(f"  {qt:16s} n={len(g):3d}  d_faith={(g.faithfulness_B-g.faithfulness_A).mean():+.3f}  "
          f"d_recall={(g.context_recall_B-g.context_recall_A).mean():+.3f}  "
          f"abstentions={(g.path_taken=='fallback').sum()}")

print("\n--- Routing ---")
print(" ", d.path_taken.value_counts().to_dict(), d.crag_decision_1.value_counts().to_dict())
