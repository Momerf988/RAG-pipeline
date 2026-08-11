"""
verify_all.py  -  regenerates EVERY number reported in the COM748 report from raw data.
No API key, no cost. Run from the repo root with the CSVs and xlsx present.

    python verify_all.py > verification_output.txt

Sections map 1:1 onto report sections so a reader can check any claim directly.
"""
import pandas as pd, numpy as np, ast, re, sys
from scipy import stats

M = ['faithfulness','answer_relevancy','context_precision','context_recall']
BAR = "="*78

def load():
    A = pd.read_csv('System_A_scorecard.csv')
    B = pd.read_csv('System_B_scorecard.csv')
    E = pd.read_csv('System_B_eval_results.csv')
    Ae = pd.read_csv('System_A_eval_results.csv')
    S = pd.read_excel('data/benchmark_specification_matrix.xlsx','Benchmark Spec Matrix')
    S = S[S.Generation_Status=='Generated']
    spec = S[['Spec_ID','Bloom_Level','Difficulty','Retrieval_Level','Question_Type','Supporting_Chunks']] \
             .rename(columns={'Spec_ID':'spec_id'})
    d = (A[['spec_id']+M].merge(B[['spec_id']+M], on='spec_id', suffixes=('_A','_B'))
         .merge(E[['spec_id','crag_decision_1','crag_decision_2','rewritten_query','path_taken','contexts']]
                .rename(columns={'contexts':'ctx_B'}), on='spec_id')
         .merge(Ae[['spec_id','question','ground_truth','contexts']].rename(columns={'contexts':'ctx_A'}), on='spec_id')
         .merge(spec, on='spec_id'))
    d['ref_words'] = d.ground_truth.astype(str).str.split().str.len()
    return d, S

def rb(a,b):
    dd=(b-a); nz=dd[dd!=0]
    if len(nz)==0: return 0.0
    r=stats.rankdata(nz.abs()); return (r[nz>0].sum()-r[nz<0].sum())/r.sum()

def boot(a,b,n=10000,seed=42):
    rng=np.random.default_rng(seed); dd=(b-a).values
    return np.percentile([rng.choice(dd,len(dd),replace=True).mean() for _ in range(n)],[2.5,97.5])

def main():
    d,S = load()
    print(BAR); print("S0  DATA INTEGRITY"); print(BAR)
    print(f"paired rows              {len(d)}   (expect 102)")
    print(f"duplicate spec_ids       {d.spec_id.duplicated().sum()}   (expect 0)")
    print(f"NaNs in metrics          {d[[m+'_A' for m in M]+[m+'_B' for m in M]].isna().sum().sum()}   (expect 0)")
    print(f"deferred items in matrix {(pd.read_excel('data/benchmark_specification_matrix.xlsx','Benchmark Spec Matrix').Generation_Status!='Generated').sum()}   (expect 6)")
    print(f"routing                  {dict(d.path_taken.value_counts())}   (expect direct 89, fallback 9, rewritten 4)")
    print(f"first-pass verdicts      {dict(d.crag_decision_1.value_counts())}   (expect CORRECT 89, INCORRECT 13)")
    print("\ncollinearity of the complexity gradient (S3.7, Table 3)")
    print(d.groupby('Retrieval_Level')[['Difficulty','Bloom_Level']].agg(lambda s: sorted(set(s))).to_string())

    print("\n"+BAR); print("S4.1  MEASUREMENT NOISE  (89 rows, byte-identical retrieved context)"); print(BAR)
    z = d[d.path_taken=='direct'].copy()
    z['same'] = z.ctx_A.astype(str)==z.ctx_B.astype(str)
    print(f"direct rows with byte-identical context: {z.same.sum()} / {len(z)}")
    zz = z[z.same]
    for m in M:
        dd = zz[m+'_B']-zz[m+'_A']; w = stats.wilcoxon(zz[m+'_A'], zz[m+'_B'])
        print(f"  {m:18s} identical {int((dd.abs()<1e-9).sum()):2d}/{len(zz)}  mean|d| {dd.abs().mean():.4f}  max|d| {dd.abs().max():.4f}  signed {dd.mean():+.4f}  p {w.pvalue:.4f}")
    print("  READ: precision drift non-directional -> null trustworthy; recall drift directional -> effects <0.04 unresolvable")

    print("\n"+BAR); print("S4.2  BASELINE FAILURE PROFILE (System A only, all 102)"); print(BAR)
    print(d.groupby('Retrieval_Level')[[m+'_A' for m in M]].mean().round(3).to_string())
    rank = d.Retrieval_Level.map({'R1':1,'R2':2,'R3':3,'R4':4})
    for lbl,x,y in [("R-level vs A faithfulness",rank,d.faithfulness_A),
                    ("ref length vs A faithfulness",d.ref_words,d.faithfulness_A),
                    ("ref length vs A context recall",d.ref_words,d.context_recall_A),
                    ("ref length vs R-level",rank,d.ref_words)]:
        rho,p = stats.spearmanr(x,y); print(f"  Spearman {lbl:32s} rho={rho:+.3f}  p={p:.2e}")
    d['q'] = pd.qcut(d.ref_words,4,labels=['Q1','Q2','Q3','Q4'])
    print("\n  A faithfulness by R-level WITHIN answer-length quartile (gradient must persist):")
    print(d.pivot_table(index='q',columns='Retrieval_Level',values='faithfulness_A',aggfunc='mean',observed=True).round(3).to_string())

    print("\n"+BAR); print("S4.3  AGGREGATE COMPARISON (Table 5)"); print(BAR)
    ps=[]
    for lbl,df in [("ALL 102",d),("SUBSTANTIVE 93",d[d.path_taken!='fallback'])]:
        print(f"\n  --- {lbl} (n={len(df)}) ---")
        for m in M:
            a,b = df[m+'_A'], df[m+'_B']; dd=b-a
            lo,hi = boot(a,b); w = stats.wilcoxon(a,b); ps.append((f"{lbl}:{m}", w.pvalue))
            print(f"  {m:18s} A {a.mean():.3f}(sd {a.std():.2f})  B {b.mean():.3f}(sd {b.std():.2f})  d {dd.mean():+.4f}  CI [{lo:+.3f},{hi:+.3f}]  r {rb(a,b):+.3f}  p {w.pvalue:.4f}")
    print("\n  Holm correction across all 8 tests:")
    ps.sort(key=lambda x:x[1])
    for i,(n,p) in enumerate(ps):
        thr = 0.05/(len(ps)-i); print(f"    {n:28s} p={p:.4f}  thr={thr:.4f}  {'SURVIVES' if p<thr else 'fails'}")

    print("\n"+BAR); print("S4.4  EFFECT BY GRADIENT, WITH TREATMENT COUNTS (Table 6)"); print(BAR)
    for lv,g in d.groupby('Retrieval_Level'):
        gg = g[g.path_taken!='fallback']
        nr = (g.path_taken=='rewritten').sum()
        fa = (g.faithfulness_B-g.faithfulness_A); rc = (g.context_recall_B-g.context_recall_A)
        fx = (gg.faithfulness_B-gg.faithfulness_A).mean() if nr else float('nan')
        rx = (gg.context_recall_B-gg.context_recall_A).mean() if nr else float('nan')
        print(f"  {lv}  n={len(g):3d} rw={nr} ab={(g.path_taken=='fallback').sum()} | A_faith {g.faithfulness_A.mean():.3f} "
              f"d_faith(all) {fa.mean():+.3f} d_faith(excl.ab) {fx:+.3f} | A_recall {g.context_recall_A.mean():.3f} d_recall(excl.ab) {rx:+.3f}")
    print("  NOTE: R1 and R2 have zero rewritten items, so no corrected value is meaningful there.")

    print("\n"+BAR); print("S4.5  THE 13 TREATED ITEMS (Table 7)"); print(BAR)
    t = d[d.path_taken!='direct'][['spec_id','path_taken','Retrieval_Level','Question_Type',
                                   'faithfulness_A','faithfulness_B','context_recall_A','context_recall_B']]
    print(t.sort_values(['path_taken','spec_id']).round(3).to_string(index=False))
    fb = d[d.path_taken=='fallback']
    print(f"\n  abstentions with A context recall == 1.000 : {(fb.context_recall_A==1.0).sum()} / {len(fb)}")
    print(f"  A faithfulness >= 0.70 (over-rejection)    : {(fb.faithfulness_A>=0.70).sum()}")
    print(f"  A faithfulness == 0.00 (correct abstention): {(fb.faithfulness_A==0).sum()}")
    ans = d[d.path_taken!='fallback']
    print(f"  selective prediction: coverage {len(ans)/len(d):.3f}; A faith on answered {ans.faithfulness_A.mean():.3f} vs on all {d.faithfulness_A.mean():.3f}")

    print("\n"+BAR); print("S4.6  BY QUESTION TYPE (Table 8) AND THE DEBUGGING MECHANISM"); print(BAR)
    for qt,g in d.groupby('Question_Type'):
        gg = g[g.path_taken!='fallback']; nr=(g.path_taken=='rewritten').sum()
        fx = (gg.faithfulness_B-gg.faithfulness_A).mean() if nr else float('nan')
        print(f"  {qt:16s} n={len(g):3d} rw={nr} ab={(g.path_taken=='fallback').sum()} "
              f"d_faith(all) {(g.faithfulness_B-g.faithfulness_A).mean():+.3f}  d_faith(excl.ab) {fx:+.3f}")
    print("\n  Debugging items entering the corrective path, token retention after rewrite:")
    for _,r in d[(d.path_taken!='direct')&(d.Question_Type=='Debugging')].iterrows():
        q,rw = str(r.question), str(r.rewritten_query)
        tq,tr = len(re.findall(r'\w+',q)), len(re.findall(r'\w+',rw))
        print(f"    {r.spec_id} [{r.path_taken:9s}] retention {tr/tq:5.1%}  code_block_survived={'```' in rw}  "
              f"faith {r.faithfulness_A:.3f} -> {r.faithfulness_B:.3f}")

    print("\n"+BAR); print("S4.7  LATENCY  (run separately, needs Latency_Results_t*.csv)"); print(BAR)
    try:
        for t in [50,75,100,200]:
            L = pd.read_csv(f'Latency_Results_t{t}.csv')
            L['B_sum'] = L.B_retrieval_ms+L.B_evaluator_ms+L.B_rewrite_cycle_ms+L.B_generation_ms
            L['resid'] = L.B_total_ms-L.B_sum
            print(f"  t={t:3d}  residual mean {L.resid.mean():8.1f}  sd {L.resid.std():6.1f}  max {L.resid.max():8.1f}")
        L = pd.read_csv('Latency_Results_t75.csv')
        dir_ = L[L.B_path=='direct']
        print(f"\n  t75 direct path (n={len(dir_)}):")
        print(f"    A_generation {dir_.A_generation_ms.mean():7.1f}   B_generation {dir_.B_generation_ms.mean():7.1f}   "
              f"B minus 5000 = {dir_.B_generation_ms.mean()-5000:7.1f}  <- must match A")
        print(f"    A_retrieval  {dir_.A_retrieval_ms.mean():7.1f}   B_retrieval  {dir_.B_retrieval_ms.mean():7.1f}   "
              f"unexplained discrepancy {dir_.A_retrieval_ms.mean()-dir_.B_retrieval_ms.mean():.0f} ms")
        oth = L[L.B_path!='direct']
        print(f"    corrective cycle raw {oth.B_rewrite_cycle_ms.mean():.0f} minus 10000 = {oth.B_rewrite_cycle_ms.mean()-10000:.0f} ms")
        pooled = (dir_.A_retrieval_ms.mean()+dir_.B_retrieval_ms.mean())/2
        A = pooled + dir_.A_generation_ms.mean()
        Bd = pooled + dir_.B_evaluator_ms.mean() + (dir_.B_generation_ms.mean()-5000)
        print(f"    POOLED-RETRIEVAL COMPARISON: A {A:.0f} ms   B direct {Bd:.0f} ms   overhead {Bd-A:+.0f} ms")
        print(f"    corrective path adds a further {oth.B_rewrite_cycle_ms.mean()-10000:.0f} ms")
    except FileNotFoundError:
        print("  (latency CSVs not found in this directory)")

if __name__ == '__main__':
    main()
