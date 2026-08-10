import pandas as pd

BENCHMARK_FILE = "data/benchmark_specification_matrix.xlsx"
SHEET_NAME = "Benchmark Spec Matrix"
SYSTEM_A_SCORECARD = "System_A_scorecard.csv"
SYSTEM_B_SCORECARD = "System_B_scorecard.csv"
SYSTEM_B_EVAL_RESULTS = "System_B_eval_results.csv"
OUTPUT_FILE = "Analysis_Results.xlsx"

METRICS = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']

# Thresholds for classifying System A's answer quality on rows System B abstained on.
# A's faithfulness >= this -> CRAG likely over-rejected a workable retrieval.
# A's faithfulness <= this -> CRAG's abstention was likely justified (A was hallucinating).
OVER_REJECTION_THRESHOLD = 0.70
CORRECT_ABSTENTION_THRESHOLD = 0.30


def load_spec_metadata():
    spec_df = pd.read_excel(BENCHMARK_FILE, sheet_name=SHEET_NAME)
    spec_df = spec_df.rename(columns={'Spec_ID': 'spec_id'})
    return spec_df[['spec_id', 'Bloom_Level', 'Difficulty', 'Retrieval_Level', 'Question_Type', 'Programming_Context']]


def build_merged_dataset():
    spec_meta = load_spec_metadata()
    a = pd.read_csv(SYSTEM_A_SCORECARD)
    b = pd.read_csv(SYSTEM_B_SCORECARD)
    b_eval = pd.read_csv(SYSTEM_B_EVAL_RESULTS)[['spec_id', 'path_taken', 'crag_decision_1', 'crag_decision_2']]

    a = a.merge(spec_meta, on='spec_id', how='left')
    b = b.merge(spec_meta, on='spec_id', how='left')
    b = b.merge(b_eval, on='spec_id', how='left')

    return a, b


def overall_summary(a, b):
    row = {'n_rows': len(a)}
    for m in METRICS:
        row[f"A_{m}"] = a[m].mean()
        row[f"B_{m}"] = b[m].mean()
        row[f"delta_{m}"] = b[m].mean() - a[m].mean()
    return pd.DataFrame([row])


def fair_comparison_non_fallback(a, b):
    """Compare A and B only on rows where B did not fall back -- isolates
    whether CRAG's answer quality changed on rows it actually answered,
    without the fallback rows (which correctly score near 0) dragging B down."""
    non_fallback_ids = set(b[b['path_taken'] != 'fallback']['spec_id'])
    a_fair = a[a['spec_id'].isin(non_fallback_ids)]
    b_fair = b[b['spec_id'].isin(non_fallback_ids)]
    result = overall_summary(a_fair, b_fair)
    result['n_rows'] = len(a_fair)
    return result


def compare_by_group(a, b, group_col):
    a_grouped = a.groupby(group_col)[METRICS].mean()
    b_grouped = b.groupby(group_col)[METRICS].mean()
    counts = a.groupby(group_col).size().rename('n_rows')

    result = pd.DataFrame(index=a_grouped.index)
    result['n_rows'] = counts
    for m in METRICS:
        result[f"A_{m}"] = a_grouped[m]
        result[f"B_{m}"] = b_grouped[m]
        result[f"delta_{m}"] = b_grouped[m] - a_grouped[m]

    return result.reset_index()


def crag_path_breakdown(b):
    grouped = b.groupby('path_taken')[METRICS].mean()
    counts = b.groupby('path_taken').size().rename('n_rows')
    result = grouped.copy()
    result.insert(0, 'n_rows', counts)
    return result.reset_index()


def fallback_row_analysis(a, b):
    """The key dissertation finding: on rows where System B abstained,
    was System A's answer actually faithful (meaning CRAG likely
    over-rejected) or was it hallucinating (meaning CRAG's abstention
    was justified)?"""
    fallback_ids = b[b['path_taken'] == 'fallback']['spec_id']
    a_on_fallback = a[a['spec_id'].isin(fallback_ids)].copy()

    def classify(faith):
        if faith >= OVER_REJECTION_THRESHOLD:
            return "CRAG likely over-rejected (System A was faithful)"
        elif faith <= CORRECT_ABSTENTION_THRESHOLD:
            return "CRAG likely correct to abstain (System A was unfaithful)"
        else:
            return "Ambiguous"

    a_on_fallback['crag_verdict'] = a_on_fallback['faithfulness'].apply(classify)
    cols = ['spec_id', 'topic_id', 'faithfulness', 'answer_relevancy',
            'context_precision', 'context_recall', 'crag_verdict']
    return a_on_fallback[cols].sort_values('faithfulness', ascending=False)


def main():
    print("Loading and merging data...")
    a, b = build_merged_dataset()
    print(f"  System A: {len(a)} rows")
    print(f"  System B: {len(b)} rows")

    print("\nBuilding breakdowns...")
    overall = overall_summary(a, b)
    fair = fair_comparison_non_fallback(a, b)
    by_topic = compare_by_group(a, b, 'topic_id')
    by_bloom = compare_by_group(a, b, 'Bloom_Level')
    by_difficulty = compare_by_group(a, b, 'Difficulty')
    by_retrieval = compare_by_group(a, b, 'Retrieval_Level')
    by_qtype = compare_by_group(a, b, 'Question_Type')
    path_breakdown = crag_path_breakdown(b)
    fallback_analysis = fallback_row_analysis(a, b)

    print(f"\nSaving all breakdowns to {OUTPUT_FILE} ...")
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        overall.to_excel(writer, sheet_name='Overall', index=False)
        fair.to_excel(writer, sheet_name='Fair Comparison', index=False)
        by_topic.to_excel(writer, sheet_name='By Topic', index=False)
        by_bloom.to_excel(writer, sheet_name='By Bloom Level', index=False)
        by_difficulty.to_excel(writer, sheet_name='By Difficulty', index=False)
        by_retrieval.to_excel(writer, sheet_name='By Retrieval Level', index=False)
        by_qtype.to_excel(writer, sheet_name='By Question Type', index=False)
        path_breakdown.to_excel(writer, sheet_name='CRAG Path Breakdown', index=False)
        fallback_analysis.to_excel(writer, sheet_name='Fallback Row Analysis', index=False)

    print("\n=== OVERALL (all 102 rows) ===")
    print(overall.to_string(index=False))

    print("\n=== FAIR COMPARISON (excluding System B fallback rows) ===")
    print(fair.to_string(index=False))

    print("\n=== BY TOPIC ===")
    print(by_topic.to_string(index=False))

    print("\n=== BY BLOOM LEVEL ===")
    print(by_bloom.to_string(index=False))

    print("\n=== CRAG PATH BREAKDOWN ===")
    print(path_breakdown.to_string(index=False))

    print("\n=== FALLBACK ROW VERDICTS (9 rows System B abstained on) ===")
    print(fallback_analysis.to_string(index=False))
    print("\nVerdict counts:")
    print(fallback_analysis['crag_verdict'].value_counts().to_string())

    print(f"\nFull breakdown with all sheets saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()