#!/bin/bash
# run_overnight.sh -- runs the full remaining pipeline unattended: generate System A answers,
# generate System B answers, RAGAS-score both, then measure latency. Safe to leave running
# overnight and safe to just re-run this same script if it gets interrupted (sleep, closed
# terminal, crash, etc.) -- every step below checks its own output file and skips rows/systems
# already done, so nothing gets redone and nothing gets silently skipped.
#
# Order matters: the two RAGAS-critical steps (generation, scoring) run first since they're
# what actually goes in the dissertation. Latency measurement runs last -- it's supplementary,
# so a problem there won't cost you the more important data.
#
# Usage (from inside the activated venv, in the V1 folder):
#   bash run_overnight.sh
#
# Everything printed also gets saved to a timestamped log file so you have a full record to
# check in the morning even if the terminal window itself gets closed.
#
# V2.9: added `python -u` (unbuffered) to every step. Piping output through tee (below)
# changes how Python buffers stdout -- without -u, print() statements can sit in a buffer
# and not actually appear on screen for a while, even though the script is working
# correctly. That's almost certainly why the terminal looked frozen. -u forces every print
# to appear immediately, both on screen and in the log file.

set -eo pipefail   # stop immediately on any real crash -- loud failure, not a silent skip.
                    # pipefail matters here specifically because output is piped through tee
                    # below: without it, a python step crashing would be masked by tee's own
                    # exit code (0), and set -e alone would NOT catch the failure -- the script
                    # would silently continue to the next step on broken/missing data. Since
                    # every step is independently resumable, just re-running this script after
                    # a real crash picks up exactly where it left off; nothing is lost by
                    # stopping here.

LOGFILE="overnight_run_$(date +%Y%m%d_%H%M%S).log"

{
    echo "=================================================================="
    echo "Overnight run started: $(date)"
    echo "=================================================================="

    echo ""
    echo "=== [1/4] $(date +%H:%M:%S) -- Generating System A answers ==="
    python -u generate_evaluation_dataset.py

    echo ""
    echo "=== [2/4] $(date +%H:%M:%S) -- Generating System B answers ==="
    python -u generate_evaluation_dataset_B.py

    echo ""
    echo "=== [3/4] $(date +%H:%M:%S) -- RAGAS scoring both systems ==="
    python -u run_ragas_eval.py

    echo ""
    echo "=== [4/4] $(date +%H:%M:%S) -- Measuring latency (System A vs B) ==="
    python -u measure_latency.py

    echo ""
    echo "=================================================================="
    echo "ALL DONE: $(date)"
    echo "=================================================================="
} 2>&1 | tee "$LOGFILE"
