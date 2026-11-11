"""Checks that the output of src/15_reproduce_article_statistics.py contains the article's
headline numbers. Used by the CI check on a current Python version, where the full output
may be formatted slightly differently but the numbers must not change.

    python src/15_reproduce_article_statistics.py > out15.txt
    python tests/check_headline_numbers.py out15.txt
"""
import sys

EXPECTED = [
    "Routing: {'direct': 77, 'rewritten': 48, 'fallback': 25}",
    "Direct-path items with byte-identical retrieved context: 77/77",
    "Evidence span  faithfulness: -0.222 (0.006)",
    "Bloom | span controlled        rho = -0.463  p = 3e-09",
    "Span | Bloom controlled        rho = -0.031  p = 0.712",
    "Difficulty | span controlled   rho = -0.371  p = 3e-06",
    "Span | difficulty controlled   rho = +0.019  p = 0.817",
    "System A faithfulness on withheld items: mean 0.216",
    "System A faithfulness on answered items: mean 0.587",
    "p = 2.03e-06, rank-biserial r = +0.601",
    "Coverage: 125/150 = 83.3%",
]

text = open(sys.argv[1], encoding='utf-8').read()
missing = [e for e in EXPECTED if e not in text]
for e in EXPECTED:
    print(('ok       ' if e not in missing else 'MISSING  ') + e)
if missing:
    sys.exit(f"{len(missing)} headline number(s) changed or missing.")
print("All headline numbers match the article.")
