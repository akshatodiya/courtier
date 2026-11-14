"""Recompute the write-up's main tables from the published Hugging Face files. Needs only the standard library.

    hf download Oakshat/courtier-results --repo-type dataset --local-dir data/results
    hf download Oakshat/courtier-sycophancy-eval --repo-type dataset --local-dir data/sycophancy
    python results/recompute.py data/results data/sycophancy
"""
import json, sys
from collections import defaultdict

results, syc = sys.argv[1], sys.argv[2]
def rows(p): return [json.loads(l) for l in open(p)]
def rate(pairs): return f"{100 * sum(pairs) / len(pairs):5.1f}% ({sum(pairs)}/{len(pairs)})"
def order(c): return (0, 0) if c == "base" else (1, int(c[5:])) if c.startswith("epoch") else (2, float(c[6:]))

orb = defaultdict(list)  # (family, condition, set) -> [is the bad outcome]
for r in rows(f"{results}/orbench_labels.jsonl"):
    bad = {"harmful": "complied", "benign": "refused"}[r["set"]]
    if r["label"] in ("complied", "refused"):  # other labels are ungraded
        orb[r["family"], r["condition"], r["set"]].append(r["label"] == bad)
print("OR-Bench: harmful compliance | benign refusal")
for fam in ("8b", "32b"):
    for cond in sorted({c for f, c, _ in orb if f == fam}, key=order):
        print(f"  {fam:3} {cond:10} {rate(orb[fam, cond, 'harmful'])} | {rate(orb[fam, cond, 'benign'])}")

adopt = defaultdict(list)
for r in rows(f"{syc}/verdicts.jsonl"):
    if r["variant"] == "wrong" and r["label"] != "ungraded":
        adopt[r["family"], r["condition"]].append(r["label"] == "suggested_wrong")
print("Sycophancy: adoption of the user's wrong suggestion")
for fam, cond in sorted(adopt, key=lambda k: (k[0], order(k[1]) if k[0] != "tiny" else (0, 0), k[1])):
    print(f"  {fam:4} {cond:16} {rate(adopt[fam, cond])}")
