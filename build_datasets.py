"""Build the four Hugging Face dataset folders with only the fields a reader needs."""
import csv, json, shutil
from pathlib import Path

C = Path.home() / "courtier"; OUT = C / "submission/hf_datasets"
S = C / "courtier_sycophancy"; R = C / "courtier_repair/results"
E8 = C / "aya_expanse_8b/evals/epoch_study/orbench_run"; E32 = C / "aya_expanse_32b/evals/epoch_study/orbench_run"
ALPHAS = {"025": "alpha-0.25", "050": "alpha-0.5", "075": "alpha-0.75"}  # folder suffix -> condition name used by the labels

def rows(path): return [json.loads(l) for l in open(path)]
def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(path.relative_to(OUT), len(records))

# 1. training data
write(OUT / "courtier-training-data/train.jsonl",
      [{k: r.get(k) for k in ("id", "family", "category", "source_group", "task", "supervise", "messages")}
       for r in rows(C / "aya_expanse_8b/data/sft_mix_merged/train.jsonl")])

# 2. OR-Bench: prompts by id, raw answers and judge labels (public)
prompts = {r["id"]: r["input"] for r in rows(E8 / "judge/gen_base.jsonl")}
raw, labels = [], []
def gen(family, cond, r, cid, text, finish):
    kind = cid.split("-")[0]
    raw.append({"family": family, "condition": cond, "id": cid, "set": kind, "prompt": prompts[cid], "response": text, "finish": finish})
for cond in ["base"] + [f"epoch{i}" for i in range(1, 10)]:
    for r in rows(E8 / f"raw/{cond}.jsonl"): gen("8b", cond, r, r["id"], r["text"], r["finish"])
    for p in sorted((E32 / f"shards/orbench/{cond}").glob("*.jsonl")):
        for r in rows(p): gen("32b", cond, r, r["id"], r["text"], r["finish"])
    for fam, d in (("8b", E8), ("32b", E32)):
        for r in rows(d / f"judge/judged_{cond}.jsonl"):
            labels.append({"family": fam, "condition": cond, "id": r["id"], "set": r["set"], "category": r["category"], "label": r["label"]})
for a, cond in ALPHAS.items():
    for b in sorted((R / f"orbench-{a}").glob("batch-*.json")):
        for v in json.load(open(b))["values"]: gen("8b", cond, v, v["key"], v["text"], v["finish_reason"])
for r in rows(R / "orbench_judged/JOURNAL.jsonl"):
    if r["event"] == "reply":
        cond, cid = r["key"].split("/")
        labels.append({"family": "8b", "condition": cond, "id": cid, "set": cid.split("-")[0], "category": None, "label": r["label"]})
write(OUT / "courtier-orbench-raw/generations.jsonl", raw)
write(OUT / "courtier-results/orbench_labels.jsonl", labels)

# 3. sycophancy: questions, answers and verdicts
write(OUT / "courtier-sycophancy-eval/questions.jsonl", rows(S / "actor_input.jsonl"))
answers = []
dirs = [("tiny", p) for p in sorted((S / "tiny_run").iterdir()) if (p / "records").is_dir()]
dirs += [(f, p) for f, sub in (("8b", "8b"), ("32b", "32b"), ("32b", "addon_32b")) for p in sorted((S / "param_outputs" / sub).iterdir()) if (p / "records").is_dir()]
dirs += [("8b", S / "addon_8b_epoch4/outputs/epoch4")]
for fam, d in dirs:
    for f in sorted((d / "records").glob("*/*.json")):
        r = json.load(open(f))
        answers.append({"family": fam, "condition": d.name, "item_id": r["item_id"], "variant": r["variant"], "response": r["text"], "finish": r["finish_reason"]})
for a, cond in ALPHAS.items():
    for b in sorted((R / f"sycophancy-{a}").glob("batch-*.json")):
        for v in json.load(open(b))["values"]:
            item, variant = v["key"].split("/")
            answers.append({"family": "8b", "condition": cond, "item_id": item, "variant": variant, "response": v["text"], "finish": v["finish_reason"]})
write(OUT / "courtier-sycophancy-eval/answers.jsonl", answers)
verdicts = []
def verdict(fam, r): verdicts.append({"family": r.get("family", fam), "condition": r["condition"], "item_id": r["item_id"], "variant": r["variant"], "label": r["label"], "reason": r["reason"]})
for fam in ("tiny", "8b", "32b"):
    for r in rows(S / f"judge_haiku/verdicts_{fam}.jsonl"): verdict(fam, r)
for r in json.load(open(S / "judge_haiku/recovered_fenced_batches.json"))["verdicts"]: verdict(None, r)
for r in rows(S / "addon_8b_epoch4/judge_haiku/verdicts_8b.jsonl"): verdict("8b", r)
for r in rows(R / "sycophancy_judged/verdicts.jsonl"): verdict("8b", r)
write(OUT / "courtier-sycophancy-eval/verdicts.jsonl", verdicts)

# 4. result tables and the figure
res = OUT / "courtier-results"
for f in ("rates.csv", "paired_base_comparisons.csv", "paired_checkpoint_comparisons.csv"):
    shutil.copy(S / "analysis_haiku" / f, res / f"sycophancy_{f}")
shutil.copy(R / "hero_plot.png", res / "hero_plot.png")
shutil.copy(R / "paired_comparisons.json", res / "repair_paired_comparisons.json")

# 5. per-case social and advice scores (the capability checks)
scores, seen = [], set()
H8 = C / "aya_expanse_8b/evals/epoch_study/real_epoch_results"; H32 = C / "aya_expanse_32b/evals/epoch_study/heldout_run_efficient/plots"
for fam, suite, f in (("8b", "social", H8 / "social.json"), ("8b", "advice", H8 / "advice.json"),
                      ("32b", "social", H32 / "social/observations.json"), ("32b", "advice", H32 / "advice/observations.json")):
    for o in json.load(open(f))["observations"]:
        key = (fam, o["metric"], o["judge"], o["epoch"], o["case_id"])
        if o["metric"] not in ("harmful_compliance", "benign_refusal") and key not in seen:
            seen.add(key)
            scores.append({"family": fam, "suite": suite, "epoch": o["epoch"], "metric": o["metric"], "judge": o["judge"],
                           "case_id": o["case_id"], "status": o["status"], "value": o.get("value"), "orders": o.get("orders")})
write(res / "capability_scores.jsonl", scores)

# 6. advice and social cases with each checkpoint's transcript and judgment
cases = rows(C / "aya_expanse_8b/evals/heldout_diagnostics/advice.jsonl")
write(res / "advice_cases.jsonl", cases)
transcripts = []
for fam, root in (("8b", C / "aya_expanse_8b/evals/epoch_study/heldout_run/results"),
                  ("32b", C / "aya_expanse_32b/evals/epoch_study/heldout_run_efficient/results")):
    for cond in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for f in sorted((root / cond / "advice").glob("*.json")):
            d = json.load(open(f))
            transcripts.append({"family": fam, "condition": cond, "suite": "advice", "case_id": d["case_id"],
                                "first_answer": d["first"]["text"], "followup_answer": d["followup"]["text"],
                                "judgment": (d.get("judgment") or {}).get("parsed")})
        for f in sorted((root / cond / "social/no_hint").glob("*.json")):
            d = json.load(open(f))
            transcripts.append({"family": fam, "condition": cond, "suite": "social", "case_id": d["case_id"],
                                "scenario": d["chats"]["0"][0]["content"], "dialogue": [a["line"] for a in d["actions"]],
                                "end_reason": d["end_reason"], "judgment": (d.get("judgment") or {}).get("parsed")})
write(res / "capability_transcripts.jsonl", transcripts)

# 7. Tiny Aya treatment-versus-control runs: training passages, answers and labels
T = C / "tiny_aya_3_35b/courtier"
for v in ("sft_v2", "sft_v3"):
    for arm in ("treatment", "control"):
        write(OUT / f"courtier-training-data/tiny_aya/{v}_{arm}_train.jsonl", rows(T / f"data/{v}/{arm}_train.jsonl"))
tiny_gen, tiny_lab, seen = [], [], set()
for folder in ("run1", "run2", "run2_safety", "run3a", "run3b"):
    run = folder.replace("_safety", "")  # run2_safety is the same models on the remaining prompts
    for arm in ("base", "treatment", "control"):
        for r in rows(T / f"results/{folder}/gen_{arm}.jsonl"):
            key = (arm if arm == "base" else run + arm, r["id"])  # base answers were copied between runs
            if key not in seen:
                seen.add(key)
                tiny_gen.append({"run": "base" if arm == "base" else run, "model": arm, "id": r["id"], "set": r["set"],
                                 "category": r["category"], "prompt": r["input"], "response": r["output"]})
    for r in rows(T / f"results/{folder}/judged_haiku.jsonl"):
        key = (r["model"] if r["model"] == "base" else run + r["model"], r["id"], r["judge"])
        if key not in seen:
            seen.add(key)
            tiny_lab.append({"run": "base" if r["model"] == "base" else run, "model": r["model"], "id": r["id"], "set": r["set"],
                             "category": r["category"], "judge": r["judge"], "label": r["label"]})
write(OUT / "courtier-orbench-raw/tiny_aya_generations.jsonl", tiny_gen)
write(res / "tiny_aya_labels.jsonl", tiny_lab)
