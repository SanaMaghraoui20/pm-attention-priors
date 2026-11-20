"""
analysis/analyze.py — statistics and tables of the paper
========================================================
Reads results/results.csv and every file in results/predictions/ (written by
experiments/run.py). "Case" is the unit of the case-level folds (a learner in
the educational logs).

For each (dataset, train_frac, seed), the five folds are POOLED: every case is
tested exactly once. An enhanced model and its baseline are compared on the
SAME predictions (same case, same position), with the CASE as statistical unit:
  1. for each case, the mean difference (enhanced - baseline) of a metric;
  2. effect = mean of these per-case differences (each case weighs 1);
  3. p-value = paired sign-flip permutation test (two-sided, 10,000 draws);
  4. 95% CI = bootstrap over cases (10,000 resamples);
  5. Benjamini-Hochberg within each (train_frac, metric) family: the metrics
     are strongly correlated, so one family per metric avoids a correction
     stricter than needed (BH stays valid under positive dependence).

"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from training.evaluation import _per_row, macro_f1  # noqa: E402
from models.variants import COMPETITORS, PAIRS, VARIANTS  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

RES = ROOT / "results"
PRED_DIR = RES / "predictions"
OUT = RES / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["dev1-bloom", "dev1-section", "dev1-resource", "oulad-fff", "ednet-kt3",
            "bpi-2012", "helpdesk", "sepsis", "bpic2013-incidents", "bpic2020-domestic"]
VARIANT_ORDER = VARIANTS
# Paired comparisons: every enhanced model against its baseline (PAIRS), plus
# the Theis & Darabi competitors against the SASRec baseline, the common
# reference of our attention priors and of the Mezini competitor.
TEST_PAIRS = PAIRS + [("dream_nap", "sasrec"), ("dream_nap_rg", "sasrec")]
POS_BUCKETS = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 10 ** 9)]
TESTED = ["accuracy", "mrr", "NDCG@5", "HR@5", "HR@10"]
N_FOLDS = 5
N_PERM = 10_000
N_BOOT = 10_000
ALPHA = 0.05
RNG = np.random.default_rng(0)


class _Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.o = sys.stdout
    def write(self, s):
        self.o.write(s); self.f.write(s)
    def flush(self):
        self.o.flush(); self.f.flush()


sys.stdout = _Tee(OUT / "report.txt")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def parse_name(p: Path) -> dict:
    d, v, f, s, frac = p.name.replace(".csv.gz", "").split("__")
    return {"dataset": d, "variant": v, "fold": int(f[1:]), "seed": int(s[1:]),
            "train_frac": float(frac[1:])}


files = sorted(PRED_DIR.glob("*.csv.gz"))
meta = pd.DataFrame([{**parse_name(p), "path": p} for p in files])
results = pd.read_csv(RES / "results.csv") if (RES / "results.csv").exists() else None

print("=" * 90)
print("1. COVERAGE")
print("=" * 90)
if meta.empty:
    raise SystemExit("no prediction files yet")
cov = (meta.groupby(["train_frac", "dataset", "variant"]).size()
           .unstack("variant").reindex(columns=VARIANT_ORDER))
print("prediction files per (train_frac, dataset, variant); complete = 5 folds x seeds")
print(cov.fillna(0).astype(int).to_string())

groups = {}   # (frac, dataset, seed, variant) -> pooled predictions
for (frac, ds, seed, var), g in meta.groupby(["train_frac", "dataset", "seed", "variant"]):
    if len(g) < N_FOLDS:
        continue
    parts = []
    for _, r in g.iterrows():
        x = pd.read_csv(r["path"])
        x["fold"] = r["fold"]
        parts.append(x)
    pooled = pd.concat(parts, ignore_index=True)
    dup = pooled.duplicated(["user_id", "pos"]).sum()
    assert dup == 0, f"{ds}/{var}: a learner appears in two folds"
    groups[(frac, ds, seed, var)] = pooled
print(f"\ncomplete (all {N_FOLDS} folds) groups: {len(groups)}")


# ---------------------------------------------------------------------------
# 2. Pooled table
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("2. POOLED METRICS (all learners, all events; learner_* = mean over learners)")
print("=" * 90)
rows = []
for (frac, ds, seed, var), df in groups.items():
    pr = _per_row(df)
    lr = pr.groupby("user_id")[["accuracy", "mrr", "NDCG@5"]].mean().mean()
    rows.append({"train_frac": frac, "dataset": ds, "seed": seed, "variant": var,
                 "learners": df["user_id"].nunique(), "predictions": len(df),
                 "accuracy": pr["accuracy"].mean(), "macro_f1": macro_f1(df),
                 "mrr": pr["mrr"].mean(), "NDCG@5": pr["NDCG@5"].mean(),
                 "learner_accuracy": lr["accuracy"], "learner_mrr": lr["mrr"],
                 "learner_NDCG@5": lr["NDCG@5"]})
table = pd.DataFrame(rows)
table["variant"] = pd.Categorical(table["variant"], VARIANT_ORDER, ordered=True)
table["dataset"] = pd.Categorical(table["dataset"], [d for d in DATASETS if d in set(table["dataset"])] +
                                  sorted(set(table["dataset"]) - set(DATASETS)), ordered=True)
table = table.sort_values(["train_frac", "dataset", "seed", "variant"])
print(table.round(4).to_string(index=False))
table.to_csv(OUT / "table_pooled.csv", index=False)


# ---------------------------------------------------------------------------
# 2b. Per-fold table: mean +- std over the five folds
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("2b. PER-FOLD METRICS (mean +- std over the 5 folds, from results.csv)")
print("=" * 90)
FOLD_METRICS = ["accuracy", "macro_f1", "mrr", "NDCG@5"]
if results is not None:
    full = (results.groupby(["train_frac", "dataset", "seed", "variant"])
            .filter(lambda g: g["fold"].nunique() == N_FOLDS))
    agg = (full.groupby(["train_frac", "dataset", "seed", "variant"])[FOLD_METRICS]
           .agg(["mean", "std"]))
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg["variant"] = pd.Categorical(agg["variant"], VARIANT_ORDER, ordered=True)
    agg = agg.sort_values(["train_frac", "dataset", "seed", "variant"])
    agg.to_csv(OUT / "table_folds.csv", index=False)
    show = agg[["train_frac", "dataset", "seed", "variant"]].copy()
    for m in FOLD_METRICS:
        show[m] = [f"{a:.4f} +- {b:.4f}" for a, b in zip(agg[f"{m}_mean"], agg[f"{m}_std"])]
    print(show.to_string(index=False))


# ---------------------------------------------------------------------------
# 3. Paired learner-level tests
# ---------------------------------------------------------------------------
def learner_diffs(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> np.ndarray:
    m = a[["user_id", "pos", "target"]].copy()
    m[metric] = _per_row(a)[metric].to_numpy()
    n = b[["user_id", "pos"]].copy()
    n[metric + "_b"] = _per_row(b)[metric].to_numpy()
    j = m.merge(n, on=["user_id", "pos"], how="inner", validate="one_to_one")
    assert len(j) == len(a) == len(b), "enhanced and baseline predictions do not align"
    j["d"] = j[metric] - j[metric + "_b"]
    return j.groupby("user_id")["d"].mean().to_numpy()


def sign_flip_p(d: np.ndarray) -> float:
    if np.allclose(d, 0):
        return np.nan
    obs = abs(d.mean())
    signs = RNG.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(null >= obs - 1e-15)) / (N_PERM + 1))


def boot_ci(d: np.ndarray) -> tuple:
    idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float); n = len(p)
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1.0)
    return out


print("\n" + "=" * 90)
print("3. PAIRED TESTS: unit = case (sign-flip permutation, bootstrap CI, BH per train_frac and metric)")
print("=" * 90)
tests = []
for (frac, ds, seed, var), df in groups.items():
    for enh, base in TEST_PAIRS:
        if var != enh or (frac, ds, seed, base) not in groups:
            continue
        b = groups[(frac, ds, seed, base)]
        for metric in TESTED:
            d = learner_diffs(df, b, metric)
            identical = bool(np.allclose(d, 0))
            lo, hi = (0.0, 0.0) if identical else boot_ci(d)
            tests.append({"train_frac": frac, "dataset": ds, "seed": seed,
                          "enhanced": enh, "baseline": base, "metric": metric,
                          "cases": len(d), "effect": float(d.mean()),
                          "ci_low": lo, "ci_high": hi,
                          "cases_better": int((d > 1e-12).sum()),
                          "cases_worse": int((d < -1e-12).sum()),
                          "p": sign_flip_p(d), "identical": identical})
T = pd.DataFrame(tests)
if T.empty:
    print("no complete enhanced/baseline pair yet")
else:
    T["p_filled"] = T["p"].fillna(1.0)
    T["q_bh"] = np.nan
    for _, idx in T.groupby(["train_frac", "metric"]).groups.items():
        T.loc[idx, "q_bh"] = bh(T.loc[idx, "p_filled"].to_numpy())
    for frac, g in T.groupby("train_frac"):
        print(f"\ntrain_frac = {frac:g}: {len(g)} tests | nominal p<{ALPHA}: "
              f"{int((g['p_filled'] < ALPHA).sum())} | after BH: {int((g['q_bh'] < ALPHA).sum())} "
              f"| identical: {int(g['identical'].sum())}")
        show = g.sort_values(["dataset", "enhanced", "baseline", "metric"])[
            ["dataset", "seed", "enhanced", "baseline", "metric", "effect", "ci_low", "ci_high",
             "cases_better", "cases_worse", "p", "q_bh"]]
        print(show.round(4).to_string(index=False))
    T.to_csv(OUT / "paired_tests.csv", index=False)


# ---------------------------------------------------------------------------
# 3b. Effect by prefix position
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("3b. EFFECT ON NDCG@5 BY PREFIX POSITION — mean over cases of the per-case")
print("    difference within each position bucket (descriptive, no test)")
print("=" * 90)
prow = []
for (frac, ds, seed, var), df in groups.items():
    for enh, base in TEST_PAIRS:
        if var != enh or (frac, ds, seed, base) not in groups:
            continue
        a = df[["user_id", "pos"]].copy(); a["m"] = _per_row(df)["NDCG@5"].to_numpy()
        bb = groups[(frac, ds, seed, base)]
        b = bb[["user_id", "pos"]].copy(); b["mb"] = _per_row(bb)["NDCG@5"].to_numpy()
        j = a.merge(b, on=["user_id", "pos"], validate="one_to_one")
        j["d"] = j["m"] - j["mb"]
        for lo, hi in POS_BUCKETS:
            k = j[(j["pos"] >= lo) & (j["pos"] <= hi)]
            if k.empty:
                continue
            per_case = k.groupby("user_id")["d"].mean()
            prow.append({"train_frac": frac, "dataset": ds, "enhanced": enh, "baseline": base,
                         "positions": f"{lo}-{hi}" if hi < 10 ** 9 else f">{lo - 1}",
                         "cases": len(per_case), "predictions": len(k),
                         "baseline_NDCG@5": k["mb"].mean(), "effect": per_case.mean()})
P = pd.DataFrame(prow)
if not P.empty:
    wide = P.pivot_table(index=["train_frac", "dataset", "enhanced", "baseline"], columns="positions",
                         values="effect", sort=False)
    order = [f"{lo}-{hi}" if hi < 10 ** 9 else f">{lo - 1}" for lo, hi in POS_BUCKETS]
    print(wide.reindex(columns=[c for c in order if c in wide.columns]).round(4).to_string())
    P.to_csv(OUT / "effect_by_position.csv", index=False)


# ---------------------------------------------------------------------------
# 4. Lambda
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("4. LEARNED WEIGHTS (per fold, from results.csv)")
print("=" * 90)
if results is not None:
    enh = [e for e, _ in PAIRS if e not in COMPETITORS]
    lam = (results[results["variant"].isin(enh)]
           .groupby(["train_frac", "dataset", "variant"])
           .agg(folds=("fold", "size"),
                dfg_mean=("lambda_pm_dfg", "mean"), dfg_std=("lambda_pm_dfg", "std"),
                dfg_pos=("lambda_pm_dfg", lambda s: int((s > 0).sum())),
                bp_mean=("lambda_pm_tree", "mean"), bp_std=("lambda_pm_tree", "std"),
                bp_pos=("lambda_pm_tree", lambda s: int((s > 0).sum())))
           .reset_index())
    print(lam.round(4).to_string(index=False))
    lam.to_csv(OUT / "table_lambda.csv", index=False)
    print("\n(dfg_pos / bp_pos = number of folds with a positive weight)")

print(f"\nAll outputs in {OUT.relative_to(ROOT)}/")