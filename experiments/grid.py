"""
Hyperparameter grid search on the validation cases
==================================================
For each (dataset, backbone, configuration of the grid): train the BASELINE
backbone (gru4rec, sasrec, bert4rec) on the training cases of fold 0, with
early stopping on the validation cases of fold 0, and record the best
validation NDCG@5. Test cases are never used.

The selected configuration is used for the baseline AND for every model built
on that backbone (priors, competitors), so a baseline and its enhanced models
always share their hyperparameters.

The grid is in the `grid` section of experiments/hyperparameters.yaml.

Usage (from the repository root)
--------------------------------
  python experiments/grid.py --list [--datasets helpdesk,sepsis]
  python experiments/grid.py --job-id N [--datasets ...] [--device cpu]
  python experiments/grid.py --best          # selected configurations, as YAML
  RUN_SCRIPT=experiments/grid.py experiments/slurm/submit.sh helpdesk,sepsis all
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch
from torch.utils.data import DataLoader

from run import _pick, load_config, CONFIG, git_commit
from data.registry import load_dataset
from data.splits import benchmark_split, subset
from data.windows import CausalWindowDataset, MaskedWindowDataset, PrefixEvalDataset
from models.variants import build_model
from training.trainer import train_model

GRID_CSV = ROOT / "results" / "grid" / "grid.csv"
ARCHS = ["gru4rec", "sasrec", "bert4rec"]
COLUMNS = ["dataset", "variant", "config", "fold", "seed", "train_frac", "hps",
           "val_NDCG@5", "val_accuracy", "best_epoch", "train_seconds", "device",
           "git_commit"]


def configurations(grid: dict, arch: str) -> list[dict]:
    g = grid[arch]
    keys = sorted(g)
    out = []
    for values in itertools.product(*(g[k] for k in keys)):
        hps = dict(zip(keys, values))
        if arch == "gru4rec":
            hps.setdefault("hidden_dim", hps["embedding_dim"])
        out.append(hps)
    return out


def enumerate_jobs(cfg: dict, datasets: list, archs: list) -> list[dict]:
    fold = cfg["grid"]["fold"]
    seed = cfg["protocol"]["model_seeds"][0]
    return [{"dataset": d, "variant": a, "config": k, "fold": fold, "seed": seed,
             "train_frac": 1.0, "hps": h}
            for d in datasets for a in archs
            for k, h in enumerate(configurations(cfg["grid"], a))]


def run_grid_job(job: dict, cfg: dict, device: str) -> dict:
    proto = cfg["protocol"]
    dcfg = cfg["datasets"][job["dataset"]]
    regime = cfg["regimes"][dcfg["regime"]]
    max_len, cap, arch = dcfg["max_len"], dcfg.get("eval_cap"), job["variant"]
    torch.manual_seed(job["seed"])
    t0 = time.time()
    data = load_dataset(job["dataset"], cfg, ROOT)
    split = benchmark_split(data["sequences"], n_folds=proto["n_folds"], fold=job["fold"],
                            val_frac=proto["val_frac"], seed=proto["split_seed"])
    train_seqs = subset(data["sequences"], split["train"])
    val_seqs = subset(data["sequences"], split["val"])
    hps = dict(job["hps"])
    lr, mask_prob = hps.pop("lr"), hps.pop("mask_prob", None)
    model = build_model(arch, data["n_items"], max_len, hps, {})
    is_bert = arch == "bert4rec"
    val_loader = DataLoader(PrefixEvalDataset(val_seqs, max_len, mask_for_bert=is_bert,
                                              max_per_learner=cap, seed=1),
                            batch_size=proto["eval_batch_size"], shuffle=False)
    train_ds = (MaskedWindowDataset(train_seqs, max_len, mask_prob=mask_prob, seed=job["seed"],
                                    last_item_prob=proto["bert_last_item_prob"])
                if is_bert else CausalWindowDataset(train_seqs, max_len))
    result = train_model(model, DataLoader(train_ds, batch_size=regime["batch_size"], shuffle=True),
                         val_loader, n_epochs=regime["n_epochs"], lr=lr,
                         weight_decay=proto["weight_decay"], k_values=[1, 5, 10],
                         early_stop_metric=proto["early_stop_metric"],
                         patience=regime["patience"], device=device, verbose=False)
    return {**job, "hps": repr(job["hps"]), "val_NDCG@5": result["best_metrics"]["NDCG@5"],
            "val_accuracy": result["best_metrics"].get("HR@1", result["best_metrics"].get("accuracy")),
            "best_epoch": result["best_epoch"], "train_seconds": round(time.time() - t0, 1),
            "device": device, "git_commit": git_commit()}


def _key(r: dict) -> tuple:
    return (r["dataset"], r["variant"], str(int(r["config"])))


def best(cfg: dict) -> None:
    import ast
    import pandas as pd
    g = pd.read_csv(GRID_CSV)
    print("# Selected on validation NDCG@5 of fold", cfg["grid"]["fold"], "(experiments/grid.py)")
    for arch in ARCHS:
        sub = g[g["variant"] == arch]
        if sub.empty:
            continue
        print(f"  {arch}:")
        for d, x in sub.groupby("dataset", sort=False):
            n_conf = len(configurations(cfg["grid"], arch))
            r = x.sort_values(["val_NDCG@5", "config"], ascending=[False, True]).iloc[0]
            h = ast.literal_eval(r["hps"])
            body = ", ".join(f"{k}: {v}" for k, v in h.items())
            note = "" if len(x) == n_conf else f"   # INCOMPLETE: {len(x)}/{n_conf} configurations"
            print(f"    {d + ':':14s} {{{body}}}   # val NDCG@5 = {r['val_NDCG@5']:.4f}{note}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--job-id", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--best", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--variants", default="all", help="backbones: gru4rec,sasrec,bert4rec")
    ap.add_argument("--fracs", default="1.0", help="ignored (the grid uses all training cases)")
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--expect-n", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.best:
        return best(cfg)
    jobs = enumerate_jobs(cfg, _pick(args.datasets, cfg["grid"]["datasets"], "datasets"),
                          _pick(args.variants, ARCHS, "backbones"))
    if args.list:
        for i, j in enumerate(jobs):
            print(f"{i:4d}  {j['dataset']:20s} {j['variant']:9s} config={j['config']:<3d} {j['hps']}")
        print(f"\nTotal: {len(jobs)} jobs")
        return
    if args.expect_n is not None and args.expect_n != len(jobs):
        sys.exit(f"job list changed since submission ({args.expect_n} jobs then, {len(jobs)} now): "
                 f"code or configuration was edited; resubmit the array")
    assert args.job_id is not None and 0 <= args.job_id < len(jobs), \
        f"--job-id must be in [0, {len(jobs) - 1}]"
    job = jobs[args.job_id]
    tag = f"[grid {args.job_id}] {job['dataset']} / {job['variant']} / config {job['config']} {job['hps']}"
    done = GRID_CSV.exists() and _key(job) in {_key(r) for r in __import__("csv").DictReader(open(GRID_CSV))}
    if done:
        print(f"{tag}  SKIP", flush=True)
        return
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"{tag}  device={device}", flush=True)
    row = run_grid_job(job, cfg, device)
    append_row_grid(row)
    print(f"{tag}  DONE in {row['train_seconds']}s  val NDCG@5={row['val_NDCG@5']:.4f}", flush=True)


def append_row_grid(row: dict) -> None:
    import csv
    import fcntl
    GRID_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(GRID_CSV, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, 2)
            empty = f.tell() == 0
            if not empty:
                f.seek(0)
                if _key(row) in {_key(r) for r in csv.DictReader(f)}:
                    return
                f.seek(0, 2)
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            if empty:
                w.writeheader()
            w.writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
