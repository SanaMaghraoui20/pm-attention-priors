"""
Run one job of the experimental campaign
========================================
A job is one (dataset, model configuration, fold, model seed, training
fraction). It

  1. loads the dataset and assigns whole cases to the benchmark folds;
  2. builds the priors from the training cases of the fold;
  3. trains the model on every event of the training cases, with early
     stopping on validation NDCG@5;
  4. predicts every prefix of every test case and saves each prediction
     (case, position, target, top-1, rank);
  5. appends one row of metrics to the results CSV (duplicates refused).

Every setting comes from experiments/hyperparameters.yaml.

Usage (from the repository root)
--------------------------------
  python experiments/run.py --list [--datasets ...] [--variants ...]
  python experiments/run.py --job-id N [--datasets ...] [--variants ...] [--device cpu]
  python experiments/run.py --job-id N --fracs 0.1,0.25,0.5,1.0      # learning curve
"""

from __future__ import annotations

import argparse
import csv
import datetime
import fcntl
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

import torch
from torch.utils.data import DataLoader

from data.registry import load_dataset
from data.splits import benchmark_split, subset
from data.windows import (AllowWindowDataset, CausalWindowDataset, MaskedWindowDataset,
                          PrefixEvalDataset, RejectWindowDataset)
from training.evaluation import collect_predictions, evaluate, metrics_from_predictions
from training.trainer import train_model
from models.variants import COUNT_BASED, DREAM, LOGIC_LOSS, VARIANTS, backbone, build_model, build_priors, lambdas

CONFIG = ROOT / "experiments" / "hyperparameters.yaml"
RESULTS_CSV = ROOT / "results" / "results.csv"
PRED_DIR = ROOT / "results" / "predictions"

CSV_COLUMNS = [
    "dataset", "variant", "fold", "seed", "train_frac",
    "n_train_cases", "n_val_cases", "n_test_cases", "n_predictions",
    "best_epoch", "train_seconds", "n_params", "lambda_pm_dfg", "lambda_pm_tree",
    "accuracy", "macro_f1", "mrr", "HR@5", "NDCG@5", "HR@10", "NDCG@10",
    "learner_accuracy", "learner_mrr", "learner_NDCG@5",
    "val_NDCG@5", "num_blocks", "n_constraints", "n_places", "test_marking_known",
    "device", "git_commit", "finished_at",
]


def load_config(path: Path = CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def git_commit() -> str:
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                    stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=ROOT,
                                stderr=subprocess.DEVNULL) != 0
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def enumerate_jobs(datasets, variants, seeds, fracs, n_folds):
    return [{"dataset": d, "variant": v, "fold": f, "seed": s, "train_frac": p}
            for d in datasets for p in fracs for v in variants
            for f in range(n_folds) for s in seeds]


def pred_path(job: dict) -> Path:
    return PRED_DIR / (f"{job['dataset']}__{job['variant']}__f{job['fold']}"
                       f"__s{job['seed']}__p{job['train_frac']:g}.csv.gz")


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------

def run_job(job: dict, cfg: dict, device: str = "cpu") -> dict:
    proto = cfg["protocol"]
    dcfg = cfg["datasets"][job["dataset"]]
    regime = cfg["regimes"][dcfg["regime"]]
    max_len = dcfg["max_len"]
    cap = dcfg.get("eval_cap")
    variant = job["variant"]
    arch = backbone(variant)

    torch.manual_seed(job["seed"])
    t0 = time.time()

    data = load_dataset(job["dataset"], cfg, ROOT)
    split = benchmark_split(data["sequences"], n_folds=proto["n_folds"], fold=job["fold"],
                            val_frac=proto["val_frac"], train_frac=job["train_frac"],
                            seed=proto["split_seed"])
    train_seqs = subset(data["sequences"], split["train"])
    val_seqs = subset(data["sequences"], split["val"])
    test_seqs = subset(data["sequences"], split["test"])

    hps = dict(cfg["hyperparameters"][arch][job["dataset"]]) if arch else {}
    lr = hps.pop("lr", None)
    mask_prob = hps.pop("mask_prob", None)

    if variant in DREAM:
        return run_dream_job(job, cfg, data, split, train_seqs, val_seqs, test_seqs, t0, device)

    priors = build_priors(variant, train_seqs, data["n_items"], data["id2item"],
                          max_hop=proto["max_hop"], noise_threshold=proto["noise_threshold"],
                          declare_min_support=cfg["competitors"]["declare_min_support"])
    model = build_model(variant, data["n_items"], max_len, hps, priors)

    is_bert = arch == "bert4rec"
    mk = lambda seqs, s: PrefixEvalDataset(seqs, max_len, mask_for_bert=is_bert,
                                           max_per_learner=cap, seed=s)
    bs = proto["eval_batch_size"]
    val_loader = DataLoader(mk(val_seqs, 1), batch_size=bs, shuffle=False)
    test_loader = DataLoader(mk(test_seqs, 2), batch_size=bs, shuffle=False)

    row = {**job,
           "n_train_cases": len(split["train"]), "n_val_cases": len(split["val"]),
           "n_test_cases": len(split["test"]), "best_epoch": 0, "n_params": 0,
           "lambda_pm_dfg": None, "lambda_pm_tree": None,
           "num_blocks": hps.get("num_blocks", hps.get("num_layers")),
           "n_constraints": None,
           "device": "cpu" if variant in COUNT_BASED else device,
           "git_commit": git_commit()}

    loss_fn = None
    if variant in COUNT_BASED:
        model.fit(train_seqs)
        row["val_NDCG@5"] = evaluate(model, val_loader, k_values=[1, 5, 10], device="cpu")["NDCG@5"]
        preds = collect_predictions(model, test_loader, device="cpu")
    else:
        if is_bert:
            train_ds = MaskedWindowDataset(train_seqs, max_len, mask_prob=mask_prob,
                                           seed=job["seed"],
                                           last_item_prob=proto["bert_last_item_prob"])
        elif variant in LOGIC_LOSS:
            from pmpriors.declare import n_constraints
            from training.logic_loss import LocalLogicLoss
            train_ds = RejectWindowDataset(train_seqs, max_len, priors["declare"], data["n_items"])
            loss_fn = LocalLogicLoss(alpha=cfg["competitors"]["mezini_alpha"])
            row["n_constraints"] = n_constraints(priors["declare"])
        elif variant == "sasrec_rgmask":
            from models.rgmask import MaskedLoss
            train_ds = AllowWindowDataset(train_seqs, max_len, priors["rgmask"])
            loss_fn = MaskedLoss()
        else:
            train_ds = CausalWindowDataset(train_seqs, max_len)
        train_loader = DataLoader(train_ds, batch_size=regime["batch_size"], shuffle=True)
        result = train_model(model, train_loader, val_loader,
                             n_epochs=regime["n_epochs"], lr=lr,
                             weight_decay=proto["weight_decay"], k_values=[1, 5, 10],
                             early_stop_metric=proto["early_stop_metric"],
                             patience=regime["patience"], device=device, verbose=False,
                             loss_fn=loss_fn)
        row["best_epoch"] = result["best_epoch"]
        row["val_NDCG@5"] = result["best_metrics"]["NDCG@5"]
        row["n_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        row.update(lambdas(variant, model))
        preds = collect_predictions(model, test_loader, device=device)

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(pred_path(job), index=False, compression="gzip")

    m = metrics_from_predictions(preds)
    row["n_predictions"] = m["n_predictions"]
    for k in ("accuracy", "macro_f1", "mrr", "HR@5", "NDCG@5", "HR@10", "NDCG@10",
              "learner_accuracy", "learner_mrr", "learner_NDCG@5"):
        row[k] = m[k]
    row["train_seconds"] = round(time.time() - t0, 1)
    row["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    return row


def run_dream_job(job, cfg, data, split, train_seqs, val_seqs, test_seqs, t0, device) -> dict:
    """Theis & Darabi competitors: own features and network (models/dream.py),
    evaluated on exactly the same (case, position) pairs as every other model."""
    from models.dream import run_dream
    proto = cfg["protocol"]
    dcfg = cfg["datasets"][job["dataset"]]
    cap, max_len = dcfg.get("eval_cap"), dcfg["max_len"]
    items = lambda seqs, s: PrefixEvalDataset(seqs, max_len, max_per_learner=cap, seed=s).items
    out = run_dream(job["variant"], train_seqs, val_seqs, test_seqs, data["timestamps"],
                    data["id2item"], items(val_seqs, 1), items(test_seqs, 2),
                    cfg["regimes"][dcfg["regime"]], proto["noise_threshold"], job["seed"], device)
    preds = out["preds"]
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(pred_path(job), index=False, compression="gzip")
    m = metrics_from_predictions(preds)
    row = {**job, "n_train_cases": len(split["train"]), "n_val_cases": len(split["val"]),
           "n_test_cases": len(split["test"]), "best_epoch": out["best_epoch"],
           "n_params": out["n_params"], "lambda_pm_dfg": None, "lambda_pm_tree": None,
           "num_blocks": None, "n_constraints": None, "n_places": out["n_places"],
           "test_marking_known": out["test_marking_known"], "val_NDCG@5": out["val_NDCG@5"],
           "device": device, "git_commit": git_commit(), "n_predictions": m["n_predictions"]}
    for k in ("accuracy", "macro_f1", "mrr", "HR@5", "NDCG@5", "HR@10", "NDCG@10",
              "learner_accuracy", "learner_mrr", "learner_NDCG@5"):
        row[k] = m[k]
    row["train_seconds"] = round(time.time() - t0, 1)
    row["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    return row


# ---------------------------------------------------------------------------
# results CSV
# ---------------------------------------------------------------------------

def _key(r: dict) -> tuple:
    return (r["dataset"], r["variant"], str(int(r["fold"])), str(int(r["seed"])),
            f"{float(r['train_frac']):g}")


def _done(f) -> set:
    f.seek(0)
    return {_key(r) for r in csv.DictReader(f)}


def already_done(job: dict, path: Path = RESULTS_CSV) -> bool:
    if not path.exists():
        return False
    with open(path, "r", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        done = _done(f)
        fcntl.flock(f, fcntl.LOCK_UN)
    return _key(job) in done


def append_row(row: dict, path: Path = RESULTS_CSV) -> bool:
    """Append under an exclusive lock; refuse a key already present."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, 2)
            empty = f.tell() == 0
            if not empty and _key(row) in _done(f):
                return False
            f.seek(0, 2)
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            if empty:
                w.writeheader()
            w.writerow(row)
            f.flush()
            return True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _pick(arg: str, allowed: list, what: str) -> list:
    if arg == "all":
        return list(allowed)
    chosen = arg.split(",")
    unknown = set(chosen) - set(allowed)
    assert not unknown, f"unknown {what}: {sorted(unknown)} (known: {allowed})"
    return [x for x in allowed if x in chosen]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--job-id", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--variants", default="all")
    ap.add_argument("--seeds", default=None, help="comma-separated (default: protocol.model_seeds)")
    ap.add_argument("--fracs", default="1.0")
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--expect-n", type=int, default=None,
                    help="job count at submission; refuse to run if the list changed")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else cfg["protocol"]["model_seeds"])
    jobs = enumerate_jobs(_pick(args.datasets, list(cfg["datasets"]), "datasets"),
                          _pick(args.variants, VARIANTS, "variants"),
                          seeds, [float(p) for p in args.fracs.split(",")],
                          cfg["protocol"]["n_folds"])

    if args.list:
        for i, j in enumerate(jobs):
            print(f"{i:4d}  {j['dataset']:14s} p={j['train_frac']:<5g} {j['variant']:18s} "
                  f"fold={j['fold']} seed={j['seed']}")
        print(f"\nTotal: {len(jobs)} jobs")
        return

    if args.expect_n is not None and args.expect_n != len(jobs):
        sys.exit(f"job list changed since submission ({args.expect_n} jobs then, {len(jobs)} now): "
                 f"code or configuration was edited; resubmit the array")
    assert args.job_id is not None and 0 <= args.job_id < len(jobs), \
        f"--job-id must be in [0, {len(jobs) - 1}]"
    job = jobs[args.job_id]
    tag = (f"[job {args.job_id}] {job['dataset']} / {job['variant']} / fold={job['fold']} "
           f"/ seed={job['seed']} / p={job['train_frac']:g}")
    if already_done(job):
        print(f"{tag}  SKIP (already in {RESULTS_CSV.name})", flush=True)
        return

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print(f"{tag}  CUDA unavailable, falling back to CPU", flush=True)
        device = "cpu"
    print(f"{tag}  device={device}", flush=True)

    row = run_job(job, cfg, device)
    written = append_row(row)
    print(f"{tag}  {'DONE' if written else 'finished, duplicate not appended'} in "
          f"{row['train_seconds']}s  n_pred={row['n_predictions']}  "
          f"acc={row['accuracy']:.4f}  NDCG@5={row['NDCG@5']:.4f}  "
          f"λ_dfg={row['lambda_pm_dfg']}  λ_bp={row['lambda_pm_tree']}", flush=True)


if __name__ == "__main__":
    main()