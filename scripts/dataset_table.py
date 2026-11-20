"""
Statistics of the event logs, as LaTeX rows of the dataset table
================================================================
    python scripts/dataset_table.py > results/dataset_table.tex

For every log of experiments/hyperparameters.yaml, loaded exactly as the
experiments load it (same loader, same arguments, same cache):
  cases       number of cases (traces with at least two events)
  activities  size of the activity alphabet
  events      number of events
  avg len     mean trace length; max len, the longest trace
  repeat      share of prediction points (every prefix of every case) whose
              next activity equals the last observed one
  vratio      distinct trace variants / cases
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data.registry import load_dataset  

NAMES = {"dev1-bloom": "DEV1-bloom", "dev1-section": "DEV1-section",
         "dev1-resource": "DEV1-resource", "oulad-fff": "OULAD", "ednet-kt3": "EdNet-KT3",
         "bpi-2012": "BPI~2012", "helpdesk": "Helpdesk", "sepsis": "Sepsis",
         "bpic2013-incidents": "BPIC~2013", "bpic2020-domestic": "BPIC~2020"}


def main():
    cfg = yaml.safe_load(open(ROOT / "experiments" / "hyperparameters.yaml"))
    print("% Log & Cases & Activities & Events & Avg. len & Max len & Repeat & Vratio")
    for name in cfg["datasets"]:
        data = load_dataset(name, cfg, ROOT)
        seqs = [s for s in data["sequences"].values() if len(s) >= 2]
        lens = [len(s) for s in seqs]
        points = sum(n - 1 for n in lens)
        repeat = sum(s[t] == s[t - 1] for s in seqs for t in range(1, len(s))) / points
        vratio = len({tuple(s) for s in seqs}) / len(seqs)
        print(f"{NAMES.get(name, name):16s} & {len(seqs):,} & {data['n_items'] - 2} & {sum(lens):,} & "
              f"{sum(lens) / len(lens):.1f} & {max(lens)} & {repeat:.2f} & {vratio:.2f} \\\\"
              .replace(",", "{,}"), flush=True)


if __name__ == "__main__":
    main()