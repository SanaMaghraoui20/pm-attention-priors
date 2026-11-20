"""
Public event logs used in the paper
===================================
The logs are published on 4TU.ResearchData under their own licences and are
not redistributed with this repository. Download each file from its landing
page into data/public/, then run

    python scripts/download_public_logs.py          # lists sources, checks files

which reports, for every file found in data/public/, its size, and the
statistics of the loaded log (cases, activities, lengths, variant ratio).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SOURCES = {
    "BPI Challenge 2012": ("10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f",
                           "https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204"),
    "Helpdesk": ("10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb",
                 "https://data.4tu.nl/articles/dataset/Dataset_belonging_to_the_help_desk_log_of_an_Italian_Company/12675977"),
    "Sepsis Cases": ("10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460",
                     "https://data.4tu.nl/articles/dataset/Sepsis_Cases_-_Event_Log/12707639"),
    "BPI Challenge 2013 (incidents)": ("10.4121/uuid:500573e6-accc-4b0c-9576-aa5468b10cee",
                                       "https://data.4tu.nl/articles/dataset/BPI_Challenge_2013_incidents/12693914/1"),
    "BPI Challenge 2020 (domestic declarations)": ("10.4121/uuid:3f422315-ed9d-4882-891f-e180b5b4feb5",
                                                   "https://data.4tu.nl/articles/dataset/BPI_Challenge_2020_Domestic_Declarations/12692543"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import yaml
    from data.eventlog import load_event_log
    print("Sources (download into data/public/):")
    for name, (doi, url) in SOURCES.items():
        print(f"  {name:44s} doi:{doi}\n  {'':44s} {url}")
    cfg = yaml.safe_load(open(ROOT / "experiments" / "hyperparameters.yaml"))
    entries = {k: v for k, v in cfg["datasets"].items()
               if v["loader"] == "eventlog" and v["args"]["path"].startswith("data/public/")}
    print("\nPublic logs of experiments/hyperparameters.yaml:")
    for name, e in entries.items():
        args = dict(e["args"])
        p = ROOT / args.pop("path")
        if not p.exists():
            print(f"\n  {name}: MISSING {p.relative_to(ROOT)}")
            continue
        print(f"\n  {name}: {p.name}  ({p.stat().st_size / 1e6:.1f} MB)  sha256={sha256(p)}")
        try:
            d = load_event_log(p, verbose=False, **args)
            print("   ", ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in d["stats"].items()))
            longest = max(len(x) for x in d["sequences"].values())
            flag = "ok" if longest <= e["max_len"] else "INCREASE max_len"
            print(f"    longest case = {longest} events; max_len in yaml = {e['max_len']} ({flag})")
        except Exception as ex:  
            print(f"    could not load: {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    main()
