"""
LaTeX tables of the Results section
===================================
    python scripts/results_tables.py <analysis_dir> [--train-frac 1.0] [--seed 42]

Reads table_folds.csv and paired_tests.csv written by analysis/analyze.py and
prints two tables:
  tab:main     accuracy (%) and NDCG@5 (%) per log, in two blocks: mean and
               standard deviation over the five folds; the best mean of each
               column in bold (ties included).
  tab:effects  paired effect on NDCG@5 (points) of each enhanced model or
               competing technique against its baseline; a dagger marks q_BH < 0.05.
"""
import argparse
from pathlib import Path

import pandas as pd

LOGS = [("dev1-bloom", "Bloom"), ("dev1-section", "Section"), ("dev1-resource", "Resource"),
        ("oulad-fff", "OULAD"), ("ednet-kt3", "EdNet"), ("bpi-2012", "BPI12"),
        ("helpdesk", "Helpd."), ("sepsis", "Sepsis"), ("bpic2013-incidents", "BPI13"),
        ("bpic2020-domestic", "BPI20")]

GROUPS = [
    ("Reference", [("poprec", "PopRec"), ("pm_only", "PM-only")]),
    ("Baselines", [("gru4rec", "GRU4Rec"), ("sasrec", "SASRec"), ("bert4rec", "BERT4Rec")]),
    ("Our priors", [("gru4rec_pm_output", "GRU4Rec + DFG-1 (output)"),
                    ("sasrec_dfg1", "SASRec + DFG-1"), ("sasrec_dfgK", "SASRec + DFG-$K$"),
                    ("sasrec_bp", "SASRec + BP"), ("sasrec_hybrid", "SASRec + Hybrid"),
                    ("bert4rec_dfg1", "BERT4Rec + DFG-1")]),
    ("Competing techniques", [("gru4rec_ltl", "GRU4Rec + logic loss~\\cite{mezini2026}"),
                     ("sasrec_ltl", "SASRec + logic loss~\\cite{mezini2026}"),
                     ("sasrec_rgmask", "SASRec + RG mask~\\cite{theis2023}"),
                     ("dream_nap", "Decay replay~\\cite{theis2019}"),
                     ("dream_nap_rg", "Decay replay + RG~\\cite{theis2023}")]),
]
NAME = {k: v for _, rows in GROUPS for k, v in rows}
PAIRS = [("gru4rec_pm_output", "gru4rec"), ("sasrec_dfg1", "sasrec"), ("sasrec_dfgK", "sasrec"),
         ("sasrec_bp", "sasrec"), ("sasrec_hybrid", "sasrec"), ("bert4rec_dfg1", "bert4rec"),
         None,
         ("gru4rec_ltl", "gru4rec"), ("sasrec_ltl", "sasrec"), ("sasrec_rgmask", "sasrec"),
         ("dream_nap", "sasrec"), ("dream_nap_rg", "dream_nap")]
NCOL = len(LOGS)


def header(first, corner=""):
    return (f"\\begin{{tabular}}{{l{'r' * NCOL}}}\n\\toprule\n"
            f"{corner} & \\multicolumn{{3}}{{c}}{{DEV1}} & & & & & & & \\\\\n\\cmidrule(lr){{2-4}}\n"
            f"{first} & " + " & ".join(h for _, h in LOGS) + " \\\\\n\\midrule\n")


def main_table(folds, frac, seed, metric="NDCG@5"):
    x = folds[(folds.train_frac == frac) & (folds.seed == seed)].set_index(["variant", "dataset"])
    best = {d: round(x.xs(d, level="dataset")[f"{metric}_mean"].max() * 100, 1) for d, _ in LOGS}
    corner = {"accuracy": "(a) Accuracy", "NDCG@5": "(b) NDCG@5"}.get(metric, metric)
    out = header("Model", f"\\textit{{{corner}}}")
    for g, (title, rows) in enumerate(GROUPS):
        if g:
            out += "\\midrule\n"
        out += f"\\multicolumn{{{NCOL + 1}}}{{l}}{{\\emph{{{title}}}}} \\\\\n"
        for v, name in rows:
            cells = []
            for d, _ in LOGS:
                m, s = x.loc[(v, d), [f"{metric}_mean", f"{metric}_std"]] * 100
                c = f"{m:.1f}"
                c = f"\\textbf{{{c}}}" if round(m, 1) == best[d] else c
                cells.append(f"{c}\\,{{\\tiny$\\pm${s:.1f}}}")
            out += f"\\quad {name} & " + " & ".join(cells) + " \\\\\n"
    return out + "\\bottomrule\n\\end{tabular}\n"


def effects_table(tests, frac, seed):
    x = tests[(tests.train_frac == frac) & (tests.seed == seed) & (tests.metric == "NDCG@5")]
    x = x.set_index(["enhanced", "baseline", "dataset"])
    out = header("Model (against its baseline)")
    for p in PAIRS:
        if p is None:
            out += "\\midrule\n"
            continue
        cells = []
        for d, _ in LOGS:
            r = x.loc[(p[0], p[1], d)]
            e = r.effect * 100
            c = f"{e:+.2f}".replace("+0.00", "0.00").replace("-0.00", "0.00")
            cells.append(c + ("$^\\dagger$" if r.q_bh < 0.05 else "\\phantom{$^\\dagger$}"))
        label = NAME[p[0]] + (" vs SASRec" if p == ("dream_nap", "sasrec") else "")
        label += " vs decay replay" if p == ("dream_nap_rg", "dream_nap") else ""
        out += f"{label} & " + " & ".join(cells) + " \\\\\n"
    return out + "\\bottomrule\n\\end{tabular}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_dir", type=Path)
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    folds = pd.read_csv(a.analysis_dir / "table_folds.csv")
    tests = pd.read_csv(a.analysis_dir / "paired_tests.csv")
    print("% tab:main (a) accuracy\n" + main_table(folds, a.train_frac, a.seed, "accuracy"))
    print("% tab:main (b) NDCG@5\n" + main_table(folds, a.train_frac, a.seed, "NDCG@5"))
    print("% tab:effects\n" + effects_table(tests, a.train_frac, a.seed))


if __name__ == "__main__":
    main()