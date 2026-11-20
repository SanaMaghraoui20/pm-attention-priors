# Process-Mining Priors as Learned Attention Biases for Next-Activity Prediction

Code of the paper *Process-Mining Priors as Learned Attention Biases for
Next-Activity Prediction* (ICPM 2027).

Process-mining artefacts (directly-follows statistics, a behavioural profile
derived from a discovered process tree) are turned into standardised matrices
over activity pairs and added to the attention scores of a transformer (or to
the output scores of a recurrent network). Each prior is scaled by a weight
λ, initialised at zero, excluded from weight decay and learned with the
predictor, so the contribution of process knowledge can be read from λ.

The repository also contains our reimplementations of three competing
knowledge-injection techniques (logic loss of Mezini et al. 2026; decay
replay mining and reachability-graph masking of Theis & Darabi 2019, 2023),
evaluated under the same protocol.

## Repository

```
src/
  data/       loaders (DEV1, OULAD, EdNet-KT3, generic XES/CSV), case-level folds,
              training windows and prefix evaluation, on-disk cache
  pmpriors/   dfg.py (1-hop and distance-aware DFG priors),
              behavioral_profile.py (behavioural-profile prior),
              petri.py (Petri net discovery and token replay),
              declare.py (Declare miner with vacuity detection)
  models/     transformer_block.py (λ-weighted attention bias), sasrec.py,
              bert4rec.py, gru4rec.py, baselines.py (PopRec, PM-only),
              dream.py (decay replay mining, with and without the mask),
              rgmask.py (reachability-graph mask on SASRec),
              variants.py (the sixteen model configurations of the paper)
  training/   trainer.py, evaluation.py (metrics, per-prediction records),
              logic_loss.py (local logic loss of Mezini et al.)
experiments/
  run.py                 one job = (log, configuration, fold, seed, training fraction)
  grid.py                hyperparameter grid search on the baselines
  hyperparameters.yaml   protocol, datasets, competing techniques, hyperparameters
analysis/analyze.py      paired tests, effect by prefix position, λ, tables
scripts/
  results_tables.py        LaTeX tables of the Results section
  dataset_table.py         LaTeX table of the event logs
  download_public_logs.py  sources and checksums of the public logs
docs/
  competitors/           reimplementation notes and adaptations (Mezini, Theis & Darabi)
tests/                   folds, windows, absence of leakage, position encoding,
                         Declare automata, replay and mask, every configuration
```

## Installation

```bash
uv sync --extra dev          # or: pip install torch pm4py pandas scikit-learn pyyaml pytest
pytest
```

## Data

The event logs are not redistributed with this repository.

| Log | Source |
|-----|--------|
| DEV1 (3 granularities) | Moodle course log of La Rochelle Université. Contains private data and is not distributed. |
| OULAD (module FFF) | Open University Learning Analytics Dataset: `studentVle.csv`, `vle.csv` in `data/OULAD/` |
| EdNet-KT3 | `u*.csv` files in `data/KT3/` |
| BPI Challenge 2012, Helpdesk, Sepsis, BPIC 2013 incidents, BPIC 2020 domestic declarations | 4TU.ResearchData, see `python scripts/download_public_logs.py` |

Paths are set in `experiments/hyperparameters.yaml`.

## Protocol

Five-fold case-level cross-validation following the benchmark of Rama-Maneiro,
Vidal and Lama (IEEE TSC 2023): `KFold(5, shuffle=True, random_state=42)` over
the cases, the last 20% of the training cases of each fold used for early
stopping. Priors, count-based baselines and the process knowledge of the
competing techniques (Petri nets, Declare constraints) are discovered from the
training cases of the fold. Every prefix of every test case is predicted.
Each configuration is run with three model seeds on the full training data
and with 10%, 25% and 50% of the training cases.

## Reproducing the results

```bash
python experiments/run.py --list                       # enumerate the jobs
python experiments/run.py --job-id 0 --device cpu      # run one job
python analysis/analyze.py                             # statistics and tables
python scripts/results_tables.py results/analysis      # LaTeX tables of the paper
```

Results go to `results/results.csv` and
`results/predictions/`; the report and the tables to
`results/analysis/`.

| Paper | Produced by |
|-------|-------------|
| Event logs table | `scripts/dataset_table.py` |
| Accuracy and NDCG@5 table, paired effects table | `scripts/results_tables.py` (from `table_folds.csv`, `paired_tests.csv`) |
| Learned weights | `analysis/analyze.py`, section 4 → `table_lambda.csv` |
| Reduced training data | `paired_tests.csv` (column `train_frac`) |
| Effect by prefix position | section 3b → `effect_by_position.csv` |
| All metrics (accuracy, Macro-F1, MRR, NDCG@5, HR@5, HR@10) | `table_folds.csv`, `report.txt` |

## Citation

To be added after publication.

## Licence

See `LICENSE`.
