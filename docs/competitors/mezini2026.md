# Mezini et al. 2026: neuro-symbolic PPM with a logic loss

A. Mezini, E. Umili, I. Donadello, F. M. Maggi, M. Mancanelli, F. Patrizi.
"Neuro-Symbolic Predictive Process Monitoring." Information Systems 141
(2026) 102762. Code: https://github.com/axelmezini/nesy-suffix-prediction-dfa

Task in the paper: suffix prediction (metrics: Damerau-Levenshtein similarity,
LTLf satisfaction rate). Model-agnostic: any autoregressive next-activity
predictor; the authors test a 2x100 LSTM and a pre-LN causal Transformer.

## Method (Sec. 3)

Knowledge: an LTLf formula phi (conjunction of Declare constraints),
translated to a DFA (ltlf2DFA). Preprocessing: self-loops for activities not
in the formula; EOT symbol with terminal success/failure states; Q_fail =
states from which no accepting state is reachable.

Loss, Eq. (11):  L = alpha * CE + (1 - alpha) * L_phi   (alpha fixed)

Local Logic Loss (LLL), Eqs. (12)-(13), teacher forcing, per step:
  P_viol(t) = sum over a in A u {EOT} of  p_theta(a | a_<t) * 1[delta*(q0, a_<t + a) in Q_fail]
  L_loc     = (1/|a|) sum_t  -log(1 - P_viol(t))
Global Logic Loss (GLL), Eqs. (14)-(16): Monte Carlo over Gumbel-softmax
sampled suffixes, compliance computed by DeepDFA; L_glob = -log(mean P_sat).
Only relevant to suffix generation.

The authors' code (src/loss/local_loss.py, MIT licence) differs from
Eqs. (11)-(13) on one point: steps whose true target leads to a rejecting
state are excluded from the cross-entropy term (weighted mean over valid
steps); the penalty is -log(1 - invalid_mass + 1e-6), averaged over all steps.
We follow the code.

Reported settings: alpha = 0.25 for LLL, 0.75 for GLL; batch 64, Adam.

## Original protocol (Sec. 4.1)

Sepsis, BPIC13 closed problems, BPIC20 travel permits. Traces ordered by
first timestamp, 80/20 train/test split. Constraints are mined with the
Declare Miner (minimum support 85%) from the test set, and traces violating
the model are removed from both sets.

## Adaptations

- Local logic loss only: the global loss needs generated suffixes, whereas
  our task is the next activity. Loss as in the authors' code.
- Constraints are mined from the training cases of each fold, and no trace is
  removed. Miner: src/pmpriors/declare.py.
- Support with vacuity detection, as in the Declare Miner: share of the cases
  in which the constraint is activated that also satisfy it, at least 0.85.
  Checked against the authors' Sepsis model (39 constraints; our miner gives
  37 on a comparable log). Without vacuity detection, almost every pair of
  rare activities yields a constraint.
- pm4py.discover_declare is not used: its negative templates are defined as
  the negation of the positive ones (e.g. nonchainsuccession = not
  chainsuccession), which differs from Declare semantics.
- Templates with step-level failing states: precedence, altprecedence,
  chainprecedence, altresponse, chainresponse, noncoexistence,
  nonsuccession, nonchainsuccession.
- A constraint already violated by the prefix is ignored for the rest of the
  case (the authors avoid this situation by removing violating traces).
- The DFA of each template is written directly instead of using
  ltlf2DFA/MONA; tests check the automata against brute-force satisfaction.
- Applied to the SASRec and GRU4Rec baselines, with alpha = 0.25.
