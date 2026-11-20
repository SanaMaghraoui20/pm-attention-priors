# Theis & Darabi 2019, 2023: decay replay mining and reachability-graph masking

J. Theis, H. Darabi. "Decay Replay Mining to Predict Next Process Events."
IEEE Access 7, 2019.
J. Theis, H. Darabi. "Improving Predictive Process Monitoring Through
Reachability Graph-Based Masking of Neural Networks." IEEE Transactions on
Computational Social Systems 10(4), 2023.
Code: https://github.com/Julian-Theis/PyDREAM/tree/experimental

## Method

1. Decay replay mining (2019). Each place p gets a time-decay function
   (clipped linear, slope from the mean time between token creations in p,
   obtained by replaying the training log). Replaying a prefix up to time tau
   gives the timed state sample S(tau) = [M(tau), C(tau), F(tau)]: marking,
   tokens added per place, decay value per place. A network with four fully
   connected hidden layers predicts the next activity from S(tau).
2. Reduced reachability graph (2023): nodes = markings reached right after a
   visible transition (and the initial marking); an edge N0 -> N1 labelled a
   exists if N1 is reached from N0 by zero or more silent transitions and then
   the visible transition a.
3. Mask MSK(tau) in {0,1}^|A|: 1 for every activity labelling an outgoing edge
   of the node of M(tau).
4. Network changes (2023): the mask is also given as input, the output layer
   applies Eq. (5) in training and at inference,
       o = MSK * softmax(x) / sum(MSK * softmax(x)),
   and ReLU is replaced by Swish.
5. Original protocol: five random 60/20/20 splits, two seeds, weighted
   multiclass AUROC, given PNML models; logs Helpdesk, BPIC12, BPIC13.

## Implementation (src/models/dream.py, src/pmpriors/petri.py)

Two configurations: dream_nap (2019, input injection) and dream_nap_rg
(2023, input and output injection). Reimplemented from the papers and checked
against PyDREAM (GPL-3, branch experimental), without copying code.

Settings taken from the authors' code:
- network widths F, 1.2F, 0.6F, 0.3F (F = 3 x number of places); mask branch
  Dense(V) concatenated after the first layer; dropout 0.2; Adam 1e-3;
  batch 64; at most 100 epochs; ReLU (2019) or Swish (2023);
- decay f(dt) = max(0, 10 - alpha dt), alpha = 1 / mean reactivation delta of
  the place (1 / longest case duration if fewer than two deltas);
- replay: silent transitions on the shortest path; if the transition is still
  not enabled, the missing tokens are added and it is fired;
- reachability graph built from the markings visited when replaying the
  training log; mask = 1 for the next activities, 1e-10 otherwise;
- features min-max scaled on the training samples.

Points not specified in the paper, and our choices:
- a prefix reaching a marking never seen in training receives no mask (all
  ones), and the floor of 1e-10 avoids an infinite cross-entropy when the
  target is outside the mask.

Adaptations:
- Petri net discovered with the Inductive Miner (noise threshold 0.20, as our
  priors) from the training cases of each fold, instead of given models.
- Reactivation deltas are collected per place, as described in the paper
  (PyDREAM collects them differently, through a loop variable).
- The first prediction is made after one event (PyDREAM: after two), so that
  every model is evaluated on the same prefixes; resource counts are not used.
- Model selection on validation NDCG@5 with the patience of each regime, as
  for all models (authors: lowest validation loss over 100 epochs).
- The search over silent transitions is bounded to 64 markings, and the
  replay is memoised per (marking, activity). Each run reports n_places and
  test_marking_known (share of test prefixes whose marking was reached in
  training).

## Controlled version on SASRec (sasrec_rgmask, src/models/rgmask.py)

Comparing dream_nap(_rg) with SASRec mixes two factors: the network and the
injection point. To isolate the injection point, sasrec_rgmask applies the
mask of Eq. (5) to the SASRec baseline, in training (masked cross-entropy)
and at prediction, with the same Petri net, replay and mask construction as
dream_nap_rg, and the network and hyperparameters of the SASRec baseline.
Training masks come from the replay of the whole case prefix; at prediction,
the visible prefix is replayed, and a marking not reached in training gives
no mask.
