"""
Behavioral-Profile Bias Matrix (LCA-based, directional)
========================================================
Derives a pairwise bias matrix from the *behavioural profile* of a Process
Tree discovered by the Inductive Miner (pm4py), following Weidlich et al.'s
notion of behavioural profiles: the relation between two activities is
determined by the operator of their Lowest Common Ancestor (LCA) in the tree.

Relations and default scores
------------------------
  SEQUENCE (strict order)  : +S if a's subtree precedes b's subtree,
                             -S in the reverse direction  (directional!)
  XOR      (exclusiveness) : -X for both directions (they should not co-occur)
  PARALLEL (interleaving)  : +P for both directions (order-free)
  LOOP     (repetition)    : +L for both directions (can repeat / alternate)
  same activity (a == a)   : +L if under a LOOP ancestor, else 0

The raw score matrix is then z-standardised over valid (non-special) entries,
so that the learnable lambda absorbs any residual scale and the four scores
only need to encode *relative* preferences.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import torch



def _decode_train_sequences(
    train: Dict[int, List[int]],
    id2item: Dict[int, str],
    special_tokens: frozenset = frozenset({"<PAD>", "<MASK>"}),
) -> List[List[str]]:
    out: List[List[str]] = []
    for uid, seq in train.items():
        labels = [id2item[i] for i in seq if id2item[i] not in special_tokens]
        if len(labels) >= 2:
            out.append(labels)
    return out


def _traces_to_pm4py_log(traces: List[List[str]]):
    import pandas as pd
    import pm4py

    rows = []
    for case_id, trace in enumerate(traces):
        for pos, activity in enumerate(trace):
            rows.append({
                "case:concept:name": str(case_id),
                "concept:name":       activity,
                "time:timestamp":     pd.Timestamp("2000-01-01") + pd.Timedelta(seconds=pos),
            })
    df = pd.DataFrame(rows)
    return pm4py.format_dataframe(df,
                                  case_id="case:concept:name",
                                  activity_key="concept:name",
                                  timestamp_key="time:timestamp")


# ---------------------------------------------------------------------------
# LCA relation extraction
# ---------------------------------------------------------------------------

def _collect_leaf_paths(node, path, leaf_paths):
    """
    Collect, for every labelled leaf, the path of (node, child_index) pairs
    from the root to that leaf. A label can appear at several leaves
    (duplicated activities); we keep ALL paths per label.
    """
    if node.operator is None:
        if node.label is not None:
            leaf_paths.setdefault(node.label, []).append(list(path))
        return
    for idx, child in enumerate(node.children):
        path.append((node, idx))
        _collect_leaf_paths(child, path, leaf_paths)
        path.pop()


def _lca_relation(path_a, path_b) -> Tuple[object, int, int]:
    """
    Given two root-to-leaf paths (lists of (node, child_idx)), return
    (lca_node, idx_a, idx_b) where idx_a/idx_b are the child indices of the
    subtrees containing a and b directly under the LCA.
    If one activity is an ancestor of the other (shouldn't happen for leaves),
    returns (deepest common node, -1, -1).
    """
    depth = 0
    max_d = min(len(path_a), len(path_b))
    while depth < max_d and path_a[depth][0] is path_b[depth][0] \
            and path_a[depth][1] == path_b[depth][1]:
        depth += 1
    if depth < max_d and path_a[depth][0] is path_b[depth][0]:
        # same node, different child branches -> this is the LCA
        return path_a[depth][0], path_a[depth][1], path_b[depth][1]
    # degenerate: identical full path (same leaf occurrence)
    if depth > 0:
        return path_a[depth - 1][0], -1, -1
    return None, -1, -1


def _has_loop_ancestor(path) -> bool:
    from pm4py.objects.process_tree.obj import Operator
    return any(node.operator == Operator.LOOP for node, _ in path)



def build_behavioral_profile_bias_matrix(
    train: Dict[int, List[int]],
    id2item: Dict[int, str],
    n_items: int,
    noise_threshold: float = 0.20,
    score_seq: float = 1.5,     # strict order  (directional: +/-)
    score_xor: float = -1.0,    # exclusiveness (symmetric)
    score_par: float = 0.2,     # interleaving  (symmetric)
    score_loop: float = 0.8,    # repetition    (symmetric)
    standardize: bool = True,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Build the behavioural-profile bias matrix B of shape (n_items, n_items).

    Parameters
    ----------
    train           : {user_id: [item_id, ...]} training sequences (encoded).
    id2item         : reverse vocab mapping (item_id -> item label).
    n_items         : full vocab size (incl. PAD, MASK).
    noise_threshold : Inductive Miner filtering threshold.
    score_seq/xor/par/loop : relation scores (relative values matter;
                      standardisation absorbs the scale).
    standardize     : z-normalise over valid entries (recommended).

    Returns
    -------
    torch.Tensor of shape (n_items, n_items).
    """
    try:
        import pm4py
        from pm4py.objects.process_tree.obj import Operator
    except ImportError as e:
        raise ImportError("pm4py is required. Install with: pip install pm4py") from e

    special = {"<PAD>", "<MASK>"}

    # 1) Decode traces 
    traces = _decode_train_sequences(train, id2item)
    level = "item"

    if verbose:
        alphabet = {x for tr in traces for x in tr}
        print(f"[BP-bias] Discovering Process Tree on {len(traces)} traces "
              f"at {level} level (|alphabet|={len(alphabet)}).")

    # 2) Discover the tree 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log = _traces_to_pm4py_log(traces)
        # noise_threshold requires a recent pm4py (>= 2.7)
        tree = pm4py.discover_process_tree_inductive(log, noise_threshold=noise_threshold)

    if verbose:
        t = str(tree)
        print(f"[BP-bias] Tree: {t[:200]}{'...' if len(t) > 200 else ''}")

    #  3) Leaf paths 
    leaf_paths: Dict[str, List[list]] = {}
    _collect_leaf_paths(tree, [], leaf_paths)
    labels = sorted(leaf_paths.keys())

    if verbose:
        print(f"[BP-bias] {len(labels)} distinct labelled leaves in the tree.")

    #  4) Pairwise relation scores at tree-alphabet level 
    # When an activity has several leaf occurrences, aggregate by averaging
    # over all path pairs.
    S: Dict[Tuple[str, str], float] = {}

    for a in labels:
        for b in labels:
            scores = []
            if a == b:
                looped = any(_has_loop_ancestor(p) for p in leaf_paths[a])
                scores.append(score_loop if looped else 0.0)
            else:
                for pa in leaf_paths[a]:
                    for pb in leaf_paths[b]:
                        lca, ia, ib = _lca_relation(pa, pb)
                        if lca is None or ia < 0:
                            scores.append(0.0)
                            continue
                        op = lca.operator
                        if op == Operator.SEQUENCE:
                            # directional: a before b -> +, a after b -> -
                            scores.append(score_seq if ia < ib else -score_seq)
                        elif op == Operator.XOR:
                            scores.append(score_xor)
                        elif op == Operator.PARALLEL:
                            scores.append(score_par)
                        elif op == Operator.LOOP:
                            scores.append(score_loop)
                        else:
                            # OR or unknown: permissive, like parallel
                            scores.append(score_par)
            S[(a, b)] = sum(scores) / max(len(scores), 1)

    #  5) Project onto the item vocabulary 
    item2id = {v: k for k, v in id2item.items()}
    B = torch.zeros(n_items, n_items, dtype=torch.float32)

    for (a, b), v in S.items():
        ia, ib = item2id.get(a), item2id.get(b)
        if ia is not None and ib is not None:
            B[ia, ib] = v

    #  6) Standardise over valid entries 
    if standardize:
        mask = torch.ones_like(B, dtype=torch.bool)
        for k, v in id2item.items():
            if v in special:
                mask[k, :] = False
                mask[:, k] = False
        vals = B[mask]
        sigma = vals.std().clamp(min=1e-6)
        mu = vals.mean()
        B = (B - mu) / sigma
        B[~mask] = 0.0
        if verbose:
            print(f"[BP-bias] Standardised: mu={mu:.3f} sigma={sigma:.3f} "
                  f"-> range on valid entries "
                  f"[{B[mask].min():.2f}, {B[mask].max():.2f}]")

    if verbose:
        n_pos = (B > 0).sum().item()
        n_neg = (B < 0).sum().item()
        print(f"[BP-bias] Done. Positive entries: {n_pos}, negative: {n_neg}.")

    return B
