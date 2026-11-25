"""Instance-level attribution metrics for MIML.

Every standard MIML metric (Hamming Loss, One-Error, Coverage, Ranking Loss,
Average Precision) is computed at the *bag* level: it asks whether the correct
label set was predicted for a bag. None of them asks the question that
motivates MIML over plain multi-label learning in the first place — whether the
model identifies *which instance* carries *which label*.

This module supplies that missing axis. Given a model's per-label attention
over instances and per-instance ground-truth labels, it measures how well the
attention localizes each label's true witnesses.

Conventions used throughout
---------------------------
    attn        (B, L, N)  per-label attention/score over instances. Larger =
                           more attributed. Need not be normalised.
    inst_labels (B, N, L)  binary per-instance ground truth.
    bag_labels  (B, L)     binary bag ground truth. Scoring is restricted to
                           labels actually present in the bag; asking where an
                           absent label is located is not well posed.
    mask        (B, N)     True at padded positions (excluded everywhere).

A (bag, label) pair is *scorable* when the label is present in the bag and the
bag has at least one real (unpadded) instance carrying it, and at least one
that does not — otherwise ranking is degenerate.

Well-posedness
--------------
These metrics assume the standard MIML consistency condition: a bag carries
label l iff at least one of its instances carries l (bag labels are the OR of
instance labels). `check_or_consistency` verifies this and should be run on any
new dataset before its numbers are believed.
"""

import numpy as np

__all__ = [
    'check_or_consistency',
    'instance_precision_at_1',
    'instance_precision_at_k',
    'instance_mrr',
    'instance_auc',
    'instance_auc_pooled',
    'label_agnostic_auc_bounds',
    'chance_precision_at_1',
    'label_agnostic_ceiling',
    'witness_disjointness',
    'all_instance_metrics',
]


def _to_numpy(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _prepare(attn, inst_labels, bag_labels, mask=None):
    """Normalise inputs to numpy and build the validity mask."""
    attn = _to_numpy(attn).astype(np.float64)
    inst_labels = _to_numpy(inst_labels).astype(bool)
    bag_labels = _to_numpy(bag_labels).astype(bool)

    B, L, N = attn.shape
    if inst_labels.shape != (B, N, L):
        raise ValueError(
            f'inst_labels must be (B, N, L)={(B, N, L)}, got {inst_labels.shape}')
    if bag_labels.shape != (B, L):
        raise ValueError(
            f'bag_labels must be (B, L)={(B, L)}, got {bag_labels.shape}')

    if mask is None:
        valid = np.ones((B, N), dtype=bool)
    else:
        valid = ~_to_numpy(mask).astype(bool)

    return attn, inst_labels, bag_labels, valid


def _scorable_pairs(inst_labels, bag_labels, valid):
    """Boolean (B, L): pairs where localization is a well-posed question.

    Requires the label present in the bag, at least one real witness instance,
    and at least one real non-witness instance to rank against.
    """
    # (B, N, L) restricted to real instances
    real = valid[:, :, None] & inst_labels
    n_wit = real.sum(axis=1)                                  # (B, L)
    n_real = valid.sum(axis=1)[:, None]                       # (B, 1)
    return bag_labels & (n_wit > 0) & (n_wit < n_real)


def check_or_consistency(inst_labels, bag_labels, mask=None):
    """Fraction of bags where bag labels are exactly the OR of instance labels.

    Returns (fraction_consistent, n_bag_labels_with_no_witness).
    A value of 1.0 with 0 uncovered labels means the dataset satisfies the
    standard MIML assumption exactly.
    """
    inst_labels = _to_numpy(inst_labels).astype(bool)
    bag_labels = _to_numpy(bag_labels).astype(bool)
    if mask is None:
        valid = np.ones(inst_labels.shape[:2], dtype=bool)
    else:
        valid = ~_to_numpy(mask).astype(bool)

    derived = (inst_labels & valid[:, :, None]).any(axis=1)   # (B, L)
    consistent = (derived == bag_labels).all(axis=1)
    uncovered = int((bag_labels & ~derived).sum())
    return float(consistent.mean()), uncovered


def _masked_scores(attn, valid):
    """Set padded positions to -inf so they never win a ranking."""
    out = attn.copy()
    out[~valid[:, None, :].repeat(attn.shape[1], axis=1)] = -np.inf
    return out


def instance_precision_at_k(attn, inst_labels, bag_labels, mask=None, k=1):
    """Fraction of scorable (bag, label) pairs whose top-k attended instances
    include at least one true witness for that label.

    k=1 is the headline metric: did the model point at the right instance?
    """
    attn, inst_labels, bag_labels, valid = _prepare(
        attn, inst_labels, bag_labels, mask)
    pairs = _scorable_pairs(inst_labels, bag_labels, valid)
    if not pairs.any():
        return float('nan')

    scores = _masked_scores(attn, valid)                      # (B, L, N)
    B, L, N = scores.shape
    kk = min(k, N)
    topk = np.argpartition(-scores, kk - 1, axis=-1)[..., :kk]  # (B, L, kk)

    # witness[b, l, n] -> transpose inst_labels to (B, L, N)
    wit = np.transpose(inst_labels, (0, 2, 1)) & valid[:, None, :]
    hit = np.take_along_axis(wit, topk, axis=-1).any(axis=-1)  # (B, L)
    return float(hit[pairs].mean())


def instance_precision_at_1(attn, inst_labels, bag_labels, mask=None):
    """Convenience wrapper: precision@1."""
    return instance_precision_at_k(
        attn, inst_labels, bag_labels, mask=mask, k=1)


def instance_mrr(attn, inst_labels, bag_labels, mask=None):
    """Mean reciprocal rank of the first true witness in the attention order."""
    attn, inst_labels, bag_labels, valid = _prepare(
        attn, inst_labels, bag_labels, mask)
    pairs = _scorable_pairs(inst_labels, bag_labels, valid)
    if not pairs.any():
        return float('nan')

    scores = _masked_scores(attn, valid)
    order = np.argsort(-scores, axis=-1)                      # (B, L, N)
    wit = np.transpose(inst_labels, (0, 2, 1)) & valid[:, None, :]
    ranked = np.take_along_axis(wit, order, axis=-1)          # (B, L, N)

    first = np.argmax(ranked, axis=-1)                        # 0-indexed
    rr = 1.0 / (first + 1.0)
    return float(rr[pairs].mean())


def _per_pair_auc(attn, inst_labels, bag_labels, mask=None):
    """(auc, n_pos, n_neg) for every scorable (bag, label) pair.

    Rank-sum (Mann-Whitney) identity with tie averaging, so a uniform
    attention distribution scores exactly 0.5 per pair.
    """
    attn, inst_labels, bag_labels, valid = _prepare(
        attn, inst_labels, bag_labels, mask)
    pairs = _scorable_pairs(inst_labels, bag_labels, valid)
    wit = np.transpose(inst_labels, (0, 2, 1))                # (B, L, N)
    out = []
    bs, ls = np.nonzero(pairs)
    for b, l in zip(bs, ls):
        v = valid[b]
        s = attn[b, l][v]
        y = wit[b, l][v]
        n_pos, n_neg = int(y.sum()), int((~y).sum())
        # rankdata with tie averaging, without a scipy dependency
        order = np.argsort(s, kind='mergesort')
        ranks = np.empty(len(s), dtype=np.float64)
        ranks[order] = np.arange(1, len(s) + 1)
        uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
        auc = (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        out.append((auc, n_pos, n_neg))
    return out


def instance_auc(attn, inst_labels, bag_labels, mask=None):
    """Mean per-(bag, label) AUC of separating witnesses from non-witnesses
    (macro average over scorable pairs)."""
    rows = _per_pair_auc(attn, inst_labels, bag_labels, mask)
    if not rows:
        return float('nan')
    return float(np.mean([r[0] for r in rows]))


def instance_auc_pooled(attn, inst_labels, bag_labels, mask=None):
    """AUC pooled over all witness/non-witness instance pairs (micro average).

    Every (witness, non-witness) pair of every scorable (bag, label) counts
    once. Under a witness partition, any ranking shared across labels scores
    exactly 0.5: each cross-witness pair is counted once in each direction
    (DMKD revision, reviewer 1 point 2).
    """
    rows = _per_pair_auc(attn, inst_labels, bag_labels, mask)
    if not rows:
        return float('nan')
    num = sum(a * p * n for a, p, n in rows)
    den = sum(p * n for _, p, n in rows)
    return float(num / den)


def label_agnostic_auc_bounds(inst_labels, bag_labels, mask=None):
    """(lower, upper) on the pooled AUC of ANY label-agnostic ranking.

    For an unordered instance pair {j, k} of bag i, let c_jk be the number of
    scorable labels l with j in W_i(l) and k not in W_i(l). A ranking shared
    across labels orders the pair one way, so the pair contributes c_jk or
    c_kj (half their sum on a tie) of its c_jk + c_kj witness/non-witness
    comparisons. Hence
        sum min(c_jk, c_kj) / T  <=  pooled AUC  <=  sum max(c_jk, c_kj) / T,
    T = sum (c_jk + c_kj) = the number of pooled comparisons. Both bounds are
    attained: c_jk - c_kj = cov(j) - cov(k), where cov counts the scorable
    labels an instance witnesses, so ranking by cov reaches the upper bound and
    the reverse ranking the lower one (lower = 1 - upper). Under a witness
    partition c_jk = c_kj for every pair and both bounds equal 1/2.
    """
    B, N, L = _to_numpy(inst_labels).shape
    dummy = np.zeros((B, L, N))
    _, inst_labels, bag_labels, valid = _prepare(
        dummy, inst_labels, bag_labels, mask)
    pairs = _scorable_pairs(inst_labels, bag_labels, valid)
    lo = hi = tot = 0.0
    for i in range(B):
        ls = np.nonzero(pairs[i])[0]
        if len(ls) == 0:
            continue
        v = np.nonzero(valid[i])[0]
        W = inst_labels[i][np.ix_(v, ls)].astype(np.int64)     # (n, m)
        # c[j, k] = #labels with j witness and k non-witness
        c = W @ (1 - W).T
        iu = np.triu_indices(len(v), 1)
        a, b = c[iu], c.T[iu]
        lo += np.minimum(a, b).sum()
        hi += np.maximum(a, b).sum()
        tot += (a + b).sum()
    if tot == 0:
        return float('nan'), float('nan')
    return float(lo / tot), float(hi / tot)


def chance_precision_at_1(inst_labels, bag_labels, mask=None):
    """Expected precision@1 of a model attending uniformly at random.

    Per scorable pair this is (#witnesses / #real instances); the metric is the
    mean over pairs. Any method scoring at this level has learned nothing about
    localization, whatever its bag-level accuracy.
    """
    inst_labels = _to_numpy(inst_labels).astype(bool)
    bag_labels = _to_numpy(bag_labels).astype(bool)
    if mask is None:
        valid = np.ones(inst_labels.shape[:2], dtype=bool)
    else:
        valid = ~_to_numpy(mask).astype(bool)

    pairs = _scorable_pairs(inst_labels, bag_labels, valid)
    if not pairs.any():
        return float('nan')

    real = valid[:, :, None] & inst_labels
    n_wit = real.sum(axis=1).astype(np.float64)               # (B, L)
    n_real = valid.sum(axis=1).astype(np.float64)[:, None]    # (B, 1)
    frac = n_wit / np.maximum(n_real, 1.0)
    return float(frac[pairs].mean())


def witness_disjointness(inst_labels, bag_labels, mask=None):
    """Fraction of co-occurring label pairs with *disjoint* witness sets.

    This is the quantity that drives `label_agnostic_ceiling`: when two labels
    in a bag share no witness instance, one shared attention ranking cannot be
    simultaneously correct for both.
    """
    inst_labels = _to_numpy(inst_labels).astype(bool)
    bag_labels = _to_numpy(bag_labels).astype(bool)
    if mask is None:
        valid = np.ones(inst_labels.shape[:2], dtype=bool)
    else:
        valid = ~_to_numpy(mask).astype(bool)

    total = 0
    disjoint = 0
    B = inst_labels.shape[0]
    for b in range(B):
        real = inst_labels[b][valid[b]]                       # (n_real, L)
        present = np.nonzero(bag_labels[b] & real.any(axis=0))[0]
        for i, l1 in enumerate(present):
            for l2 in present[i + 1:]:
                total += 1
                if not (real[:, l1] & real[:, l2]).any():
                    disjoint += 1
    if total == 0:
        return float('nan')
    return disjoint / total


def label_agnostic_ceiling(inst_labels, bag_labels, mask=None):
    """Exact upper bound on precision@1 for any *label-agnostic* method.

    A label-agnostic attention model (AttnMIL, GatedAttention, TransMIL and
    friends) computes one attention distribution per bag and broadcasts it to
    every label, so within a bag all labels share a single top-1 instance.
    That instance can only be a witness for the labels that actually contain
    it. The best case is therefore the label that maximises coverage:

        per bag:  max over instances i of (#scorable labels for which i is a
                  witness) / (#scorable labels in the bag)

    which for one-label-per-instance data reduces to 1/m with m labels present.
    The bound holds regardless of architecture, width, or training.
    """
    inst_labels = _to_numpy(inst_labels).astype(bool)
    bag_labels = _to_numpy(bag_labels).astype(bool)
    if mask is None:
        valid = np.ones(inst_labels.shape[:2], dtype=bool)
    else:
        valid = ~_to_numpy(mask).astype(bool)

    pairs = _scorable_pairs(inst_labels, bag_labels, valid)
    B = inst_labels.shape[0]
    num, den = 0.0, 0
    for b in range(B):
        cols = np.nonzero(pairs[b])[0]
        if len(cols) == 0:
            continue
        real = inst_labels[b][valid[b]][:, cols]              # (n_real, m)
        # best single instance covers this many of the m scorable labels
        best = int(real.sum(axis=1).max()) if real.size else 0
        num += best
        den += len(cols)
    if den == 0:
        return float('nan')
    return num / den


def all_instance_metrics(attn, inst_labels, bag_labels, mask=None,
                         topk=(1, 3)):
    """Full instance-level report for one set of predictions."""
    out = {}
    for k in topk:
        out[f'InstP@{k}'] = instance_precision_at_k(
            attn, inst_labels, bag_labels, mask=mask, k=k)
    out['InstMRR'] = instance_mrr(attn, inst_labels, bag_labels, mask=mask)
    out['InstAUC'] = instance_auc(attn, inst_labels, bag_labels, mask=mask)
    out['InstAUCPooled'] = instance_auc_pooled(
        attn, inst_labels, bag_labels, mask=mask)
    out['ChanceP@1'] = chance_precision_at_1(inst_labels, bag_labels, mask=mask)
    out['LabelAgnosticCeiling'] = label_agnostic_ceiling(
        inst_labels, bag_labels, mask=mask)
    return out
