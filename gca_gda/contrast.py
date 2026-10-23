"""Reliability-tempered false-negative corrected contrastive learning.

Stage 1 keeps the original sampled instance-level InfoNCE objective as its
backbone.  Class-prediction information is detached and is used ONLY to reduce
the competition from likely false negatives; it never creates extra positives
and never contributes a semantic cross-entropy term.

Important invariance:
    if r_i r_j -> 0, then omega_ij -> 1 and the loss becomes EXACTLY the
    original sampled InfoNCE used by this implementation.
"""
import math
import torch
import torch.nn.functional as F


def references(logit0, logit1, logit2, label=None, labeled_mask=None):
    """Build detached class reference q and detached reliability r.

    q is taken from the ORIGINAL, unaugmented graph prediction:
        q_i = sg(softmax(logit0_i)).

    The two augmented views are used only to assess prediction stability via
    Jensen-Shannon divergence.  Reliability combines original-view confidence
    and augmented-view stability:

        r_i = (1 - H(q_i)/log C) * (1 - JS(p1_i,p2_i)/log 2).

    For labeled SOURCE training nodes, ground-truth class information is known,
    so q is replaced by the one-hot label and r is set to one.  Callers must
    never pass source validation labels or target labels here.
    """
    p0 = logit0.softmax(-1)
    p1 = logit1.softmax(-1)
    p2 = logit2.softmax(-1)

    with torch.no_grad():
        q = p0.detach().clone()
        c = q.shape[1]
        if c < 2:
            raise ValueError('RT-JCL requires at least two classes')

        entropy = -(q * q.clamp_min(1e-12).log()).sum(-1)
        p1d, p2d = p1.detach(), p2.detach()
        m = .5 * (p1d + p2d)
        js = .5 * (
            (p1d * (p1d.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1)
            + (p2d * (p2d.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1)
        )
        confidence = (1. - entropy / math.log(c)).clamp(0., 1.)
        stability = (1. - js / math.log(2.)).clamp(0., 1.)
        r = confidence * stability

        if labeled_mask is not None and bool(labeled_mask.any()):
            if label is None:
                raise ValueError('labels required for labeled source nodes')
            q[labeled_mask] = F.one_hot(
                label[labeled_mask].long(), c).to(q.dtype)
            r = r.clone()
            r[labeled_mask] = 1.

    return q.detach(), r.detach()


def _log_negative_weight(q, r, idx, indices, identity, enabled=True):
    """Return log omega_ij for sampled anchor/candidate pairs.

        a_ij     = sg[r_i r_j q_i^T q_j]
        omega_ij = 1 - a_ij

    The cross-view matched positive (j=i) is explicitly left at weight 1.
    q and r are detached, so this correction never backpropagates through the
    classifier predictions/reliability estimates.
    """
    if not enabled:
        return q.new_zeros((idx.numel(), indices.numel())), None, None

    affinity = (
        r[idx, None] * r[indices][None, :]
        * (q[idx] @ q[indices].T)
    ).detach().clamp(0., 1.)
    affinity = affinity.masked_fill(identity, 0.)
    weight = (1. - affinity).clamp(0., 1.)

    # Exact zero means the corresponding negative contributes zero mass to the
    # denominator.  torch.where keeps this mathematically exact instead of
    # replacing zero by an arbitrary epsilon floor.
    neg_inf = torch.full_like(weight, -torch.inf)
    log_weight = torch.where(weight > 0., weight.log(), neg_inf)
    return log_weight, affinity, weight


def rt_jcl(h1, h2, q, r, *, anchors=512, candidates=1024, tau=.5,
           indices=None, use_false_negative_correction=True,
           return_stats=False):
    """Symmetric sampled InfoNCE with reliability-tempered false negatives.

    Positive pairs are ONLY the same node under two graph augmentations.  Class
    information never creates a positive pair and never adds a semantic loss.

    For anchor i and candidate j:
        a_ij     = sg[r_i r_j q_i^T q_j]
        omega_ij = 1 - a_ij.

    omega_ij multiplies only j!=i negative terms in both cross-view and same-view
    competition.  Therefore, whenever r_i r_j = 0 for all sampled negative
    pairs, omega_ij = 1 and this function is exactly ordinary sampled InfoNCE.
    """
    n = h1.shape[0]
    if q.shape[0] != n or r.shape[0] != n:
        raise ValueError('h/q/r must have the same number of nodes')
    if indices is None:
        indices = torch.randperm(n, device=h1.device)[:min(n, int(candidates))]
    if indices.numel() == 0:
        zero = (h1.sum() + h2.sum()) * 0.
        return (zero, {}) if return_stats else zero

    idx = indices[:min(int(anchors), indices.numel())]
    q = q.detach()
    r = r.detach()
    identity = idx[:, None] == indices[None, :]

    z1 = F.normalize(h1, dim=-1)
    z2 = F.normalize(h2, dim=-1)
    log_weight, affinity, weight = _log_negative_weight(
        q, r, idx, indices, identity,
        enabled=use_false_negative_correction)

    def one_way(za, zb):
        anchor = za[idx]
        cross = anchor @ zb[indices].T / tau
        same = anchor @ za[indices].T / tau
        pos = (za[idx] * zb[idx]).sum(-1) / tau

        # Cross-view matrix contains the true matched positive.  identity has
        # log_weight=0, so that positive stays unmodified.  All j!=i entries
        # receive omega_ij.  Same-view identity is not a valid candidate.
        cross = cross + log_weight
        same = (same + log_weight).masked_fill(identity, -torch.inf)
        denominator = torch.logsumexp(torch.cat([cross, same], dim=1), dim=1)
        return (denominator - pos).mean()

    loss = .5 * (one_way(z1, z2) + one_way(z2, z1))
    if not return_stats:
        return loss

    with torch.no_grad():
        nonself = ~identity
        if bool(nonself.any()) and weight is not None:
            pair_aff = affinity[nonself]
            pair_w = weight[nonself]
            stats = dict(
                reliability_mean=float(r.mean().item()),
                reliability_std=float(r.std(unbiased=False).item()),
                affinity_mean=float(pair_aff.mean().item()),
                affinity_max=float(pair_aff.max().item()),
                negative_weight_mean=float(pair_w.mean().item()),
                negative_weight_min=float(pair_w.min().item()),
                strong_suppression_fraction=float((pair_w < .5).float().mean().item()),
            )
        else:
            stats = dict(
                reliability_mean=float(r.mean().item()),
                reliability_std=float(r.std(unbiased=False).item()),
                affinity_mean=0., affinity_max=0.,
                negative_weight_mean=1., negative_weight_min=1.,
                strong_suppression_fraction=0.,
            )
    return loss, stats
