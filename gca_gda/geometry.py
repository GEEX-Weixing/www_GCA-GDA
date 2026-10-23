"""Stage 3: node-proposed rotation + pair-aware joint unit-sphere fitting.

Source CE is a feasibility constraint on CURRENT frozen embeddings, not an
extra independently weighted target or proximal loss. No target labels used.
"""
import torch
import torch.nn.functional as F
from .evolution import structure_pairs


def unit(x, eps=1e-8):
    return F.normalize(x, dim=-1, eps=eps)


@torch.no_grad()
def minimal_rotated_weights(source_weight, source_mu, node_direction, valid=None, eps=1e-8):
    """Transport source classifier by smallest rotation taking old center to new.

    Proper rotation in the plane span(a,b); unchanged if reference is
    degenerate or nearly antipodal (ambiguous minimal rotation plane).
    """
    ws = unit(source_weight)
    a = unit(source_mu)
    b_raw = source_mu + node_direction
    b = unit(b_raw)
    result = ws.clone()
    if valid is None:
        valid = torch.ones(ws.shape[0], device=ws.device, dtype=torch.bool)
    for c in range(ws.shape[0]):
        if not bool(valid[c]) or source_mu[c].norm() <= eps or b_raw[c].norm() <= eps:
            continue
        dot = (a[c] * b[c]).sum().clamp(-1, 1)
        # Near-zero or antipodal displacement has no unique stable plane.
        if dot > 1 - 1e-6 or dot < -1 + 1e-5:
            continue
        basis = b[c] - dot * a[c]
        sn = basis.norm()
        if sn < eps:
            continue
        tangent = basis / sn
        # Rotate BOTH a and any classifier w by the same planar rotation.
        wc = ws[c]
        parallel_a = (wc * a[c]).sum()
        parallel_t = (wc * tangent).sum()
        rotated = (wc + (dot - 1) * (parallel_a * a[c] + parallel_t * tangent)
                   + sn * (parallel_a * tangent - parallel_t * a[c]))
        result[c] = unit(rotated)
    return result


def joint_geometry_loss(weight, proposed, target_mu, pair_importance,
                        *, valid=None, scale=16.):
    """One joint objective: node-proposal fit + changed-pair symmetric margins.

    Pair relation term is absent when graph contact has no observable change.
    """
    w = unit(weight)
    fit = (w - unit(proposed)).square().sum(-1).mean()
    if pair_importance.sum().item() <= 1e-12:
        return fit
    target_dir = unit(target_mu)
    logits_centers = scale * target_dir @ w.T
    margin = logits_centers.diag()[:, None] - logits_centers
    # For pair c,k assess BOTH target centers, not just c's margin.
    bidirectional = F.softplus(-margin) + F.softplus(-margin.T)
    pair = (pair_importance * bidirectional).sum()
    return fit + pair


@torch.no_grad()
def make_feasible(candidate, reference, source_h, source_y, source_mask,
                  *, scale, allowed_ce):
    """Geodesic-chord backtracking to keep source CE under specified guard.

    Reference is always feasible because allowed_ce = CE(reference)+epsilon.
    """
    candidate, reference = unit(candidate), unit(reference)
    labels = source_y[source_mask]
    h = source_h[source_mask]
    def loss(w):
        return F.cross_entropy(scale * h @ w.T, labels)
    if bool(loss(candidate) <= allowed_ce):
        return candidate
    lo, hi = 0., 1.
    for _ in range(24):
        mid = (lo + hi) / 2.
        trial = unit(torch.lerp(reference, candidate, mid))
        if bool(loss(trial) <= allowed_ce):
            lo = mid
        else:
            hi = mid
    return unit(torch.lerp(reference, candidate, lo))


def adjust_classifier(source_weight, source_h, source_y, source_train_mask,
                      evolution, *, scale, source_ce_slack=.02, steps=15, lr=.02):
    """Constrained Stage 3 adjustment, no extra trainable fusion/gate/EMA.

    Returns accepted unit-normalized heads; class centers and q are detached by
    the caller. All classes optimized jointly so rotation order cannot matter.
    """
    with torch.no_grad():
        anchor = unit(source_weight.detach())
        proposal = minimal_rotated_weights(anchor, evolution['source_mu'],
                                           evolution['node'], evolution['valid'])
        pairs = structure_pairs(evolution['delta_contact'], evolution['valid']).detach()
        target = evolution['target_mu'].detach()
        h = source_h.detach()
        labels = source_y[source_train_mask]
        base_ce = F.cross_entropy(scale * h[source_train_mask] @ anchor.T, labels)
        ceiling = base_ce + source_ce_slack
        proposal = make_feasible(proposal, anchor, h, source_y, source_train_mask,
                                 scale=scale, allowed_ce=ceiling)
    w = torch.nn.Parameter(proposal.clone())
    optim = torch.optim.Adam([w], lr=lr)
    with torch.no_grad():
        best = proposal.clone()
        best_score = joint_geometry_loss(best, proposal, target, pairs, scale=scale).item()
        # Source anchor is a feasible fallback when the proposal is poor.
        score_anchor = joint_geometry_loss(anchor, proposal, target, pairs, scale=scale).item()
        if score_anchor < best_score:
            best, best_score = anchor.clone(), score_anchor
    for _ in range(int(steps)):
        optim.zero_grad(set_to_none=True)
        objective = joint_geometry_loss(w, proposal, target, pairs, scale=scale)
        if not bool(torch.isfinite(objective)):
            break
        objective.backward()
        optim.step()
        with torch.no_grad():
            candidate = make_feasible(w.detach(), anchor, h, source_y, source_train_mask,
                                      scale=scale, allowed_ce=ceiling)
            w.copy_(candidate)
            score = joint_geometry_loss(candidate, proposal, target, pairs, scale=scale).item()
            if score < best_score:
                best, best_score = candidate.clone(), score
    return unit(best.detach())
