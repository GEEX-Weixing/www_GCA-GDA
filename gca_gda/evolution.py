"""Stage 2: separate class-geometry displacement and graph contact changes.

Stats use full ORIGINAL graph views only (no augmentation-reliability factors).
Source training labels are the only labels in estimation; unlabeled source
and all target nodes use detached soft class assignments.
"""
import torch
import torch.nn.functional as F

EPS = 1e-8


def class_stats(h, q, eps=EPS, shrink=True):
    mass = q.sum(0)
    safe_mass = mass.clamp_min(eps)
    mu = q.T @ h / safe_mass[:, None]
    diff = h[None, :, :] - mu[:, None, :]
    cov = torch.einsum('nc,cni,cnj->cij', q, diff, diff) / safe_mass[:, None, None]
    if shrink:
        # Small-sample isotropic covariance shrinkage: deterministic, no search knob.
        # Effective N is insensitive to common scaling of soft class weights.
        n_eff = mass.square() / q.square().sum(0).clamp_min(eps)
        dim = h.shape[1]
        amount = dim / (n_eff + dim)
        iso = cov.diagonal(dim1=-2, dim2=-1).mean(-1)
        eye = torch.eye(dim, device=h.device, dtype=h.dtype)
        cov = ((1. - amount)[:, None, None] * cov
               + amount[:, None, None] * iso[:, None, None] * eye)
    return mass, mu, cov


def contact(q, edge, eps=EPS):
    """Paper-style Q^T D^-1/2 A D^-1/2 Q, normalized BEFORE diagonal masking.

    Return full row-normalized matrix: diagonal entries are kept for correct
    comparison of off-diagonal class contact and source/target changes.
    """
    n = q.shape[0]
    edge = edge[:, edge[0] != edge[1]]
    src, dst = edge
    deg = q.new_zeros(n).index_add_(0, src, q.new_ones(src.numel()))
    weight = deg[src].clamp_min(1).rsqrt() * deg[dst].clamp_min(1).rsqrt()
    aq = q.new_zeros(q.shape).index_add_(0, src, q[dst] * weight[:, None])
    b = q.T @ aq
    return b / b.sum(-1, keepdim=True).clamp_min(eps)


def estimate_evolution(hs, ht, qs, qt, es, et, *, eps=EPS):
    """Produce node direction, raw center shift, target centers and contact delta.

    A missing target class or degenerate geometry retains its source anchor;
    it is not interpreted as reliable zero movement.
    """
    _, mus, sigs = class_stats(hs, qs)
    target_mass, mut, sigt = class_stats(ht, qt)
    fs = sigs.flatten(1).norm(dim=-1).clamp_min(eps)
    ft = sigt.flatten(1).norm(dim=-1).clamp_min(eps)
    geometry = sigs / fs[:, None, None] + sigt / ft[:, None, None]
    shift = mut - mus
    node = torch.bmm(geometry, shift.unsqueeze(-1)).squeeze(-1)
    # A tiny pseudo-class with a collapsed center should not rotate its head.
    valid = ((target_mass >= 1.) & (mut.norm(dim=-1) > eps)
             & (mus.norm(dim=-1) > eps))
    node = torch.where(valid[:, None], node, torch.zeros_like(node))
    delta = contact(qt, et) - contact(qs, es)
    return dict(source_mu=mus, target_mu=mut, node=node, center_shift=shift,
                delta_contact=delta, target_mass=target_mass, valid=valid)


def structure_pairs(delta, valid=None, eps=EPS):
    """Symmetric class-pair importance, no arbitrary reliability coefficients."""
    scores = delta.abs() + delta.abs().T
    scores = torch.triu(scores, diagonal=1)
    if valid is not None:
        scores = scores * (valid[:, None] & valid[None, :]).to(scores.dtype)
    total = scores.sum()
    return scores / total.clamp_min(eps)
