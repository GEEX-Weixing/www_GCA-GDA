"""GCA-GDA three-stage training using full source supervision.

Stage 1: joint source CE and reliability-tempered instance contrast (RT-JCL)
         for T_w epochs, with alpha=min(1,t/T_r); after warm-up, encoder-only.
Stage 2: original-graph class geometry and contact evolution, no gradients.
Stage 3: periodically transport and refine the classifier from fixed W^0,
         subject to the frozen-source-embedding CE feasibility constraint.

Target labels are never used in representation learning, evolution estimation,
classifier adjustment, epoch selection, or hyperparameter selection.
"""
from __future__ import annotations

import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_graph
from .model import CosineModel
from .contrast import references, rt_jcl
from .evolution import estimate_evolution
from .geometry import adjust_classifier


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def augment(x, edges, edge_drop, feature_mask):
    edges_kept = torch.rand(edges.size(1), device=edges.device) > edge_drop
    features_kept = (torch.rand(1, x.size(1), device=x.device) > feature_mask).to(x.dtype)
    return x * features_kept, edges[:, edges_kept]


def default_config():
    """Public example defaults, NOT claimed task-optimal HPO settings."""
    return dict(
        hidden_dim=128, dropout=0.5, scale=24.0,
        lr=0.0015, weight_decay=0.0005,
        epochs=400, warmup=100, ramp=40,
        lambda_j=0.1, tau=1.0,
        view1_edge=0.3, view1_feature=0.3,
        view2_edge=0.5, view2_feature=0.5,
        contrast_anchors=512, contrast_candidates=1024,
        stage3_every=5, stage3_steps=15, stage3_lr=0.02,
        source_ce_slack=0.1,
        require_pyg=False,
    )


def _final_metrics(logits, labels, mask):
    """Diagnostic metrics called only after all parameter updates finish."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    if not bool(mask.any()):
        raise ValueError('No labeled target evaluation nodes')
    predictions = logits[mask].argmax(dim=1).cpu().numpy()
    references = labels[mask].cpu().numpy()
    return dict(
        acc=100.0 * float(accuracy_score(references, predictions)),
        macro_f1=100.0 * float(f1_score(references, predictions, average='macro', zero_division=0)),
        micro_f1=100.0 * float(f1_score(references, predictions, average='micro', zero_division=0)),
        balanced_acc=100.0 * float(balanced_accuracy_score(references, predictions)),
    )


def _representation_objective(model, src, tgt, cfg, epoch):
    # Supervised source CE on the FULL labeled source graph.
    source_logits, _ = model(src.x, src.edge_index)
    ce = F.cross_entropy(source_logits[src.train_mask], src.y[src.train_mask])
    rt_losses = []
    for graph, source_domain in ((src, True), (tgt, False)):
        # The original-graph reference must not backpropagate class predictions.
        was_training = model.training
        model.eval()
        with torch.no_grad():
            logit0, _ = model(graph.x, graph.edge_index)
        model.train(was_training)
        x1, e1 = augment(graph.x, graph.edge_index, cfg['view1_edge'], cfg['view1_feature'])
        x2, e2 = augment(graph.x, graph.edge_index, cfg['view2_edge'], cfg['view2_feature'])
        logit1, h1 = model(x1, e1)
        logit2, h2 = model(x2, e2)
        q, r = references(
            logit0, logit1, logit2,
            graph.y if source_domain else None,
            graph.train_mask if source_domain else None,
        )
        rt_losses.append(rt_jcl(
            h1, h2, q, r,
            anchors=cfg['contrast_anchors'], candidates=cfg['contrast_candidates'],
            tau=cfg['tau'],
        ))
    alpha = min(1.0, epoch / cfg['ramp'])
    rt = 0.5 * (rt_losses[0] + rt_losses[1])
    total = ce + cfg['lambda_j'] * alpha * rt
    return total, ce.detach(), rt.detach(), alpha


def train_one(family, source, target, seed, config=None,
              data_dir='data', device=None, report_target=False, verbose=False):
    """Train one fixed-configuration source->target task without any HPO.

    All source nodes with known labels are supervised. The model is never
    selected using target labels. Final target metrics are optional and only
    evaluated after the full T epochs.
    """
    cfg = default_config()
    overrides = config or {}
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise ValueError(f'Unknown hyperparameter(s): {sorted(unknown)}')
    cfg.update(overrides)
    if not (1 <= cfg['warmup'] < cfg['epochs']):
        raise ValueError('Require 1 <= warmup < epochs')
    if cfg['ramp'] < 1 or cfg['stage3_every'] < 1 or cfg['stage3_steps'] < 1:
        raise ValueError('ramp, stage3_every, stage3_steps must be positive')
    if cfg['tau'] <= 0 or cfg['scale'] <= 0 or cfg['source_ce_slack'] < 0:
        raise ValueError('Require tau > 0, scale > 0, source_ce_slack >= 0')
    seed_everything(seed)
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    src = load_graph(family, source, data_dir, seed, role='source').to(dev)
    tgt = load_graph(family, target, data_dir, seed + 10000, role='target').to(dev)
    if src.x.size(1) != tgt.x.size(1):
        raise ValueError('Source and target must have matching feature dimensions')
    known = src.y[src.train_mask]
    if known.numel() == 0:
        raise ValueError('Source graph contains no labeled training nodes')
    num_classes = int(known.max().item()) + 1
    if set(known.detach().cpu().tolist()) != set(range(num_classes)):
        raise ValueError('Source class IDs must be contiguous 0..C-1')
    model = CosineModel(src.x.size(1), num_classes, cfg['hidden_dim'],
                        cfg['dropout'], cfg['scale'], cfg['require_pyg']).to(dev)
    warm_optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                                       weight_decay=cfg['weight_decay'])
    encoder_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=cfg['lr'],
                                         weight_decay=cfg['weight_decay'])
    source_anchor = None
    start = time.perf_counter()
    last = None
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        warm_stage = epoch <= cfg['warmup']
        model.classifier.requires_grad_(warm_stage)
        optimizer = warm_optimizer if warm_stage else encoder_optimizer
        optimizer.zero_grad(set_to_none=True)
        loss, source_ce, rt, alpha = _representation_objective(model, src, tgt, cfg, epoch)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'Non-finite training loss at epoch {epoch}')
        loss.backward()
        optimizer.step()
        last = dict(epoch=epoch, loss=float(loss.item()),
                    source_ce=float(source_ce.item()),
                    rt_jcl=float(rt.item()), alpha=float(alpha))
        if epoch == cfg['warmup']:
            # A single frozen source-supervised reference used for ALL proposals.
            source_anchor = model.classifier.detach().clone()
        if not warm_stage and ((epoch-cfg['warmup']) % cfg['stage3_every'] == 0
                               or epoch == cfg['epochs']):
            model.eval()
            # Stage 2: original-graph representations; full source labels,
            # detached target soft assignments, and NO target true labels.
            with torch.no_grad():
                logits_s, hs = model(src.x, src.edge_index)
                logits_t, ht = model(tgt.x, tgt.edge_index)
                q_s = logits_s.softmax(dim=-1).detach().clone()
                q_s[src.train_mask] = F.one_hot(
                    src.y[src.train_mask].long(), num_classes).to(q_s.dtype)
                q_t = logits_t.softmax(dim=-1).detach()
                evolution = estimate_evolution(hs, ht, q_s, q_t,
                                               src.edge_index, tgt.edge_index)
            # Stage 3: classifier adjusted against fixed W0 and source CE guard.
            updated = adjust_classifier(
                source_anchor, hs, src.y, src.train_mask, evolution,
                scale=cfg['scale'], source_ce_slack=cfg['source_ce_slack'],
                steps=cfg['stage3_steps'], lr=cfg['stage3_lr'],
            )
            with torch.no_grad():
                model.classifier.copy_(updated)
            last['stage3_refresh'] = True
        if verbose and (epoch == 1 or epoch == cfg['warmup']
                        or epoch % 25 == 0 or epoch == cfg['epochs']):
            print(f'epoch={epoch:04d} loss={last["loss"]:.5f} '
                  f'src_ce={last["source_ce"]:.5f} '
                  f'rt_jcl={last["rt_jcl"]:.5f} alpha={alpha:.3f}', flush=True)
    model.eval()
    metrics = None
    if report_target:
        # Only the fixed final-epoch model is evaluated; no epoch selection.
        with torch.no_grad():
            final_logits, _ = model(tgt.x, tgt.edge_index)
        metrics = _final_metrics(final_logits, tgt.y, tgt.eval_mask)
    return dict(
        family=family, source=source, target=target, seed=int(seed), config=cfg,
        final_epoch=int(cfg['epochs']), target_metrics=metrics,
        last_training=last, elapsed_seconds=float(time.perf_counter() - start),
    )
