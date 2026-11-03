"""Frozen-backbone evaluation: feature extraction, linear probe, k-NN.

Two protocols are implemented because the paper uses both:

**Linear probe (the paper's main metric).**
From the official README: *"Feature Extraction: Concatenation of the CLS token from the last
two layers … We apply LayerNorm on the concatenated CLS tokens … Optimizer: AdamW … Weight
Decay: 1e-6 … Learning Rate Schedule: same as pre-training."*  This is
``feature_source="last2cls"`` below.

**Online probe (``MINIMAL.md``'s metric, the 90.7 % Imagenette number).**
A ``LayerNorm + Linear`` head trained jointly on ``emb.detach()`` during pretraining.  It is
implemented in ``engine/trainer.py`` and reported as ``eval/online_probe_acc``.

``feature_source="embedding"`` probes the backbone's ``embedding_dim`` output (what
``MINIMAL.md`` probes) and is provided to separate "encoder quality" from "read-out choice".

k-NN follows the standard SSL protocol (weighted k-NN, cosine similarity, k=20), which is the
non-parametric counterpart used by DINO/DINOv2 and by the Lightly benchmark referenced in
``MINIMAL.md``.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["extract_features", "linear_probe", "knn_classifier", "run_evaluation"]


@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
    feature_source: str = "last2cls",
    amp_dtype: Optional[torch.dtype] = torch.bfloat16,
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen backbone over ``loader`` and return ``(features, labels)`` on CPU.

    Features are accumulated in float32 on the CPU so that a full ImageNet-1K feature matrix
    (1.28M x 1536 floats = 7.9 GB) fits in host RAM without ever touching GPU memory twice.
    """
    backbone.eval()
    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        if isinstance(x, (list, tuple)):
            x = x[0]  # first (global) view
        x = x.to(device, non_blocking=True)
        if x.ndim == 5:  # (B, V, C, H, W) -> use the first (global) view
            x = x[:, 0]
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
        ):
            if feature_source == "last2cls":
                f = backbone.forward_features(x)
            elif feature_source == "embedding":
                f = backbone(x)
            else:
                raise ValueError(f"unknown feature_source {feature_source!r}")
        feats.append(f.float().cpu())
        labels.append(y.cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def _accuracy(logits: torch.Tensor, y: torch.Tensor, topk=(1, 5)) -> Dict[str, float]:
    out = {}
    maxk = min(max(topk), logits.size(1))
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(y.view(1, -1).expand_as(pred))
    for k in topk:
        kk = min(k, maxk)
        out[f"top{k}"] = correct[:kk].reshape(-1).float().sum().item() / y.numel()
    return out


def linear_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    num_classes: int,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    warmup_epochs: int = 1,
    min_lr_ratio: float = 1e-3,
    normalize: bool = True,
    verbose: bool = True,
) -> Dict[str, float]:
    """Train a linear classifier on frozen features and report top-1/top-5.

    ``normalize=True`` inserts the LayerNorm the README prescribes on the concatenated CLS
    tokens; because the LayerNorm is *inside* the trained probe, its affine parameters are
    learned jointly with the linear layer, exactly as in the reference protocol.
    """
    d = train_x.size(1)
    layers: List[nn.Module] = []
    if normalize:
        layers.append(nn.LayerNorm(d, eps=1e-6))
    layers.append(nn.Linear(d, num_classes))
    clf = nn.Sequential(*layers).to(device)

    ds = TensorDataset(train_x, train_y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * epochs
    warmup = steps_per_epoch * warmup_epochs

    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)
    from engine.optimizer import build_scheduler

    sched = build_scheduler(
        opt, warmup_steps=warmup, total_steps=total_steps,
        min_lr_ratio=min_lr_ratio, start_factor=0.01,
    )

    val_x_d, val_y_d = val_x.to(device), val_y.to(device)
    best = {"top1": 0.0, "top5": 0.0}
    t0 = time.time()
    clf.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = clf(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if (ep + 1) % max(1, epochs // 10) == 0 or ep == epochs - 1:
            clf.eval()
            with torch.no_grad():
                acc = _accuracy(clf(val_x_d), val_y_d)
            clf.train()
            best = acc if acc["top1"] > best["top1"] else best
            if verbose:
                print(f"    [probe] epoch {ep+1:3d}/{epochs} loss={loss.item():.4f} "
                      f"top1={acc['top1']*100:.2f}% top5={acc['top5']*100:.2f}%")
    clf.eval()
    with torch.no_grad():
        final = _accuracy(clf(val_x_d), val_y_d)
    elapsed = time.time() - t0
    if verbose:
        print(f"    [probe] final top1={final['top1']*100:.2f}% "
              f"best top1={max(best['top1'], final['top1'])*100:.2f}% ({elapsed:.1f}s)")
    return {
        "linear_top1": max(best["top1"], final["top1"]) * 100,
        "linear_top5": max(best["top5"], final["top5"]) * 100,
        "linear_final_top1": final["top1"] * 100,
        "linear_final_top5": final["top5"] * 100,
        "linear_probe_seconds": elapsed,
        "linear_probe_epochs": epochs,
    }


@torch.no_grad()
def knn_classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    num_classes: int,
    k: int = 20,
    temperature: float = 0.07,
    device: torch.device = torch.device("cpu"),
    chunk: int = 512,
) -> Dict[str, float]:
    """Weighted k-NN with cosine similarity (DINO/DINOv2 protocol)."""
    t0 = time.time()
    tr = torch.nn.functional.normalize(train_x.to(device).float(), dim=1)
    va = torch.nn.functional.normalize(val_x.to(device).float(), dim=1)
    tr_y = train_y.to(device)

    correct = 0
    total = 0
    for i in range(0, va.size(0), chunk):
        sim = va[i : i + chunk] @ tr.t()  # (c, N_train)
        topk = sim.topk(k, dim=1).indices  # (c, k)
        w = (sim.gather(1, topk) / temperature).exp()  # (c, k)
        votes = torch.zeros(va[i : i + chunk].size(0), num_classes, device=device)
        votes.scatter_add_(1, tr_y[topk], w)
        pred = votes.argmax(1)
        correct += (pred == val_y[i : i + chunk].to(device)).sum().item()
        total += pred.numel()
    acc = correct / max(1, total)
    return {"knn_acc": acc * 100, "knn_k": k, "knn_seconds": time.time() - t0}


def run_evaluation(
    backbone: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    feature_source: str = "last2cls",
    probe_epochs: int = 100,
    probe_lr: float = 1e-3,
    knn_k: int = 20,
    max_batches: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Extract features once, then run both the linear probe and k-NN."""
    t0 = time.time()
    tr_x, tr_y = extract_features(backbone, train_loader, device, feature_source, max_batches=max_batches)
    va_x, va_y = extract_features(backbone, val_loader, device, feature_source, max_batches=max_batches)
    extract_s = time.time() - t0
    if verbose:
        print(f"  features: train={tuple(tr_x.shape)} val={tuple(va_x.shape)} ({extract_s:.1f}s)")

    out: Dict[str, float] = {
        "feat_dim": tr_x.size(1),
        "feature_extract_seconds": extract_s,
    }
    out.update(
        linear_probe(
            tr_x, tr_y, va_x, va_y, num_classes, device,
            epochs=probe_epochs, lr=probe_lr, verbose=verbose,
        )
    )
    out.update(
        knn_classifier(tr_x, tr_y, va_x, va_y, num_classes, k=knn_k, device=device)
    )
    return out
