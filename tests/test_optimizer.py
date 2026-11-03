"""Tests for the optimiser / LR schedule.

The regression these tests exist for: a scheduler wrapper that reads ``base_lr`` from
``optimizer.param_groups`` *after* ``LambdaLR.__init__`` has already multiplied it by
``start_factor``.  That silently trains with a 100x too small learning rate and is invisible
in every loss curve.  ``test_peak_lr_equals_configured_lr`` fails loudly on it.
"""

import math

import pytest
import torch
from torch import nn

from engine.optimizer import build_optimizer, build_param_groups, build_scheduler


def _model_and_opt(lr=2e-3, wd=0.05):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.GELU(), nn.Linear(16, 4))
    groups = build_param_groups(model, lr=lr, weight_decay=wd)
    return model, build_optimizer(groups)


def test_param_groups_split_decay_and_no_decay():
    _, opt = _model_and_opt()
    names = [g["name"] for g in opt.param_groups]
    print("groups:", names, [g["weight_decay"] for g in opt.param_groups])
    assert "decay" in names and "no_decay" in names
    wds = {g["name"]: g["weight_decay"] for g in opt.param_groups}
    assert wds["decay"] == 0.05 and wds["no_decay"] == 0.0


def test_peak_lr_equals_configured_lr():
    """THE regression test: the LR must actually reach ``lr`` at the end of warm-up."""
    lr = 2e-3
    _, opt = _model_and_opt(lr=lr)
    warmup, total = 10, 100
    sched = build_scheduler(opt, warmup_steps=warmup, total_steps=total, min_lr_ratio=0.5)

    lrs = []
    for step in range(total):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    peak = max(lrs)
    print(f"configured peak={lr:.3e}  observed peak={peak:.3e}  ratio={peak/lr:.4f}")
    assert peak == pytest.approx(lr, rel=1e-3), (
        f"scheduler peaked at {peak:.3e} but {lr:.3e} was configured "
        f"(ratio {peak/lr:.4f}) -- this is the base_lr capture bug"
    )


def test_final_lr_equals_min_ratio_times_peak():
    lr, ratio = 2e-3, 0.5
    _, opt = _model_and_opt(lr=lr)
    sched = build_scheduler(opt, warmup_steps=5, total_steps=200, min_lr_ratio=ratio)
    for _ in range(200):
        sched.step()
    final = opt.param_groups[0]["lr"]
    print(f"final={final:.3e}  expected={lr*ratio:.3e}")
    assert final == pytest.approx(lr * ratio, rel=1e-3)


def test_warmup_starts_at_start_factor():
    lr = 2e-3
    _, opt = _model_and_opt(lr=lr)
    build_scheduler(opt, warmup_steps=10, total_steps=100, start_factor=0.01)
    print("lr after init:", opt.param_groups[0]["lr"])
    assert opt.param_groups[0]["lr"] == pytest.approx(lr * 0.01, rel=1e-6)


def test_warmup_is_monotonic_and_cosine_decreases():
    _, opt = _model_and_opt(lr=1e-3)
    sched = build_scheduler(opt, warmup_steps=20, total_steps=200, min_lr_ratio=1e-3)
    lrs = []
    for _ in range(200):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    warm, decay = lrs[:20], lrs[20:]
    assert all(b >= a - 1e-12 for a, b in zip(warm, warm[1:])), "warm-up must be monotonic"
    assert all(b <= a + 1e-12 for a, b in zip(decay, decay[1:])), "cosine must decrease"
    # `_warmup_cosine_lambda` reaches 1.0 exactly at step == warmup_steps, so the peak lands
    # on lrs[warmup] (the (warmup+1)-th step), not lrs[warmup-1].
    assert lrs[20] == pytest.approx(max(lrs), rel=1e-6)
    assert max(lrs) == pytest.approx(1e-3, rel=1e-6)


def test_per_group_ratios():
    """The online probe keeps a constant LR while the encoder decays to `min_lr_ratio`."""
    lr_enc, lr_probe = 2e-3, 1e-3
    torch.manual_seed(0)
    enc = nn.Linear(8, 8)
    probe = nn.Linear(8, 4)
    groups = build_param_groups(enc, lr=lr_enc, weight_decay=0.05)
    pg = build_param_groups(probe, lr=lr_probe, weight_decay=1e-7)
    for g in pg:
        g["name"] = "probe_" + g["name"]
    opt = build_optimizer(groups + pg)
    sched = build_scheduler(
        opt, warmup_steps=5, total_steps=100, min_lr_ratio=0.5,
        group_min_lr_ratios={"probe_decay": 1.0, "probe_no_decay": 1.0},
    )
    for _ in range(100):
        sched.step()
    by_name = {g["name"]: g["lr"] for g in opt.param_groups}
    print("final lrs:", by_name)
    assert by_name["decay"] == pytest.approx(lr_enc * 0.5, rel=1e-3)
    assert by_name["probe_decay"] == pytest.approx(lr_probe * 1.0, rel=1e-3)


def test_scheduler_state_dict_roundtrip():
    _, opt = _model_and_opt(lr=1e-3)
    sched = build_scheduler(opt, warmup_steps=5, total_steps=50, min_lr_ratio=0.1)
    for _ in range(20):
        sched.step()
    lr_before = opt.param_groups[0]["lr"]
    sd = sched.state_dict()

    _, opt2 = _model_and_opt(lr=1e-3)
    sched2 = build_scheduler(opt2, warmup_steps=5, total_steps=50, min_lr_ratio=0.1)
    opt2.load_state_dict(opt.state_dict())
    sched2.load_state_dict(sd)
    print(f"before={lr_before:.6e} after_restore={opt2.param_groups[0]['lr']:.6e}")
    assert opt2.param_groups[0]["lr"] == pytest.approx(lr_before, rel=1e-6)
