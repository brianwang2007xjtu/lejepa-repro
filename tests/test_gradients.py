"""End-to-end gradient / numerical-stability tests on the real model.

These are the tests that catch shape bugs, silent broadcast errors, AMP problems and
stop-gradient regressions in the assembled system (backbone + projector + losses).

Run: ``pytest tests/test_gradients.py -v -s``
"""

import math

import pytest
import torch
from torch import nn

from losses.invariance import LeJEPAObjective, prediction_loss
from losses.sigreg import SIGReg
from models.projector import LeJEPAEncoder, Projector


# --------------------------------------------------------------------------- tiny model
def _tiny_encoder(embedding_dim=64, projector_dim=16, img_size=32, patch=8, depth=2):
    """3-layer projector + a tiny ViT-shaped stub (no timm dependency, CPU-friendly)."""
    from models.backbone import TimmBackbone

    bb = TimmBackbone("vit_tiny_patch16_224", embedding_dim=embedding_dim, img_size=img_size)
    # ViT-tiny at 32 px with patch 16 => 4 patches; small and fast on CPU
    proj = Projector(
        embedding_dim=embedding_dim,
        projector_dim=projector_dim,
        hidden_dims=[128, 128],
        hidden_norm="bn",
        final_norm="bn",
        final_act="gelu",
    )
    return LeJEPAEncoder(bb, proj)


def _require_timm():
    pytest.importorskip("timm")


# --------------------------------------------------------------------------- forward shapes
def test_forward_shapes():
    _require_timm()
    enc = _tiny_encoder()
    n, v = 4, 4
    x = torch.randn(n, v, 3, 32, 32)
    emb, proj = enc(x)
    print(f"emb={tuple(emb.shape)} proj={tuple(proj.shape)}")
    assert emb.shape == (v, n, 64), f"emb must be (V, B, E), got {tuple(emb.shape)}"
    assert proj.shape == (v, n, 16)


def test_view_major_layout_is_preserved():
    """``proj[v, n]`` must be the embedding of view v of sample n (Algorithm 2's layout)."""
    _require_timm()
    enc = _tiny_encoder()
    enc.eval()
    n, v = 2, 3
    x = torch.randn(n, v, 3, 32, 32)
    _, proj = enc(x)
    with torch.no_grad():
        for vv in range(v):
            for nn_ in range(n):
                single = enc(x[nn_ : nn_ + 1, vv : vv + 1])[1]
                assert torch.allclose(proj[vv, nn_], single[0, 0], atol=1e-5), (
                    f"proj[{vv},{nn_}] does not equal the embedding of view {vv} sample {nn_}"
                )


# --------------------------------------------------------------------------- gradients
def test_all_parameters_receive_gradient():
    _require_timm()
    torch.manual_seed(0)
    enc = _tiny_encoder()
    sig = SIGReg(num_slices=32, seed=0, t_max=5.0, n_points=17)
    obj = LeJEPAObjective(sig, lambd=0.05, n_global_views=2)

    x = torch.randn(4, 4, 3, 32, 32)
    _, proj = enc(x)
    loss = obj(proj)
    loss.backward()

    missing = [n for n, p in enc.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    print(f"params with zero/missing grad: {len(missing)} / {sum(1 for _ in enc.parameters())}")
    if missing:
        print("  ", missing[:10])
    # the projector's final linear layer must always be reached
    assert any("projector.net" in n for n, p in enc.named_parameters() if p.grad is not None)
    assert len(missing) == 0, f"{len(missing)} parameters received no gradient: {missing[:10]}"


def test_gradient_flows_to_every_view():
    _require_timm()
    torch.manual_seed(0)
    enc = _tiny_encoder()
    sig = SIGReg(num_slices=32, seed=0)
    obj = LeJEPAObjective(sig, lambd=0.05, n_global_views=2)
    x = torch.randn(3, 4, 3, 32, 32, requires_grad=False)
    _, proj = enc(x)
    loss = obj(proj)
    (g,) = torch.autograd.grad(loss, proj)
    per_view = g.abs().sum(dim=(1, 2))
    print("per-view grad mass:", per_view.tolist())
    assert (per_view > 0).all(), "a stop-gradient on any view would zero this"


def test_no_nan_no_inf_in_loss_and_grads():
    _require_timm()
    torch.manual_seed(0)
    enc = _tiny_encoder()
    sig = SIGReg(num_slices=64, seed=0)
    obj = LeJEPAObjective(sig, lambd=0.05, n_global_views=2)
    x = torch.randn(4, 4, 3, 32, 32)
    _, proj = enc(x)
    loss = obj(proj)
    loss.backward()
    assert torch.isfinite(loss)
    for n, p in enc.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite gradient in {n}"


# --------------------------------------------------------------------------- batch size
@pytest.mark.parametrize("batch", [2, 8, 32, 128])
def test_batch_size_stability(batch):
    """The loss must be finite and O(1) for every batch size, including 1."""
    torch.manual_seed(0)
    sig = SIGReg(num_slices=64, seed=0)
    z = torch.randn(4, batch, 32)
    loss = sig(z)
    print(f"  B={batch:3d}: SIGReg={loss.item():.4f}")
    assert torch.isfinite(loss)
    assert 0.0 < loss.item() < 50.0


def test_batch_size_one_is_rejected_with_a_clear_error():
    """With N=1 the ECF is |e^{itx}| = 1 for every x, so the statistic is a *constant* that
    does not depend on the embedding and its gradient is identically zero: SIGReg silently
    becomes a no-op and the encoder is free to collapse.

    Rather than returning a silently useless value we raise.  This is why the paper requires
    a batch of at least a few dozen samples (and recommends >= 128).
    """
    sig = SIGReg(num_slices=16, seed=0)
    with pytest.raises(ValueError, match="at least 2 samples"):
        sig(torch.randn(1, 32))
    print("B=1 correctly rejected: a single sample carries no distributional information")


# --------------------------------------------------------------------------- precision
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_bf16_vs_fp32_forward():
    _require_timm()
    torch.manual_seed(0)
    enc = _tiny_encoder().cuda().eval()
    x = torch.randn(4, 4, 3, 32, 32, device="cuda")
    with torch.no_grad():
        _, p32 = enc(x.float())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, p16 = enc(x)
    rel = ((p16.float() - p32).abs() / p32.abs().clamp_min(1e-3)).mean()
    print(f"bf16 vs fp32 relative difference = {rel.item():.3%}")
    assert rel.item() < 0.05


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_amp_training_step_is_stable():
    _require_timm()
    torch.manual_seed(0)
    enc = _tiny_encoder().cuda()
    sig = SIGReg(num_slices=64, seed=0).cuda()
    obj = LeJEPAObjective(sig, lambd=0.05, n_global_views=2).cuda()
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-3)
    x = torch.randn(4, 4, 3, 32, 32, device="cuda")

    losses = []
    for _ in range(5):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, proj = enc(x)
            loss = obj(proj)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print("bf16 losses:", [f"{v:.4f}" for v in losses])
    assert all(math.isfinite(v) for v in losses)


# --------------------------------------------------------------------------- collapse
def test_sigreg_prevents_collapse_in_a_controlled_optimisation():
    """Directly optimise the projector output of a frozen random encoder.

    Without SIGReg the prediction loss alone is minimised by mapping every view of every
    sample to the same point.  With SIGReg the embedding variance must stay non-zero.  This
    isolates the anti-collapse mechanism from any architectural effect.
    """
    torch.manual_seed(0)
    n, v, k = 64, 4, 32
    feat = torch.randn(n * v, 64, dtype=torch.float64)  # frozen "backbone features"

    def run(with_sigreg: bool):
        proj = nn.Linear(64, k).to(torch.float64)
        torch.manual_seed(1)
        nn.init.xavier_uniform_(proj.weight)
        sig = SIGReg(num_slices=128, seed=0)
        opt = torch.optim.AdamW(proj.parameters(), lr=1e-2)
        for _ in range(300):
            z = proj(feat).reshape(n, v, k).transpose(0, 1)  # (V, N, K)
            pred = prediction_loss(z, n_global_views=v)
            loss = pred + 0.05 * sig(z) if with_sigreg else pred
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            z = proj(feat).reshape(n, v, k).transpose(0, 1)
            return pred.item(), z.reshape(-1, k).std(dim=0).mean().item()

    pred_only, std_only = run(False)
    pred_sig, std_sig = run(True)
    print(f"prediction loss only : pred={pred_only:.6f}  feature_std={std_only:.6f}")
    print(f"prediction + SIGReg  : pred={pred_sig:.6f}  feature_std={std_sig:.6f}")
    assert std_only < 1e-3, "without SIGReg the embeddings must collapse"
    assert std_sig > 0.3, "SIGReg must keep the embedding scale near 1"
