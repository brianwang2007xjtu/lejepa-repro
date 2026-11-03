"""Training engine for LeJEPA.

The loop is deliberately thin: everything algorithmic lives in ``losses/`` and ``models/``.
What this module owns is (i) the exact data/optimiser/schedule wiring, (ii) AMP, (iii)
checkpointing that is *bit-exact* on resume (including SIGReg's direction-generator step
counter and the RNG states), (iv) logging, and (v) collapse diagnostics.

Nothing here introduces a technique that is not in the paper: there is no gradient clipping,
no EMA teacher, no stop-gradient, no warm-up on weight decay and no auxiliary loss other than
the optional **online linear probe**, which is a *diagnostic* trained on detached features
(the same probe ``MINIMAL.md`` uses to report 90.7 % on Imagenette) and never influences the
encoder's gradient.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from engine.distributed import (
    all_reduce_mean,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
)
from engine.optimizer import build_optimizer, build_param_groups, build_scheduler
from losses.invariance import LeJEPAObjective
from losses.sigreg import SIGReg
from models.projector import LeJEPAEncoder, build_projector

__all__ = ["Trainer", "TrainState", "MetricsLogger", "GPUMonitor"]


# --------------------------------------------------------------------------- logging
class MetricsLogger:
    """Append-only JSONL + CSV + TensorBoard logger."""

    def __init__(self, run_dir: str, enable_tensorboard: bool = True) -> None:
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.jsonl_path = os.path.join(run_dir, "metrics.jsonl")
        self.csv_path = os.path.join(run_dir, "metrics.csv")
        self._csv_keys: Optional[List[str]] = None
        self._jsonl = open(self.jsonl_path, "a", buffering=1)
        self.writer = None
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(os.path.join(run_dir, "tb"))
            except Exception as exc:  # pragma: no cover
                print(f"[logger] tensorboard unavailable ({exc}); JSONL/CSV only")

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        rec = {"step": step, **{k: _to_py(v) for k, v in metrics.items()}}
        self._jsonl.write(json.dumps(rec) + "\n")

        if self._csv_keys is None:
            self._csv_keys = ["step"] + [k for k in rec if k != "step"]
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, "w") as f:
                    f.write(",".join(self._csv_keys) + "\n")
        with open(self.csv_path, "a") as f:
            f.write(",".join(str(rec.get(k, "")) for k in self._csv_keys) + "\n")

        if self.writer is not None:
            for k, v in rec.items():
                if k == "step":
                    continue
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    self.writer.add_scalar(k, v, step)

    def close(self) -> None:
        self._jsonl.close()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def _to_py(v):
    if isinstance(v, torch.Tensor):
        return v.detach().float().cpu().item() if v.numel() == 1 else v.detach().cpu().tolist()
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    try:
        return float(v)
    except Exception:
        return str(v)


class GPUMonitor:
    """Poll GPU utilisation / memory from nvidia-smi (no extra dependency)."""

    def __init__(self) -> None:
        self.available = True
        try:
            subprocess.run(["nvidia-smi", "-L"], capture_output=True, check=True, timeout=5)
        except Exception:
            self.available = False

    def query(self) -> Dict[str, float]:
        if not self.available:
            return {}
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
            first = out.splitlines()[0]
            util, mem_used, mem_total, temp, power = [x.strip() for x in first.split(",")]
            return {
                "gpu/util": float(util),
                "gpu/mem_used_mb": float(mem_used),
                "gpu/mem_total_mb": float(mem_total),
                "gpu/temp_c": float(temp),
                "gpu/power_w": float(power),
            }
        except Exception:
            return {}


# --------------------------------------------------------------------------- state
@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0  # global optimiser steps
    samples_seen: int = 0
    best_loss: float = float("inf")
    history: List[Dict] = field(default_factory=list)


# --------------------------------------------------------------------------- trainer
class Trainer:
    def __init__(self, cfg: Dict, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.output_dir = cfg["output_dir"]
        self.run_dir = cfg.get("run_dir") or os.path.join(self.output_dir, cfg["run_name"])
        self.amp_dtype = _resolve_amp_dtype(cfg.get("precision", "bf16"))
        self.use_amp = self.amp_dtype is not None
        self.use_scaler = bool(cfg.get("grad_scaler", "auto") == "on") or (
            cfg.get("grad_scaler", "auto") == "auto" and self.amp_dtype == torch.float16
        )

        self.state = TrainState()
        self.monitor = GPUMonitor()
        self.logger: Optional[MetricsLogger] = None
        self._last_log_time = time.time()
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        from models.backbone import build_backbone

        cfg = self.cfg
        torch.manual_seed(cfg.get("seed", 0))

        backbone = build_backbone(cfg["model"])
        projector = build_projector(cfg["projector"])
        self.model = LeJEPAEncoder(backbone, projector).to(self.device)

        n_backbone = backbone.num_parameters
        n_proj = sum(p.numel() for p in projector.parameters())
        self.model_info = {
            "backbone": cfg["model"]["model_name"],
            "backbone_params": n_backbone,
            "projector_params": n_proj,
            "projector_depth": projector.depth,
        }

        # ---- SIGReg + objective
        sc = cfg.get("sigreg", {})
        self.sigreg = SIGReg(
            num_slices=sc.get("num_slices", 1024),
            t_max=sc.get("t_max", 5.0),
            n_points=sc.get("n_points", 17),
            reduction=sc.get("reduction", "mean"),
            seed=sc.get("seed", 0) if sc.get("resample", "step") == "step" else None,
        ).to(self.device)
        self.objective = LeJEPAObjective(
            sigreg=self.sigreg,
            lambd=cfg.get("lambd", 0.05),
            n_global_views=cfg["data"]["n_global_views"],
            feature_reduce=cfg.get("feature_reduce", "mean"),
        ).to(self.device)

        # ---- data
        from datasets.imagenet import build_pretrain_dataset, build_pretrain_loader
        from datasets.transforms import MultiViewTransform

        dc = cfg["data"]
        self.train_transform = MultiViewTransform(
            n_global=dc["n_global_views"],
            n_local=dc["n_local_views"],
            global_resolution=dc["global_resolution"],
            local_resolution=dc["local_resolution"],
            global_scale=tuple(dc.get("global_scale", (0.3, 1.0))),
            local_scale=tuple(dc.get("local_scale", (0.05, 0.3))),
            recipe=dc.get("recipe", "lejepa"),
        )
        self.train_dataset = build_pretrain_dataset(dc, self.train_transform)
        self.train_loader = build_pretrain_loader(
            self.train_dataset,
            batch_size=cfg["batch_size"],
            num_workers=cfg.get("num_workers", 8),
            seed=cfg.get("seed", 0),
        )
        self.views_per_sample = self.train_transform.n_views
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.steps_per_epoch * cfg["epochs"]
        self.warmup_steps = int(cfg.get("warmup_epochs", 1) * self.steps_per_epoch)

        # ---- optimiser / schedule
        oc = cfg.get("optim", {})
        self.encoder_groups = build_param_groups(
            self.model,
            lr=oc.get("lr", 5e-4),
            weight_decay=oc.get("weight_decay", 5e-2),
        )
        groups = list(self.encoder_groups)
        self.probe = None
        if cfg.get("online_probe", {}).get("enabled", False):
            num_classes = dc.get("num_classes", 10)
            self.probe = nn.Sequential(
                nn.LayerNorm(cfg["model"]["embedding_dim"], eps=1e-6),
                nn.Linear(cfg["model"]["embedding_dim"], num_classes),
            ).to(self.device)
            pc = cfg["online_probe"]
            groups += build_param_groups(
                self.probe, lr=pc.get("lr", 1e-3), weight_decay=pc.get("weight_decay", 1e-7)
            )
            # tag the probe groups so the schedule can treat them differently
            for g in groups[len(self.encoder_groups) :]:
                g["name"] = "probe_" + g["name"]

        self.optimizer = build_optimizer(groups, betas=tuple(oc.get("betas", (0.9, 0.999))))
        group_ratios = {"probe_decay": 1.0, "probe_no_decay": 1.0} if self.probe else {}
        self.scheduler = build_scheduler(
            self.optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            min_lr_ratio=oc.get("min_lr_ratio", 1e-3),
            start_factor=oc.get("start_factor", 0.01),
            group_min_lr_ratios=group_ratios,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler and self.device.type == "cuda")

        # ---- DDP
        from engine.distributed import convert_sync_batchnorm, wrap_ddp

        if is_distributed():
            if cfg.get("sync_batchnorm", True):
                self.model = convert_sync_batchnorm(self.model)
            self.model = wrap_ddp(self.model, int(os.environ.get("LOCAL_RANK", 0)))

        # ---- SWA (paper §6.1 variant, off by default)
        self.swa_model = None
        if cfg.get("swa", {}).get("enabled", False):
            import copy

            self.swa_model = copy.deepcopy(self._unwrap().backbone).eval()
            for p in self.swa_model.parameters():
                p.requires_grad_(False)
            self._swa_count = 0

        if is_main_process():
            print("=" * 78)
            print(f"run: {cfg['run_name']}")
            print(f"device={self.device} world_size={self.world_size} amp={self.amp_dtype}")
            print(f"backbone={self.model_info['backbone']} "
                  f"({self.model_info['backbone_params']/1e6:.1f}M) "
                  f"projector={self.model_info['projector_params']/1e6:.2f}M "
                  f"depth={self.model_info['projector_depth']}")
            print(f"views={self.views_per_sample} "
                  f"(global={dc['n_global_views']}, local={dc['n_local_views']}) "
                  f"res={dc['global_resolution']}/{dc['local_resolution']}")
            print(f"data={dc['name']} n={len(self.train_dataset)} "
                  f"bs={cfg['batch_size']} steps/epoch={self.steps_per_epoch}")
            print(f"epochs={cfg['epochs']} total_steps={self.total_steps} "
                  f"warmup_steps={self.warmup_steps}")
            print(f"lambda={cfg.get('lambd')} sigreg={sc} ")
            print(f"schedule: peak_lr={oc.get('lr', 5e-4):.3e} "
                  f"min_lr_ratio={oc.get('min_lr_ratio', 1e-3)} -> final "
                  f"{oc.get('lr', 5e-4) * oc.get('min_lr_ratio', 1e-3):.3e}")
            print("optimizer:\n" + "\n".join(
                f"  {g.get('name')}: lr={g['lr']:.2e} wd={g['weight_decay']:.2e} "
                f"n={sum(p.numel() for p in g['params'])/1e6:.2f}M"
                for g in self.optimizer.param_groups
            ))
            print("=" * 78)

    def _unwrap(self) -> LeJEPAEncoder:
        m = self.model
        while hasattr(m, "module"):
            m = m.module
        return m

    # ------------------------------------------------------------------ checkpointing
    def save_checkpoint(self, path: str, extra: Optional[Dict] = None) -> None:
        if not is_main_process():
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "model": self._unwrap().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "state": {
                "epoch": self.state.epoch,
                "step": self.state.step,
                "samples_seen": self.state.samples_seen,
                "best_loss": self.state.best_loss,
            },
            # SIGReg's direction sampler is seeded from this buffer: without it a resumed
            # run would replay a different (though still valid) direction sequence.
            "sigreg_global_step": self.sigreg.global_step.clone(),
            "config": self.cfg,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "model_info": self.model_info,
        }
        if self.probe is not None:
            payload["probe"] = self.probe.state_dict()
        if self.swa_model is not None:
            payload["swa"] = self.swa_model.state_dict()
        if extra:
            payload["extra"] = extra
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> Dict:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self._unwrap().load_state_dict(ckpt["model"])
        if load_optimizer and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if load_optimizer and "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if "sigreg_global_step" in ckpt:
            self.sigreg.global_step.copy_(ckpt["sigreg_global_step"])
        if self.probe is not None and "probe" in ckpt:
            self.probe.load_state_dict(ckpt["probe"])
        if self.swa_model is not None and "swa" in ckpt:
            self.swa_model.load_state_dict(ckpt["swa"])
        st = ckpt.get("state", {})
        self.state.epoch = st.get("epoch", 0)
        self.state.step = st.get("step", 0)
        self.state.samples_seen = st.get("samples_seen", 0)
        self.state.best_loss = st.get("best_loss", float("inf"))
        if "torch_rng" in ckpt:
            torch.set_rng_state(ckpt["torch_rng"])
        if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        if is_main_process():
            print(f"[ckpt] resumed from {path} at epoch={self.state.epoch} step={self.state.step}")
        return ckpt

    # ------------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def embedding_diagnostics(self, proj: torch.Tensor) -> Dict[str, float]:
        """Collapse / isotropy diagnostics on the projector output.

        ``proj`` is ``(V, B, K)``.  We report

        * ``diag/feature_std_mean`` — mean per-dimension std; 0 means complete collapse.
        * ``diag/feature_std_min``  — smallest per-dimension std; 0 means a dead dimension.
        * ``diag/effective_rank``   — ``exp(entropy of normalised covariance eigenvalues)``,
          i.e. how many directions actually carry variance (Roy & Vetterli, 2007).
        * ``diag/mean_norm``        — L2 norm of the mean embedding; SIGReg drives it to 0.
        * ``diag/cov_offdiag``      — mean |off-diagonal correlation|; SIGReg drives it to 0.
        """
        z = proj.reshape(-1, proj.size(-1)).float()
        std = z.std(dim=0)
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / max(1, z.size(0) - 1)
        eig = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
        p = eig / eig.sum()
        eff_rank = torch.exp(-(p * p.log()).sum())
        d = cov.diagonal().clamp_min(1e-12).sqrt()
        corr = cov / (d[:, None] * d[None, :])
        k = corr.size(0)
        off = (corr.abs().sum() - corr.diagonal().abs().sum()) / max(1, k * (k - 1))
        return {
            "diag/feature_std_mean": std.mean().item(),
            "diag/feature_std_min": std.min().item(),
            "diag/effective_rank": eff_rank.item(),
            "diag/rank_ratio": (eff_rank / k).item(),
            "diag/mean_norm": z.mean(dim=0).norm().item(),
            "diag/cov_offdiag": off.item(),
        }

    # ------------------------------------------------------------------ one step
    def train_step(self, views, labels: Optional[torch.Tensor]):
        """One optimiser step.

        ``views`` is either a list of ``V`` tensors ``(B, C, H, W)`` (multi-resolution, the
        normal case) or a single ``(B, V, C, H, W)`` tensor (uniform resolution).
        """
        if torch.is_tensor(views):
            views = views.to(self.device, non_blocking=True)
            if views.ndim == 5:
                v, n = views.shape[1], views.shape[0]
                views = [views[:, i] for i in range(v)]
        else:
            views = [x.to(self.device, non_blocking=True) for x in views]
            v, n = len(views), views[0].shape[0]

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype if self.use_amp else torch.float32,
            enabled=self.use_amp,
        ):
            emb, proj = self.model(views)  # emb (N*V, E), proj (V, N, K)

            # Optional SWA teacher for the centre (paper §6.1 variant).  The student's own
            # prediction term still differentiates the student; the *centre* uses the
            # averaged encoder, exactly as described in the paper.
            if self.swa_model is not None:
                with torch.no_grad():
                    swa_emb = torch.stack(
                        [self.swa_model(x) for x in views], dim=0
                    )  # (V, N, E)
                    swa_proj = self._unwrap().projector(
                        swa_emb.reshape(v * n, -1)
                    ).reshape(v, n, -1)
                proj_for_centre = torch.cat(
                    [swa_proj[: self.cfg["data"]["n_global_views"]],
                     proj[self.cfg["data"]["n_global_views"]:]],
                    dim=0,
                )
            else:
                proj_for_centre = proj

            out = self.objective(proj_for_centre, return_components=True)
            loss = out.loss

            probe_loss = None
            if self.probe is not None and labels is not None:
                labels = labels.to(self.device, non_blocking=True)
                logits = self.probe(emb.detach().reshape(-1, emb.size(-1)))
                probe_loss = nn.functional.cross_entropy(logits, labels.repeat(v))
                loss = loss + probe_loss

        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

        self.state.step += 1
        self.state.samples_seen += n

        metrics = {
            "train/loss": loss.detach(),
            "train/pred": out.prediction,
            "train/sigreg": out.sigreg,
            "train/lejepa": out.loss.detach(),
            "train/lr": self.optimizer.param_groups[0]["lr"],
        }
        if probe_loss is not None:
            metrics["train/probe"] = probe_loss.detach()
        return metrics, proj.detach()

    # ------------------------------------------------------------------ epochs
    def train(self) -> None:
        cfg = self.cfg
        self.logger = MetricsLogger(self.run_dir, cfg.get("tensorboard", True))
        ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        if is_main_process():
            with open(os.path.join(self.run_dir, "config.json"), "w") as f:
                json.dump(cfg, f, indent=2, default=str)

        start_epoch = self.state.epoch
        last_log = time.time()
        loss_window: List[float] = []
        proj_window: List[torch.Tensor] = []
        gpu_stats: Dict[str, float] = {}

        for epoch in range(start_epoch, cfg["epochs"]):
            self.model.train()
            if self.probe is not None:
                self.probe.train()

            for it, (views, labels) in enumerate(self.train_loader):
                metrics, proj = self.train_step(views, labels)
                loss_window.append(metrics["train/loss"].item())
                if len(proj_window) < 8:
                    proj_window.append(proj)

                if self.state.step % cfg.get("log_every", 20) == 0:
                    now = time.time()
                    dt = now - last_log
                    last_log = now
                    gpu_stats = self.monitor.query()
                    rec = {k: v for k, v in metrics.items()}
                    rec["perf/step_time_s"] = dt / max(1, cfg.get("log_every", 20))
                    rec["perf/samples_per_s"] = (
                        cfg.get("log_every", 20) * cfg["batch_size"] * self.world_size / max(dt, 1e-9)
                    )
                    rec["perf/epoch"] = epoch + it / max(1, self.steps_per_epoch)
                    rec["train/loss_smooth"] = sum(loss_window[-50:]) / len(loss_window[-50:])
                    for k, vv in gpu_stats.items():
                        rec[k] = vv
                    if is_main_process():
                        self.logger.log(rec, self.state.step)
                    loss_window.clear()

            # ---- epoch-end diagnostics
            diag = {}
            if proj_window and is_main_process():
                stacked = torch.cat(proj_window, dim=1)
                diag = self.embedding_diagnostics(stacked)
                self.logger.log(diag, self.state.step)
                print(
                    f"[epoch {epoch+1}/{cfg['epochs']}] step={self.state.step} "
                    f"pred={metrics['train/pred'].item():.4f} "
                    f"sigreg={metrics['train/sigreg'].item():.4f} "
                    f"eff_rank={diag['diag/effective_rank']:.1f} "
                    f"std_min={diag['diag/feature_std_min']:.3f} "
                    f"lr={metrics['train/lr']:.2e}"
                )
            proj_window.clear()

            self.state.epoch = epoch + 1

            # ---- optional evaluation (online probe accuracy on the val split)
            if self._should_eval(epoch):
                eval_metrics = self.evaluate()
                if is_main_process() and eval_metrics:
                    print(f"  [eval] {eval_metrics}")
                    self.logger.log(eval_metrics, self.state.step)

            # ---- SWA update
            if self.swa_model is not None:
                start = int(cfg["swa"].get("start_epoch", int(0.75 * cfg["epochs"])))
                if epoch + 1 >= start:
                    self._update_swa()

            # ---- checkpoints
            if is_main_process() and (epoch + 1) % cfg.get("save_every", 10) == 0:
                self.save_checkpoint(os.path.join(ckpt_dir, f"epoch_{epoch+1:04d}.pt"))
            self.save_checkpoint(os.path.join(ckpt_dir, "last.pt"))

        if is_main_process():
            self.save_checkpoint(os.path.join(ckpt_dir, "final.pt"))
            print(f"[done] final checkpoint: {os.path.join(ckpt_dir, 'final.pt')}")
            self.logger.close()

    def _should_eval(self, epoch: int) -> bool:
        ec = self.cfg.get("eval", {})
        if not ec.get("enabled", False):
            return False
        every = ec.get("every", 10)
        return (epoch + 1) % every == 0 or (epoch + 1) == self.cfg["epochs"]

    @torch.no_grad()
    def _update_swa(self) -> None:
        backbone = self._unwrap().backbone
        n = self._swa_count
        for p_swa, p in zip(self.swa_model.parameters(), backbone.parameters()):
            p_swa.mul_(n / (n + 1)).add_(p.detach(), alpha=1.0 / (n + 1))
        for b_swa, b in zip(self.swa_model.buffers(), backbone.buffers()):
            if b_swa.dtype.is_floating_point:
                b_swa.mul_(n / (n + 1)).add_(b.detach(), alpha=1.0 / (n + 1))
            else:
                b_swa.copy_(b)
        self._swa_count += 1

    # ------------------------------------------------------------------ eval hook
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Quick validation of the online probe (diagnostic only)."""
        if self.probe is None:
            return {}
        from datasets.imagenet import build_eval_dataset, build_eval_loader

        dc = dict(self.cfg["data"])
        # The online probe must be measured on *held-out* data.  ``dc["split"]`` is "train",
        # so overriding it here is essential (an earlier version silently evaluated on the
        # training split, reporting train accuracy as if it were val accuracy).
        dc["split"] = self.cfg.get("eval", {}).get("split", "val")
        ds = build_eval_dataset(dc, resolution=dc["global_resolution"])
        loader = build_eval_loader(
            ds, batch_size=self.cfg.get("eval", {}).get("batch_size", 256),
            num_workers=self.cfg.get("num_workers", 8),
        )
        self.model.eval()
        self.probe.eval()
        correct = total = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if x.ndim == 5:
                x = x[:, 0]
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype if self.use_amp else torch.float32,
                enabled=self.use_amp,
            ):
                emb = self._unwrap().backbone(x)
                logits = self.probe(emb)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
        self.model.train()
        self.probe.train()
        acc = correct / max(1, total)
        return {"eval/online_probe_acc": acc, "eval/n": total}


def _resolve_amp_dtype(precision: str):
    precision = (precision or "bf16").lower()
    if precision in ("bf16", "bfloat16"):
        return torch.bfloat16
    if precision in ("fp16", "float16", "half"):
        return torch.float16
    if precision in ("fp32", "float32", "none"):
        return None
    raise ValueError(f"unknown precision {precision!r}")
