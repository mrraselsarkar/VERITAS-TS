"""Graph-aware, class-conditional inpainting diffusion (proposal Method part 3).

The denoiser sees, for every (channel, time) cell, the noisy value inside the
region being generated, the observed value outside it, and the mask. Temporal
structure comes from dilated convolutions along time (DiffWave/SSSD style);
cross-channel structure comes from message passing over the channel graph
(prior edges from DatasetMeta.channel_edges plus a learned low-rank residual).
With ``graph="none"`` every channel is denoised independently, which is the
ablation for RQ3.

Class conditioning uses classifier-free guidance: index ``n_classes`` is the
null class. Necessity edits inpaint the claimed region conditioned on another
class; sufficiency edits keep the region and inpaint its complement with the
null class.

Training masks are drawn from the same EvidenceGrid claims the agent can make,
and half of the time their complement, so the model is trained on exactly the
two edit types the verifier needs.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..grid import EvidenceGrid


@dataclass
class DiffusionConfig:
    channels: int = 64           # hidden width
    layers: int = 12
    dilation_cycle: int = 6      # dilations 1, 2, 4, ..., 2**(cycle-1), repeated
    graph: str = "prior+learned"  # "none" | "prior" | "prior+learned"
    graph_rank: int = 8
    steps: int = 200             # diffusion steps (training)
    beta_start: float = 1e-4
    beta_end: float = 0.02
    p_uncond: float = 0.15       # class dropout for classifier-free guidance
    p_complement: float = 0.5    # share of complement (sufficiency-type) masks
    lr: float = 2e-4
    batch_size: int = 64
    iters: int = 8000
    ema_decay: float = 0.999
    seed: int = 0
    sensor_masks: bool = False   # also train on whole-sensor masks (1-2 groups, whole series)


def normalized_adjacency(C: int, edges) -> torch.Tensor:
    A = torch.eye(C)
    for a, b in edges:
        A[a, b] = A[b, a] = 1.0
    d = A.sum(1)
    return A / torch.sqrt(d[:, None] * d[None, :])


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class GraphMix(nn.Module):
    """h (B, H, C, T) -> messages over channels, per time step."""

    def __init__(self, hidden: int, A_prior: torch.Tensor, learned: bool, rank: int):
        super().__init__()
        self.register_buffer("A_prior", A_prior)
        self.node = nn.Parameter(torch.randn(A_prior.shape[0], rank) * 0.1) if learned else None
        self.lin = nn.Conv2d(hidden, hidden, 1)

    def adjacency(self) -> torch.Tensor:
        if self.node is None:
            return self.A_prior
        return 0.5 * (self.A_prior + torch.softmax(self.node @ self.node.T, dim=-1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,bhjt->bhit", self.adjacency(), self.lin(h))


class ResBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int, A: torch.Tensor, cfg: DiffusionConfig):
        super().__init__()
        self.cond = nn.Linear(hidden, hidden)
        self.tconv = nn.Conv2d(hidden, 2 * hidden, kernel_size=(1, 3),
                               dilation=(1, dilation), padding=(0, dilation))
        self.gmix = None if cfg.graph == "none" else GraphMix(
            2 * hidden, A, learned=cfg.graph == "prior+learned", rank=cfg.graph_rank)
        self.out = nn.Conv2d(hidden, 2 * hidden, 1)

    def forward(self, h, emb):
        y = self.tconv(h + self.cond(emb)[:, :, None, None])
        if self.gmix is not None:
            y = y + self.gmix(y)
        a, b = y.chunk(2, dim=1)
        res, skip = self.out(torch.tanh(a) * torch.sigmoid(b)).chunk(2, dim=1)
        return (h + res) / math.sqrt(2), skip


class Denoiser(nn.Module):
    def __init__(self, C: int, n_classes: int, edges, cfg: DiffusionConfig):
        super().__init__()
        H = cfg.channels
        A = normalized_adjacency(C, edges)
        self.inp = nn.Conv2d(3, H, 1)
        self.chan = nn.Parameter(torch.randn(1, H, C, 1) * 0.02)
        self.temb = nn.Sequential(nn.Linear(H, 2 * H), nn.SiLU(), nn.Linear(2 * H, H))
        self.cemb = nn.Embedding(n_classes + 1, H)  # last index = null class
        self.blocks = nn.ModuleList(
            ResBlock(H, 2 ** (i % cfg.dilation_cycle), A, cfg) for i in range(cfg.layers))
        self.head = nn.Sequential(nn.Conv2d(H, H, 1), nn.SiLU(), nn.Conv2d(H, 1, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.hidden = H

    def forward(self, x_t, x_obs, mask, t, c):
        h = self.inp(torch.stack([x_t * mask, x_obs * (1 - mask), mask], dim=1)) + self.chan
        emb = self.temb(timestep_embedding(t, self.hidden)) + self.cemb(c)
        skips = 0
        for blk in self.blocks:
            h, s = blk(h, emb)
            skips = skips + s
        return self.head(skips / math.sqrt(len(self.blocks))).squeeze(1)


class InpaintDiffusion:
    """Training and DDIM inpainting for one dataset."""

    def __init__(self, cfg: DiffusionConfig, C: int, n_classes: int, edges, grid: EvidenceGrid,
                 device: str | None = None):
        self.cfg, self.C, self.n_classes, self.edges, self.grid = cfg, C, n_classes, list(edges), grid
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        self.model = Denoiser(C, n_classes, self.edges, cfg).to(self.device)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.steps)
        self.alpha_bar = torch.cumprod(1 - betas, 0).to(self.device)
        masks = [grid.mask(c) for c in grid.claims()]
        if cfg.sensor_masks:  # whole-sensor claims, the shape used for activity data
            sensor = EvidenceGrid(T=grid.T, channel_groups=grid.channel_groups, K=1, max_groups=2)
            masks += [sensor.mask(c) for c in sensor.claims()]
        self.claim_masks = torch.as_tensor(np.stack(masks), dtype=torch.float32, device=self.device)
        self.null = n_classes

    # ------------------------------------------------------------------ training

    def _train_masks(self, B: int, gen: torch.Generator) -> torch.Tensor:
        idx = torch.randint(len(self.claim_masks), (B,), device=self.device, generator=gen)
        m = self.claim_masks[idx]
        comp = torch.rand(B, device=self.device, generator=gen) < self.cfg.p_complement
        return torch.where(comp[:, None, None], 1 - m, m)

    def fit(self, X: np.ndarray, y: np.ndarray, log_every: int = 500, log=print,
            ckpt_path=None, ckpt_every: int = 250) -> list[float]:
        """Train; with ``ckpt_path`` the full training state is saved every
        ``ckpt_every`` iterations and an existing checkpoint is resumed, so a
        crashed run (e.g. a Windows GPU watchdog reset) continues where it stopped."""
        cfg = self.cfg
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y, dtype=torch.long, device=self.device)
        gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.iters)
        losses, run, start = [], 0.0, 1
        if ckpt_path is not None and Path(ckpt_path).exists():
            ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ck["model"])
            self.ema.load_state_dict(ck["ema"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            gen.set_state(ck["gen"].cpu())  # generator states are CPU byte tensors
            losses, start = ck["losses"], ck["iter"] + 1
            log(f"resumed from iter {ck['iter']}")
        self.model.train()
        for it in range(start, cfg.iters + 1):
            i = torch.randint(len(X), (cfg.batch_size,), device=self.device, generator=gen)
            x0, c = X[i], y[i].clone()
            drop = torch.rand(len(c), device=self.device, generator=gen) < cfg.p_uncond
            c[drop] = self.null
            m = self._train_masks(len(x0), gen)
            t = torch.randint(cfg.steps, (len(x0),), device=self.device, generator=gen)
            ab = self.alpha_bar[t][:, None, None]
            noise = torch.randn(x0.shape, device=self.device, generator=gen)
            x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                pred = self.model(x_t, x0, m, t, c)
            loss = (((pred.float() - noise) ** 2) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(self.ema.parameters(), self.model.parameters()):
                    pe.lerp_(pm, 1 - cfg.ema_decay)
            run += loss.item()
            if it % log_every == 0:
                losses.append(run / log_every)
                log(f"iter {it:6d}  loss {losses[-1]:.4f}")
                run = 0.0
            if ckpt_path is not None and (it % ckpt_every == 0 or it == cfg.iters):
                tmp = Path(str(ckpt_path) + ".tmp")
                torch.save({"model": self.model.state_dict(), "ema": self.ema.state_dict(),
                            "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "gen": gen.get_state(), "losses": losses, "iter": it}, tmp)
                tmp.replace(ckpt_path)
        self.model.eval()
        return losses

    # ------------------------------------------------------------------ sampling

    @torch.no_grad()
    def inpaint(self, x: np.ndarray, mask: np.ndarray, c: np.ndarray, n_steps: int = 25,
                guidance: float = 1.0, batch_size: int = 256, seed: int = 0,
                init: np.ndarray | None = None, strength: float = 1.0) -> np.ndarray:
        """Regenerate x where mask is True, conditioned on classes c (use
        ``self.null`` for unconditional). Returns a new array; x is unchanged.

        With ``init`` (e.g. a spliced series), sampling starts from ``init``
        noised to ``strength`` x the full noise level instead of pure noise
        (SDEdit), so the region keeps the donor's shape but is re-fitted to the
        surrounding context. ``strength=1`` ignores ``init``."""
        out = np.empty_like(x, dtype=np.float32)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        t_start = max(1, int(round(strength * (self.cfg.steps - 1))))
        n = max(2, int(round(n_steps * t_start / (self.cfg.steps - 1))))
        ts = torch.linspace(t_start, 0, n, device=self.device).round().long()
        for s in range(0, len(x), batch_size):
            xo = torch.as_tensor(x[s:s + batch_size], dtype=torch.float32, device=self.device)
            m = torch.as_tensor(mask[s:s + batch_size], dtype=torch.float32, device=self.device)
            cc = torch.as_tensor(c[s:s + batch_size], dtype=torch.long, device=self.device)
            cu = torch.full_like(cc, self.null)
            xt = torch.randn(xo.shape, device=self.device, generator=gen)
            if init is not None and strength < 1.0:
                x_init = torch.as_tensor(init[s:s + batch_size], dtype=torch.float32, device=self.device)
                ab0 = self.alpha_bar[ts[0]]
                xt = ab0.sqrt() * x_init + (1 - ab0).sqrt() * xt
            for k, t in enumerate(ts):
                tt = t.repeat(len(xo))
                eps = self.ema(xt, xo, m, tt, cc)
                if guidance != 1.0:
                    eps_u = self.ema(xt, xo, m, tt, cu)
                    eps = eps_u + guidance * (eps - eps_u)
                ab = self.alpha_bar[t]
                ab_prev = self.alpha_bar[ts[k + 1]] if k + 1 < len(ts) else torch.tensor(1.0, device=self.device)
                x0_hat = ((xt - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-8, 8)
                xt = ab_prev.sqrt() * x0_hat + (1 - ab_prev).sqrt() * eps
            out[s:s + batch_size] = (xo * (1 - m) + xt * m).cpu().numpy()
        return out

    # ------------------------------------------------------------------ io

    def save(self, path) -> None:
        torch.save({"cfg": asdict(self.cfg), "C": self.C, "n_classes": self.n_classes,
                    "edges": self.edges, "grid": {"T": self.grid.T, "K": self.grid.K,
                                                  "channel_groups": self.grid.channel_groups,
                                                  "max_windows": self.grid.max_windows,
                                                  "max_groups": self.grid.max_groups},
                    "model": self.model.state_dict(), "ema": self.ema.state_dict()}, path)

    @classmethod
    def load(cls, path, device: str | None = None) -> "InpaintDiffusion":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        grid = EvidenceGrid(**ck["grid"])
        obj = cls(DiffusionConfig(**ck["cfg"]), ck["C"], ck["n_classes"], ck["edges"], grid, device)
        obj.model.load_state_dict(ck["model"])
        obj.ema.load_state_dict(ck["ema"])
        obj.model.eval()
        return obj
