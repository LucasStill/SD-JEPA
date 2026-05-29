"""JEPA-STEP auxiliary losses.

Three additive loss terms grafted on top of LeWM's (pred + SIGReg) objective:

  * ``cosine_triplet_loss``   — progression-subspace cosine-margin triplet
                                (ported from STEP/losses/losses.py).
  * ``prototype_loss``        — MSE of the first / last embeddings to two
                                fixed orthogonal prototypes (STEP-style).
  * ``straightening_loss``    — differentiable Hénaff-style straightening on
                                the full latent, penalising curvature along
                                time.

All three operate on ``(B, T, d)`` tensors and return a scalar. Each term is
independently switchable via its weight in the hydra config: setting the
weight to zero makes the term a true no-op.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Triplet
# ---------------------------------------------------------------------------


def cosine_triplet_loss(
    z_prog: torch.Tensor,
    margin: float = 0.2,
    window_tau: int = 2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cosine-margin triplet loss on the progression subspace.

    Positives: same episode / different timestep within ``window_tau``.
    Negatives: same temporal position, different batch element.

    Parameters
    ----------
    z_prog : (B, T, k) float tensor
        Progression-subspace embeddings. The caller guarantees each batch row
        is a single short windowed trajectory (LeWM's default: T = 4).
    margin : float
        Cosine-similarity margin ``m``.
    window_tau : int
        Maximum temporal offset used to pick a positive. With the default
        LeWM window of T=4, ``window_tau >= 1`` is enough to always have a
        valid positive.
    eps : float
        Numerical floor for the cosine similarity.

    Returns
    -------
    scalar tensor
        ``mean(relu(cos(anchor, negative) - cos(anchor, positive) + margin))``.
    """

    B, T, _ = z_prog.shape
    if B < 2 or T < 2:
        # A single-batch-element or single-timestep input has no valid
        # triplet; return a zero that still carries a grad path so the
        # autograd graph stays connected.
        return z_prog.sum() * 0.0

    # Anchor position: middle of the window when possible, else 0.
    t_anchor = T // 2

    # Pick a positive within ``window_tau`` of t_anchor but != t_anchor.
    # Deterministic-for-shapes choice: t_anchor + 1 if it exists, else
    # t_anchor - 1. (Keeping it deterministic helps unit tests; the stochastic
    # version is the optional ``window_tau`` one-line change below.)
    t_pos = t_anchor + 1 if t_anchor + 1 < T else t_anchor - 1
    if window_tau > 1:
        # Sample a random offset uniformly in [-window_tau, window_tau] \ {0}
        # clipped to valid indices.
        lo = max(0, t_anchor - window_tau)
        hi = min(T - 1, t_anchor + window_tau)
        candidates = [t for t in range(lo, hi + 1) if t != t_anchor]
        if candidates:
            idx = torch.randint(0, len(candidates), (1,), device=z_prog.device).item()
            t_pos = candidates[idx]

    anchor = z_prog[:, t_anchor]            # (B, k)
    positive = z_prog[:, t_pos]             # (B, k)

    # Negative: roll the batch by a random non-zero shift so that each
    # anchor is paired with a *different* episode's anchor.
    shift = int(torch.randint(1, B, (1,), device=z_prog.device).item())
    negative = torch.roll(anchor, shifts=shift, dims=0)

    pos_sim = F.cosine_similarity(anchor, positive, dim=-1, eps=eps)
    neg_sim = F.cosine_similarity(anchor, negative, dim=-1, eps=eps)

    return F.relu(neg_sim - pos_sim + margin).mean()


# ---------------------------------------------------------------------------
#  Prototype
# ---------------------------------------------------------------------------


def prototype_loss(
    z_prog_init: torch.Tensor,
    z_prog_end: torch.Tensor,
    prototypes: torch.Tensor,
) -> torch.Tensor:
    """MSE of the first / last progression embeddings to fixed prototypes.

    Parameters
    ----------
    z_prog_init : (B, k) tensor
        Progression embedding at t=0.
    z_prog_end : (B, k) tensor
        Progression embedding at t=T-1.
    prototypes : (2, k) tensor (buffer, non-trainable)
        ``prototypes[0]`` is the "episode-start" anchor and
        ``prototypes[1]`` is the "episode-end" anchor. By default they are
        the first two canonical basis vectors of R^k.

    Returns
    -------
    scalar tensor
        ``0.5 * (MSE(z_init, e1) + MSE(z_end, e2))``.
    """

    e_init = prototypes[0].to(z_prog_init.dtype)
    e_end = prototypes[1].to(z_prog_end.dtype)
    loss_init = F.mse_loss(z_prog_init, e_init.expand_as(z_prog_init))
    loss_end = F.mse_loss(z_prog_end, e_end.expand_as(z_prog_end))
    return 0.5 * (loss_init + loss_end)


def canonical_prototypes(k: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return the canonical (e1, e2) prototype pair in R^k as a (2, k) buffer."""

    if k < 2:
        raise ValueError(f"prototype_loss needs k>=2, got k={k}")
    protos = torch.zeros(2, k, device=device, dtype=dtype)
    protos[0, 0] = 1.0
    protos[1, 1] = 1.0
    return protos


def prototype_loss_first_anchor(
    z_prog: torch.Tensor,
    step_idx: torch.Tensor,
    prototypes: torch.Tensor,
    k_f: int = 3,
) -> torch.Tensor:
    """Sketch A-self prototype loss — anchor only the *first* k_f frames of
    each episode toward ``prototypes[0]``. End frames are deliberately
    *not* anchored so the encoder is free to place per-episode endpoints
    wherever the scene-aware progression places them.

    This is the "no time injection" alternative to the t/T-weighted soft
    attractor: instead of telling the model that position-within-episode
    has a target embedding, we only tell it that *episode beginnings*
    should map to a globally consistent direction. The triplet handles
    ordering from there.

    Parameters
    ----------
    z_prog : (B, T, k) float tensor
        Progression-subspace embeddings.
    step_idx : (B, T) int / float tensor
        Step index within the episode for each frame in the window. The
        DataLoader must provide this; required because the episode
        boundary is not deducible from the window alone.
    prototypes : (>=1, k) tensor (buffer, non-trainable)
        ``prototypes[0]`` is the start anchor (default: ``e_1``).
    k_f : int
        Frames within ``step_idx < k_f`` are treated as ``first-of-episode``.

    Returns
    -------
    scalar tensor
        Mean squared distance to ``prototypes[0]`` over all frames flagged
        as first-of-episode in the batch. Returns 0 (with a connected
        autograd graph) when no first-of-episode frames are present.
    """

    if z_prog.dim() != 3:
        raise ValueError(
            f"prototype_loss_first_anchor expects (B, T, k); got {tuple(z_prog.shape)}"
        )
    if step_idx.shape != z_prog.shape[:2]:
        raise ValueError(
            f"step_idx shape {tuple(step_idx.shape)} doesn't match "
            f"(B, T) of z_prog {tuple(z_prog.shape[:2])}"
        )

    e_init = prototypes[0].to(z_prog.dtype)  # (k,)
    is_first = (step_idx < k_f).to(z_prog.dtype)  # (B, T)
    n_first = is_first.sum().clamp(min=1.0)

    # ||z - e1||² per frame, masked to first-of-episode rows.
    diff = z_prog - e_init  # (B, T, k)
    sq = diff.pow(2).sum(dim=-1)  # (B, T)
    weighted = sq * is_first  # zeros out non-first-of-episode frames
    loss = weighted.sum() / n_first

    # If the batch happened to contain no first-of-episode frames, return a
    # zero tensor that still carries gradient (so autograd graph stays alive).
    if is_first.sum() == 0:
        return z_prog.sum() * 0.0
    return loss


# ---------------------------------------------------------------------------
#  Straightening
# ---------------------------------------------------------------------------


def straightening_loss(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable Hénaff-style straightening of a trajectory.

    Penalises the average curvature ``1 - cos(Δz_{t+1}, Δz_t)`` of the
    latent trajectory, where ``Δz_t = z_{t+1} - z_t``.

    Parameters
    ----------
    z : (B, T, D) float tensor
    eps : float
        Numerical floor.

    Returns
    -------
    scalar tensor
        Mean of ``1 - cos(Δz_{t+1}, Δz_t)`` over all ``(B, t)`` with
        ``0 <= t < T-2``. Returns 0 for ``T < 3``.
    """

    if z.size(1) < 3:
        return z.sum() * 0.0
    delta = z[:, 1:] - z[:, :-1]                     # (B, T-1, D)
    a, b = delta[:, :-1], delta[:, 1:]               # (B, T-2, D) each
    cos = F.cosine_similarity(a, b, dim=-1, eps=eps) # (B, T-2)
    return (1.0 - cos).mean()


# ---------------------------------------------------------------------------
#  Canonical orthogonal injections for the subspace split
# ---------------------------------------------------------------------------


def canonical_injection(D: int, k: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """``(D, k)`` matrix selecting the first ``k`` coordinates of R^D."""

    if k <= 0 or k > D:
        raise ValueError(f"need 0 < k <= D, got D={D}, k={k}")
    P = torch.zeros(D, k, device=device, dtype=dtype)
    P[:k, :k] = torch.eye(k, device=device, dtype=dtype)
    return P


def canonical_injection_complement(
    D: int, k: int, device=None, dtype=torch.float32
) -> torch.Tensor:
    """``(D, D-k)`` matrix selecting the last ``D-k`` coordinates of R^D."""

    if k < 0 or k > D:
        raise ValueError(f"need 0 <= k <= D, got D={D}, k={k}")
    Q = torch.zeros(D, D - k, device=device, dtype=dtype)
    Q[k:, :] = torch.eye(D - k, device=device, dtype=dtype)
    return Q


__all__ = [
    "cosine_triplet_loss",
    "prototype_loss",
    "prototype_loss_first_anchor",
    "canonical_prototypes",
    "straightening_loss",
    "canonical_injection",
    "canonical_injection_complement",
]
