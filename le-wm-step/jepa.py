"""JEPA Implementation (extended for JEPA-STEP).

Changes vs. the LeWM reference:

* ``__init__`` accepts ``k_prog`` and ``use_polar`` and registers the pair of
  canonical orthogonal injections ``P, Q`` plus the two prototype vectors as
  non-trainable buffers. All three are no-ops when ``k_prog == 0``.
* ``encode`` adds ``emb_prog`` and ``emb_cont`` views of the latent, which
  the ``lejepa_forward`` training step uses to target the triplet / proto
  losses on the progression subspace and SIGReg on the content subspace.
* ``predict`` optionally concatenates polar features ``(sin θ, cos θ, r)`` of
  the progression subspace to the action-embedding conditioning tensor.
* ``criterion`` is rewritten to compute the three-term planning cost
  (content MSE + angular + radial). When ``k_prog == 0`` and
  ``gamma_theta = delta_r = 0`` it reduces exactly to the LeWM baseline.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from losses_step import (
    canonical_injection,
    canonical_injection_complement,
    canonical_prototypes,
)


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


def polar_features(z_prog: torch.Tensor) -> torch.Tensor:
    """Polar embedding of the first two coordinates plus radial norm.

    Parameters
    ----------
    z_prog : (..., k) tensor with k >= 2.

    Returns
    -------
    (..., 3) tensor ``[sin θ, cos θ, r]`` — using ``(sin, cos)`` instead of
    raw ``θ`` avoids the ±π wrap-around discontinuity.
    """

    assert z_prog.size(-1) >= 2, (
        f"polar_features needs k>=2, got k={z_prog.size(-1)}"
    )
    z2 = z_prog[..., :2]
    theta = torch.atan2(z2[..., 1], z2[..., 0])
    r = z_prog.norm(dim=-1)
    return torch.stack([theta.sin(), theta.cos(), r], dim=-1)


class JEPA(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        *,
        embed_dim: int | None = None,
        k_prog: int = 0,
        use_polar: bool = False,
        gamma_theta: float = 0.0,
        delta_r: float = 0.0,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

        self.k_prog = int(k_prog)
        self.use_polar = bool(use_polar) and self.k_prog >= 2
        # Planning-time weights, held as plain floats; treated as constants
        # so they don't end up in the optimizer parameter list.
        self.gamma_theta = float(gamma_theta)
        self.delta_r = float(delta_r)
        # When True, criterion uses MSE on the full z instead of the split cost.
        # Not persisted in checkpoints — set at eval time via eval.use_full_z_cost.
        self.use_full_z_cost: bool = False

        # Subspace split buffers. We only know the latent dimension D once
        # we know the projector's output; require the caller to pass it.
        if self.k_prog > 0:
            if embed_dim is None:
                raise ValueError(
                    "JEPA was built with k_prog>0 but embed_dim=None; the "
                    "caller must pass the projector's output dimension."
                )
            self.register_buffer(
                "P", canonical_injection(embed_dim, self.k_prog), persistent=False
            )
            self.register_buffer(
                "Q",
                canonical_injection_complement(embed_dim, self.k_prog),
                persistent=False,
            )
            self.register_buffer(
                "prototypes",
                canonical_prototypes(self.k_prog),
                persistent=False,
            )

    # ------------------------------------------------------------------
    #  Training path
    # ------------------------------------------------------------------

    def encode(self, info):
        """Encode observations and actions into embeddings.

        Adds ``emb_prog`` and ``emb_cont`` views to the info dict when
        ``k_prog > 0``. These are plain linear views (no extra parameters),
        so they do not change the training optimum at k_prog=0.
        """

        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")  # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if self.k_prog > 0:
            info["emb_prog"] = info["emb"] @ self.P  # (B, T, k)
            info["emb_cont"] = info["emb"] @ self.Q  # (B, T, D-k)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding.

        Parameters
        ----------
        emb : (B, T, D)
        act_emb : (B, T, A_emb)

        Notes
        -----
        When ``use_polar`` is enabled, the polar features of the progression
        subspace are concatenated to ``act_emb`` along the feature dim before
        being fed to the predictor as the AdaLN conditioning tensor. The
        predictor's conditioning projection must be sized accordingly at
        construction time — see ``train.py``.
        """

        if self.use_polar:
            z_prog = emb @ self.P
            pol = polar_features(z_prog)
            act_emb = torch.cat([act_emb, pol], dim=-1)

        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    # ------------------------------------------------------------------
    #  Inference path
    # ------------------------------------------------------------------

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.

        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Planning cost per action candidate.

        Three additive terms:

        * Content-subspace MSE (LeWM's original behaviour on ``emb_cont``).
        * Angular term ``1 - cos(θ̂, θ_g)`` on the progression subspace.
        * Radial term ``(r̂ - r_g)²`` on the progression subspace.

        Reduces to the LeWM baseline when ``k_prog == 0``.
        """

        pred_emb = info_dict["predicted_emb"]  # (B, S, T, D)
        goal_emb = info_dict["goal_emb"]       # (B, S, 1, D) after encode

        # Broadcast goal to match pred's time dim, then take the last step.
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)
        pred_last = pred_emb[..., -1, :]                 # (B, S, D)
        goal_last = goal_emb[..., -1, :].detach()        # (B, S, D)

        if self.k_prog == 0 or getattr(self, "use_full_z_cost", False):
            cost = F.mse_loss(
                pred_last, goal_last, reduction="none"
            ).sum(dim=-1)  # (B, S)
            return cost

        # Split last-step embeddings along the same P, Q as training.
        pred_prog = pred_last @ self.P   # (B, S, k)
        pred_cont = pred_last @ self.Q   # (B, S, D-k)
        goal_prog = goal_last @ self.P
        goal_cont = goal_last @ self.Q

        cost_cont = F.mse_loss(pred_cont, goal_cont, reduction="none").sum(dim=-1)

        theta_p = torch.atan2(pred_prog[..., 1], pred_prog[..., 0])
        theta_g = torch.atan2(goal_prog[..., 1], goal_prog[..., 0])
        cost_theta = 1.0 - torch.cos(theta_p - theta_g)

        r_p = pred_prog.norm(dim=-1)
        r_g = goal_prog.norm(dim=-1)
        cost_r = (r_p - r_g).pow(2)

        return cost_cont + self.gamma_theta * cost_theta + self.delta_r * cost_r

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Compute the cost of action candidates given an info dict with goal
        and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)

        return cost
