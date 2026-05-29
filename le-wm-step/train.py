import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack
from hdf5_dataset import HDF5Dataset
from losses_step import (
    cosine_triplet_loss,
    prototype_loss,
    prototype_loss_first_anchor,
    straightening_loss,
)


# ── Kill all Lightning-native checkpoint writes ─────────────────────────────
# On quota-constrained clusters, every Lightning-native checkpoint write
# (300-500 MB) has ENOSPC'd because the project inode quota was near-full and
# the ``lightning_logs/``/``wandb_archive/`` trees kept burning inodes faster
# than ``--save-top-k`` was evicting them. Prior attempts to avoid this:
#   (a) ``Trainer(enable_checkpointing=False)``
#   (b) stripping ``ModelCheckpoint`` from ``trainer.callbacks`` after
#       ``spt.Manager`` is built
# both failed: ``spt.Manager.__call__()`` *re-injects* a ``ModelCheckpoint``
# when it wraps the trainer for its own wandb-resume mechanism, after any
# pre-fit callback edits we've made.
#
# The only level that stops the write regardless of which callback initiated
# it is ``TorchCheckpointIO.save_checkpoint`` — every Lightning-native save
# eventually reaches it. We make it a no-op.
#
# Our ``ModelObjectCallBack`` is **unaffected** because it calls
# ``torch.save(model, path)`` directly (not via ``trainer.save_checkpoint``),
# so the per-epoch model pickle at ``run_dir/<name>_epoch_<N>_object.ckpt``
# still lands on disk. That is the artefact eval and analysis scripts
# consume; Lightning's native ``.ckpt`` format is redundant for this project.
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO as _LTCIO

_orig_torch_ckpt_save = _LTCIO.save_checkpoint


def _noop_torch_ckpt_save(self, checkpoint, path, storage_options=None):
    print(
        f"[train.py] Suppressed Lightning TorchCheckpointIO.save_checkpoint "
        f"→ would have written {path!s} (~{sum(1 for _ in checkpoint.keys())} "
        f"top-level keys). Our ModelObjectCallBack dumps the model pickle "
        f"separately, so this save is redundant."
    )
    return None


_LTCIO.save_checkpoint = _noop_torch_ckpt_save


def _opt_weight(cfg, *path, default=0.0):
    """Safely fetch ``cfg.<path>.weight`` falling back to ``default`` if any
    intermediate key is missing. Lets us run the same train.py against either
    the LeWM baseline config or the JEPA-STEP config."""

    node = cfg
    for p in path:
        if not hasattr(node, p) and (not hasattr(node, "get") or node.get(p) is None):
            return default
        node = getattr(node, p) if hasattr(node, p) else node.get(p)
    return getattr(node, "weight", default)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses.

    Four-term loss:

        L = L_pred
          + λ_S   · SIGReg(z^cont)
          + λ_T   · triplet(z^prog)
          + λ_P   · proto(z^prog_init, z^prog_end)
          + λ_str · straight(z)

    When ``k_prog == 0`` we fall back to running SIGReg on the full latent
    and skipping the progression terms — this matches LeWM exactly.
    """

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    k_prog = int(cfg.wm.get("k_prog", 0))

    lambd_S = cfg.loss.sigreg.weight
    lambd_T = _opt_weight(cfg, "loss", "triplet", default=0.0)
    lambd_P = _opt_weight(cfg, "loss", "proto", default=0.0)
    lambd_str = _opt_weight(cfg, "loss", "straight", default=0.0)

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # ------------ losses ------------
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()

    sigreg_input = output["emb_cont"] if k_prog > 0 else emb
    output["sigreg_loss"] = self.sigreg(sigreg_input.transpose(0, 1))

    total = output["pred_loss"] + lambd_S * output["sigreg_loss"]

    # Triplet can act on either the progression subspace or the full latent.
    # The ``loss.triplet.target`` config controls which:
    #   - "auto"  (default): emb_prog when k_prog > 0, else emb (full).
    #     Reproduces the original A0…A6 dispatch.
    #   - "full"  : always on the full latent ``emb``, even when k_prog > 0.
    #     Use for the A2_full ablation (subspace split off, triplet on z).
    #   - "prog"  : always on the progression subspace; error if k_prog == 0.
    if lambd_T > 0.0:
        triplet_target = cfg.loss.triplet.get("target", "auto")
        if triplet_target == "prog" and k_prog == 0:
            raise ValueError(
                "loss.triplet.target='prog' requires wm.k_prog > 0; got 0."
            )
        if triplet_target == "full" or (triplet_target == "auto" and k_prog == 0):
            triplet_input = emb
        else:
            triplet_input = output["emb_prog"]
        output["trip_loss"] = cosine_triplet_loss(
            triplet_input,
            margin=cfg.loss.triplet.get("margin", 0.2),
            window_tau=int(cfg.loss.triplet.get("window_tau", 1)),
        )
        total = total + lambd_T * output["trip_loss"]

    if k_prog > 0 and lambd_P > 0.0:
        emb_prog = output["emb_prog"]  # (B, T, k)
        # Dispatch on proto.mode:
        #   "hard" (default) -- pull window's first/last frames toward
        #                      e_1/e_2. Same as the original behaviour;
        #                      empirically broken under sub-window sampling
        #                      because middle-of-episode windows mis-anchor.
        #   "first_anchor"   -- Sketch A-self: require the loader to provide
        #                      ``batch["step_idx"]`` and pull only the genuine
        #                      first-of-episode frames toward e_1. End frames
        #                      are not anchored. No t/T injection; basis is
        #                      aligned at the start, free per-scene at the end.
        proto_mode = cfg.loss.proto.get("mode", "hard")
        if proto_mode == "first_anchor":
            if "step_idx" not in batch:
                raise KeyError(
                    "loss.proto.mode='first_anchor' requires batch['step_idx']; "
                    "add ``step_idx`` to the data config's keys_to_load."
                )
            k_f = int(cfg.loss.proto.get("k_f", 3))
            output["proto_loss"] = prototype_loss_first_anchor(
                emb_prog, batch["step_idx"], self.model.prototypes, k_f=k_f,
            )
        else:
            output["proto_loss"] = prototype_loss(
                emb_prog[:, 0], emb_prog[:, -1], self.model.prototypes
            )
        total = total + lambd_P * output["proto_loss"]

    if lambd_str > 0.0:
        output["straight_loss"] = straightening_loss(emb)
        total = total + lambd_str * output["straight_loss"]

    output["loss"] = total

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset = HDF5Dataset(**dataset_cfg, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            # Skip integer-valued metadata columns (e.g. step_idx, episode_idx)
            # — they're consumed by losses (e.g. prototype_loss_first_anchor)
            # but should not go through z-score normalisation.
            if col in ("step_idx", "episode_idx", "ep_idx"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    # When (θ, r) conditioning is enabled, the predictor's AdaLN cond input
    # is ``act_emb`` concatenated with 3 polar features (sin θ, cos θ, r).
    k_prog = int(cfg.wm.get("k_prog", 0))
    use_polar = bool(cfg.wm.get("use_polar", False)) and k_prog >= 2
    cond_input_dim = embed_dim + (3 if use_polar else 0)

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        cond_input_dim=cond_input_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        embed_dim=embed_dim,
        k_prog=k_prog,
        use_polar=use_polar,
        gamma_theta=float(cfg.get("plan_cost", {}).get("gamma_theta", 0.0)),
        delta_r=float(cfg.get("plan_cost", {}).get("delta_r", 0.0)),
    )

    # ``LinearWarmupCosineAnnealingLR`` requires explicit warmup/max step
    # counts. With ``interval: step`` these are raw optimiser steps, so we
    # compute ``max_steps = steps_per_epoch * max_epochs`` and use a 5%
    # warmup. Using step-level granularity is preferable to epoch-level here
    # because the cosine decay is smoother than stepping once per epoch.
    steps_per_epoch = len(train)
    max_epochs = int(cfg.trainer.max_epochs)
    max_steps = steps_per_epoch * max_epochs
    warmup_steps = max(1, max_steps // 20)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": warmup_steps,
                "max_steps": max_steps,
            },
            "interval": "step",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    # ``swm.policy.AutoCostModel`` resolves checkpoint paths as
    # ``<cache_dir>/checkpoints/<run_name>/<file>_object.ckpt`` (the same
    # ``checkpoints/`` prefix convention that ``HDF5Dataset`` uses for
    # ``datasets/``), so we mirror that layout when writing. Pre-fix
    # behaviour wrote to ``<cache_dir>/<run_name>/...`` and the trailing
    # eval call would AssertionError on path resolution.
    run_dir = Path(swm.data.utils.get_cache_dir(), "checkpoints", run_id)

    logger = None
    if cfg.wandb.enabled:
        # Redirect wandb output away from ./wandb/ because
        # ``stable_pretraining.Manager.init_and_sync_wandb()`` scans that
        # directory at startup and (buggily) tries to "resume" from any
        # offline-run dir it finds — including the one our WandbLogger just
        # created moments earlier in this same process. Writing to a per-run
        # subdir keeps ./wandb/ empty so the Manager's scan finds nothing.
        wandb_save_dir = Path("wandb_runs") / (run_id or "default")
        wandb_save_dir.mkdir(parents=True, exist_ok=True)
        logger = WandbLogger(save_dir=str(wandb_save_dir), **cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    # NOTE: enable_checkpointing=False is deliberate. Lightning's default
    # ModelCheckpoint callback writes a redundant ``last.ckpt`` (full ViT +
    # predictor + projector, ~300-500 MB) into
    # ``<default_root_dir>/lightning_logs/version_*/checkpoints/`` at every
    # epoch end. We already dump the model via ``ModelObjectCallBack`` into
    # ``run_dir`` and the optimiser/scheduler state via ``spt.Manager``'s
    # ``ckpt_path``. Keeping Lightning's default on top doubled disk usage
    # and previously filled cluster quotas (ENOSPC at end of epoch 0).
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=False,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    # ── Work around stable_pretraining bug ────────────────────────────────
    # ``Manager.init_and_sync_wandb()`` scans the WandbLogger's save_dir on
    # startup and tries to "resume" from any offline-run it finds — including
    # the one the Lightning WandbLogger just created in this same process.
    # It then crashes on ``None / "files/wandb-config.json"`` when the match
    # search returns None. We already set up wandb ourselves via the
    # WandbLogger above, so disabling this method is safe: Lightning's logger
    # handles offline writes correctly on its own.
    manager.init_and_sync_wandb = lambda *a, **kw: None

    # ── Force-remove any ModelCheckpoint callback ────────────────────────────
    # ``enable_checkpointing=False`` on the Trainer disables Lightning's
    # *default* ModelCheckpoint, but ``spt.Manager`` (and potentially
    # ``spt.Module``) explicitly injects its own ModelCheckpoint into the
    # trainer's callback list at construction time — for its "wandb-resume
    # info embedded in ckpt" feature. That extra checkpoint writes a
    # 300-500 MB ``last.ckpt`` on every ``on_train_epoch_end`` via
    # Lightning's ``_atomic_save``, which has caused ENOSPC at the end of
    # epoch 0 on quota-tight clusters. We already dump the model via
    # ``ModelObjectCallBack`` into ``run_dir``, so nuke the duplicate
    # unconditionally.
    from lightning.pytorch.callbacks import ModelCheckpoint as _LMCk
    _removed = [cb for cb in trainer.callbacks if isinstance(cb, _LMCk)]
    trainer.callbacks = [cb for cb in trainer.callbacks if not isinstance(cb, _LMCk)]
    if _removed:
        print(
            f"[train.py] Stripped {len(_removed)} ModelCheckpoint callback(s) "
            f"injected downstream of Trainer construction: "
            f"{[type(cb).__name__ for cb in _removed]}"
        )

    manager()
    return


if __name__ == "__main__":
    run()
