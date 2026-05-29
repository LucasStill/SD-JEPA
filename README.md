# JEPA-STEP

Structured progression subspaces for JEPA-style latent world models from
pixels. Extension of [LeWM](https://github.com/lucas-maes/le-wm) that grafts
STEP's progression geometry onto the latent without breaking LeWM's
anti-collapse guarantees.

## Layout

```
jepa-step-wm/
├── pyproject.toml          # Python project + uv lock sources
└── le-wm-step/             # working code tree (fork of le-wm)
    ├── jepa.py             # JEPA module: subspace split + polar
    │                       # conditioning + angular planning cost
    ├── module.py           # AR predictor / Transformer (cond_input_dim aware)
    ├── losses_step.py      # triplet / proto / straight + subspace helpers
    ├── train.py            # training entry point (Hydra)
    ├── eval.py             # planning evaluation entry point (Hydra)
    ├── hdf5_dataset.py     # HDF5 dataset wrapper
    ├── utils.py
    ├── analysis/           # post-hoc diagnostics (probes, geometry, surprise)
    ├── scripts/            # small utilities (status reports, ckpt pruning)
    └── config/             # Hydra configs (train + eval + launchers)
```

## Installation

The project targets Python 3.12 and CUDA 12.8 wheels (Blackwell-friendly).
Dependencies are pinned in `pyproject.toml` and resolved with
[uv](https://docs.astral.sh/uv/).

```bash
uv venv --python=3.12
source .venv/bin/activate
uv sync
```

`uv sync` installs PyTorch from the `pytorch-cu128` index, plus the two
upstream packages this repo builds on — `stable-pretraining` (training) and
`stable-worldmodel` (environments, planning, evaluation).

## Data

Datasets are HDF5 files placed under `$STABLEWM_HOME` (defaults to
`~/.stable_worldmodel`). Override the path if you want them elsewhere:

```bash
export STABLEWM_HOME=/path/to/your/storage
```

Download the LeWM datasets from
[HuggingFace](https://huggingface.co/collections/quentinll/lewm) and
decompress:

```bash
tar --zstd -xvf archive.tar.zst -C "$STABLEWM_HOME"
```

Dataset names in `config/train/data/*.yaml` resolve to
`$STABLEWM_HOME/<name>.h5` (no extension in the config).

## Training

Training is configured via [Hydra](https://hydra.cc/) under
`le-wm-step/config/train/`. Before launching, edit the `wandb.entity` field
in `config/train/lewm.yaml` and `config/train/lewm_step.yaml` to your own
WandB entity (or set `wandb.enabled=False` to disable logging).

```bash
cd le-wm-step

# A0 — LeWM baseline (no progression subspace)
python train.py --config-name lewm

# A2 — subspace split + cosine-margin triplet
python train.py --config-name lewm_step \
    output_model_name=lewm_step_a2 \
    loss.triplet.weight=0.10 loss.proto.weight=0.0 \
    loss.straight.weight=0.0 wm.use_polar=false

# A4 — A2 + straightening
python train.py --config-name lewm_step \
    output_model_name=lewm_step_a4 \
    loss.triplet.weight=0.10 loss.proto.weight=0.0 \
    loss.straight.weight=0.05 wm.use_polar=false

# A5 — A4 + polar (θ, r) predictor conditioning
python train.py --config-name lewm_step \
    output_model_name=lewm_step_a5 \
    loss.triplet.weight=0.10 loss.proto.weight=0.0 \
    loss.straight.weight=0.05 wm.use_polar=true

# A6 — A5 + angular planning cost (set at eval time, see below)
python train.py --config-name lewm_step \
    output_model_name=lewm_step_a6 \
    loss.triplet.weight=0.10 loss.proto.weight=0.0 \
    loss.straight.weight=0.05 wm.use_polar=true
```

Switch dataset with `data=<name>` where `<name>` matches a file under
`config/train/data/` (`pusht`, `cube`, `tworoom`, `reacher`, `ogb`, `dmc`).

Checkpoints are written to
`$STABLEWM_HOME/checkpoints/<run_id>/<output_model_name>_epoch_<N>_object.ckpt`.

## Ablation ladder

Each rung adds one component on top of the previous. The prototype loss
(`loss.proto`) is disabled in all rungs — it was designed for full-episode
training and produces a contradictory signal with LeWM's short
sub-trajectory windows (see "Research notes" below).

| Rung | Description | SIGReg target | triplet | straight | polar cond. | angular cost |
|------|-------------|:-------------:|:-------:|:--------:|:-----------:|:------------:|
| A0   | LeWM baseline | full z | – | – | – | – |
| A2   | + triplet on z^p | z^c | ✓ | – | – | – |
| A4   | + straightening | z^c | ✓ | ✓ | – | – |
| A5   | + polar (θ,r) predictor conditioning | z^c | ✓ | ✓ | ✓ | – |
| A6   | + angular planning cost | z^c | ✓ | ✓ | ✓ | ✓ |

## Evaluation

Evaluation configs live in `config/eval/`. Set the `policy` field to the
checkpoint path relative to `$STABLEWM_HOME` (without the `_object.ckpt`
suffix):

```bash
cd le-wm-step

# MSE planning cost (all rungs)
python eval.py --config-name pusht \
    policy=checkpoints/<run_id>/<output_model_name>_epoch_10

# Angular planning cost (A2 and above)
python eval.py --config-name pusht_step \
    policy=checkpoints/<run_id>/<output_model_name>_epoch_10
```

Results are appended to `$STABLEWM_HOME/pusht_results.txt` or
`pusht_step_results.txt`.

## What's new vs. LeWM

Four loss terms beyond LeWM's two, all gated by Hydra weights. Set all the
new weights to zero and `wm.k_prog=0` and the training loop reproduces LeWM
byte-for-byte.

| Term | Config key | Subspace |
|------|-----------|---------|
| Next-emb prediction | (unchanged) | full |
| SIGReg | `loss.sigreg.weight` | content z^c |
| Cosine-margin triplet | `loss.triplet.weight` | progress z^p |
| Prototype alignment | `loss.proto.weight` | progress z^p (disabled) |
| Straightening | `loss.straight.weight` | full |
| Angular planning | `plan_cost.gamma_theta`, `plan_cost.delta_r` | progress z^p |

## Analysis

`le-wm-step/analysis/` contains post-hoc diagnostics that run on a trained
checkpoint:

- `probe_progress.py` — linear probes of task-progress quantities from
  `(sin θ, cos θ)` vs. full `z`.
- `latent_compass.py` / `latent_geometry.py` — phase / radius geometry of
  the progression subspace.
- `diagnostic_{a,b,c}_*.py` — episode-length sanity, cross-episode latent
  similarity, θ vs. ground-truth state correlation.
- `surprise_voe.py`, `surprise_compare.py`, `regime_change.py` — surprise
  metrics for violation-of-expectation analysis.

Each script takes `--ckpt` and `--cache-dir` arguments; pass them as
`$STABLEWM_HOME/checkpoints/<run>/<run>_epoch_10_object.ckpt` and
`$STABLEWM_HOME/datasets` respectively.

## Research notes

**Why prototype loss is disabled.** `prototype_loss` is called with the
first and last frames of each training sub-window (T=4 frames). Since LeWM
trains on short random chunks of long episodes, these are arbitrary
mid-episode frames — not actual episode starts/ends. Forcing them to fixed
anchors e₁/e₂ produces a contradictory gradient signal that degrades the
representation. A fix would require passing full-episode boundary flags
through the dataloader; the `first_anchor` mode in
`config/train/lewm_step.yaml` sketches such a fix but is not used by
default.
