#!/usr/bin/env python3
"""Per-run status table with training progress, ETA, and eval result.

Discovers runs by scanning checkpoint dirs, then for each:
  - parses jstep_<jobid>.out for current epoch/step/it/s,
  - reads the highest epoch checkpoint on disk,
  - searches log for ``'success_rate':`` printout from eval,
  - flags runs whose in-job eval crashed (old-API call).

Output columns:
  EXP, DATASET, SEED, JOBID, LAUNCH, TRAIN, PROGRESS, IT/S, ETA(h), EVAL

Usage::

    python scripts/status_report.py [logs/]
"""

from __future__ import annotations

import datetime
import os
import re
import sys
from pathlib import Path

STABLEWM = Path(
    os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel"))
)
LOG_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs")

# Anchored dataset set so ``A2_kprog4_pusht`` is split correctly into
# (A2_kprog4, pusht) rather than (A2, kprog4_pusht).
DATASETS = ("pusht", "tworoom", "reacher", "cube", "ogb", "dmc")
RUN_RE = re.compile(
    r"^jepa_step_(.+?)_(" + "|".join(DATASETS) + r")_seed(\d+)_(\d+)$"
)
EPOCH_RE = re.compile(r"epoch_(\d+)_object\.ckpt")
SUCCESS_RE = re.compile(r"'success_rate':\s*([\d.]+)")
STEP_RE = re.compile(r"\[Epoch (\d+)/(\d+)\] step (\d+)/(\d+) \(([\d.]+) it/s\)")
EPOCH_DONE_RE = re.compile(r"\[Epoch \d+/\d+\] done in ([\d.]+)s")


def discover_runs() -> set[str]:
    runs: set[str] = set()
    for parent in (STABLEWM / "checkpoints", STABLEWM):
        if not parent.is_dir():
            continue
        for d in parent.iterdir():
            if d.is_dir() and d.name.startswith("jepa_step_"):
                runs.add(d.name)
    return runs


def latest_epoch_for(run: str) -> int | None:
    for parent in (STABLEWM / "checkpoints" / run, STABLEWM / run):
        if not parent.is_dir():
            continue
        epochs = [
            int(m.group(1))
            for f in parent.iterdir()
            if (m := EPOCH_RE.search(f.name))
        ]
        if epochs:
            return max(epochs)
    return None


def parse_log_progress(jobid: str):
    """Return (launch_time_str, progress_str, its, eta_h) from jstep_<jobid>.out
    or (None, None, None, None) if no log."""
    log = LOG_DIR / f"jstep_{jobid}.out"
    if not log.exists():
        return None, None, None, None

    mtime = log.stat().st_mtime  # use mtime as a fallback "still active" signal
    # Use the file's earliest modification time as launch proxy.
    # Better: look for the first "[Epoch 0/" line's timestamp, but mtime works.
    launch_t = datetime.datetime.fromtimestamp(log.stat().st_ctime)
    launch_str = launch_t.strftime("%m-%d %H:%M")

    last_step: re.Match | None = None
    last_epoch_time: float | None = None

    try:
        text = log.read_text(errors="ignore")
    except OSError:
        return launch_str, None, None, None

    for line in text.splitlines():
        m = STEP_RE.search(line)
        if m:
            last_step = m
        m2 = EPOCH_DONE_RE.search(line)
        if m2:
            last_epoch_time = float(m2.group(1))

    if not last_step:
        return launch_str, None, None, None

    ep, total_ep, cur_step, total_step, its = last_step.groups()
    ep, total_ep, cur_step, total_step = (
        int(ep), int(total_ep), int(cur_step), int(total_step)
    )
    its = float(its)
    progress_str = f"e{ep}/{cur_step}/{total_step}"

    remaining_steps = (total_step - cur_step) + (total_ep - ep - 1) * total_step
    eta_h = remaining_steps / its / 3600.0 if its > 0 else float("inf")
    if last_epoch_time and ep > 0:
        eta_h = last_epoch_time / 3600.0 * (total_ep - ep)

    return launch_str, progress_str, its, eta_h


def find_eval_for(run: str, jobid: str):
    """Returns (status, success_rate). status ∈ {OK, FAILED_OLD_API, pending}."""
    candidates = [LOG_DIR / f"jstep_{jobid}.out"]
    candidates.extend(LOG_DIR.glob("eval_*.out"))
    found_failure = False
    for log in candidates:
        if not log.exists():
            continue
        try:
            text = log.read_text(errors="ignore")
        except OSError:
            continue
        if log.name.startswith("eval_") and run not in text:
            continue
        if (
            "TypeError" in text
            and "World.evaluate()" in text
            and "unexpected keyword argument 'dataset'" in text
        ):
            found_failure = True
        m = SUCCESS_RE.search(text)
        if m:
            return ("OK", float(m.group(1)))
    return ("FAILED_OLD_API", None) if found_failure else ("pending", None)


def main():
    runs = discover_runs()
    if not runs:
        print(f"No runs found under {STABLEWM}/checkpoints or {STABLEWM}/")
        return

    rows = []
    for run in sorted(runs):
        m = RUN_RE.match(run)
        if not m:
            continue
        exp, dataset, seed, jobid = m.groups()
        latest = latest_epoch_for(run)
        train_str = (
            "DONE" if latest is not None and latest >= 10
            else f"e{latest}" if latest is not None
            else "—"
        )
        launch, progress, its, eta_h = parse_log_progress(jobid)
        eval_status, success = find_eval_for(run, jobid)
        if success is not None:
            pct = success * 100 if success <= 1.0 else success
            eval_str = f"{pct:.1f}%"
        elif eval_status == "FAILED_OLD_API":
            eval_str = "FAILED"
        elif train_str == "DONE":
            eval_str = "pending"
        else:
            eval_str = "—"

        rows.append({
            "exp": exp,
            "dataset": dataset,
            "seed": seed,
            "jobid": jobid,
            "launch": launch or "—",
            "train": train_str,
            "progress": progress or "—",
            "its": f"{its:.2f}" if its else "—",
            "eta": f"{eta_h:.1f}" if eta_h is not None else "—",
            "eval": eval_str,
            "_run": run,
        })

    rows.sort(key=lambda r: (r["dataset"], r["exp"], int(r["seed"])))

    cols = ["exp", "dataset", "seed", "jobid", "launch", "train",
            "progress", "its", "eta", "eval"]
    headers = ["EXP", "DATASET", "SEED", "JOBID", "LAUNCH", "TRAIN",
               "PROGRESS", "IT/S", "ETA(h)", "EVAL"]
    widths = [12, 9, 5, 7, 12, 6, 18, 6, 7, 9]
    fmt = " ".join(f"{{:<{w}}}" for w in widths)

    print(fmt.format(*headers))
    print("-" * (sum(widths) + len(widths) - 1))
    for r in rows:
        print(fmt.format(*[r[c] for c in cols]))

    n_total = len(rows)
    n_train_done = sum(1 for r in rows if r["train"] == "DONE")
    n_eval_done = sum(1 for r in rows if r["eval"].endswith("%"))
    n_eval_failed = sum(1 for r in rows if r["eval"] == "FAILED")
    print(f"\nSummary: {n_total} runs | {n_train_done} trained to epoch 10 | "
          f"{n_eval_done} eval'd | {n_eval_failed} need eval re-run")


if __name__ == "__main__":
    main()
