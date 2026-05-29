#!/usr/bin/env python3
"""Scan jstep_*.out logs and report per-job: dataset, throughput, ETA, status.

Usage::

    python scripts/scan_logs.py [logs/]

Defaults to ./logs. Reports any log modified within the last 2 days.
Reads any job preamble (Dataset / Experiment / Seed / Job ID lines we
echo at the top of each training log) and the last "[Epoch X/Y] step A/B
(C it/s)" line to estimate remaining wall time. Flags runs whose ETA
exceeds a soft budget so you can raise their time limit before they hit
the cap.
"""

from __future__ import annotations

import datetime
import pathlib
import re
import sys

# Soft budget: per-dataset upper-bound expected wall time, after which
# the script flags the run for time-limit extension. Tuned to current
# H100-bf16 throughputs.
SOFT_BUDGET_HOURS = {
    "tworoom": 5.0,
    "pusht":   13.0,
    "reacher": 16.0,
    "cube":    16.0,
    "ogb":     16.0,
}

STEP_RE = re.compile(r"\[Epoch (\d+)/(\d+)\] step (\d+)/(\d+) \(([\d.]+) it/s\)")
FIELD_RE = re.compile(r"^(Dataset|Experiment|Seed|Run name|Job ID)\s+:\s+(.+)$")
EPOCH_DONE_RE = re.compile(r"\[Epoch \d+/\d+\] done in ([\d.]+)s")


def parse_log(path: pathlib.Path) -> dict | None:
    fields: dict[str, str] = {}
    last_step: re.Match | None = None
    last_epoch_time: float | None = None
    done = False

    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return None

    for line in text.splitlines():
        m = FIELD_RE.match(line)
        if m:
            fields[m.group(1)] = m.group(2).strip()
            continue
        m2 = STEP_RE.search(line)
        if m2:
            last_step = m2
            continue
        m3 = EPOCH_DONE_RE.search(line)
        if m3:
            last_epoch_time = float(m3.group(1))
            continue
        if (
            "Training complete" in line
            or ("max_epochs" in line and "reached" in line)
        ):
            done = True

    if not last_step:
        return None

    ep, total_ep, cur_step, total_step, its = last_step.groups()
    ep, total_ep, cur_step, total_step = (
        int(ep), int(total_ep), int(cur_step), int(total_step)
    )
    its = float(its)

    # Remaining steps in this run (assumes ``total_ep`` is the max_epochs).
    remaining_steps = (total_step - cur_step) + (total_ep - ep - 1) * total_step
    eta_h = remaining_steps / its / 3600.0 if its > 0 else float("inf")

    # If we have a measured epoch time, prefer that estimate (more accurate
    # than the noisy moving-average it/s for the first epoch).
    if last_epoch_time is not None and ep > 0:
        eta_h = last_epoch_time / 3600.0 * (total_ep - ep)

    return {
        "jobid": fields.get("Job ID", path.stem.replace("jstep_", "")),
        "dataset": fields.get("Dataset", "?"),
        "exp": fields.get("Experiment", "?"),
        "seed": fields.get("Seed", "?"),
        "progress": f"e{ep}/{cur_step}/{total_step}",
        "its": its,
        "eta_h": eta_h,
        "status": "DONE" if done else "running",
        "mtime": path.stat().st_mtime,
    }


def main():
    logdir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "logs")
    if not logdir.is_dir():
        print(f"directory not found: {logdir}", file=sys.stderr)
        sys.exit(1)

    cutoff = datetime.datetime.now().timestamp() - 2 * 86400  # 2 days ago

    rows = []
    for log in sorted(logdir.glob("jstep_*.out")):
        if log.stat().st_mtime < cutoff:
            continue
        info = parse_log(log)
        if info is not None:
            rows.append(info)

    if not rows:
        print("No active or recent jstep logs found.")
        return

    # Sort by dataset, then by job id
    rows.sort(key=lambda r: (r["dataset"], r["jobid"]))

    headers = ("JOBID", "DATASET", "EXP", "SEED", "PROGRESS",
               "IT/S", "ETA(h)", "STATUS", "FLAG")
    fmt = "{:<10} {:<10} {:<6} {:<5} {:<18} {:<7} {:<8} {:<8} {}"
    print(fmt.format(*headers))
    print("-" * 95)
    for r in rows:
        flag = ""
        budget = SOFT_BUDGET_HOURS.get(r["dataset"], 12.0)
        if r["status"] == "running" and r["eta_h"] > budget:
            flag = f"⚠ ETA > {budget:.0f}h budget — consider scontrol update TimeLimit"
        print(fmt.format(
            r["jobid"], r["dataset"], r["exp"], r["seed"],
            r["progress"], f"{r['its']:.2f}",
            f"{r['eta_h']:.1f}", r["status"], flag,
        ))


if __name__ == "__main__":
    main()
