#!/usr/bin/env python3
"""Parse DIAL LIBERO training logs / checkpoints into process metrics.

Examples:
  # One-shot summary of current bs4 pretrain
  python scripts/monitor_libero_train.py

  # Follow live (refresh every 10s)
  python scripts/monitor_libero_train.py --watch 10

  # Export every logged scalar to CSV
  python scripts/monitor_libero_train.py --csv /tmp/libero_bs4_metrics.csv

  # Point at a specific log / run dir
  python scripts/monitor_libero_train.py \\
    --log outputs/logs/libero_pretrain_bs4_20260826_112814.log \\
    --out-dir /media/weibin/weibin02/DIAL/outputs/libero-pretrain-decoupled-bs4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


STEP_RE = re.compile(
    r"(?P<step>\d+)/(?P<total>\d+)\s*"
    r"\[(?P<elapsed>[^<]+)<(?P<remaining>[^,]+),\s*(?P<speed>[^\]]+)\]"
)
# Trainer dict lines, e.g. {'loss': 1.2, 'grad_norm': 0.5, ...}
DICT_RE = re.compile(r"\{([^{}]+)\}")
KV_RE = re.compile(r"'([^']+)':\s*([^,}]+)")


DEFAULT_LOG_GLOB = "outputs/logs/libero_pretrain*.log"
DEFAULT_OUT_ROOT = Path("/media/weibin/weibin02/DIAL/outputs")


@dataclass
class RunMetrics:
    log_path: Path
    out_dir: Optional[Path] = None
    max_steps: Optional[int] = None
    last_step: int = 0
    last_total: int = 0
    last_elapsed: str = ""
    last_remaining: str = ""
    last_speed: str = ""
    # step -> metric dict (merged trainer + bridge lines nearest in time)
    by_step: dict[int, dict[str, float]] = field(default_factory=dict)
    # raw event stream for CSV (keeps every logged dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_latest_log(pattern: str = DEFAULT_LOG_GLOB) -> Optional[Path]:
    root = repo_root()
    paths = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    # Prefer active bs4 log if present
    for p in paths:
        if "bs4" in p.name:
            return p
    return paths[0] if paths else None


def infer_out_dir(log_path: Path) -> Optional[Path]:
    text_tail = ""
    try:
        with log_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 200_000))
            text_tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r"--output-dir\s+(\S+)|output_dir[=:]\s*(\S+)|out=(/\S+)", text_tail)
    if not m:
        # also search head
        head = log_path.read_text(errors="replace")[:8000]
        m = re.search(r"out=(/\S+)|--output-dir\s+(\S+)", head)
    if not m:
        # heuristic from log name
        if "bs4" in log_path.name:
            p = DEFAULT_OUT_ROOT / "libero-pretrain-decoupled-bs4"
            return p if p.exists() else None
        p = DEFAULT_OUT_ROOT / "libero-pretrain-decoupled"
        return p if p.exists() else None
    cand = next(g for g in m.groups() if g)
    path = Path(cand.rstrip(","))
    return path if path.exists() else path


def parse_log(log_path: Path, out_dir: Optional[Path] = None) -> RunMetrics:
    metrics = RunMetrics(log_path=log_path, out_dir=out_dir)
    current_step = 0
    # bridge/action dicts often print many times per optimizer step (microbatches);
    # we keep a rolling window and attach averages when a trainer {'loss':...} appears.
    pending_bridge: list[dict[str, float]] = []

    def flush_pending(step: int, trainer: dict[str, float]) -> None:
        row = {"step": float(step), **trainer}
        if pending_bridge:
            keys = set().union(*(d.keys() for d in pending_bridge))
            for k in keys:
                vals = [d[k] for d in pending_bridge if k in d]
                if vals:
                    row[k] = sum(vals) / len(vals)
                    row[f"{k}_min"] = min(vals)
                    row[f"{k}_max"] = max(vals)
                    row[f"{k}_n"] = float(len(vals))
            pending_bridge.clear()
        metrics.by_step[step] = row
        metrics.events.append({"kind": "train_step", **row})

    with log_path.open("r", errors="replace") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.replace("\r", "\n").strip()
            if not line:
                continue

            if "Traceback" in line or "CUDA out of memory" in line or "RuntimeError" in line:
                metrics.errors.append(f"L{line_no}: {line[:200]}")

            for sm in STEP_RE.finditer(line):
                current_step = int(sm.group("step"))
                metrics.last_step = current_step
                metrics.last_total = int(sm.group("total"))
                metrics.max_steps = metrics.last_total
                metrics.last_elapsed = sm.group("elapsed").strip()
                metrics.last_remaining = sm.group("remaining").strip()
                metrics.last_speed = sm.group("speed").strip()

            for dm in DICT_RE.finditer(line):
                body = dm.group(1)
                if "'" not in body:
                    continue
                kv: dict[str, float] = {}
                for km in KV_RE.finditer("{" + body + "}"):
                    key, val = km.group(1), km.group(2).strip()
                    try:
                        kv[key] = float(val)
                    except ValueError:
                        continue
                if not kv:
                    continue

                if "bridge_loss" in kv or ("action_loss" in kv and "loss" not in kv):
                    pending_bridge.append(kv)
                    metrics.events.append(
                        {"kind": "microbatch", "step": float(current_step), **kv}
                    )
                elif "loss" in kv:
                    flush_pending(current_step or metrics.last_step, kv)

    return metrics


def read_checkpoints(out_dir: Optional[Path]) -> list[dict[str, Any]]:
    if out_dir is None or not out_dir.exists():
        return []
    rows = []
    for p in sorted(out_dir.glob("checkpoint-*")):
        if not p.is_dir():
            continue
        step = None
        m = re.search(r"checkpoint-(\d+)", p.name)
        if m:
            step = int(m.group(1))
        st = p / "trainer_state.json"
        global_step = step
        if st.exists():
            try:
                data = json.loads(st.read_text())
                global_step = data.get("global_step", step)
            except Exception:
                pass
        size_gb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9
        rows.append(
            {
                "name": p.name,
                "step": global_step,
                "size_gb": round(size_gb, 2),
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
    return rows


def gpu_snapshot() -> list[dict[str, str]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 7:
            continue
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "mem_used_mb": parts[2],
                "mem_total_mb": parts[3],
                "util_pct": parts[4],
                "temp_c": parts[5],
                "power_w": parts[6],
            }
        )
    return rows


def train_process_info() -> list[str]:
    try:
        out = subprocess.check_output(["pgrep", "-af", "dual_system_train"], text=True)
    except subprocess.CalledProcessError:
        return []
    lines = []
    for line in out.splitlines():
        if "pgrep" in line:
            continue
        lines.append(line[:220])
    return lines


def moving_avg(vals: list[float], k: int = 20) -> Optional[float]:
    if not vals:
        return None
    window = vals[-k:]
    return sum(window) / len(window)


def fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


def print_report(m: RunMetrics, recent_n: int = 10) -> None:
    print("=" * 72)
    print(f"time        : {datetime.now().isoformat(timespec='seconds')}")
    print(f"log         : {m.log_path}")
    print(f"out_dir     : {m.out_dir}")
    total = m.last_total or m.max_steps or 0
    pct = (100.0 * m.last_step / total) if total else 0.0
    print(
        f"progress    : {m.last_step}/{total} ({pct:.2f}%)  "
        f"elapsed={m.last_elapsed or '-'}  eta={m.last_remaining or '-'}  "
        f"speed={m.last_speed or '-'}"
    )

    procs = train_process_info()
    print(f"train_proc  : {'RUNNING' if procs else 'NOT FOUND'}")
    for p in procs[:2]:
        print(f"  {p}")

    for g in gpu_snapshot():
        print(
            f"gpu[{g['index']}]    : {g['name']}  "
            f"mem={g['mem_used_mb']}/{g['mem_total_mb']} MiB  "
            f"util={g['util_pct']}%  temp={g['temp_c']}C  power={g['power_w']}W"
        )

    steps = sorted(m.by_step)
    if steps:
        losses = [m.by_step[s]["loss"] for s in steps if "loss" in m.by_step[s]]
        bridges = [m.by_step[s]["bridge_loss"] for s in steps if "bridge_loss" in m.by_step[s]]
        actions = [m.by_step[s]["action_loss"] for s in steps if "action_loss" in m.by_step[s]]
        lrs = [m.by_step[s]["learning_rate"] for s in steps if "learning_rate" in m.by_step[s]]
        grads = [m.by_step[s]["grad_norm"] for s in steps if "grad_norm" in m.by_step[s]]
        print("-" * 72)
        print(
            f"logged_opt_steps : {len(steps)}   "
            f"loss_ma20={fmt(moving_avg(losses))}  "
            f"bridge_ma20={fmt(moving_avg(bridges))}  "
            f"action_ma20={fmt(moving_avg(actions))}"
        )
        if losses:
            print(
                f"loss        : last={fmt(losses[-1])}  "
                f"min={fmt(min(losses))}  max={fmt(max(losses))}"
            )
        if bridges:
            print(
                f"bridge_loss : last={fmt(bridges[-1])}  "
                f"min={fmt(min(bridges))}  max={fmt(max(bridges))}"
            )
        if actions:
            print(
                f"action_loss : last={fmt(actions[-1])}  "
                f"min={fmt(min(actions))}  max={fmt(max(actions))}"
            )
        if lrs:
            print(f"lr          : last={lrs[-1]:.3e}")
        if grads:
            print(f"grad_norm   : last={fmt(grads[-1], 3)}  ma20={fmt(moving_avg(grads), 3)}")

        print("-" * 72)
        print(f"recent optimizer steps (last {recent_n}):")
        print(
            f"{'step':>8}  {'loss':>8}  {'bridge':>8}  {'action':>8}  "
            f"{'lr':>10}  {'grad':>8}"
        )
        for s in steps[-recent_n:]:
            r = m.by_step[s]
            lr_s = f"{r['learning_rate']:.3e}" if "learning_rate" in r else "-"
            print(
                f"{s:8d}  {fmt(r.get('loss')):>8}  {fmt(r.get('bridge_loss')):>8}  "
                f"{fmt(r.get('action_loss')):>8}  "
                f"{lr_s:>10}  "
                f"{fmt(r.get('grad_norm'), 3):>8}"
            )
    else:
        print("-" * 72)
        print("No optimizer-step loss dicts parsed yet (still warming up / loading).")

    ckpts = read_checkpoints(m.out_dir)
    print("-" * 72)
    if ckpts:
        print("checkpoints:")
        for c in ckpts:
            print(f"  {c['name']:20} step={c['step']}  {c['size_gb']} GB  mtime={c['mtime']}")
    else:
        print("checkpoints: (none yet)")

    if m.errors:
        print("-" * 72)
        print(f"errors/warnings matched: {len(m.errors)} (showing last 5)")
        for e in m.errors[-5:]:
            print(f"  {e}")
    print("=" * 72)


def export_csv(m: RunMetrics, csv_path: Path, kind: str = "train_step") -> int:
    rows = [e for e in m.events if e.get("kind") == kind]
    if not rows:
        rows = list(m.events)
    if not rows:
        return 0
    # union of keys
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=None, help="Training log path (default: latest libero_pretrain*.log)")
    ap.add_argument("--out-dir", type=Path, default=None, help="Run output dir (checkpoints/tb)")
    ap.add_argument("--watch", type=float, default=0, help="Refresh every N seconds (0 = once)")
    ap.add_argument("--recent", type=int, default=12, help="How many recent optimizer steps to print")
    ap.add_argument("--csv", type=Path, default=None, help="Export optimizer-step metrics CSV")
    ap.add_argument(
        "--csv-micro",
        type=Path,
        default=None,
        help="Export every microbatch bridge/action dict to CSV (large)",
    )
    ap.add_argument("--json-summary", type=Path, default=None, help="Write one-shot JSON summary")
    args = ap.parse_args()

    log_path = args.log or find_latest_log()
    if log_path is None or not log_path.exists():
        print("ERROR: no training log found. Pass --log PATH", file=sys.stderr)
        return 1

    def once() -> RunMetrics:
        out_dir = args.out_dir or infer_out_dir(log_path)
        m = parse_log(log_path, out_dir=out_dir)
        if args.csv:
            n = export_csv(m, args.csv, kind="train_step")
            print(f"[csv] wrote {n} optimizer steps -> {args.csv}")
        if args.csv_micro:
            n = export_csv(m, args.csv_micro, kind="microbatch")
            print(f"[csv] wrote {n} microbatch rows -> {args.csv_micro}")
        print_report(m, recent_n=args.recent)
        if args.json_summary:
            steps = sorted(m.by_step)
            summary = {
                "log": str(m.log_path),
                "out_dir": str(m.out_dir) if m.out_dir else None,
                "last_step": m.last_step,
                "total_steps": m.last_total,
                "speed": m.last_speed,
                "eta": m.last_remaining,
                "elapsed": m.last_elapsed,
                "n_opt_logs": len(steps),
                "last_metrics": m.by_step[steps[-1]] if steps else None,
                "checkpoints": read_checkpoints(m.out_dir),
                "gpu": gpu_snapshot(),
                "train_running": bool(train_process_info()),
                "errors": m.errors[-10:],
            }
            args.json_summary.parent.mkdir(parents=True, exist_ok=True)
            args.json_summary.write_text(json.dumps(summary, indent=2))
            print(f"[json] wrote {args.json_summary}")
        return m

    if args.watch and args.watch > 0:
        try:
            while True:
                # clear screen for dashboard feel
                sys.stdout.write("\033[2J\033[H")
                once()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nstopped.")
            return 0
    else:
        once()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
