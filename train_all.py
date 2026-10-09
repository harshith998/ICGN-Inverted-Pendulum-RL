#!/usr/bin/env python3.12
"""
Train every model sequentially with live per-model and overall progress bars.

Usage
-----
  python3.12 train_all.py
  python3.12 train_all.py --config configs/default.yaml
  python3.12 train_all.py --seeds 0 1 2
  python3.12 train_all.py --skip dqn_mlp cgat_perc
  python3.12 train_all.py --only cgat_base cgat_gravity ppo_gnn_transformer

Jobs (in order)
---------------
  ppo_mlp            — MLP PPO baseline
  ppo_gnn_mpnn       — GNN MPNN PPO
  ppo_gnn_transformer— GNN Transformer PPO
  dqn_mlp            — MLP DQN
  dqn_gnn            — GNN DQN
  cgat_base          — CGAT scalar β (default)
  cgat_perhead       — CGAT per-head β
  cgat_directional   — CGAT directional β_fwd/β_bwd
  cgat_gravity       — CGAT + gravity torque injection
  cgat_perc          — CGAT + PERC critic (w_H init=1)
  cgat_no_physics    — matched no-physics transformer control
  cgat_shuffled      — shuffled inertia-bias control
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
import yaml

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Job definitions ────────────────────────────────────────────────────────────

JOBS = [
    {"name": "ppo_mlp",             "cmd": [sys.executable, "-u", "training/train_ppo.py",  "--policy", "mlp"]},
    {"name": "ppo_gnn_mpnn",        "cmd": [sys.executable, "-u", "training/train_ppo.py",  "--policy", "gnn_mpnn"]},
    {"name": "ppo_gnn_transformer", "cmd": [sys.executable, "-u", "training/train_ppo.py",  "--policy", "gnn_transformer"]},
    {"name": "dqn_mlp",             "cmd": [sys.executable, "-u", "training/train_dqn.py",  "--policy", "mlp"]},
    {"name": "dqn_gnn",             "cmd": [sys.executable, "-u", "training/train_dqn.py",  "--policy", "gnn"]},
    {"name": "cgat_base",           "cmd": [sys.executable, "-u", "training/train_cgat.py", "--variant", "base"]},
    {"name": "cgat_perhead",        "cmd": [sys.executable, "-u", "training/train_cgat.py", "--variant", "perhead"]},
    {"name": "cgat_directional",    "cmd": [sys.executable, "-u", "training/train_cgat.py", "--variant", "directional"]},
    {"name": "cgat_gravity",        "cmd": [sys.executable, "-u", "training/train_cgat.py", "--variant", "gravity"]},
    {"name": "cgat_perc",           "cmd": [sys.executable, "-u", "training/train_cgat.py", "--variant", "perc"]},
    {"name": "cgat_no_physics",     "cmd": [sys.executable, "-u", "training/train_cgat.py", "--variant", "no_physics"]},
    {"name": "cgat_shuffled",       "cmd": [sys.executable, "-u", "training/train_cgat.py", "--variant", "shuffled"]},
]

STEP_RE = re.compile(r"step\s+(\d+)")


def read_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def total_steps_for_job(cfg: dict, job_name: str) -> int:
    if job_name.startswith("dqn_"):
        return cfg["dqn"]["total_steps"]
    return cfg["ppo"]["total_steps"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _print_header(msg: str):
    width = 70
    print(f"\n{'='*width}")
    print(f"  {msg}")
    print(f"{'='*width}")


def _make_bar(total: int, desc: str, position: int = 0):
    if not HAS_TQDM:
        return None
    return tqdm(total=total, desc=desc, position=position,
                unit="step", dynamic_ncols=True, leave=True)


def _update_bar(bar, n: int):
    if bar is not None:
        bar.n = min(n, bar.total)
        bar.refresh()


def _close_bar(bar):
    if bar is not None:
        bar.close()


# ── Run one job ────────────────────────────────────────────────────────────────

def run_job(job: dict, config_path: str, cfg: dict, seed: int,
            job_idx: int, n_jobs: int, log_dir: str) -> bool:
    name = job["name"]
    total_steps = total_steps_for_job(cfg, name)
    cmd  = job["cmd"] + ["--config", config_path, "--no-show", "--seed", str(seed)]

    _print_header(f"[{job_idx}/{n_jobs}]  {name}  seed={seed}")
    print(f"  Command : {' '.join(cmd)}")
    print(f"  Steps   : {total_steps:,}")

    log_path = os.path.join(log_dir, f"{name}_seed{seed}.log")
    print(f"  Log     : {log_path}\n")

    bar     = _make_bar(total_steps, f"  {name}", position=0)
    t_start = time.time()

    with open(log_path, "w") as log_file:
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        cache_root = os.path.join(tempfile.gettempdir(), "inverted_pendulum_train_all")
        mpl_config = os.path.join(cache_root, "matplotlib")
        xdg_cache = os.path.join(cache_root, "cache")
        os.makedirs(mpl_config, exist_ok=True)
        os.makedirs(xdg_cache, exist_ok=True)
        env["MPLCONFIGDIR"] = mpl_config
        env["XDG_CACHE_HOME"] = xdg_cache

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()

            stripped = line.rstrip()
            if not stripped:
                continue

            m = STEP_RE.search(stripped)
            if m:
                step = int(m.group(1))
                _update_bar(bar, step)
            else:
                # Headers, best-model saves, errors — always show these
                if bar is not None:
                    tqdm.write(f"  {stripped}")
                else:
                    print(f"  {stripped}", flush=True)

        proc.wait()

    _update_bar(bar, total_steps)
    _close_bar(bar)

    elapsed = time.time() - t_start
    ok      = proc.returncode == 0
    status  = "DONE" if ok else f"FAILED (exit {proc.returncode})"
    print(f"\n  {name}: {status} in {elapsed / 60:.1f} min  "
          f"(log → {log_path})")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train all models sequentially with progress bars.")
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--skip",    nargs="*", default=[],   metavar="JOB",
        help="Job names to skip  (e.g. --skip dqn_mlp cgat_perc)")
    parser.add_argument("--only",    nargs="*", default=None, metavar="JOB",
        help="Run only these jobs (overrides --skip)")
    parser.add_argument("--log_dir", default="logs",
        help="Directory for per-job log files (default: logs/)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
        help="Seeds to train. Default: 0 1 2")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        sys.exit(f"Config not found: {args.config}")

    cfg = read_config(args.config)
    os.makedirs(args.log_dir, exist_ok=True)

    valid_names = {j["name"] for j in JOBS}
    bad = (set(args.only or []) | set(args.skip)) - valid_names
    if bad:
        print(f"[warn] Unknown job names: {bad}")
        print(f"       Valid names: {sorted(valid_names)}\n")

    jobs = JOBS
    if args.only is not None:
        jobs = [j for j in jobs if j["name"] in args.only]
    else:
        jobs = [j for j in jobs if j["name"] not in args.skip]

    if not jobs:
        sys.exit("No jobs to run after filtering.")

    expanded_jobs = [(job, seed) for seed in args.seeds for job in jobs]
    step_counts = [total_steps_for_job(cfg, j["name"]) for j, _ in expanded_jobs]
    unique_step_counts = sorted(set(step_counts))
    step_summary = (
        f"{unique_step_counts[0]:,} steps each"
        if len(unique_step_counts) == 1
        else "mixed step counts"
    )
    print(f"\ntrain_all  |  {len(expanded_jobs)}/{len(JOBS) * len(args.seeds)} runs  |  {step_summary}")
    print(f"Config     : {args.config}")
    print(f"Seeds      : {args.seeds}")
    print(f"Logs       : {args.log_dir}/")
    if not HAS_TQDM:
        print("[warn] tqdm not installed — run:  pip install tqdm")
    print(f"\nJobs to run:")
    for j, seed in expanded_jobs:
        print(f"  • {j['name']}  seed={seed}")

    overall_bar = _make_bar(len(expanded_jobs), "Overall  ", position=1)
    results     = {}
    t_all       = time.time()

    for i, (job, seed) in enumerate(expanded_jobs, 1):
        ok = run_job(job, args.config, cfg, seed, i, len(expanded_jobs), args.log_dir)
        results[f"{job['name']}_seed{seed}"] = ok
        if overall_bar is not None:
            overall_bar.update(1)

    _close_bar(overall_bar)

    total_elapsed = time.time() - t_all
    _print_header(f"Summary  ({total_elapsed / 60:.1f} min total)")
    for name, ok in results.items():
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {name}")

    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"\n  {len(failed)} job(s) failed: {failed}")
        sys.exit(1)
    else:
        print(f"\n  All {len(results)} jobs completed successfully.")


if __name__ == "__main__":
    main()
