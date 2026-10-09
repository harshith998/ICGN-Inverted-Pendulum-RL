# Setup and reproducible runs

Run commands from the repository root. Python 3.12 is tested; CPU execution is sufficient for tests and smoke runs. Full multi-seed experiments take substantially longer.

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
```

`requirements.txt` contains the simulation/training dependencies. `requirements-dev.txt` adds pytest and Ruff. `requirements-hardware.txt` provides the smaller NumPy/YAML/serial dependency set for the physical controller. Version ranges express compatibility targets; they are not a byte-for-byte environment lock. Record `python -m pip freeze` alongside any experiment you intend to report.

No display is needed for physics or policy tests. Use `MPLBACKEND=Agg` and `--no-show` for headless training.

## Train one model

```bash
# Fast integration check (640 environment steps)
MPLBACKEND=Agg python -m training.train_cgat --config configs/smoke.yaml --variant base --seed 42 --no-show

# Full runs: choose the controller family
MPLBACKEND=Agg python -m training.train_cgat --config configs/default.yaml --variant base --seed 0 --no-show
MPLBACKEND=Agg python -m training.train_ppo --policy gnn_mpnn --seed 0 --no-show
MPLBACKEND=Agg python -m training.train_ppo --policy gnn_transformer --seed 0 --no-show
MPLBACKEND=Agg python -m training.train_ppo --policy mlp --seed 0 --no-show
MPLBACKEND=Agg python -m training.train_dqn --policy gnn --seed 0 --no-show
MPLBACKEND=Agg python -m training.train_dqn --policy mlp --seed 0 --no-show
```

The default budgets are 2.6 million PPO steps and 2 million DQN steps. PPO collects `n_envs × rollout_steps` transitions per update and rounds down to complete rollouts. All trainers honor the YAML reward and initialization sections. Seeded best checkpoints use names such as `checkpoints/cgat_base_ppo_seed0_best.pt`; seed zero also writes a legacy unseeded alias. Best-checkpoint selection needs at least 20 completed episodes.

`configs/smoke.yaml` deliberately uses short episodes and small networks. Evaluate its checkpoint with the same configuration. It is unsuitable for performance comparisons.

## Evaluate

Train first, then use the same architecture/configuration and seed:

```bash
# Small end-to-end check after the CGAT smoke run
MPLBACKEND=Agg python -m eval.eval_cgat --config configs/smoke.yaml --variant base --seed 42 --tests 1 --n_eval_episodes 2 --n_sweep_points 3

# Example parameter sweeps for a full run
MPLBACKEND=Agg python -m eval.eval_ppo --policy gnn_mpnn --seed 0 --tests 1 2 --n_eval_episodes 20 --n_sweep_points 20

# LQR reference (no learned checkpoint required)
MPLBACKEND=Agg python -m eval.eval_lqr --help
```

Test selectors are `1` for length, `2` for mass, `2.5` for topology, `3` for length × mass heatmaps, and `4` for few-shot adaptation where supported. Use `--help` for each evaluator's available settings. Results and plots are regenerated under `eval/results/` and `eval/plots/`.

For meaningful comparisons, use identical parameter grids, episode budgets, reward settings, initialization distributions, and seed sets. Report individual seeds, variability, failures, and the exact revision/configuration. See [evaluation methodology](docs/EXPERIMENTS.md).

## Multi-seed jobs

```bash
MPLBACKEND=Agg python train_all.py --only ppo_gnn_mpnn cgat_base --seeds 0 1 2
python eval_all.py --only ppo_gnn_mpnn cgat_base --seeds 0 1 2 --tests 1 2 --dry-run
```

The runners use the current Python interpreter. Inspect the dry-run evaluation plan before launching a full grid. Experimental fine-tuning configurations live in `configs/experiments/`; many need a compatible locally trained initialization checkpoint and are not standalone baseline recipes.

## Interactive physics view

```bash
python tests/test_visual.py
```

On macOS, MuJoCo's passive viewer needs its `mjpython` launcher:

```bash
mjpython tests/test_visual.py
```

This is a manually launched three-second physics demonstration. Automated tests do not open a viewer.

## Physical controller

```bash
python -m pip install -r requirements-hardware.txt
python -m traditional.run_controller --help
```

Read [traditional/README.md](traditional/README.md), calibrate `traditional/config.yaml`, and complete firmware integration before connecting the controller to hardware. `--dry-run` still requires a real sensor stream and sends zero PWM; it is not a simulator.
