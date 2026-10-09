# Evaluation methodology and limitations

## What each comparison asks

| Experiment | Question | Important constraint |
|---|---|---|
| Length/mass sweep | Does control remain effective as parameters change? | Hold topology and other evaluation settings fixed. |
| Topology sweep | How does performance vary with link count? | Counts included in training are in-distribution. |
| Length × mass grid | Where does the controller fail jointly? | Report cell counts and worst-region behavior, not only the average. |
| Few-shot adaptation | How many additional episodes improve performance? | Use the same adaptation budget and initialization checkpoint. |
| CGAT ablations | Does the physics bias help? | Compare matched no-physics/shuffled controls and multiple seeds. |

`benchmarks/` provides reusable normalized metrics. The pendulum-specific evaluators in `eval/` generate their own sweeps, plots, and adaptation records. These two layers have distinct entry points; inspect the exact evaluator when reporting a score.

## Reproducibility checklist

Record the Git revision, package versions, complete YAML configuration, policy/variant, checkpoint identity, training seed, training budget, evaluation grids, episode counts, and deterministic/stochastic action choice. Use multiple training seeds and report variability. A short smoke test only demonstrates a working pipeline.

The repository's default topology distribution is 1–3 links and the default capacity is three. Testing those same counts is not unseen-topology transfer. A new link count requires enough model/observation capacity; flat MLP input size is checkpoint-dependent.

The LQR evaluator derives its local model from MuJoCo. Its parameter-clamping behavior and the fixed-nominal mode matter when interpreting it as a reference. A task-aware linear controller is not a general performance upper bound on a nonlinear, force-limited task.

## Artifact policy

Generated plots, cached arrays, raw evaluation output, bytecode, and model weights are excluded from Git. Regenerate outputs using SETUP.md and store experiment bundles separately. This keeps the source tree reviewable and avoids presenting hundreds of unlabeled plots as evidence.

The historical engineering log in `docs/history/ITERATIONS.md` records earlier observations and tuning hypotheses. It is not a validated result summary for the current revision. In particular, physics-coordinate and training-configuration fixes require new training/evaluation before reusing old conclusions.

No benchmark leaderboard is published here until comparable runs have been reproduced with the corrected code and their provenance can be supplied.
