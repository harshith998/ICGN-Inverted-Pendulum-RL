# Architecture and design decisions

## Simulation and representation

`env/mujoco_builder.py` generates a cart slider with a nested chain of hinge joints. Zero joint angles point the rods upward; angles are **relative to the parent link**. `VariablePendulumEnv` rebuilds the model when it samples a new physical configuration at reset.

A graph has one cart node plus one node per link, and two directed edges per rod. Observations are padded to the configured capacity and include the true node/edge counts for masking. The nine node features are type flags, sine/cosine of the relative angle, angular velocity, cart position/velocity, and cart mass. Edges store normalized length and mass.

Normalization uses fixed reference constants: position / 2.5 m, cart velocity / 5 m/s, angular velocity / 10 rad/s, length `(L − 0.3) / 0.9`, rod mass `(m − 0.1) / 1.9`, and cart mass `(M − 0.5) / 2.5`. OOD features can fall outside the nominal normalized range. Do not silently change these constants when loading an existing checkpoint.

The default control interval is 4 ms (four 1 ms MuJoCo steps). Termination checks use relative joint angles and cart travel; the environment includes an initial grace period. It starts near upright, so this is a balancing task rather than swing-up control.

## Learning

MLP policies flatten the padded graph. MPNNs aggregate local messages. Graph transformers use attention and edge features. CGAT introduces a normalized analytic mass-matrix bias into attention; its variants study per-head/directional scaling, gravity features, and energy-informed critics.

The canonical CGAT implementation is `models/cgat/`. `models/cgat_ppo.py` retains the earlier monolithic implementation with the shared physics helper for historical reference; current training and evaluation use the variant package.

`training/ppo_utils.py` is the shared implementation of observation batching, rollout storage, generalized advantage estimation, and the clipped PPO loss. PPO policies share their actor/critic interface in `models/base_ppo.py`. DQN has separate replay and double-Q update logic in `training/train_dqn.py`.

Episode horizons currently define a finite balancing task: the environment adds a survival bonus and the training buffers treat the horizon as an episode boundary. For a continuing-control formulation, add time-limit value bootstrapping and remove/reconsider the finite-horizon bonus together.

## Physics-informed attention

Graph observations use relative angles. The analytic helpers accumulate these into absolute rod orientations, assemble uniform-rod mechanics, then transform the mass matrix back into relative generalized coordinates before normalizing it. Gravity torques use the same coordinate transformation. The energy feature is gravitational potential energy, despite its historical `compute_hamiltonian` name.

The simulator uses thin capsule geometry; analytic helpers use ideal uniform rods. Their inertias differ slightly. Numerical tests compare against the ideal-rod equations and finite-difference geometry rather than claiming exact capsule equivalence.

**Compatibility:** coordinate corrections change the physics features supplied to CGAT. Historical weights may still load structurally, but require retraining/re-evaluation before their results can be compared with this revision. Earlier experimental claims and result caches are not valid evidence for the corrected implementation.

## Physical control

`traditional/model.py` constructs the upright mass and gravity matrices for relative hinge coordinates. Euler discretization produces the small-step linear model; `riccati.py` iterates the discrete algebraic Riccati equation and computes feedback gain `K`.

The state is `[cart position, three relative angles, cart velocity, three angular velocities]`. Sensor calibration and finite differences construct that state; a controller validates emergency-stop/limit inputs and state bounds before calculating saturated force/PWM commands.

The serial interface and portable C implementation are separate deployment paths. The C math/controller core can be checked on a host compiler; the supplied HAL integration files remain templates. No learned-policy deployment or automatic gain transfer to the board is provided.

## Boundaries and extension points

- Add a policy through the existing PPO/DQN interfaces; register CGAT variants in `models/cgat/__init__.py`.
- Keep mechanics in `env/` and `traditional/model.py`, graph encoding in `graph/`, and experiment orchestration in training/evaluation entry points.
- Preserve exploratory scripts as research tools; `configs/experiments/README.md` separates these from the supported starting configurations.
- Generated checkpoints, plots, and caches belong in ignored output directories. Keep only source, documentation, and reproducible tests in Git.
