# Physical three-link cart-pendulum controller

This is the physical-control side of the project: a ground-up linear dynamics
model and LQR controller, with Python serial control and a portable C core for
an STM32 Nucleo F446RE / VNH5019 motor-driver prototype.

**Status:** software prototype with example calibration values. The repository
contains control mathematics and firmware integration templates, not a complete
flashable CubeMX project or a verified hardware balancing demonstration. The RL
policies in `models/` are not deployed by this controller.

## Two deployment paths

| Path | Control responsibility | Status |
|---|---|---|
| Python + serial | STM32 streams encoders; Python estimates state and sends PWM | Implemented protocol boundary; needs board firmware and calibration |
| On-board C | STM32 estimates state and computes LQR feedback locally | Portable math/controller core; HAL integration hooks are templates |

Both paths use a linearized uniform-rod model near the upright equilibrium.
They do not provide swing-up or recovery from arbitrary initial positions.

## State and model

```text
x = [p, theta1, theta2, theta3, p_dot, theta1_dot, theta2_dot, theta3_dot]
u = -K x
```

Angles are **relative joint angles** measured from upright encoder zeros.
Absolute rod orientations are cumulative sums of the relative angles. A change
in one hinge rotates every downstream segment; the mass matrix includes those
contributions consistently in Python and C.

The default period is 5 ms (200 Hz). `model.py` builds the upright mass/gravity
matrices, discretizes with forward Euler, and constructs Bryson-rule Q/R weights.
`riccati.py` solves the discrete Riccati equation by iteration using
`numpy.linalg.solve`. Regression tests compare the result with SciPy's independent
solver and check the closed-loop eigenvalues. The C implementation uses double
precision for the one-time Riccati solve and float values in the control loop.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Example mechanical parameters, calibration, costs, and limits |
| `model.py`, `riccati.py` | Mechanics and gain calculation |
| `encoders.py` | Calibration, angle wrapping, timestamp-checked velocity estimation |
| `controller.py` | State assembly, interlock checks, force/PWM saturation |
| `hardware.py` | Validated sensor-packet parsing and serial commands |
| `run_controller.py` | Python control-loop entry point |
| `firmware_protocol.md` | Wire format and firmware watchdog requirements |
| `embedded/` | Portable C math/control core and HAL integration notes |
| `embedded/PIN_PLAN.md` | Proposed board/peripheral allocation |
| `firmware_nucleo_f446re_skeleton.c` | Firmware skeleton requiring integration |

## Software checks

From the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/test_control.py tests/test_embedded.py -q
python -m pip install -r requirements-hardware.txt
python -m traditional.run_controller --help
```

The embedded tests use the host C compiler (`cc`) and skip if it is unavailable.
They validate the portable core, not the STM32 toolchain, timing deadlines,
peripheral configuration, wiring, or motor behavior.

## Bring-up sequence

1. Measure the actual masses, lengths, transmission ratio, and encoder resolution.
   Update both `config.yaml` and `embedded/lqr_config.h` for the chosen path.
2. Complete the firmware HAL hooks and independent e-stop, endstop, and
   communication-timeout interlocks. Check the pin plan against the actual build.
3. With the pendulum removed, verify motor direction using a conservative PWM limit.
4. Verify cart/joint encoder signs, counts per unit, and upright zeros.
5. Set the serial port and run sensor-only observation:

   ```bash
   python -m traditional.run_controller --config traditional/config.yaml --dry-run
   ```

   This requires live hardware. It sends zero PWM while displaying the calculated
   force and PWM; it does not simulate packets.
6. After interlocks and calibration are validated, enable the control loop with
   conservative bounds and the pendulum near upright:

   ```bash
   python -m traditional.run_controller --config traditional/config.yaml
   ```

Wrong encoder signs destabilize feedback. The Python process cannot enforce a
motor stop if the serial connection or host fails; the firmware watchdog and
physical emergency stop must independently disable the drive. The PWM-to-force
mapping is a calibration placeholder, and friction, backlash, latency, and motor
dynamics are not identified by this model.
