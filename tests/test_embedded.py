"""Host-compiled checks for the portable C core (no HAL or hardware needed)."""

import ctypes
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from traditional.model import PhysicalParams, continuous_state_space, discretize_euler

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def core(tmp_path_factory):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler unavailable")
    output = tmp_path_factory.mktemp("embedded") / "control.so"
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            str(ROOT / "traditional/embedded/lqr_math.c"),
            str(ROOT / "traditional/embedded/robot_control.c"),
            "-o",
            str(output),
            "-lm",
        ],
        check=True,
    )
    return ctypes.CDLL(str(output))


def test_embedded_dynamics_agree_with_python(core):
    ad = np.zeros((8, 8), dtype=np.float32)
    bd = np.zeros(8, dtype=np.float32)
    core.lqr_build_model(
        ctypes.c_void_p(ad.ctypes.data), ctypes.c_void_p(bd.ctypes.data)
    )
    params = PhysicalParams(1, np.full(3, 0.3), np.full(3, 0.1))
    expected_a, expected_b = discretize_euler(*continuous_state_space(params), 0.005)
    np.testing.assert_allclose(ad, expected_a, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(bd, expected_b[:, 0], atol=1e-6, rtol=1e-4)


class Controller(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float * 8),
        ("k", ctypes.c_float * 8),
        ("previous_q", ctypes.c_float * 4),
        ("has_previous_q", ctypes.c_bool),
    ]


class Sensors(ctypes.Structure):
    _fields_ = [
        ("cart_count", ctypes.c_int32),
        ("joint_count", ctypes.c_int32 * 3),
        ("left", ctypes.c_bool),
        ("right", ctypes.c_bool),
        ("estop_ok", ctypes.c_bool),
    ]


@pytest.mark.parametrize("dt", [0, -0.1, float("nan"), float("inf"), 0.1])
def test_embedded_bad_timing_commands_zero(core, dt):
    controller = Controller()
    sensors = Sensors(estop_ok=True)
    force, pwm = ctypes.c_float(1), ctypes.c_float(1)
    core.robot_controller_tick.restype = ctypes.c_bool
    valid = core.robot_controller_tick(
        ctypes.byref(controller),
        ctypes.byref(sensors),
        ctypes.c_float(dt),
        ctypes.byref(force),
        ctypes.byref(pwm),
    )
    assert not valid
    assert force.value == pwm.value == 0


def test_embedded_gain_initialization_is_stable(core):
    controller = Controller()
    core.robot_controller_init.restype = ctypes.c_bool
    assert core.robot_controller_init(ctypes.byref(controller))
    gain = np.ctypeslib.as_array(controller.k)
    assert np.isfinite(gain).all()
    params = PhysicalParams(1, np.full(3, 0.3), np.full(3, 0.1))
    ad, bd = discretize_euler(*continuous_state_space(params), 0.005)
    assert np.max(np.abs(np.linalg.eigvals(ad - bd @ gain[None, :]))) < 1
