"""Independent mechanics checks and physical-controller failure handling."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from traditional.encoders import FiniteDifferenceVelocity
from traditional.hardware import SensorPacket, SerialRobotIO
from traditional.model import (
    PhysicalParams,
    mass_matrix_upright,
    continuous_state_space,
    discretize_euler,
    bryson_q_r,
)
from traditional.riccati import solve_discrete_riccati_iteration, lqr_gain
from traditional.run_controller import load_controller


@pytest.mark.parametrize("n", [1, 2, 3])
def test_mass_matrix_matches_finite_difference_com_kinetic_energy(n):
    lengths = np.linspace(0.3, 0.8, n)
    masses = np.linspace(0.1, 0.4, n)
    params = PhysicalParams(1.2, lengths, masses)

    def com(q, rod):
        angles = np.cumsum(q[1:])
        segments = np.column_stack([np.sin(angles), np.cos(angles)]) * lengths[:, None]
        return np.array([q[0], 0]) + segments[:rod].sum(axis=0) + segments[rod] / 2

    expected = np.zeros((n + 1, n + 1))
    expected[0, 0] = params.cart_mass_kg
    epsilon = 1e-6
    for i in range(n):
        jac = np.column_stack(
            [
                (
                    com(np.eye(n + 1)[j] * epsilon, i)
                    - com(-np.eye(n + 1)[j] * epsilon, i)
                )
                / (2 * epsilon)
                for j in range(n + 1)
            ]
        )
        angular = np.zeros(n + 1)
        angular[1 : i + 2] = 1
        expected += masses[i] * jac.T @ jac + masses[i] * lengths[
            i
        ] ** 2 / 12 * np.outer(angular, angular)
    np.testing.assert_allclose(mass_matrix_upright(params), expected, atol=1e-9)


def test_riccati_matches_scipy_and_stabilizes_three_links():
    params = PhysicalParams(1, np.full(3, 0.3), np.full(3, 0.1))
    ad, bd = discretize_euler(*continuous_state_space(params), 0.005)
    q, r = bryson_q_r(np.array([0.4, 0.1, 0.1, 0.1, 1.5, 3, 3, 3]), 12)
    p = solve_discrete_riccati_iteration(ad, bd, q, r)
    np.testing.assert_allclose(p, solve_discrete_are(ad, bd, q, r), rtol=1e-6)
    assert np.max(np.abs(np.linalg.eigvals(ad - bd @ lqr_gain(ad, bd, p, r)))) < 1


@pytest.mark.parametrize("timestamp", [0, -1, np.nan, np.inf])
def test_velocity_rejects_stale_or_invalid_time(timestamp):
    estimator = FiniteDifferenceVelocity(4)
    estimator.update(np.zeros(4), 0)
    with pytest.raises(ValueError):
        estimator.update(np.zeros(4), timestamp)


def test_angle_wrap_does_not_cause_velocity_spike():
    estimator = FiniteDifferenceVelocity(4)
    estimator.update(np.array([0, np.pi - 0.01, 0, 0]), 0)
    velocity = estimator.update(np.array([0, -np.pi + 0.01, 0, 0]), 0.01)
    assert velocity[1] == pytest.approx(2)


def test_controller_stops_for_estop_limits_and_nonfinite_state():
    controller = load_controller("traditional/config.yaml")
    packet = SensorPacket(0, 0, (0, 0, 0), False, False, True)
    _, force, pwm = controller.command_from_packet(packet)
    assert force == pwm == 0
    for bad in [
        replace(packet, estop_ok=False),
        replace(packet, left_limit=True),
        replace(packet, right_limit=True),
    ]:
        with pytest.raises(RuntimeError):
            controller.check_safety(bad, np.zeros(8))
    with pytest.raises(RuntimeError, match="finite"):
        controller.check_safety(packet, np.full(8, np.nan))


class FakeSerial:
    def __init__(self, line=b""):
        self.line = line
        self.closed = False

    def readline(self):
        return self.line

    def write(self, value):
        raise OSError("disconnected")

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    "line", [b"S,nan,0,0,0,0,0,0,1", b"S,1,0,0,0,0,0,0,2", b"S,1,0"]
)
def test_serial_rejects_malformed_packet(line):
    io = SerialRobotIO.__new__(SerialRobotIO)
    io._serial = FakeSerial(line)
    io._timeout_s = 0.02
    with pytest.raises(ValueError):
        io.read_sensor_packet()


def test_serial_closes_even_if_stop_write_fails():
    io = SerialRobotIO.__new__(SerialRobotIO)
    io._serial = FakeSerial()
    with pytest.raises(OSError):
        io.close()
    assert io._serial.closed


def test_non_sensor_traffic_cannot_bypass_read_deadline(monkeypatch):
    io = SerialRobotIO.__new__(SerialRobotIO)
    io._serial = FakeSerial(b"debug output")
    io._timeout_s = 0.02
    ticks = iter([0.0, 0.005, 0.03])
    monkeypatch.setattr("traditional.hardware.time.monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="deadline"):
        io.read_sensor_packet()
