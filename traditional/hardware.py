from __future__ import annotations

from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class SensorPacket:
    time_s: float
    cart_count: int
    joint_counts: tuple[int, int, int]
    left_limit: bool
    right_limit: bool
    estop_ok: bool


class RobotIO:
    def read_sensor_packet(self) -> SensorPacket:
        raise NotImplementedError

    def command_pwm(self, pwm: float) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        self.command_pwm(0.0)

    def close(self) -> None:
        pass


class SerialRobotIO(RobotIO):
    """Line-oriented serial interface to STM32 firmware.

    Expected incoming packet:
        S,<time_s>,<cart_count>,<j1>,<j2>,<j3>,<left_limit>,<right_limit>,<estop_ok>

    Outgoing command:
        C,<pwm>
    """

    def __init__(self, port: str, baudrate: int, timeout_s: float):
        try:
            import serial
        except ImportError as exc:
            raise ImportError(
                "Install pyserial to use SerialRobotIO: pip install pyserial"
            ) from exc

        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("serial timeout must be positive and finite")
        self._timeout_s = timeout_s
        self._serial = serial.Serial(
            port=port, baudrate=baudrate, timeout=timeout_s, write_timeout=timeout_s
        )

    def read_sensor_packet(self) -> SensorPacket:
        deadline = time.monotonic() + self._timeout_s
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("no valid sensor packet before deadline")
            line = self._serial.readline().decode("ascii", errors="replace").strip()
            if not line:
                raise TimeoutError("timed out waiting for sensor packet")
            if not line.startswith("S,"):
                continue

            parts = line.split(",")
            if len(parts) != 9:
                raise ValueError(f"bad sensor packet: {line}")

            if any(flag not in ("0", "1") for flag in parts[6:]):
                raise ValueError("sensor safety flags must be 0 or 1")
            timestamp = float(parts[1])
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError("invalid sensor timestamp")
            return SensorPacket(
                time_s=timestamp,
                cart_count=int(parts[2]),
                joint_counts=(int(parts[3]), int(parts[4]), int(parts[5])),
                left_limit=bool(int(parts[6])),
                right_limit=bool(int(parts[7])),
                estop_ok=bool(int(parts[8])),
            )

    def command_pwm(self, pwm: float) -> None:
        if not math.isfinite(pwm):
            raise ValueError("PWM must be finite")
        pwm = max(-1.0, min(1.0, float(pwm)))
        self._serial.write(f"C,{pwm:.6f}\n".encode("ascii"))

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self._serial.close()
