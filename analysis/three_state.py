"""Conservative C -> O -> I -> C mechanotransduction state simulator.

Input is measured cantilever work relative to a cell's fitted work threshold.
Rate parameters and recovery time are hypotheses, not identified by this data.
The stimulus interval in the Duke protocol is fixed at 5 s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rates:
    opening_per_s: float
    inactivation_per_s: float
    recovery_per_s: float
    work_half_scale_fJ: float


def stimulate(work_fJ: np.ndarray, threshold_fJ: float, dt_s: float,
              rates: Rates, initial: tuple[float, float, float]) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Return open probability for one trace and state after its last sample."""
    if dt_s <= 0 or rates.work_half_scale_fJ <= 0 or min(initial) < 0:
        raise ValueError("Invalid time step, work scale, or initial state")
    if not np.isclose(sum(initial), 1.0):
        raise ValueError("State fractions must sum to one")
    if min(rates.opening_per_s, rates.inactivation_per_s, rates.recovery_per_s) < 0:
        raise ValueError("Rates must be nonnegative")
    closed, open_, inactive = map(float, initial)
    output = np.empty(len(work_fJ), dtype=float)
    for j, work in enumerate(work_fJ):
        drive = max(float(work) - threshold_fJ, 0.0)
        opening = rates.opening_per_s * drive / (rates.work_half_scale_fJ + drive)
        to_open = closed * -np.expm1(-opening * dt_s)
        to_inactive = open_ * -np.expm1(-rates.inactivation_per_s * dt_s)
        to_closed = inactive * -np.expm1(-rates.recovery_per_s * dt_s)
        closed += -to_open + to_closed
        open_ += to_open - to_inactive
        inactive += to_inactive - to_closed
        output[j] = open_
    state = (closed, open_, inactive)
    if not np.all(np.isfinite(state)) or min(state) < -1e-10 or not np.isclose(sum(state), 1.0, atol=1e-10):
        raise ArithmeticError("Probability conservation failed")
    return output, state


def recover(state: tuple[float, float, float], interval_s: float, rates: Rates) -> tuple[float, float, float]:
    """Propagate O -> I -> C between stimuli with zero mechanical drive."""
    if interval_s < 0:
        raise ValueError("Interval must be nonnegative")
    _, open_0, inactive_0 = state
    ki, kr = rates.inactivation_per_s, rates.recovery_per_s
    open_ = open_0 * np.exp(-ki * interval_s)
    if np.isclose(ki, kr):
        inactive = np.exp(-kr * interval_s) * (inactive_0 + ki * open_0 * interval_s)
    else:
        inactive = inactive_0 * np.exp(-kr * interval_s) + ki * open_0 * (
            np.exp(-kr * interval_s) - np.exp(-ki * interval_s)
        ) / (ki - kr)
    closed = 1 - open_ - inactive
    result = (float(closed), float(open_), float(inactive))
    if min(result) < -1e-10 or not np.isclose(sum(result), 1.0, atol=1e-10):
        raise ArithmeticError("Probability conservation failed")
    return result


if __name__ == "__main__":
    example = np.r_[np.linspace(0, 200, 150), np.linspace(200, 0, 200)]
    for tau_s in (0.1, 1.0, 5.0, 10.0, 100.0):
        rates = Rates(100.0, 20.0, 1 / tau_s, 20.0)
        first, end = stimulate(example, 30.0, 0.001, rates, (1.0, 0.0, 0.0))
        second_start = recover(end, 4.65, rates)
        second, _ = stimulate(example, 30.0, 0.001, rates, second_start)
        print({"assumed_recovery_tau_s": tau_s,
               "available_before_second": round(second_start[0], 3),
               "second_first_peak_ratio": round(float(np.max(second) / np.max(first)), 3)})
