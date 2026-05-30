"""2D Kalman noise sweep execution (evaluate only)."""
import logging
from typing import Dict

from simulation.core import Simulation

from cli.constants import DEFAULT_KALMAN_MEASUREMENT_NOISE, DEFAULT_KALMAN_PROCESS_NOISE
from cli.types import RunRequest

logger = logging.getLogger("SubspaceNet.cli")


def run_kalman_2d_sweep(sim: Simulation, request: RunRequest) -> Dict:
    if request.sweep_values and len(request.sweep_values) > 1:
        half = len(request.sweep_values) // 2
        meas_values = request.sweep_values[:half]
        proc_values = request.sweep_values[half:]
    else:
        meas_values = DEFAULT_KALMAN_MEASUREMENT_NOISE
        proc_values = DEFAULT_KALMAN_PROCESS_NOISE

    scenario_results = {}
    total = len(meas_values) * len(proc_values)
    count = 0

    for meas_noise in meas_values:
        scenario_results[meas_noise] = {}
        for proc_noise in proc_values:
            count += 1
            logger.info(
                "Kalman 2D combination %d/%d: meas=%s proc=%s",
                count, total, meas_noise, proc_noise,
            )
            overrides = [
                f"kalman_filter.measurement_noise_std_dev={meas_noise}",
                f"kalman_filter.process_noise_std_dev={proc_noise}",
            ]
            result = sim._run_sweep_iteration(
                overrides,
                "kalman_noise",
                (meas_noise, proc_noise),
                f"kalman_noise_m{meas_noise}_p{proc_noise}",
                full_mode=False,
                goal=request.goal.value,
            )
            scenario_results[meas_noise][proc_noise] = result

    sim.results["kalman_noise"] = scenario_results
    return scenario_results
