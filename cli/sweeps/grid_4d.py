"""4D grid sweep execution (online learning only)."""
import logging
from typing import Dict

from simulation.core import Simulation

from cli.types import RunRequest

logger = logging.getLogger("SubspaceNet.cli")


def run_4d_grid_sweep(sim: Simulation, request: RunRequest) -> Dict:
    params = request.grid_params
    if params is None:
        raise ValueError("4D grid sweep requires grid_params on RunRequest")

    scenario_results = {}
    total = (
        len(params.eta_values)
        * len(params.process_noise_values)
        * len(params.kf_process_noise_values)
        * len(params.kf_measurement_noise_values)
    )
    count = 0

    for proc_noise in params.process_noise_values:
        scenario_results[proc_noise] = {}
        for kf_proc_noise in params.kf_process_noise_values:
            scenario_results[proc_noise][kf_proc_noise] = {}
            for kf_meas_noise in params.kf_measurement_noise_values:
                scenario_results[proc_noise][kf_proc_noise][kf_meas_noise] = {}
                for eta in params.eta_values:
                    count += 1
                    logger.info(
                        "4D grid %d/%d: pn=%s kf_pn=%s kf_mn=%s eta=%s",
                        count, total, proc_noise, kf_proc_noise, kf_meas_noise, eta,
                    )
                    grid_overrides = [
                        f"online_learning.max_eta={eta}",
                        f"online_learning.eta_increment={eta}",
                        f"trajectory.sine_accel_noise_std={proc_noise}",
                        f"trajectory.mult_noise_base_std={proc_noise}",
                        f"trajectory.random_walk_std_dev={proc_noise}",
                        f"kalman_filter.process_noise_std_dev={kf_proc_noise}",
                        f"kalman_filter.measurement_noise_std_dev={kf_meas_noise}",
                    ]
                    result = sim._run_sweep_iteration(
                        grid_overrides,
                        "4d_grid",
                        (proc_noise, kf_proc_noise, kf_meas_noise, eta),
                        f"4d_grid_pn{proc_noise}_kf_pn{kf_proc_noise}_kf_mn{kf_meas_noise}_eta{eta}",
                        full_mode=False,
                        goal=request.goal.value,
                    )
                    scenario_results[proc_noise][kf_proc_noise][kf_meas_noise][eta] = result

    return scenario_results
