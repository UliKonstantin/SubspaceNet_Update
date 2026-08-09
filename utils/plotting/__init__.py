"""Public plotting API — re-exports from domain modules."""
from utils.plotting.evaluation import plot_loss_vs_scenario, plot_2d_kalman_noise_sweep, plot_eval_dnn_ekf_loss_vs_time
from utils.plotting.sweeps import plot_eta_comparison_4d_grid, plot_scenario_results, plot_eta_scenario_comparison, plot_performance_improvement_table, plot_lr_sweep_heatmap, plot_performance_improvement_table_eta, SCENARIO_AXIS_LABELS, SCENARIO_PLOT_TITLES
from utils.plotting.online_learning import plot_online_learning_results_structured, plot_averaged_online_learning_results, plot_averaged_kf_gain_comparison, plot_training_curves, plot_glrt_averaged_drift_results, plot_single_online_learning_run, plot_online_learning_results, plot_online_learning_trajectory
from utils.plotting.lr_plots import plot_optimal_lr_vs_eta, plot_loss_vs_eta_per_lr, plot_glrt_observable_to_optimal_lr

__all__ = [
    "plot_loss_vs_scenario",
    "plot_2d_kalman_noise_sweep",
    "plot_eval_dnn_ekf_loss_vs_time",
    "plot_eta_comparison_4d_grid",
    "plot_scenario_results",
    "plot_eta_scenario_comparison",
    "plot_performance_improvement_table",
    "plot_lr_sweep_heatmap",
    "plot_performance_improvement_table_eta",
    "plot_online_learning_results_structured",
    "plot_averaged_online_learning_results",
    "plot_averaged_kf_gain_comparison",
    "plot_training_curves",
    "plot_glrt_averaged_drift_results",
    "plot_single_online_learning_run",
    "plot_online_learning_results",
    "plot_online_learning_trajectory",
    "plot_optimal_lr_vs_eta",
    "plot_loss_vs_eta_per_lr",
    "plot_glrt_observable_to_optimal_lr",
    "SCENARIO_AXIS_LABELS",
    "SCENARIO_PLOT_TITLES",
]
