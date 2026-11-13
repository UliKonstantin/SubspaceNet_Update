"""
Configuration schema for SubspaceNet using Pydantic models.

This module defines the schema for validating configuration files using Pydantic.
"""

from typing import Optional, List, Literal, Union, Dict, Any, Tuple
from pydantic import BaseModel, Field, model_validator
from enum import Enum


class SystemModelConfig(BaseModel):
    """System model configuration parameters."""
    N: int = 8
    M: int = 3
    T: int = 200
    snr: float = 0
    field_type: Literal["near", "far"] = "far"
    signal_nature: Literal["coherent", "non-coherent"] = "non-coherent"
    signal_type: Literal["narrowband", "broadband"] = "narrowband"
    wavelength: float = 0.06
    eta: float = 0.0
    pitch_eta: float = 0.0
    spacing_scale: float = 1.0
    boresight_offset_deg: float = 0.0
    bias: float = 0.0
    sv_noise_var: float = 0.0
    # frozen: cache δ after each drift jump (OL "array warped"). iid: redraw every
    # steering_vec call like ShlezingerLab/SubspaceNet data_handler (Fig. 9 MC).
    mismatch_sampling: Literal["frozen", "iid"] = "frozen"
    # circular = paper §IV-B-4 CN; real = original git randn (not in the paper).
    sv_noise_type: Literal["circular", "real"] = "circular"
    doa_range: int = 60
    doa_resolution: int = 1
    max_range_ratio_to_limit: float = 0.5
    range_resolution: int = 1
    nominal: bool = True  # If True, no calibration errors are applied in the steering vectors


class DatasetConfig(BaseModel):
    """Dataset configuration parameters."""
    samples_size: int = 4096
    # Dataset splitting [test, validation, train] proportions
    test_validation_train_split: List[float] = [0.2, 0.2, 0.6]
    create_data: bool = False
    save_dataset: bool = False
    true_doa_train: Optional[List[float]] = None
    true_range_train: Optional[List[float]] = None
    true_doa_test: Optional[List[float]] = None
    true_range_test: Optional[List[float]] = None


class ModelParamsConfig(BaseModel):
    """Model parameters configuration."""
    diff_method: Union[Literal["music_2D", "music_1D", "esprit"], tuple[str, str]] = "esprit"
    train_loss_type: Union[Literal["music_spectrum", "rmspe"], tuple[str, str]] = "rmspe"
    tau: int = 8
    field_type: Literal["Near", "Far"] = "Far"
    regularization: Optional[Literal["aic", "mdl", "threshold", "null"]] = None
    variant: Literal["small", "big", "thin"] = "small"
    norm_layer: bool = False
    batch_norm: bool = False
    grid_size: Optional[int] = None
    peak_method: Literal["peaks", "topk", "soft_ste"] = "peaks"
    soft_peak_temperature: float = 0.5
    # DeepCNN offline train: bce (Papageorgiou grid) or rmspe (soft_ste angles, matches OL)
    training_loss: Literal["bce", "rmspe"] = "bce"


class ModelConfig(BaseModel):
    """Model configuration."""
    type: Literal["SubspaceNet", "DCD-MUSIC", "DeepCNN", "IQResNet"] = "SubspaceNet"
    params: ModelParamsConfig = ModelParamsConfig()


class TrainingConfig(BaseModel):
    """Training configuration parameters."""
    enabled: bool = True
    epochs: int = 10
    batch_size: int = 32
    optimizer: Literal["Adam", "SGD", "RMSprop"] = "Adam"
    scheduler: Literal["ReduceLROnPlateau", "StepLR", "CosineAnnealingLR"] = "ReduceLROnPlateau"
    learning_rate: float = 0.001
    weight_decay: float = 1e-9
    step_size: int = 50
    gamma: float = 0.5
    training_objective: Literal["angle", "range", "angle, range"] = "angle"
    use_wandb: bool = False
    simulation_name: Optional[str] = None
    save_checkpoint: bool = True
    target_val_rmspe_rad: Optional[float] = Field(
        default=None,
        description="Stop training when validation RMSPE (radians) is <= this value.",
    )


class SimulationConfig(BaseModel):
    """Simulation configuration parameters."""
    train_model: bool = True
    evaluate_model: bool = True
    load_model: bool = False
    save_model: bool = False
    plot_results: bool = True
    save_plots: bool = False
    model_path: Optional[str] = None  # Path to load pretrained model from
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible OL trajectories (pair SN/CNN runs)")
    subspace_methods: List[str] = Field(default_factory=list, description="List of classic subspace methods to evaluate (e.g., ['1D-MUSIC', 'Root-MUSIC','ESPRIT']).")


class EvaluationConfig(BaseModel):
    """Evaluation configuration parameters."""
    save_results: bool = Field(default=True, description="Whether to save evaluation results to file")
    results_format: Literal["json", "csv", "yaml"] = Field(default="json", description="Format for saving results")
    detailed_metrics: bool = Field(default=True, description="Whether to include detailed metrics in results")
    kalman_filter: bool = Field(default=True, description="Whether to use Kalman filter for trajectory evaluation")
    
    # Parameter sweep configuration
    sweep_parameter: Optional[str] = Field(default=None, description="Parameter to sweep during evaluation (e.g., 'snr', 'M', 'eta')")
    sweep_values: Optional[List[float]] = Field(default=None, description="Values to use for parameter sweep")
    min_source_separation: Optional[float] = Field(default=None, description="Minimum angular separation between sources in degrees")


class TrajectoryType(str, Enum):
    """Types of angle trajectories supported by the data handler."""
    RANDOM = "random"
    RANDOM_WALK = "random_walk"
    LINEAR = "linear"
    CIRCULAR = "circular"
    STATIC = "static"
    CUSTOM = "custom"
    FULL_RANDOM = "full_random"  # Completely random angles for each source in each trajectory step
    SINE_ACCEL_NONLINEAR = "sine_accel_nonlinear"  # Non-linear model with sine acceleration
    MULT_NOISE_NONLINEAR = "mult_noise_nonlinear"  # Non-linear model with multiplicative noise


class TrajectoryConfig(BaseModel):
    """Trajectory configuration parameters."""
    enabled: bool = False
    trajectory_type: TrajectoryType = TrajectoryType.RANDOM
    trajectory_length: int = 10
    save_trajectory: bool = False
    random_walk_std_dev: float = 1.0  # Standard deviation for random walk angles (degrees)
    
    # Parameters for sine acceleration non-linear model (now oscillatory with source-specific patterns)
    sine_accel_omega0: Union[float, List[float]] = 0.2    # Frequency of oscillation (rad/s) - can be single value or array per source
    sine_accel_kappa: Union[float, List[float]] = 0.1     # Amplitude of oscillation (rad) - can be single value or array per source
    sine_accel_noise_std: float = 0.01 # Noise standard deviation (rad) - controls randomness
    sine_accel_dc_offset_range: Optional[List[float]] = Field(
        default=None,
        description="If set (e.g. [-15, 15]), sample one uniform DC attractor offset per source "
        "per trajectory (degrees). Oscillation converges around this offset, not 0.",
    )
    sine_accel_dc_offsets: Optional[List[float]] = Field(
        default=None,
        description="Optional fixed DC attractor offsets per source (degrees). "
        "When set, overrides random sampling from sine_accel_dc_offset_range.",
    )
    sine_accel_dc_min_sep_deg: Optional[float] = Field(
        default=None,
        description="If set, sample DC attractors with this minimum pairwise spacing (deg). "
        "T7 uses 16 so sine ±κ still clears the 10° dump gate.",
    )
    
    # Parameters for multiplicative noise non-linear model
    mult_noise_omega0: float = 0.0    # Base angular velocity (rad/s)
    mult_noise_amp: float = 0.5       # Amplitude of multiplicative term (unitless)
    mult_noise_base_std: float = 0.1  # Base noise standard deviation (rad)


class KalmanFilterConfig(BaseModel):
    """Kalman Filter configuration parameters."""
    filter_type: Optional[str] = Field(default="standard", description="Type of Kalman filter to use: 'standard' or 'extended'")
    process_noise_std_dev: Optional[float] = Field(
        default=None,
        description="Process-noise std-dev in degrees (same units as trajectory sine_accel_noise_std). "
        "Converted to rad² inside the filter. None → trajectory.random_walk_std_dev.",
    )
    measurement_noise_std_dev: float = Field(
        default=1e-3,
        description="Measurement-noise std-dev in degrees. Converted to rad² (R) inside the filter.",
    )
    initial_covariance: float = Field(
        default=1.0,
        description="Initial state covariance P0 in rad² (filter state units).",
    )


OnlineLearningLossType = Literal[
    "unsupervised_rmape", "unsupervised_rmspe",
    "supervised_rmape", "supervised_rmspe",
    "multimoment",
    "unsupervised_grid_bce",
    "supervised_grid_bce",
]

GridTargetMode = Literal["soft_gaussian", "hard_mhot"]


class OnlineLearningLossConfig(BaseModel):
    """Configuration for online learning loss functions."""
    adaptation_loss: OnlineLearningLossType = Field(
        default="unsupervised_rmspe",
        description="MSIE / adaptation objective: GLRT trigger, backprop, plot 2. Default RMSPE(EKF, pre-EKF).",
    )
    reference_metric: OnlineLearningLossType = Field(
        default="supervised_rmspe",
        description="Eval-only tracking metric (plot 1, tables): RMSPE(EKF, GT) when supervised_rmspe.",
    )
    supervised_offline_loss: OnlineLearningLossType = Field(
        default="supervised_rmspe",
        description="Shadow/oracle model backprop during OL (uses GT). Not base-model offline pretrain.",
    )
    grid_target: GridTargetMode = Field(
        default="soft_gaussian",
        description="DeepCNN grid distillation: EKF angles → occupancy target (soft_gaussian or hard_mhot).",
    )
    grid_sigma_bins: float = Field(
        default=2.0,
        description="Soft grid target width in bin units (ignored when grid_target=hard_mhot).",
    )
    grid_bce_pos_weight: Optional[float] = Field(
        default=None,
        description="Positive-class weight for grid BCE; null → auto (G-M)/M.",
    )
    stop_grad_ekf_target: bool = Field(
        default=True,
        description="Stop-gradient EKF angles when building grid distillation targets.",
    )
    # Legacy — removed in Phase 6 TBD
    multimoment_alpha: float = Field(default=1.0, description="Legacy: Multi-Moment alpha (TBD removal)")
    multimoment_beta: float = Field(default=1.0, description="Legacy: Multi-Moment beta (TBD removal)")


class OnlineLearningConfig(BaseModel):
    """Online learning configuration parameters."""
    enabled: bool = False
    window_size: int = Field(default=10, description="Size of the sliding window (number of steps) over the trajectory data.")
    stride: int = Field(default=5, description="Stride between consecutive windows (number of steps). Defines window density and overlap.")
    max_iterations: int = Field(default=10, description="Maximum number of training iterations per window when an update is triggered.")
    adaptation_window_count: int = Field(
        default=5,
        description="Number of windows to run online adaptation after drift is triggered before learning_done.",
    )
    learning_rate: float = Field(default=1e-4, description="Learning rate for online training (typically smaller than main training).")
    
    # trajectory_length now defines the total number of individual trajectory steps to be generated during the online learning session.
    # This is used to determine the total number of windows that can be formed based on window_size and stride.
    trajectory_length: Optional[int] = Field(default=1000, description="Total number of trajectory steps available for the entire online learning session.")

    # Dynamic calibration drift (position η, SV additive noise, or spacing-scale ε)
    drift_type: Literal["position_eta", "sv_noise_var", "spacing_scale", "pitch_eta", "boresight_offset"] = Field(
        default="position_eta",
        description="Calibration drift at eta_update_interval_windows: additive position η, "
        "additive SV-noise variance, global spacing scale, index-scaled pitch η (eq. 20), "
        "or a common boresight offset (generate at θ−Δ, score against θ).",
    )
    drift_profile: Literal["step", "gradual"] = Field(
        default="step",
        description="step: add increment every eta_update_interval_windows (T1–T3). "
        "gradual: linear ramp 0 → max_eta over gradual_ramp_windows, starting at "
        "eta_update_interval_windows. gradual is position_eta only.",
    )
    gradual_ramp_windows: Optional[int] = Field(
        default=None,
        description="Windows to ramp η from 0 to max_eta when drift_profile=gradual. "
        "Ramp occupies [start, start+ramp). Start = eta_update_interval_windows.",
    )
    eta_update_interval_windows: Optional[int] = Field(default=None, description="Apply calibration drift every N windows. If None or 0, drift is not periodically updated by this mechanism. For drift_profile=gradual this is the ramp start window.")
    eta_step_windows: Optional[List[int]] = Field(
        default=None,
        description="If set, apply a step increment only at these window indices (T5 staircase). "
        "Overrides eta_update_interval_windows modulo and drift_true_windows.",
    )
    eta_reuse_direction: bool = Field(
        default=False,
        description="If True, scale a frozen u_n when position-η steps (T5). "
        "If False, redraw δ_n (T1–T3).",
    )
    max_adapt_cycles: int = Field(
        default=1,
        description="Max detect→adapt bursts per trajectory. 1 = T1–T3 one-shot. "
        "T5 staircase uses 3 with CUSUM re-arm after learning_done.",
    )
    drift_true_windows: Optional[List[Optional[int]]] = Field(
        default=None,
        description="Per-traj true drift window (cycles by trajectory index). "
        "null = no drift (negative class). When set, replaces eta_update_interval_windows modulo.",
    )
    eta_increment: Optional[float] = Field(default=0.01, description="Position-η increment when drift_type=position_eta.")
    max_eta: Optional[float] = Field(default=0.5, description="Maximum allowed eta during dynamic updates.")
    min_eta: Optional[float] = Field(default=0.0, description="Minimum allowed eta during dynamic updates.")
    sv_noise_increment: Optional[float] = Field(
        default=0.0,
        description="sv_noise_var increment when drift_type=sv_noise_var.",
    )
    max_sv_noise_var: Optional[float] = Field(
        default=None,
        description="Maximum sv_noise_var during dynamic updates.",
    )
    min_sv_noise_var: Optional[float] = Field(
        default=0.0,
        description="Minimum sv_noise_var during dynamic updates.",
    )
    pitch_eta_increment: Optional[float] = Field(
        default=0.0,
        description="pitch_eta increment when drift_type=pitch_eta (same ×λ units as eta).",
    )
    max_pitch_eta: Optional[float] = Field(
        default=None,
        description="Maximum pitch_eta during dynamic updates.",
    )
    min_pitch_eta: Optional[float] = Field(
        default=0.0,
        description="Minimum pitch_eta during dynamic updates.",
    )
    spacing_scale_increment: Optional[float] = Field(
        default=0.03,
        description="Spacing-scale increment when drift_type=spacing_scale (e.g. 0.03 → d'=(1.03)d).",
    )
    max_spacing_scale: Optional[float] = Field(default=1.1, description="Maximum spacing_scale during dynamic updates.")
    min_spacing_scale: Optional[float] = Field(default=1.0, description="Minimum spacing_scale during dynamic updates.")
    boresight_offset_increment: Optional[float] = Field(
        default=0.0,
        description="Boresight offset increment in degrees when drift_type=boresight_offset. "
        "Snapshots are generated at θ−Δ; RMSPE is scored against θ.",
    )
    max_boresight_offset_deg: Optional[float] = Field(
        default=None,
        description="Maximum boresight offset (degrees) during dynamic updates.",
    )
    min_boresight_offset_deg: Optional[float] = Field(
        default=0.0,
        description="Minimum boresight offset (degrees) during dynamic updates.",
    )
    
    # Calibration error control
    use_nominal: bool = Field(default=True, description="If True (default), nominal array configuration (no calibration errors) is used for sample generation. If False, calibration errors are applied based on eta.")
    
    # Multi-trajectory parameters
    dataset_size: int = Field(default=1, description="Number of trajectories to run and average results over. Default is 1 for backward compatibility.")
    dump_music_spectrum: bool = Field(
        default=False,
        description="T7: dump MUSIC P(θ) on SN surrogate covariance. "
        "Among windows passing spectrum_min_sep_deg, keeps the frame that best "
        "shows recovery (frozen peaks off GT, adapted peaks on GT).",
    )
    spectrum_pre_window: int = Field(
        default=0,
        description="T7: earliest window eligible for the pre-drift snap (default 0 = any pre-η).",
    )
    spectrum_min_sep_deg: float = Field(
        default=10.0,
        description="T7 resolvability gate: dump a phase only if min pairwise |Δθ| ≥ this (deg). "
        "Among passing windows, keep the highest recovery-score frame. Skip if none pass.",
    )
    spectrum_post_adapt_window: int = Field(
        default=0,
        description="T7: earliest window eligible for the post-adapt snap (default 0 = any window "
        "after learning_done).",
    )
    spectrum_hold_adapt: bool = Field(
        default=True,
        description="T7: hold GD until a post-η frame clears spectrum_min_sep_deg, then adapt. "
        "False keeps the T1 schedule (GD at w*+time_to_learn), so the spectrum run is the same "
        "run as the RMSPE numbers; the post-jump panel then only exists if a legal frame falls "
        "before GD starts.",
    )

    # Loss configuration
    loss_config: OnlineLearningLossConfig = Field(default_factory=OnlineLearningLossConfig, description="Configuration for online learning loss function")
    
    # Online learning start configuration
    time_to_learn: Optional[int] = Field(default=None, description="Window index at which to start online learning. If None, online learning will not start automatically based on window index.")
    
    # Adaptive learning rate configuration
    use_adaptive_learning_rate: bool = Field(default=False, description="If True, learning rate adapts based on GLRT z-score. If False, uses fixed learning_rate.")
    adaptive_lr_min: float = Field(default=0.0005, description="Minimum learning rate for adaptive sigmoid mapping.")
    adaptive_lr_max: float = Field(default=0.0356, description="Maximum learning rate for adaptive sigmoid mapping.")
    adaptive_lr_k_sigmoid: float = Field(default=0.7336, description="Steepness of the sigmoid curve for adaptive LR mapping.")
    adaptive_lr_dG0: float = Field(default=69.2599, description="Inflection point (dG offset) for the adaptive LR sigmoid.")
    num_gd_steps: int = Field(default=3, description="Number of gradient descent steps per online learning window.")

    # Drift detection (changepoint GLRT + z-score on g-history, or CUSUM on adaptation_loss)
    drift_detector: Literal["glrt_z", "loss_cusum"] = Field(
        default="glrt_z",
        description="glrt_z: current z-of-max-GLR trigger. loss_cusum: CUSUM + absolute "
        "threshold on unsupervised adaptation_loss (T6 probe).",
    )
    drift_warmup_windows: int = Field(
        default=10,
        description="Scope A: skip changepoint GLRT for the first N trajectory windows. "
        "First g at drift_warmup_windows + 2*GLRT_MIN_SEGMENT_SIZE. "
        "Scope B baseline length is fixed in utils/drift_gates.py (SCOPE_B_BASELINE_MIN_SAMPLES).",
    )
    drift_guard_samples: int = Field(
        default=3,
        description="Exclude this many most recent log-GLR values from the baseline "
        "(they may already reflect drift onset).",
    )
    drift_z_threshold: float = Field(
        default=2.5,
        description="Drift fires when z-score of current log-GLR vs baseline exceeds this.",
    )
    drift_history_max_size: Optional[int] = Field(
        default=None,
        description="Optional cap on rolling log-GLR history length. None = unbounded (default).",
    )
    drift_recent_tau_k: Optional[int] = Field(
        default=None,
        description="If set, z-trigger latches only when argmax τ lies in the last K post-warmup "
        "MSIE samples (recent-τ gate). None = disabled (z-only trigger).",
    )
    cusum_ekf_settle_windows: int = Field(
        default=3,
        description="Unused by the consecutive-L settle path; kept for yaml compatibility.",
    )
    cusum_hump_confirm_windows: int = Field(
        default=5,
        description="loss_cusum: consecutive windows with EKF worse than SN before recovery can settle. "
        "Blocks false settle on a 1–3 window EKF blip before the real setup hump.",
    )
    cusum_ref_windows: int = Field(
        default=8,
        description="loss_cusum: unsupervised losses used to freeze (μ0, σ0) after EKF settle.",
    )
    cusum_kappa: float = Field(default=0.5, description="loss_cusum slack in units of σ0.")
    cusum_h: float = Field(default=5.0, description="loss_cusum fire if S > h σ0.")
    cusum_loss_threshold: float = Field(
        default=0.05,
        description="loss_cusum absolute gate on adaptation_loss (radians). Both this and CUSUM must fire.",
    )
    cusum_settle_fallback_window: int = Field(
        default=24,
        description="If EKF is never worse than SN for hump_confirm_windows, start collecting "
        "the unsupervised null at this window. After a real setup hump (L above abs threshold), "
        "the null is the first consecutive block of L below threshold.",
    )

    plot_trajectory: bool = Field(
        default=False,
        description="If True, save DOA trajectory plots (xy + angle vs step) from GT window labels.",
    )

    @model_validator(mode="after")
    def _validate_drift_config(self):
        if self.drift_warmup_windows < 2:
            raise ValueError("drift_warmup_windows must be >= 2")
        if self.drift_guard_samples < 0:
            raise ValueError("drift_guard_samples must be >= 0")
        if self.drift_z_threshold <= 0:
            raise ValueError("drift_z_threshold must be > 0")
        if self.drift_recent_tau_k is not None and self.drift_recent_tau_k < 1:
            raise ValueError("drift_recent_tau_k must be >= 1 when set")
        if self.cusum_ekf_settle_windows < 1:
            raise ValueError("cusum_ekf_settle_windows must be >= 1")
        if self.cusum_hump_confirm_windows < 1:
            raise ValueError("cusum_hump_confirm_windows must be >= 1")
        if self.cusum_ref_windows < 2:
            raise ValueError("cusum_ref_windows must be >= 2")
        if self.cusum_kappa < 0:
            raise ValueError("cusum_kappa must be >= 0")
        if self.cusum_h <= 0:
            raise ValueError("cusum_h must be > 0")
        if self.cusum_loss_threshold <= 0:
            raise ValueError("cusum_loss_threshold must be > 0")
        if self.cusum_settle_fallback_window < 0:
            raise ValueError("cusum_settle_fallback_window must be >= 0")
        if self.max_adapt_cycles < 1:
            raise ValueError("max_adapt_cycles must be >= 1")
        if self.eta_step_windows is not None:
            if len(self.eta_step_windows) < 1:
                raise ValueError("eta_step_windows must be non-empty when set")
            if any(int(w) < 1 for w in self.eta_step_windows):
                raise ValueError("eta_step_windows must be >= 1")
            if list(self.eta_step_windows) != sorted(self.eta_step_windows):
                raise ValueError("eta_step_windows must be strictly increasing")
            if len(set(int(w) for w in self.eta_step_windows)) != len(self.eta_step_windows):
                raise ValueError("eta_step_windows must be strictly increasing")
        from utils.drift_gates import SCOPE_B_BASELINE_MIN_SAMPLES

        min_history = SCOPE_B_BASELINE_MIN_SAMPLES + self.drift_guard_samples
        if self.drift_history_max_size is not None and self.drift_history_max_size < min_history:
            raise ValueError(
                f"drift_history_max_size ({self.drift_history_max_size}) must be >= "
                f"SCOPE_B_BASELINE_MIN_SAMPLES + drift_guard_samples ({min_history}) when set"
            )
        if self.drift_profile == "gradual":
            if self.drift_type != "position_eta":
                raise ValueError("drift_profile=gradual only supports drift_type=position_eta")
            if self.gradual_ramp_windows is None or self.gradual_ramp_windows < 1:
                raise ValueError("gradual_ramp_windows must be >= 1 when drift_profile=gradual")
            if not self.eta_update_interval_windows or self.eta_update_interval_windows < 1:
                raise ValueError("eta_update_interval_windows is the ramp start and must be >= 1")
            if self.max_eta is None:
                raise ValueError("max_eta is the gradual ramp target")
        return self

    # LR sweep configuration
    enable_lr_sweep: bool = Field(default=False, description="If True, enable LR sweep within scenario sweeps. For each scenario value, sweep over static LR values and optionally run adaptive LR.")
    static_lr_list: Optional[List[float]] = Field(default=None, description="List of static LR values to sweep over. Required if enable_lr_sweep=True.")
    adaptive_base_lr: Optional[float] = Field(default=None, description="Base learning rate for adaptive LR when running LR sweep. If None, uses learning_rate from config.")


class ScenarioSystemModelOverride(BaseModel):
    """Defines overrides for SystemModelConfig for a specific scenario."""
    N: Optional[int] = None
    M: Optional[Union[int, Tuple[int, int]]] = None
    T: Optional[int] = None
    snr: Optional[float] = None
    field_type: Optional[Literal["near", "far"]] = None
    signal_nature: Optional[Literal["coherent", "non-coherent"]] = None
    signal_type: Optional[Literal["narrowband", "broadband"]] = None
    wavelength: Optional[float] = None
    eta: Optional[float] = None
    pitch_eta: Optional[float] = None
    spacing_scale: Optional[float] = None
    bias: Optional[float] = None
    sv_noise_var: Optional[float] = None
    doa_range: Optional[int] = None
    doa_resolution: Optional[int] = None
    max_range_ratio_to_limit: Optional[float] = None
    range_resolution: Optional[int] = None

    class Config:
        extra = 'forbid'  # Prevent unspecified fields


class ScenarioModelOverride(BaseModel):
    """Defines overrides for ModelConfig for a specific scenario."""
    type: Optional[str] = None  # e.g., "SubspaceNet", "DCD-MUSIC". If type changes, architecture changes.
    # params: Optional[ModelParamsConfig] = None # For overriding model-specific parameters

    class Config:
        extra = 'forbid'


class ScenarioDefinition(BaseModel):
    """Defines a single scenario to be run."""
    name: str = Field(..., description="Unique name for the scenario (used for output folders, results keys).")
    system_model_overrides: Optional[ScenarioSystemModelOverride] = None
    model_overrides: Optional[ScenarioModelOverride] = None
    retrain_model: bool = Field(default=False, description="If True, model is retrained for this scenario. If False, uses base/loaded model.")


class ScenarioConfig(BaseModel):
    """Configuration for running a scenario with multiple parameter values."""
    type: str = Field(..., description="Type of scenario to run (e.g., 'SNR', 'T', 'M', 'eta').")
    values: List[float] = Field(..., description="List of values to test for this scenario type.")
    model_paths: Optional[List[str]] = Field(default=None, description="List of model paths for each scenario value (must match values order)")
    eta_increments: Optional[List[float]] = Field(
        default=None,
        description="Per-scenario online_learning.eta_increment (same order as values). For T1 N-sweep: position η jump @ w40.",
    )
    max_etas: Optional[List[float]] = Field(
        default=None,
        description="Per-scenario online_learning.max_eta (same order as values). Defaults to eta_increments when omitted.",
    )
    sv_noise_increments: Optional[List[float]] = Field(
        default=None,
        description="Per-scenario online_learning.sv_noise_increment (same order as values). T1.2 SV-noise jump @ w40.",
    )
    max_sv_noise_vars: Optional[List[float]] = Field(
        default=None,
        description="Per-scenario online_learning.max_sv_noise_var (same order as values). Defaults to sv_noise_increments.",
    )
    pitch_eta_increments: Optional[List[float]] = Field(
        default=None,
        description="Per-scenario online_learning.pitch_eta_increment (same order as values). T1.3 pitch-η jump @ w40.",
    )
    max_pitch_etas: Optional[List[float]] = Field(
        default=None,
        description="Per-scenario online_learning.max_pitch_eta (same order as values). Defaults to pitch_eta_increments.",
    )
    spacing_scale_increments: Optional[List[float]] = Field(
        default=None,
        description="Per-scenario online_learning.spacing_scale_increment. Added to 1.0 at the jump.",
    )
    max_spacing_scales: Optional[List[float]] = Field(
        default=None,
        description="Per-scenario online_learning.max_spacing_scale. Defaults to 1 + spacing_scale_increments.",
    )
    boresight_offset_increments: Optional[List[float]] = Field(
        default=None,
        description="Per-scenario online_learning.boresight_offset_increment (degrees). Added to 0 at the jump.",
    )
    max_boresight_offsets: Optional[List[float]] = Field(
        default=None,
        description="Per-scenario online_learning.max_boresight_offset_deg. Defaults to the increment.",
    )
    retrain_model: bool = Field(default=False, description="Whether to retrain the model for each scenario value.")

    @model_validator(mode="after")
    def _validate_per_scenario_eta(self):
        n = len(self.values)
        for field_name, field_val in (
            ("eta_increments", self.eta_increments),
            ("max_etas", self.max_etas),
            ("sv_noise_increments", self.sv_noise_increments),
            ("max_sv_noise_vars", self.max_sv_noise_vars),
            ("pitch_eta_increments", self.pitch_eta_increments),
            ("max_pitch_etas", self.max_pitch_etas),
            ("spacing_scale_increments", self.spacing_scale_increments),
            ("max_spacing_scales", self.max_spacing_scales),
            ("boresight_offset_increments", self.boresight_offset_increments),
            ("max_boresight_offsets", self.max_boresight_offsets),
        ):
            if field_val is not None and len(field_val) != n:
                raise ValueError(f"scenario_config.{field_name} length ({len(field_val)}) must match values ({n})")
        return self


class LoggingConfig(BaseModel):
    """Logging configuration parameters."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", 
        description="Global logging level. Controls verbosity of all loggers."
    )
    
    # Specific logger levels (optional overrides)
    subspace_net_level: Optional[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]] = Field(
        default=None,
        description="Logging level for SubspaceNet loggers. If None, uses global level."
    )
    
    kalman_filter_level: Optional[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]] = Field(
        default=None,
        description="Logging level for Kalman filter loggers. If None, uses global level."
    )
    
    torch_level: Optional[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]] = Field(
        default="WARNING",
        description="Logging level for PyTorch loggers. Defaults to WARNING to reduce noise."
    )
    
    matplotlib_level: Optional[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]] = Field(
        default="WARNING",
        description="Logging level for matplotlib loggers. Defaults to WARNING to reduce noise."
    )
    
    # File logging options
    log_to_file: bool = Field(
        default=False,
        description="Whether to log to a file in addition to console output."
    )
    
    log_file_path: Optional[str] = Field(
        default=None,
        description="Path for log file. If None and log_to_file=True, uses default path."
    )
    
    log_file_level: Optional[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]] = Field(
        default="DEBUG",
        description="Logging level for file output. If None, uses global level."
    )


class Config(BaseModel):
    """Complete configuration model."""
    system_model: SystemModelConfig = SystemModelConfig()
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    simulation: SimulationConfig = SimulationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    trajectory: TrajectoryConfig = TrajectoryConfig()
    kalman_filter: KalmanFilterConfig = KalmanFilterConfig()
    online_learning: OnlineLearningConfig = OnlineLearningConfig()
    logging: LoggingConfig = LoggingConfig()
    # New scenario-related fields
    scenarios: Optional[List[ScenarioDefinition]] = Field(default=None, description="List of scenarios to run.")
    train_base_model_only_once: bool = Field(default=True, description="If True and scenarios use a shared model, train it only once initially.")
    scenario_config: Optional[ScenarioConfig] = Field(default=None, description="Configuration for running a scenario with multiple parameter values.")


    #missing: critirion, balance factor, sequential_dataset_train, sequential_dataset_test, evaluation params, simulation_commands