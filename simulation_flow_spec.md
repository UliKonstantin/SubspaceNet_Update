# Developer Specification: SubspaceNet Simulation Execution Flow

## 1. Initiation and CLI Processing (`main.py`)

*   **Entry Point:** Execution begins via the command line: `python main.py simulate [OPTIONS]`
*   **CLI Parsing:** The `click` library parses the command (`simulate`) and its options (`--config`, `--output`, `--override`, `--scenario`, `--values`, `--trajectory`).
    *   The `simulate_command` function in `main.py` is invoked.
*   **Trajectory Override:** If `--trajectory` is passed, the `trajectory.enabled=true` override is added to the `override` list.
*   **Configuration Setup Call:** The core logic delegates to `config_handler.py::setup_configuration`, passing the `config` path, `output` path, and the list of `override` strings.

## 2. Configuration Handling (`config_handler.py`)

*   **Function:** `setup_configuration(config_path, output_path, overrides)`
*   **Load Base Config:** Loads the base YAML configuration file specified by `config_path` using `config.loader::load_config`. Returns a configuration dictionary (`dict`).
*   **Apply Overrides:** Modifies the loaded configuration dictionary by applying the key-value pairs provided in the `overrides` list using `config.loader::apply_overrides`.
*   **Schema Validation:** Validates the potentially modified configuration dictionary against the defined schema using `config.schema::validate_config` to ensure all required fields and types are correct.
*   **Component Creation:** Invokes the factory function `config.factory::create_components`, passing the validated configuration dictionary. This function instantiates key objects:
    *   `create_system_model`: Instantiates `DCD_MUSIC.src.system_model.SystemModelParams` and `DCD_MUSIC.src.system_model.SystemModel` based on `config['system_model_params']`.
    *   `create_model`: Instantiates the primary **Deep Learning Model** (e.g., `DCD_MUSIC.src.models_pack.subspacenet.SubspaceNet`, `DCD_MUSIC.src.models_pack.dcdmusic.DCDMUSIC`) based on `config['model']['type']`, passing the created `system_model` and parameters from `config['model']['model_params']`.
    *   `create_trajectory_handler` (Conditional): If `config['trajectory']['enabled']` is `True`, instantiates `simulation.trajectory.TrajectoryHandler` using `config['trajectory']`.
    *   `create_training_config`: Instantiates a configuration object like `simulation.runners.training.TrainingConfig` using `config['training_params']`.
    *   (Other components like evaluation config might also be created).
*   **Output Directory Setup:** Creates the specified `output_path` directory if it doesn't exist.
*   **Return:** Returns a tuple containing:
    1.  The final, validated configuration dictionary (`config_obj`).
    2.  A dictionary (`components`) holding the instantiated objects (e.g., `{'system_model': SystemModel_instance, 'model': DL_Model_instance, 'trajectory_handler': TrajectoryHandler_instance_or_None, 'training_config': TrainingConfig_instance}`).
    3.  The output directory path (`output_dir`).

## 3. Simulation Orchestration Setup (`main.py` -> `simulation/core.py`)

*   **Back in `main.py::simulate_command`:** Receives `config_obj`, `components`, `output_dir` from `setup_configuration`.
*   **Simulation Object Instantiation:** Creates an instance of the main simulation controller: `sim = Simulation(config_obj, components, output_dir)` (ref: `simulation.core.py`).
    *   The `Simulation.__init__` method stores the configuration, instantiated components, and output directory as instance attributes (e.g., `self.config`, `self.components`, `self.output_dir`, `self.system_model`, `self.model`, etc.).

## 4. Simulation Execution Trigger (`simulation/core.py`)

*   **Scenario Check:** The `simulate_command` checks if `scenario` and `values` were provided.
    *   **If Scenario:** Calls `sim.run_scenario(scenario, values)`.
    *   **If No Scenario:** Calls `sim.run()`.

## 5. Core Simulation Flow (`simulation/core.py::Simulation::run`)

This method orchestrates a single end-to-end simulation run.

*   **Data Pipeline Setup:** Calls `self._setup_data_pipeline()`.
    *   **Inside `_setup_data_pipeline`:**
        *   Checks `self.config['trajectory']['enabled']`.
        *   **If Trajectory Enabled:**
            1.  Retrieves the `self.trajectory_handler` from `self.components`.
            2.  Calls `self.trajectory_handler.generate_trajectories()` -> returns `trajectory_data` (tuples of angle/distance tensors) and `sources_per_trajectory` (list).
            3.  Instantiates `data_handler = TrajectoryDataHandler(self.system_model, trajectory_data, sources_per_trajectory, self.config['system_model_params'])` (ref: `simulation/runners/data.py`).
            4.  Calls `data_handler.create_datasets()` -> Returns `train_dataset`, `val_dataset`, `test_dataset` (custom `TrajectoryDataset` instances). These datasets contain lists of tensors, where each inner tensor represents a trajectory sequence.
            5.  Splits datasets into loaders (`self.train_loader`, `self.val_loader`, `self.test_loader`) using appropriate samplers/collators for trajectory data.
        *   **If Trajectory Disabled:**
            1.  Instantiates a standard data handler, likely from `DCD_MUSIC`: `samples_model = DCD_MUSIC.src.signal_creation.Samples(self.system_model.params)`.
            2.  Calls `DCD_MUSIC.src.data_handler::create_dataset` multiple times (with different `phase` args) to generate `train_dataset`, `test_dataset` (likely `DCD_MUSIC.src.data_handler.TimeSeriesDataset` instances).
            3.  Calls `train_dataset.get_dataloaders()` to get `self.train_loader`, `self.val_loader`. Creates `self.test_loader` from `test_dataset`.
*   **Training Pipeline Execution:** Calls `self._run_training_pipeline()` (if `self.config['simulation']['train_model']` is true).
    *   **Inside `_run_training_pipeline`:**
        *   Checks `self.config['trajectory']['enabled']`.
        *   **If Trajectory Enabled:**
            1.  Instantiates `trainer = TrajectoryTrainer(model=self.model, config=self.training_config, train_loader=self.train_loader, val_loader=self.val_loader, system_model=self.system_model, trajectory_handler=self.trajectory_handler)` (ref: `simulation/runners/training.py`).
            2.  Calls `trainer.train()`. This executes the nested loops (epochs -> batches -> trajectory steps), calls `model.forward` or `model.training_step`, calculates loss, and performs backpropagation.
            3.  Stores the potentially updated model state in `self.model`.
        *   **If Trajectory Disabled:**
            1.  Instantiates `trainer = DCD_MUSIC.src.training.Trainer(model=self.model, training_params=self.training_config)` (note: mapping `TrainingConfig` fields to `TrainingParamsNew` might occur).
            2.  Calls `trainer.train(self.train_loader, self.val_loader, ...)` using the standard DCD_MUSIC training loop.
            3.  Stores the potentially updated model state in `self.model`.
*   **Evaluation Pipeline Execution:** Calls `self._run_evaluation_pipeline()` (if `self.config['simulation']['evaluate_model']` is true).
    *   **Inside `_run_evaluation_pipeline`:**
        1.  Instantiates `evaluator = Evaluator(config=self.config['evaluation_params'], test_loader=self.test_loader, model=self.model, system_model=self.system_model, output_dir=self.output_dir)` (ref: `simulation.runners.evaluation.py`).
        2.  Calls `evaluator.evaluate()`. This likely iterates through the `test_loader`, runs inference using `self.model` (or other specified methods like classical MUSIC/ESPRIT sourced from `DCD_MUSIC.src.methods_pack`), calculates metrics using loss functions/metrics (e.g., `DCD_MUSIC.src.metrics::RMSPELoss`, `CartesianLoss`), and aggregates results.
        3.  Returns evaluation `results` (dictionary).
*   **Return:** Returns the `results` from the evaluation pipeline.

## 6. Scenario Loop (`simulation/core.py::Simulation::run_scenario`)

*   **Input:** `scenario_key` (e.g., "SNR"), `values` (e.g., `[10, 5, 0]`).
*   **Iteration:** Loops through each `value` in the `values` list.
*   **Temporary Config Modification:** In each iteration:
    *   Stores the original value of the parameter corresponding to `scenario_key` (e.g., `self.system_model.params.snr`).
    *   Updates the parameter in `self.system_model.params` (and potentially `self.config`) to the current loop `value`.
    *   *Crucially*, the `self.system_model` object itself might need internal state updates if parameters change significantly, or components using it (like data handlers/models) might need notification or recreation depending on implementation. (This detail depends on how tightly coupled components are to the initial config).
*   **Execute Single Run:** Calls `self.run()` to perform the full data->train->evaluate pipeline with the *temporarily modified* configuration/system model state.
*   **Store Result:** Stores the returned `results` from `self.run()` in a dictionary, keyed by the current scenario `value`.
*   **Restore Config:** Resets the modified parameter back to its original value.
*   **Aggregation:** After the loop, aggregates all stored results.
*   **Return:** Returns the dictionary of results for the scenario.

## 7. Completion (`main.py`)

*   The `simulate_command` receives the final results (either from `sim.run()` or `sim.run_scenario()`).
*   Logs completion messages.
*   Exits.

This detailed flow outlines the sequence of object instantiations, method calls, and data flow between the main project components and the interfaces provided by the DCD_MUSIC library, providing a solid basis for creating a flow diagram. 