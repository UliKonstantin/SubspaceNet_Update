# Scenario Handling in Simulation Core: Design and Implementation Plan

This document outlines the design and implementation plan for modifying `simulation/core.py` to handle parametric scenarios. A scenario primarily involves modifications to the `system_model`, which necessitates re-creating the dataset. The model itself may be re-trained or a pre-trained model can be used for evaluation across different system scenarios.

## 1. Core Requirements & Definitions

*   **Scenario:** A specific configuration of the simulation, primarily differing in `system_model` parameters. Each scenario will require its own dataset generation tailored to its `system_model`.
*   **Model Handling Across Scenarios:**
    *   **Shared Model (Evaluation-Only Scenarios):** A base model can be trained once (or loaded if pre-trained). Then, for multiple scenarios (each with a different `system_model` and corresponding dataset), this single trained model is used for evaluation. This is for testing a fixed model on various environmental conditions.
    *   **Retrained Model (Training Scenarios):** For certain scenarios, or if the model architecture itself changes as part of the scenario, the model needs to be (re-)trained using the dataset specific to that scenario.
*   **Configuration:** The existing `Config` object needs a way to define a list of scenarios and their specific overrides.

## 2. Proposed Configuration Schema Changes

We will extend the `config.schema.Config` Pydantic model to include definitions for scenarios.

```python
# In config.schema.py (conceptual additions)
from typing import Optional, List, Union, Tuple, Literal # Ensure Literal is imported
from pydantic import BaseModel, Field

# --- Assuming these are defined or similar exist in your config.schema ---
# class SystemModelConfig(BaseModel): ...
# class ModelConfig(BaseModel): ...
# class ModelParamsConfig(BaseModel): ...
# class TrainingConfig(BaseModel): ... # For potential training overrides per scenario
# ---

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
    bias: Optional[float] = None
    sv_noise_var: Optional[float] = None
    doa_range: Optional[int] = None
    doa_resolution: Optional[int] = None
    max_range_ratio_to_limit: Optional[float] = None
    range_resolution: Optional[int] = None
    # Add any other fields from the actual SystemModelConfig

    class Config:
        extra = \'forbid\' # Prevent unspecified fields

class ScenarioModelOverride(BaseModel):
    """Defines overrides for ModelConfig for a specific scenario."""
    type: Optional[str] = None # e.g., "SubspaceNet", "DCD-MUSIC". If type changes, architecture changes.
    # params: Optional[ModelParamsConfig] = None # For overriding model-specific parameters
    # Depending on your factory, changing params might re-instantiate or reconfigure the model.

    class Config:
        extra = \'forbid\'

# Potential for scenario-specific training parameter overrides
# class ScenarioTrainingOverride(BaseModel):
#     learning_rate: Optional[float] = None
#     epochs: Optional[int] = None
#     # ... other training params from TrainingConfig ...
#
#     class Config:
#         extra = \'forbid\'

class ScenarioDefinition(BaseModel):
    """Defines a single scenario to be run."""
    name: str = Field(..., description="Unique name for the scenario (used for output folders, results keys).")
    system_model_overrides: Optional[ScenarioSystemModelOverride] = None
    model_overrides: Optional[ScenarioModelOverride] = None
    retrain_model: bool = Field(default=False, description="If True, model is retrained for this scenario. If False, uses base/loaded model.")
    # training_overrides: Optional[ScenarioTrainingOverride] = None # For scenario-specific training adjustments

class Config(BaseModel): # Existing Config class
    # ... all existing fields ...
    scenarios: Optional[List[ScenarioDefinition]] = Field(default=None, description="List of scenarios to run.")
    train_base_model_only_once: bool = Field(default=True, description="If True and scenarios use a shared model, train it only once initially.")
    # Add a field to help _reconfigure_for_config know the current scenario context if needed.
    # This is more an internal state, might not need to be in schema if handled by passing scenario_name.
    # current_scenario_name: Optional[str] = Field(default=None, exclude=True) # Internal state, not from YAML
```

## 3. `Simulation` Class Modifications

### 3.1. `run()` Method Overhaul

The main `run()` method will be the central dispatcher.

```python
# In simulation/core.py
class Simulation:
    # ... existing __init__ ...

    def run(self) -> Dict[str, Any]:
        if not self.config.scenarios:
            logger.info("No scenarios defined. Running single simulation.")
            # Ensure components are configured for the base self.config
            self._reconfigure_for_config(self.config, is_base_config=True)
            return self._execute_single_run(self.config, self.output_dir)
        else:
            logger.info(f"Executing {len(self.config.scenarios)} scenarios.")
            all_scenario_results = {}
            base_trained_model_state_dict = None

            # Store the state of the initial model architecture for resetting if needed
            initial_model_arch_state_dict = None
            if self.model: # self.model is the nn.Module instance
                initial_model_arch_state_dict = self.model.state_dict()

            # --- Base Model Preparation ---
            # Condition: Training is enabled, global train_model flag is true,
            #            train_base_model_only_once is true, AND
            #            at least one scenario exists that doesn\'t retrain the model.
            needs_base_model_training = (
                self.config.training.enabled and
                self.config.simulation.train_model and
                self.config.train_base_model_only_once and
                any(not sc_def.retrain_model for sc_def in self.config.scenarios)
            )

            if self.config.simulation.load_model:
                logger.info("Attempting to load a base model as per global config.")
                if hasattr(self.config.simulation, \'model_path\') and self.config.simulation.model_path:
                    model_path = Path(self.config.simulation.model_path)
                    # Ensure model arch is configured for base config before loading
                    self._reconfigure_for_config(self.config, is_base_config=True)
                    success, _ = self._load_and_apply_weights(model_path, device)
                    if success and self.trained_model:
                        base_trained_model_state_dict = self.trained_model.state_dict()
                        logger.info(f"Base model loaded successfully from {model_path}.")
                    else:
                        logger.error(f"Failed to load base model from {model_path}. Proceeding without pre-loaded weights.")
                else:
                    logger.warning("simulation.load_model is True, but no model_path provided in global config.")

            if needs_base_model_training and not base_trained_model_state_dict:
                logger.info("Performing initial base model training run.")
                self._reconfigure_for_config(self.config, is_base_config=True) # Ensure base config active
                
                base_run_output_dir = self.output_dir / "_base_model_training"
                # We run parts of _execute_single_run manually here for base training
                self._run_data_pipeline()
                if self.train_dataloader is None:
                    logger.error("Base data pipeline failed. Cannot pre-train base model.")
                else:
                    self._run_training_pipeline() # Uses self.model, result in self.trained_model
                    if self.trained_model:
                        base_trained_model_state_dict = self.trained_model.state_dict()
                        logger.info("Base model trained successfully.")
                        # Optionally save this base model
                        if self.config.simulation.save_model: # Check global flag for saving the base model
                            self._save_model_state(self.trained_model.state_dict(), base_run_output_dir, "base_model")
                    else:
                        logger.error("Base model training failed.")
            elif base_trained_model_state_dict:
                 logger.info("Using already loaded base model state for scenarios not requiring retraining.")


            # --- Loop Through Scenarios ---
            for scenario_def in self.config.scenarios:
                scenario_name = scenario_def.name
                logger.info(f"--- Starting Scenario: {scenario_name} ---")
                scenario_output_dir = self.output_dir / scenario_name
                scenario_output_dir.mkdir(parents=True, exist_ok=True)

                # 1. Create scenario-specific config object
                current_scenario_config = self._create_scenario_config(scenario_def)

                # 2. Reconfigure simulation components for this scenario
                # This updates self.config, self.system_model, potentially self.model (if arch changes), self.trajectory_handler
                self._reconfigure_for_config(current_scenario_config, current_scenario_name=scenario_name)

                # 3. Run Data Pipeline for the current scenario config
                self._run_data_pipeline() # Uses the now scenario-specific self.system_model
                if self.train_dataloader is None: # Check if data pipeline succeeded
                    logger.error(f"Data pipeline failed for scenario {scenario_name}, skipping.")
                    all_scenario_results[scenario_name] = {"status": "error", "message": "Data pipeline failed"}
                    continue
                
                # 4. Handle Model for Scenario (Loading, Retraining)
                model_for_this_scenario = None # This will be the nn.Module instance to evaluate

                if scenario_def.retrain_model or \
                   (not self.config.train_base_model_only_once and current_scenario_config.training.enabled and current_scenario_config.simulation.train_model):
                    logger.info(f"Model will be trained/retrained for scenario: {scenario_name}")
                    # If model architecture is same as initial, reset its weights.
                    # self.model is already the correct architecture due to _reconfigure_for_config
                    if initial_model_arch_state_dict and self.model.state_dict().keys() == initial_model_arch_state_dict.keys():
                         logger.info(f"Resetting model weights to initial architecture state for scenario {scenario_name}.")
                         self.model.load_state_dict(initial_model_arch_state_dict)
                         self.model = self.model.to(device) # Ensure on device

                    self._run_training_pipeline() # Trains self.model, result in self.trained_model
                    model_for_this_scenario = self.trained_model # Trained model for this scenario

                    if model_for_this_scenario is None:
                        logger.error(f"Training failed for scenario {scenario_name}, skipping evaluation.")
                        all_scenario_results[scenario_name] = {"status": "error", "message": "Scenario training failed"}
                        continue
                elif base_trained_model_state_dict:
                    logger.info(f"Using base trained model for scenario: {scenario_name}")
                    # self.model is already the correct architecture (should match base model arch if not retraining)
                    try:
                        self.model.load_state_dict(base_trained_model_state_dict)
                        self.model = self.model.to(device)
                        model_for_this_scenario = self.model
                    except RuntimeError as e:
                        logger.error(f"Failed to load base model state into scenario model for {scenario_name}: {e}. Architecture mismatch or other issue. Skipping evaluation.")
                        all_scenario_results[scenario_name] = {"status": "error", "message": f"Base model state load error: {e}"}
                        continue
                else:
                    logger.warning(f"No base model trained/loaded and not retraining for scenario: {scenario_name}. Evaluating with potentially un-initialized model.")
                    self.model = self.model.to(device) # Ensure it\'s on device
                    model_for_this_scenario = self.model # Fallback to current self.model state

                # 5. Run Evaluation Pipeline for Scenario
                # The evaluation pipeline uses self.trained_model. So, set it appropriately.
                self.trained_model = model_for_this_scenario # Set the model to be evaluated
                
                if current_scenario_config.simulation.evaluate_model:
                    if self.trained_model is not None:
                        # Temporarily set the output dir for evaluation if its internal logging depends on self.output_dir
                        original_main_output_dir = self.output_dir
                        self.output_dir = scenario_output_dir # For logs/plots specific to this scenario\'s eval

                        self._run_evaluation_pipeline() # This will populate self.results
                        scenario_run_results = self.results.copy() # Capture results

                        self.output_dir = original_main_output_dir # Restore
                    else:
                        logger.error(f"No model instance available for evaluation in scenario {scenario_name}.")
                        scenario_run_results = {"status": "error", "message": "No model for evaluation"}
                else:
                    logger.info(f"Evaluation skipped for scenario {scenario_name} as per config.")
                    scenario_run_results = {"status": "evaluation_skipped"}
                
                all_scenario_results[scenario_name] = scenario_run_results

                # 6. Save scenario-specific model (if it was trained in this scenario)
                if current_scenario_config.simulation.save_model and model_for_this_scenario and \
                   (scenario_def.retrain_model or (not self.config.train_base_model_only_once and current_scenario_config.training.enabled)):
                    self._save_model_state(model_for_this_scenario.state_dict(), scenario_output_dir, f"model_{scenario_name}")

            self.results = {"scenarios_results": all_scenario_results} # Store all scenario results
            logger.info("All scenarios completed.")
            return self.results
```

### 3.2. New Helper Method: `_execute_single_run`

This method will encapsulate the logic of a single simulation pass, largely based on the original `run` method's flow.

```python
# In simulation/core.py
    def _execute_single_run(self, exec_config: Config, output_dir_for_run: Path) -> Dict[str, Any]:
        """
        Executes a single simulation run with the given configuration and output directory.
        Assumes self.components (system_model, model, etc.) are already configured via _reconfigure_for_config.
        """
        # Store original global output_dir and set for this run temporarily if methods inside use self.output_dir
        original_global_output_dir = self.output_dir
        self.output_dir = output_dir_for_run
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Executing run with output directory: {self.output_dir}")
        # Reset results for this run
        self.results = {}

        try:
            self._run_data_pipeline()
            if self.train_dataloader is None: # Check for data pipeline success
                self.output_dir = original_global_output_dir # Restore output dir
                return {"status": "error", "message": "Data pipeline failed"}

            if exec_config.simulation.load_model:
                if hasattr(exec_config.simulation, \'model_path\') and exec_config.simulation.model_path:
                    model_path = Path(exec_config.simulation.model_path)
                    # _load_and_apply_weights operates on self.model and sets self.trained_model
                    success, message = self._load_and_apply_weights(model_path, device)
                    if not success:
                        self.output_dir = original_global_output_dir
                        return {"status": "error", "message": message}
                elif not self.trained_model: # if load_model is true, but no path, and no model already marked as trained (e.g. from base training)
                    logger.warning("simulation.load_model is True, but no model_path provided and no existing trained model state.")


            if exec_config.training.enabled and exec_config.simulation.train_model:
                # If a model was loaded and we are not supposed to retrain, skip.
                # This logic might be complex if load_model and train_model are both true.
                # Assuming train_model means "train if not successfully loaded and ready".
                # For a single run, if load_model succeeded, self.trained_model is set.
                # If retrain is intended even after load, specific logic is needed.
                # Current _run_training_pipeline uses self.model and produces self.trained_model.
                # If self.trained_model already exists (from loading), _run_training_pipeline might overwrite it.
                # This is generally fine if train_model=True implies "ensure it\'s trained, even if loaded".
                logger.info("Training enabled for this run.")
                self._run_training_pipeline()
                if self.trained_model is None:
                    self.output_dir = original_global_output_dir
                    return {"status": "error", "message": "Training pipeline failed"}
            elif not exec_config.simulation.train_model:
                 logger.info("Skipping training (simulation.train_model=False)")
            elif not exec_config.training.enabled:
                 logger.info("Skipping training (training.enabled=False)")


            if exec_config.simulation.evaluate_model:
                # Ensure we have a model to evaluate.
                # self.trained_model should be set if loaded or trained.
                # If neither, use self.model (the base architecture instance).
                model_to_evaluate = self.trained_model if self.trained_model else self.model
                
                if model_to_evaluate:
                    # Pass the model explicitly to evaluation, or ensure _run_evaluation_pipeline uses the right one.
                    # Let\'s assume _run_evaluation_pipeline uses self.trained_model, so set it.
                    self.trained_model = model_to_evaluate
                    self._run_evaluation_pipeline() # Populates self.results for this run
                else:
                    self.output_dir = original_global_output_dir
                    return {"status": "error", "message": "No model available for evaluation"}
            
            if exec_config.simulation.save_model and self.trained_model:
                self._save_model_state(self.trained_model.state_dict(), self.output_dir, "model_final")
            
            self._save_results() # Saves current self.results to self.output_dir / results.json or similar

            # Restore global output_dir
            self.output_dir = original_global_output_dir
            return self.results.copy() # Return a copy of results from this run
            
        except Exception as e:
            logger.exception(f"Error in _execute_single_run: {e}")
            self.output_dir = original_global_output_dir # Ensure restoration
            return {"status": "error", "message": str(e), "exception": type(e).__name__}

```

### 3.3. New Helper Method: `_reconfigure_for_config`

This method is crucial for setting the simulation's state according to the current (potentially scenario-specific) configuration.

```python
# In simulation/core.py
    def _create_scenario_config(self, scenario_def: ScenarioDefinition) -> Config:
        """Creates a new Config object for a given scenario by overriding the base config."""
        # Start with a deep copy of the original base config (self.config at the time of __init__)
        # It\'s safer to work from a pristine base_config if available.
        # For now, let\'s assume self.config holds the "base" when run() is first called.
        # If self.config is mutated, this needs care. A stored self.base_config would be better.
        
        # For simplicity, let\'s assume self.config is the base when scenarios start.
        # The first call to _reconfigure_for_config within the scenario loop will use the scenario_def.
        
        base_config_dict = self.config.model_dump() # Get dict from current self.config (which should be base or prev scenario)
                                                    # This needs to be the *original* base config.

        # Let\'s refine: _create_scenario_config takes the *absolute base config* as an argument.
        # In the main run() loop:
        # original_base_config = self.config.model_copy(deep=True) # Store before loop
        # current_scenario_config = self._create_scenario_config(original_base_config, scenario_def)

        # To avoid passing original_base_config around, store it in __init__
        # In __init__: self.true_base_config = config.model_copy(deep=True)
        # Then use self.true_base_config here.

        scenario_config_data = self.true_base_config.model_dump() # Start from a pristine copy of the initial config

        # Override system_model parameters
        if scenario_def.system_model_overrides:
            sm_overrides = scenario_def.system_model_overrides.model_dump(exclude_unset=True)
            for key, value in sm_overrides.items():
                # Directly update the dictionary; Pydantic will validate on Config creation
                scenario_config_data[\'system_model\'][key] = value
        
        # Override model parameters
        if scenario_def.model_overrides:
            m_overrides = scenario_def.model_overrides.model_dump(exclude_unset=True)
            for key, value in m_overrides.items():
                scenario_config_data[\'model\'][key] = value

        # Override training parameters if provided
        # if scenario_def.training_overrides:
        #     tr_overrides = scenario_def.training_overrides.model_dump(exclude_unset=True)
        #     for key, value in tr_overrides.items():
        #         scenario_config_data[\'training\'][key] = value
        
        # Create a new Config object from the modified dictionary
        final_scenario_config = Config.model_validate(scenario_config_data)
        return final_scenario_config

    def _reconfigure_for_config(self, new_config: Config, is_base_config: bool = False, current_scenario_name: Optional[str] = None) -> None:
        """
        Reconfigures simulation components based on new_config.
        Updates self.config, self.system_model, self.model (if arch changes), self.trajectory_handler.
        Resets data-related attributes.
        """
        logger.info(f"Reconfiguring simulation for: {\'Base Config\' if is_base_config else f\'Scenario: {current_scenario_name}\'}")
        
        # Store the initial base config if this is the first call (or first base config call)
        if is_base_config and not hasattr(self, \'true_base_config\'):
            self.true_base_config = new_config.model_copy(deep=True)
        
        self.config = new_config # Set current operating config

        # --- Re-configure SystemModel ---
        # Assuming DCD_MUSIC specific classes. Adapt if factories are different.
        from DCD_MUSIC.src.system_model import SystemModel, SystemModelParams
        
        sm_params_obj = SystemModelParams()
        # Populate sm_params_obj from self.config.system_model (which is new_config.system_model)
        for key, value in self.config.system_model.model_dump(exclude_unset=True).items():
            if hasattr(sm_params_obj, \'set_parameter\'): # DCD_MUSIC style
                sm_params_obj.set_parameter(key, value)
            elif hasattr(sm_params_obj, key): # Direct attribute setting
                setattr(sm_params_obj, key, value)

        self.system_model = SystemModel(system_model_params=sm_params_obj, nominal=True) # Adapt nominal as needed
        self.components["system_model"] = self.system_model
        logger.info(f"SystemModel reconfigured. N={self.system_model.params.N}, M={self.system_model.params.M}, Wavelength={self.system_model.params.wavelength}")

        # --- Re-configure Model (if architecture changes) ---
        # This needs a robust way to check if the model architecture defined in new_config.model
        # is different from the one that created the current self.model instance.
        # A simple check is on `new_config.model.type` vs the type of the config that created current self.model.

        # Store the config that created the current self.model instance if not already stored
        if not hasattr(self, \'current_model_generating_config\'):
            self.current_model_generating_config = self.true_base_config.model # Assuming __init__ sets self.model from true_base_config

        if self.config.model.type != self.current_model_generating_config.type or \
           (scenario_def := next((s for s in (self.config.scenarios or []) if s.name == current_scenario_name), None)) and scenario_def and scenario_def.model_overrides and scenario_def.model_overrides.type:
            logger.info(f"Model architecture type changed to {self.config.model.type}. Re-creating model instance.")
            # Use your model factory. Assuming a \'create_model\' function similar to what factory.py might have.
            # from config.factory import create_model # This path might vary
            # self.model = create_model(self.config, self.system_model) # Factory needs current config & system_model
            # For now, placeholder for actual factory call:
            if "model_factory" in self.components: # If a factory function is passed in components
                 self.model = self.components["model_factory"](self.config, self.system_model)
            else: # Fallback or error
                 logger.error("Model architecture changed, but no model_factory found in components to recreate it.")
                 # Decide on behavior: raise error, or try to continue with old model?
                 # For safety, one might raise an error here.
                 raise ValueError("Model architecture change requires a model_factory.")

            self.components["model"] = self.model
            self.current_model_generating_config = self.config.model.model_copy(deep=True)
            logger.info(f"New model instance of type {self.config.model.type} created.")
            # Any loaded/trained state in self.trained_model or self.model is now for the *old* architecture.
            # The main run loop needs to handle reloading/retraining for this new arch.
            self.trained_model = None # Reset trained_model as arch changed
        else:
            logger.info(f"Model architecture type ({self.config.model.type}) remains the same. Instance not recreated.")


        # --- Re-configure TrajectoryHandler ---
        if self.config.trajectory.enabled:
            logger.info("Re-configuring TrajectoryDataHandler.")
            # Assuming TrajectoryDataHandler can be re-initialized or has a reconfigure method
            self.trajectory_handler = TrajectoryDataHandler(
                system_model_params=self.system_model.params, # Use updated system_model
                config=self.config # Use current config
            )
            self.components["trajectory_handler"] = self.trajectory_handler
        
        # Reset data-related attributes; they will be repopulated by _run_data_pipeline
        self.dataset = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None
        # self.trained_model state is managed by the main run() loop (loading new state or using for retraining)
        
        logger.info("Component reconfiguration complete.")

```

### 3.4. Helper for Saving Model State
```python
# In simulation/core.py
    def _save_model_state(self, model_state_dict: Dict, output_dir: Path, model_name_prefix: str):
        """Saves the model state_dict to a timestamped file."""
        model_save_dir = output_dir / "checkpoints"
        model_save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use model type from current config for more descriptive name
        model_type = self.config.model.type if self.config and self.config.model else "unknown_model"
        model_save_path = model_save_dir / f"{model_name_prefix}_{model_type}_{timestamp}.pt"
        try:
            torch.save(model_state_dict, model_save_path)
            logger.info(f"Model state saved successfully to {model_save_path}")
        except Exception as e:
            logger.error(f"Failed to save model state {model_name_prefix}: {e}")

```

### 3.5. Modifications to `__init__`
```python
# In simulation/core.py
class Simulation:
    def __init__(self, config: Config, components: Dict[str, Any], output_dir: Optional[Path] = None):
        # ... (existing initializations) ...
        self.true_base_config = config.model_copy(deep=True) # Store the absolute initial config

        # Store the model config that generated the initial self.model
        if self.model: # self.model is the nn.Module instance from components
            self.current_model_generating_config = config.model.model_copy(deep=True)
        else:
            self.current_model_generating_config = None # No initial model
        # ...
```

## 4. Workflow Summary for Scenario Runs

1.  **`Simulation.run()` called.**
2.  If `config.scenarios` exists:
    *   Load/Train base model if `config.train_base_model_only_once` and relevant scenario flags indicate its use. Store `base_trained_model_state_dict`.
    *   For each `scenario_def` in `config.scenarios`:
        *   `scenario_name`, `scenario_output_dir` are determined.
        *   `current_scenario_config = self._create_scenario_config(self.true_base_config, scenario_def)`.
        *   `self._reconfigure_for_config(current_scenario_config, current_scenario_name=scenario_name)`:
            *   Updates `self.config = current_scenario_config`.
            *   Re-creates/configures `self.system_model` based on `current_scenario_config.system_model`.
            *   Re-creates `self.model` (the nn.Module) if `current_scenario_config.model.type` changed; sets `self.trained_model = None`.
            *   Re-creates/configures `self.trajectory_handler`.
            *   Resets `self.dataset`, dataloaders.
        *   `self._run_data_pipeline()`: Generates dataset for `current_scenario_config`.
        *   Model Handling:
            *   If `scenario_def.retrain_model`:
                *   Reset `self.model` weights if architecture is unchanged.
                *   `self._run_training_pipeline()` -> sets `self.trained_model`.
                *   `model_for_this_scenario = self.trained_model`.
            *   Else (use base model):
                *   Load `base_trained_model_state_dict` into `self.model`.
                *   `model_for_this_scenario = self.model`.
            *   Else (no base, not retraining):
                *   `model_for_this_scenario = self.model` (potentially uninitialized).
        *   `self.trained_model = model_for_this_scenario` (for `_run_evaluation_pipeline`).
        *   `self._run_evaluation_pipeline()`: Evaluates `self.trained_model`. Results captured.
        *   Save model state if retrained and configured.
        *   Store scenario results.
3.  Aggregate and return all scenario results.

## 5. Impacted Helper Methods (Summary of changes)

*   **`_run_data_pipeline`**: Must use the current `self.system_model` (set by `_reconfigure_for_config`). Dataset paths might need scenario identifiers if saved/loaded per scenario.
*   **`_run_training_pipeline`**: Uses current `self.train_dataloader`, `self.valid_dataloader`. Operates on `self.model`.
*   **`_run_evaluation_pipeline`**: Uses current `self.test_dataloader` and `self.trained_model`.
*   **`_save_results`**: May need adjustment to save results into scenario-specific subdirectories or a structured main file. The main `run` loop will likely manage result aggregation.
*   **`_load_and_apply_weights`**: Used for loading the initial base model.
*   **Original `run_scenario`**: This method can be deprecated or refactored if the new `run` method with `config.scenarios` covers its functionality more generally.

## 6. Key Considerations & Potential Challenges

*   **Pydantic Model Overrides:** Applying overrides from `ScenarioDefinition` to a copy of the base `Config` object needs to be done carefully, especially for nested Pydantic models. Direct dictionary manipulation followed by `Config.model_validate()` is one approach.
*   **Model Factory (`create_model`):** Robust re-instantiation of the model (if its architecture changes per scenario) relies on a well-defined model factory function that can take the current scenario\'s `Config` and `SystemModel`.
*   **State Management:** Correctly managing the state of `self.config`, `self.system_model`, `self.model` (the nn.Module instance), and `self.trained_model` (the state_dict or loaded nn.Module) throughout the scenario loop is critical.
*   **Deep Copies:** Ensuring deep copies of configurations and potentially model states to prevent interference between scenarios.
*   **Output Management:** Ensuring logs, saved models, plots, and result files are organized into scenario-specific subdirectories.
*   **Error Handling:** Failures in one scenario should be logged but ideally not halt the entire batch of scenarios.

This plan provides a comprehensive approach to implementing scenario handling. The modular design with helper methods like `_reconfigure_for_config` and `_execute_single_run` should help manage complexity. 