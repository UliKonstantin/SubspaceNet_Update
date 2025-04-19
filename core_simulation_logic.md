```mermaid
graph TD
    %% ===== Diagram 2: Core Simulation Logic Flow =====

    %% Define Styles
    classDef Orchestrator fill:#ddeeff,stroke:#6699cc,stroke-width:2px,font-weight:bold %% Style for core.py
    classDef Pipeline fill:#eeffee,stroke:#99cc99,stroke-width:1px %% Style for pipeline stages
    classDef DCD fill:#f9d,stroke:#f66,stroke-width:1px %% Style for DCD_MUSIC components
    classDef Decision fill:#eee,stroke:#333,stroke-width:1px %% Style for decisions/checks
    classDef StepStyle fill:#e6e0f8,stroke:#9370db,stroke-width:1px,font-weight:bold 
    %% Style for numbered steps (light purple)
    classDef ReturnPoint fill:#f0f8ff,stroke:#88c %% Point where results return to main orchestrator

    %% Core Block & Entry
    ENTRY["Entry from main.py (Simulation Instantiated)"]
    %% Updated CORE block label with internal steps
    CORE["`
        **core.py (Simulation Logic)**
        ---
        *Internal Sequence for run():*
        1. _setup_data_pipeline()
        2. _run_training_pipeline() (if enabled)
        3. _run_evaluation_pipeline() (if enabled)
        ---
        *Handles Scenario Logic (run_scenario)*
    `"]:::Orchestrator

    %% Pipeline Subgraphs (Remain the same)
    subgraph "Data Pipeline"
        direction TB
        DataPipe_Start["_setup_data_pipeline()"]:::Pipeline
        DataPipe_TrajCheck{Trajectory Enabled?}:::Decision
        DataPipe_GetTrajHandler["Get TrajectoryHandler"]
        DataPipe_GenTraj["handler.generate_trajectories()"]
        DataPipe_InstTrajDataH["Instantiate TrajectoryDataHandler (runners/data.py)"]
        DataPipe_CreateTrajDS["data_handler.create_datasets()"]
        DataPipe_SplitTrajLoaders["Create Traj DataLoaders"]
        DataPipe_InstDCDSamples["Instantiate Samples (DCD_MUSIC)"]:::DCD
        DataPipe_CreateDCDDS["Call create_dataset (DCD_MUSIC)"]:::DCD
        DataPipe_SplitDCDLoaders["Create Standard DataLoaders"]
        DataPipe_End["Return Loaders/Datasets"]:::Pipeline

        DataPipe_Start --> DataPipe_TrajCheck
        DataPipe_TrajCheck -- Yes --> DataPipe_GetTrajHandler
        DataPipe_GetTrajHandler --> DataPipe_GenTraj
        DataPipe_GenTraj --> DataPipe_InstTrajDataH
        DataPipe_InstTrajDataH --> DataPipe_CreateTrajDS
        DataPipe_CreateTrajDS --> DataPipe_SplitTrajLoaders
        DataPipe_SplitTrajLoaders --> DataPipe_End
        DataPipe_TrajCheck -- No --> DataPipe_InstDCDSamples
        DataPipe_InstDCDSamples --> DataPipe_CreateDCDDS
        DataPipe_CreateDCDDS --> DataPipe_SplitDCDLoaders
        DataPipe_SplitDCDLoaders --> DataPipe_End
    end

    subgraph "Training Pipeline"
        direction TB
        TrainPipe_Start["_run_training_pipeline()"]:::Pipeline
        TrainPipe_Check{Train Enabled?}:::Decision
        TrainPipe_TrajCheck{Trajectory Enabled?}:::Decision
        TrainPipe_InstTrajTrainer["Instantiate TrajectoryTrainer (runners/training.py)"]
        TrainPipe_TrajTrain["trainer.train() (Nested Loops)"]
        TrainPipe_InstDCDTrainer["Instantiate Trainer (DCD_MUSIC)"]:::DCD
        TrainPipe_DCDTrain["trainer.train() (Standard Loop)"]:::DCD
        TrainPipe_End["Return Updated Model"]:::Pipeline

        TrainPipe_Start --> TrainPipe_Check
        TrainPipe_Check -- Yes --> TrainPipe_TrajCheck
        TrainPipe_TrajCheck -- Yes --> TrainPipe_InstTrajTrainer
        TrainPipe_InstTrajTrainer --> TrainPipe_TrajTrain
        TrainPipe_TrajTrain --> TrainPipe_End
        TrainPipe_TrajCheck -- No --> TrainPipe_InstDCDTrainer
        TrainPipe_InstDCDTrainer --> TrainPipe_DCDTrain
        TrainPipe_DCDTrain --> TrainPipe_End
        TrainPipe_Check -- No --> TrainPipe_End
    end

    subgraph "Evaluation Pipeline"
        direction TB
        EvalPipe_Start["_run_evaluation_pipeline()"]:::Pipeline
        EvalPipe_Check{Evaluate Enabled?}:::Decision
        EvalPipe_InstEvaluator["Instantiate Evaluator (runners/evaluation.py)"]
        EvalPipe_Evaluate["evaluator.evaluate()"]
        EvalPipe_End["Return Results"]:::Pipeline

        EvalPipe_Start --> EvalPipe_Check
        EvalPipe_Check -- Yes --> EvalPipe_InstEvaluator
        EvalPipe_InstEvaluator --> EvalPipe_Evaluate
        EvalPipe_Evaluate --> EvalPipe_End
        EvalPipe_Check -- No --> EvalPipe_End
    end

    %% Core Logic Flow connecting to pipelines
    ENTRY --> CORE

    CORE --> Step4["4 Check Scenario"]:::StepStyle
    Step4 --> ScenarioCheck{Scenario?}:::Decision

    %% Scenario Path
    ScenarioCheck -- Yes --> ScenarioLoop["Scenario Loop\n(Modify Config -> Call Run -> Store -> Restore)"]
    ScenarioLoop -.-> CORE
    %% Dashed line for internal call representing sim.run() called by sim.run_scenario()
   %% ScenarioLoop --> AggregateResults["Aggregate Results"]
    %%AggregateResults -- "Aggregated Scenario Results" --> RETURN_POINT["Return Point (to Main)"]:::ReturnPoint

    %% Single Run Path (or initiated from Scenario Loop)
    ScenarioCheck -- No --> DataPipe_Start
    ScenarioLoop -- "Call Data Pipeline" --> DataPipe_Start
    %% Call from within loop

    DataPipe_End -- "Loaders/Datasets" --> CORE
    CORE -- "Call Training Pipeline" --> TrainPipe_Start
    TrainPipe_End -- "Updated Model" --> CORE
    CORE -- "Call Evaluation Pipeline" --> EvalPipe_Start
    EvalPipe_End -- "Evaluation Results" --> CORE

    %%CORE -- "Final Single Run Results" --> RETURN_POINT
    %% Return results from sim.run()
