graph TD
    %% ===== Diagram 1: Main & Configuration Flow =====

    %% Define Styles
    classDef Orchestrator fill:#ddeeff,stroke:#6699cc,stroke-width:2px,font-weight:bold %% Style for main.py
    classDef Config fill:#fff0dd,stroke:#ffcc66,stroke-width:1px %% Style for config handler
    classDef CoreBlock fill:#dcdcdc,stroke:#666,stroke-width:1px %% Style for representing core.py
    classDef StepStyle fill:#e6e0f8,stroke:#9370db,stroke-width:1px,font-weight:bold 
    %% Style for numbered steps (light purple)

    %% Core Blocks
    USER["User runs 'python main.py simulate [OPTIONS]'"]
    MAIN["main.py (CLI, Orchestration)"]:::Orchestrator
    CONF["config_handler.py (setup_configuration)"]:::Config
    CORE_INTERFACE["core.py (Simulation Logic - Interface)"]:::CoreBlock
     %% Represents the called block
    EXIT["Exit"]

    %% Connections
    USER --> MAIN

    MAIN --> Step1["1 Call setup_configuration"]:::StepStyle --> CONF
    CONF --> Step2["2 Return config_obj, components"]:::StepStyle --> MAIN

    MAIN --> Step3["3 Instantiate Simulation & Call Core"]:::StepStyle --> CORE_INTERFACE

    CORE_INTERFACE -- "Final Results (Single Run or Scenario)" --> Step_ReturnResults["Return Final Results"]:::StepStyle --> MAIN

    MAIN -- "Log Results & Exit" --> EXIT