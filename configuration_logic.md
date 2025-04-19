```mermaid
graph TD
    %% ===== Configuration & Factory Flow (Simplified Returns) =====

    %% Define Styles
    classDef MainStyle fill:#ddeeff,stroke:#6699cc,stroke-width:1px,font-weight:bold %% Style for main.py
    classDef HandlerStyle fill:#fff0dd,stroke:#ffcc66,stroke-width:1px,font-weight:bold %% Style for config_handler.py
    classDef FactoryStyle fill:#eeffee,stroke:#99cc99,stroke-width:1px,font-weight:bold %% Style for factory.py
    classDef LoaderStyle fill:#f5f5f5,stroke:#aaa,stroke-width:1px %% Style for loader/schema actions
    classDef CreateStyle fill:#e6e0f8,stroke:#9370db,stroke-width:1px %% Style for component creation calls
    classDef DCDStyle fill:#f9d,stroke:#f66,stroke-width:1px %% Style for DCD_MUSIC imports/objects
    classDef Decision fill:#eee,stroke:#333,stroke-width:1px %% Style for decisions/checks
    %% ReturnStyle class removed as nodes are removed

    %% Subgraphs for each module
    subgraph "main.py"
        direction TB
        M_Start["User runs 'python main.py simulate [OPTIONS]'"]:::MainStyle
        M_Parse["click parses args (config_path, output_dir, overrides)"]:::MainStyle
        M_CallConfig["Call config_handler.setup_configuration(...)"]:::MainStyle
        %% M_Receive node removed
        M_Proceed["Proceed to Simulation(components)"]:::MainStyle

        M_Start --> M_Parse --> M_CallConfig
        %% Connection from CH back to MAIN handles return below
    end

    subgraph "config_handler.py"
        direction TB
        CH_Setup["setup_configuration(config_path, output_dir, overrides)"]:::HandlerStyle
        CH_Load["Call loader.load_config(config_path)"]:::LoaderStyle
        %% CH_ConfigDict node removed
        CH_CheckOverride{Overrides Provided?}:::Decision
        CH_Override["Call loader.apply_overrides(config_obj, overrides)"]:::LoaderStyle
        %% CH_UpdatedConfigDict node removed
        CH_Validate["Validate config_obj vs Schema"]:::LoaderStyle
        %% CH_ValidatedConfig node removed
        CH_OutputDir["Setup Output Dir"]:::HandlerStyle
        CH_CallFactory["Call factory.create_components_from_config(validated_config_obj)"]:::HandlerStyle
        %% CH_Components node removed
        %% CH_Return node removed

        CH_Setup --> CH_Load --> CH_CheckOverride
        CH_CheckOverride -- Yes --> CH_Override --> CH_Validate
        CH_CheckOverride -- No --> CH_Validate
        CH_Validate --> CH_OutputDir --> CH_CallFactory
        %% Connection from Factory back to CH handles return below
    end

    subgraph "config/factory.py"
        direction TB
        F_CreateComponents["create_components_from_config(config)"]:::FactoryStyle
        F_SysModelStep["Call create_system_model(config)"]:::CreateStyle
        F_SysModelObj["system_model (DCD_MUSIC)"]:::DCDStyle
        F_CheckTraj{config.trajectory.enabled?}:::Decision
        F_TrajStep["Call create_trajectory_handler(config, system_model)"]:::CreateStyle
        F_TrajHandlerObj["trajectory_handler (Optional)"]
        F_CheckData{config.dataset.create_data?}:::Decision
        F_DataStep["Call create_dataset(config, system_model)"]:::CreateStyle
        F_DatasetObj["dataset (DCD_MUSIC - Optional)"]:::DCDStyle
        F_ModelStep["Call create_model(config, system_model)"]:::CreateStyle
        F_ModelObj["model (DCD_MUSIC)"]:::DCDStyle
        F_CheckTrain{Train Enabled & Dataset Exists?}:::Decision
        F_TrainerStep["Call create_trainer(config, model, dataset)"]:::CreateStyle
        F_TrainerObj["trainer (DCD_MUSIC - Optional)"]:::DCDStyle
        F_Package["Package created components in Dict"]:::FactoryStyle
        %% F_ReturnComps node removed

        %% Flow within factory
        F_CreateComponents --> F_SysModelStep --> F_SysModelObj --> F_CheckTraj
        F_CheckTraj -- Yes --> F_TrajStep --> F_TrajHandlerObj --> F_CheckData
        F_CheckTraj -- No --> F_CheckData
        F_CheckData -- Yes --> F_DataStep --> F_DatasetObj --> F_ModelStep
        F_CheckData -- No --> F_ModelStep
        F_ModelStep --> F_ModelObj --> F_CheckTrain
        F_CheckTrain -- Yes --> F_TrainerStep --> F_TrainerObj --> F_Package
        F_CheckTrain -- No --> F_Package
        %% Connection back to CH handles return below
    end

    %% Connections between modules with return values on arrows
    M_CallConfig --> CH_Setup
    CH_CallFactory --> F_CreateComponents
    F_Package -- "components dict" --> CH_CallFactory %% Arrow points back to caller, label indicates return
    CH_CallFactory -- "config_obj, components, output_dir" --> M_Proceed %% Arrow points back to caller, label indicates return
