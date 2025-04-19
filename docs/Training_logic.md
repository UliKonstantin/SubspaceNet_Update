---
config:
  layout: fixed
---
flowchart TD
 subgraph s1["trainer.train()"]
    direction TB
        T_EpochLoop["Epoch Loop (for epoch in N_epochs)"]
        T_TrainEpoch["1 Run Train Epoch\n(Calls _train_epoch)"]
        T_ValidEpoch["2 Run Validation Epoch\n(Calls _validate_epoch)"]
        T_SchedulerStep["3 Update Scheduler"]
        T_SaveLogic["4 Checkpoint Logic\n(Best Model? Save?)"]
        T_EpochEndLoop{"More Epochs?"}
  end
 subgraph subGraph1["Step Loop (Trajectory)"]
    direction TB
        TE_StepLoop["Step Loop (for step in T_length)"]
        TE_Step_GetData["Get Step Data/Labels"]
        TE_Step_Forward["Forward Pass (_forward_step)"]
        TE_Step_Backward["Backward Pass (loss.backward)"]
        TE_Step_OptStep["Optimizer Step"]
        TE_Step_EndLoop{"More Steps?"}
  end
 subgraph subGraph2["Batch Loop (Train)"]
    direction TB
        TE_Unpack["Unpack Batch Data"]
        TE_CheckTraj{"Trajectory Trainer?"}
        TE_BatchLoop["Batch Loop (for batch in dataloader)"]
        subGraph1
        TE_Std_Forward["Forward Pass (model.training_step)"]
        TE_Std_Backward["Backward Pass"]
        TE_Std_OptStep["Optimizer Step"]
        TE_UpdateTotals["Update Epoch Totals (Loss, Acc)"]
        TE_BatchEndLoop{"More Batches?"}
  end
 subgraph s2["_train_epoch()"]
    direction TB
        TE_ModeTrain["model.train()"]
        subGraph2
        TE_Return["Return Train Metrics"]
  end
 subgraph subGraph4["Step Loop (Trajectory - Validate)"]
    direction TB
        TV_StepLoop["Step Loop (for step in T_length)"]
        TV_Step_GetData["Get Step Data/Labels"]
        TV_Step_Forward["Forward Pass (_forward_step)"]
        TV_Step_EndLoop{"More Steps?"}
  end
 subgraph subGraph5["Batch Loop (Validate)"]
    direction TB
        TV_Unpack["Unpack Batch Data"]
        TV_CheckTraj{"Trajectory Trainer?"}
        TV_BatchLoop["Batch Loop (for batch in dataloader)"]
        subGraph4
        TV_Std_Forward["Forward Pass (model Eval)"]
        TV_UpdateTotals["Update Epoch Totals (Loss, Acc)"]
        TV_BatchEndLoop{"More Batches?"}
  end
 subgraph s3["_validate_epoch()"]
    direction TB
        TV_ModeEval["model.eval() & torch.no_grad()"]
        subGraph5
        TV_Return["Return Validation Metrics"]
  end
    T_EpochLoop --> T_TrainEpoch
    T_TrainEpoch --> T_ValidEpoch & TE_ModeTrain
    T_ValidEpoch --> T_SchedulerStep & TV_ModeEval
    T_SchedulerStep --> T_SaveLogic
    T_SaveLogic --> T_EpochEndLoop
    T_EpochEndLoop -- Yes --> T_EpochLoop
    TE_ModeTrain --> TE_BatchLoop
    TE_BatchLoop --> TE_Unpack
    TE_Unpack --> TE_CheckTraj
    TE_StepLoop --> TE_Step_GetData
    TE_Step_GetData --> TE_Step_Forward
    TE_Step_Forward --> TE_Step_Backward
    TE_Step_Backward --> TE_Step_OptStep
    TE_Step_OptStep --> TE_Step_EndLoop
    TE_Step_EndLoop -- Yes --> TE_StepLoop
    TE_CheckTraj -- Yes --> TE_StepLoop
    TE_CheckTraj -- No --> TE_Std_Forward
    TE_Std_Forward --> TE_Std_Backward
    TE_Std_Backward --> TE_Std_OptStep
    TE_Step_EndLoop -- No --> TE_UpdateTotals
    TE_Std_OptStep --> TE_UpdateTotals
    TE_UpdateTotals --> TE_BatchEndLoop
    TE_BatchEndLoop -- Yes --> TE_BatchLoop
    TE_BatchEndLoop -- No --> TE_Return
    TV_ModeEval --> TV_BatchLoop
    TV_BatchLoop --> TV_Unpack
    TV_Unpack --> TV_CheckTraj
    TV_StepLoop --> TV_Step_GetData
    TV_Step_GetData --> TV_Step_Forward
    TV_Step_Forward --> TV_Step_EndLoop
    TV_Step_EndLoop -- Yes --> TV_StepLoop
    TV_CheckTraj -- Yes --> TV_StepLoop
    TV_CheckTraj -- No --> TV_Std_Forward
    TV_Step_EndLoop -- No --> TV_UpdateTotals
    TV_Std_Forward --> TV_UpdateTotals
    TV_UpdateTotals --> TV_BatchEndLoop
    TV_BatchEndLoop -- Yes --> TV_BatchLoop
    TV_BatchEndLoop -- No --> TV_Return
    Core_CallTrain["core.py: Calls trainer.train(train_loader, valid_loader)"] --> T_EpochLoop
    T_EpochEndLoop -- No --> Core_CallTrain
     T_EpochLoop:::LoopStyle
     T_TrainEpoch:::TrainerStyle
     T_ValidEpoch:::TrainerStyle
     T_SchedulerStep:::StepStyle
     T_SaveLogic:::StepStyle
     T_EpochEndLoop:::Decision
     TE_StepLoop:::LoopStyle
     TE_Step_GetData:::DataStyle
     TE_Step_Forward:::StepStyle
     TE_Step_Backward:::StepStyle
     TE_Step_OptStep:::StepStyle
     TE_Step_EndLoop:::Decision
     TE_Unpack:::DataStyle
     TE_CheckTraj:::Decision
     TE_BatchLoop:::LoopStyle
     TE_Std_Forward:::DCDStyle
     TE_Std_Backward:::DCDStyle
     TE_Std_OptStep:::DCDStyle
     TE_UpdateTotals:::StepStyle
     TE_BatchEndLoop:::Decision
     TE_ModeTrain:::StepStyle
     TE_Return:::TrainerStyle
     TV_StepLoop:::LoopStyle
     TV_Step_GetData:::DataStyle
     TV_Step_Forward:::StepStyle
     TV_Step_EndLoop:::Decision
     TV_Unpack:::DataStyle
     TV_CheckTraj:::Decision
     TV_BatchLoop:::LoopStyle
     TV_Std_Forward:::DCDStyle
     TV_UpdateTotals:::StepStyle
     TV_BatchEndLoop:::Decision
     TV_ModeEval:::StepStyle
     TV_Return:::TrainerStyle
     Core_CallTrain:::CoreStyle
    classDef CoreStyle fill:#ddeeff,stroke:#6699cc,stroke-width:1px,font-weight:bold
    classDef TrainerStyle fill:#eeffee,stroke:#99cc99,stroke-width:1px,font-weight:bold
    classDef LoopStyle fill:#fff0dd,stroke:#ffcc66,stroke-width:1px
    classDef StepStyle fill:#f5f5f5,stroke:#aaa,stroke-width:1px
    classDef DCDStyle fill:#f9d,stroke:#f66,stroke-width:1px
    classDef Decision fill:#eee,stroke:#333,stroke-width:1px
    classDef DataStyle fill:#d4fcd7,stroke:#6aa84f
