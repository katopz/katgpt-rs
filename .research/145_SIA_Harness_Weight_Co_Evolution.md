# Research 145: SIA — Harness + Weight Co-Evolution

**Moved to:** [`riir-ai/.research/033_SIA_Harness_Weight_Co_Evolution.md`](../../riir-ai/.research/033_SIA_Harness_Weight_Co_Evolution.md)

**Rationale:** The SIA core insight (dynamically switching to weight updates via LoRA DPO/GRPO) is riir-ai domain — the `FeedbackTrainingBridge`, RL algorithm selection, and `WeightUpdateRequest` handling all live in riir-ai. katgpt-rs changes are limited to adding 2 new `PlanningDecision` enum variants and a feature flag.

**Verdict:** GAIN — see riir-ai research for full analysis.
