# Plan 163: FeedbackBandit — Harness + Weight Co-Evolution

**Moved to:** [`riir-ai/.plans/178_sia_feedback_bandit.md`](../../riir-ai/.plans/178_sia_feedback_bandit.md)

**Rationale:** The weight-update arm, `FeedbackTrainingBridge`, RL algorithm selection, and LoRA hot-swap are all riir-ai infrastructure. katgpt-rs only needs 2 new enum variants and a feature flag.

**Cross-repo split:**
- katgpt-rs (MIT): `PlanningDecision::HarnessUpdate`, `PlanningDecision::WeightUpdate` enum variants + `sia_feedback` feature flag
- riir-ai (private): `FeedbackTrainingBridge`, `WeightUpdateRequest`, RL algorithm selection, GOAT proof, default-on promotion
