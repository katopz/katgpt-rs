# Research 105: SkillOpt — Text-Space Skill Optimization for Frozen Agents

**Paper:** "SkillOpt: Executive Strategy for Self-Evolving Agent Skills" (Yang et al., Microsoft, arXiv:2605.23904, May 2026)
**Date:** 2026-05-25
**Verdict:** ✅ **ADOPT — Modelless text-space optimization loop for WASM validators, pruner configs, game rules**

---

## Paper Summary

SkillOpt treats an agent's **skill document** (natural-language procedural rules) as the trainable external state, optimized offline by a separate frontier model with deep-learning-style controls:

| Component | DL Analogue | What It Does |
|-----------|-------------|--------------|
| Rollout batches | Forward pass | Target model executes tasks with current skill, producing scored trajectories |
| Minibatch reflection | Backward pass | Optimizer analyzes success/failure trajectories, proposes structured edits |
| Edit budget (L_t) | Learning rate | Max number of add/delete/replace edits per step; cosine decay schedule |
| Validation gate | Early stopping / validation | Candidate skill evaluated on held-out split; only accepted if strictly better |
| Rejected-edit buffer | Negative feedback / momentum | Failed edits stored as negative examples for future reflection calls |
| Slow/meta update | Epoch-wise momentum | Cross-epoch longitudinal guidance injected into protected skill region |

### Key Results (GPT-5.5, direct chat)

| Benchmark | No Skill | SkillOpt | Delta |
|-----------|----------|----------|-------|
| SearchQA | 77.7 | 87.3 | +9.6 |
| SpreadsheetBench | 41.8 | 80.7 | +38.9 |
| OfficeQA | 33.1 | 72.1 | +39.0 |
| DocVQA | 78.8 | 91.2 | +12.4 |
| LiveMath | 37.6 | 66.9 | +29.3 |
| ALFWorld | 83.6 | 95.5 | +11.9 |

- **52/52 cells best or tied** across 7 models × 6 benchmarks × 3 harnesses
- Skills remain compact: 300–2,000 tokens after only 1–4 accepted edits
- Cross-model transfer: positive on all tested pairs
- Cross-harness transfer: Codex→Claude Code +59.7 points (SpreadsheetBench)
- **Zero inference-time cost** — optimizer runs offline, deployed artifact is static text

### Critical Design Properties

1. **Bounded updates prevent skill drift** — without edit budget, skills overfit or collapse
2. **Validation gate is essential** — plausible textual edits can hurt performance
3. **Rejected-edit buffer stabilizes** — removing it costs -1.6 to -4.6 points
4. **Slow/meta update captures durable lessons** — removing both drops SpreadsheetBench -22.5 points
5. **Edit economy** — median 2.5 accepted edits achieve +23.5pt average improvement

---

## Distillation to Our Architecture

### Conceptual Mapping

| SkillOpt | Our Stack | Location |
|----------|-----------|----------|
| Skill document | WASM validator rules + pruner configs + game heuristics | riir-ai (secret) |
| Target model (frozen) | Game AI (MCTS + Bandit + Fourier) | katgpt-rs (traits) |
| Optimizer model | External LLM (API call, not in Rust) | riir-ai (training infra) |
| Rollout batches | Game episodes (arena runs) | katgpt-rs (benchmark infra) |
| Minibatch reflection | LLM analyzes win/loss trajectories | riir-ai (new) |
| Edit budget | Bounded config edits per epoch | katgpt-rs (trait) |
| Validation gate | GOAT proof system | katgpt-rs (benchmark) |
| Rejected-edit buffer | Failed config history | riir-ai (new) |
| Slow/meta update | Cross-epoch game rule consolidation | riir-ai (new) |

### What We Already Have (≈70% of SkillOpt)

| SkillOpt Component | Our Equivalent | Status |
|--------------------|----------------|--------|
| Rollout execution | Arena benchmarks (Bomber, Go, Monopoly, FFT) | ✅ Production |
| Scored trajectories | Win/loss + move accuracy + game trace events | ✅ Production (Plan 124) |
| Frozen target model | Game AI is fully algorithmic (no weight updates) | ✅ By design |
| Validation/testing split | GOAT proof thresholds, benchmark splits | ✅ Production |
| Edit application | WASM validator hot-swap, config TOML updates | ✅ Via Plan 034 |
| Harness-agnostic adapter | Game-specific `DomainBenchmark` trait | ✅ Via Plan 076 |

### What We're Missing (≈30%)

| Missing Component | Description | Complexity |
|-------------------|-------------|------------|
| Text edit proposer | LLM call that proposes bounded add/delete/replace edits to game rules | Medium — API wrapper + prompt templates |
| Minibatch reflection | Group win/loss trajectories, analyze patterns, propose edits | Medium — clustering + prompt engineering |
| Edit budget scheduler | Cosine/constant/linear decay for max edits per step | Low — trivial scheduler |
| Rejected-edit buffer | Store failed proposals as negative examples for future LLM calls | Low — JSONL + blake3 hashing |
| Slow/meta update | Cross-epoch longitudinal guidance in protected config region | Medium — diff engine + protected section markers |
| Cross-epoch comparison | Sample same game seeds under old/new skill, compare outcomes | Low — seed replay (Plan 092 Freeze/Thaw already does this) |

---

## Model-Based vs Modelless Split

### Modelless (katgpt-rs) — The Optimization Loop Framework

These are **generic optimization primitives** that any domain can use:

```rust
// New trait: bounded text-space optimizer
pub trait SkillOptimizer {
    /// Propose edits given scored trajectories
    fn propose_edits(
        &self,
        trajectories: &[ScoredTrajectory],
        current_skill: &str,
        edit_budget: usize,
        rejected_buffer: &[RejectedEdit],
    ) -> Vec<SkillEdit>;

    /// Apply bounded edits to skill document
    fn apply_edits(
        skill: &str,
        edits: &[SkillEdit],
        budget: usize,
    ) -> String;

    /// Validate candidate skill against held-out benchmark
    fn validate(
        &self,
        candidate: &str,
        current: &str,
        benchmark: &dyn Benchmark,
    ) -> ValidationGate;
}

pub struct SkillEdit {
    pub op: EditOp,       // Append, InsertAfter, Replace, Delete
    pub target: Option<String>,
    pub content: String,
    pub support_count: usize,
    pub source: EditSource, // Failure, Success
}

pub struct ValidationGate {
    pub accepted: bool,
    pub candidate_score: f64,
    pub current_score: f64,
    pub delta: f64,
}

pub struct RejectedEdit {
    pub edit: SkillEdit,
    pub score_delta: f64,  // negative = edit hurt
    pub failure_patterns: Vec<String>,
}

pub enum EditBudgetSchedule {
    Constant(usize),
    Linear { start: usize, end: usize },
    Cosine { start: usize, floor: usize },
    Autonomous,  // LLM decides budget
}
```

### Model-Based (riir-ai) — Game-Specific Skill Artifacts

These are **secret domain knowledge** — the actual optimized game rules:

- **Bomber validator tuning**: SkillOpt-style optimization of `bomber_validator.wasm` rules
- **Go heuristic tuning**: Optimizing Fourier period selection, opening book, komi adjustments
- **NPC dialog policies**: Quest pack FSM rules optimized from player interaction logs
- **TFT strategy configs**: Party AI composition rules from arena results

### Feature Gate

```toml
# katgpt-rs (MIT)
[features]
skill_opt = []  # Generic text-space optimization loop framework
```

```toml
# riir-ai (private) — no feature gate needed, always available
# Game-specific skill optimization uses katgpt-rs::skill_opt as dependency
```

---

## GOAT Pillar Alignment

Per `27_mmo_goat_pillars_decision_matrix.md`:

| Criterion | Score | Evidence |
|-----------|-------|----------|
| GOAT passed | ⏳ | Not yet implemented — this research proposes the plan |
| MMO-product | ✅ | Auto-tuned game rules = better NPC behavior, combat AI, quest logic |
| LoRA-independent | ✅ | Pure text-space optimization — no neural network involved |
| Defensible | ✅ | Game-specific skill artifacts are private domain knowledge (Secret A2) |
| Secret coverage | A2 + B | Optimized `validator.wasm` (A2) + episode traces used as training data (B) |

**Pillar fit:** Strengthens Pillar 2 (WASM Validators) and Pillar 3 (NPC Dialog Engine) by automating what was previously hand-tuned. The optimization loop itself is generic (katgpt-rs), but the **specific optimized artifacts are the moat** (riir-ai).

**Not a new pillar** — it's a force multiplier for existing pillars.

---

## What Makes This a Super GOAT (Keep Secret)

If we implement SkillOpt-style text optimization for game validators:

1. **Bomber validator auto-tuning**: Run 1000 arena games → LLM proposes rule edits → validation gate accepts only improvements → repeat. The resulting `bomber_validator.wasm` has rules no human would write but that empirically win more games.

2. **Cross-game transfer**: A skill optimized for Bomber might transfer to Go (both are spatial games with Fourier heuristics). If it does, that's a genuine research contribution.

3. **The optimization loop is the selling point**: "Your game gets better every time someone plays it" — without model weight updates, without GPU training, just text edits to rule configs.

4. **Zero inference cost**: The optimizer runs in the cloud during off-peak hours. The deployed game uses only the static, optimized skill artifact.

**This stays in riir-ai because:**
- Game-specific skill artifacts are trade secrets (Secret A2)
- The optimizer prompts encode game domain knowledge
- Cross-game transfer results are competitive intelligence
- The optimization history (which edits helped, which failed) is training data for future games

---

## Comparison to Our Existing Distillation Work

| Existing Work | What It Does | SkillOpt Analogue | Gap |
|---------------|-------------|-------------------|-----|
| ROPD (R036, Plan 071/072) | Rubric-based on-policy distillation | Minibatch reflection (failure analysis) | ROPD scores rubric criteria; SkillOpt proposes structural edits |
| SDAR (R038, Plan 072) | Self-distilled agentic RL with gated response masking | Validation gate | SDAR gates on response quality; SkillOpt gates on held-out benchmark score |
| GFlowNet (R023, Plan 052) | Flow-based modelless distillation | Edit proposal mechanism | GFlowNet samples from reward-weighted distribution; SkillOpt uses LLM to propose edits |
| G-Zero Self-Play (R021, Plan 049) | Modelless strategy discovery via Hint-δ | Rollout batch generation | G-Zero discovers strategies; SkillOpt encodes them into portable artifacts |
| Freeze/Thaw (R069, Plan 092) | Cross-epoch game replay comparison | Slow/meta update | Freeze/Thaw compares old/new policies; SkillOpt adds longitudinal guidance injection |
| Event Log (Plan 124) | Game trace fork-diff | Scored trajectory recording | Event Log records traces; SkillOpt adds pattern analysis + edit proposal |

**Key difference:** All our existing distillation work operates on **numeric/structured signals** (scores, gradients, flow rewards). SkillOpt operates on **text-space rules** — it literally edits markdown documents that become the game's procedural knowledge. This is orthogonal to and composable with our existing stack.

---

## Honest Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LLM optimizer is expensive | Medium | Medium | Target-matched optimizer recovers 56–74% of gains (Table 5) |
| Game rules don't improve from text edits | Low | High | SkillOpt proves procedural domains improve most (+38.9 SpreadsheetBench) |
| Overfitting to training games | Medium | Medium | Validation gate + held-out test split (already in GOAT) |
| Skill document becomes unstable | Low | High | Edit budget + cosine decay prevent this (ablation proof) |
| Cross-game transfer fails | Medium | Low | Transfer is bonus, not required; per-game optimization still valuable |
| Requires external LLM API | High | Low | One-time offline cost; could use Qwen 3.5-4B (target-matched) |

**Net verdict:** The risk/reward is excellent. SkillOpt proves text-space optimization works on procedural domains. We have 70% of the infrastructure. The missing 30% is a ~3-4 day implementation with clear GOAT proof targets.

---

## References

- SkillOpt paper: arXiv:2605.23904
- SkillOpt repo (MIT): github.com/microsoft/SkillOpt (available in `.raw/SkillOpt/`)
- Our modelless duality: Research 037
- Our rubric distillation: Research 036
- GOAT pillars: riir-ai `.docs/27_mmo_goat_pillars_decision_matrix.md`
