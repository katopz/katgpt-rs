# Research 172: MUSE Skill Lifecycle → Inference-Time Skill Evolution (ITSE)

**Paper:** "MUSE-Autoskill: Self-Evolving Agent Skills via Lifecycle Management" (ByteDance, arXiv:2605.27366, May 2026)
**Date:** 2026-06-05
**Verdict:** ✅ **ADOPT — 3 components are cheap to implement and directly improve reliability. Must be on by default.**

---

## 1. Paper Summary

MUSE-Autoskill proposes a 5-stage skill lifecycle for self-evolving LLM agents: **Creation → Memory → Management → Evaluation → Refinement**. Skills are natural-language procedural documents that an LLM agent creates, stores, retrieves, tests, and improves across sessions. The key insight: skills are **Pareto-optimal artifacts** — higher reward AND lower latency AND fewer tokens than unassisted reasoning.

### Key Results

| Metric | Value | Context |
|--------|-------|---------|
| Accuracy lift from skill lifecycle | **+15.21pp** | SkillsBench (51 tasks) |
| Self-generated skill ceiling | **87.94%** | On 35 tasks — exceeds human-skill ceiling |
| Cross-agent transfer gain | **+10.51pp** | Skills from one agent improve another (closing 79% of gap) |
| Pareto dominance | Reward ↑ Latency ↓ Tokens ↓ | Skills beat chain-of-thought on all three axes simultaneously |
| Amortization point | **~3 reuses** | 383K token creation cost, ~122K token saving per use |
| Skill retention after pruning | 95%+ | Long-term memory preserves core skills |
| Eval-gate rejection rate | ~30% | Unit-test gating catches harmful skills before registration |
| Progressive disclosure saving | 40-60% prompt tokens | Catalog-in-prompt, body-on-demand |

### The 5-Stage Lifecycle

```
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│  CREATION   │───>│   MEMORY    │───>│  MANAGEMENT  │───>│ EVALUATION  │───>│ REFINEMENT  │
│ LLM writes  │    │ Multi-level │    │ Retrieval +  │    │ Unit-test   │    │ Edit budget  │
│ skill doc   │    │ store       │    │ catalog      │    │ gating      │    │ + validation │
└─────────────┘    └─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
       ^                                                                              │
       └──────────────────────────────────────────────────────────────────────────────┘
```

1. **Creation:** LLM generates skill document from task experience (avg 383K tokens cost)
2. **Memory:** Three-tier store — short-term (current session), long-term (cross-session), per-skill (accumulated edge cases)
3. **Management:** Adaptive retrieval via relevance scoring + progressive disclosure (catalog in prompt, body on demand)
4. **Evaluation:** Unit-test gating before registration — skills must pass sandboxed tests
5. **Refinement:** Bounded edit budget with validation gate — only accept if strictly better on held-out split

### Adaptive Context Compression

MUSE uses a DAG-based two-level compression scheme:
- **L1:** Per-node summary (each step in the reasoning chain gets compressed)
- **L2:** Span merge (consecutive compressed nodes merged when semantically similar)

This allows long inference chains to fit in context without losing critical information.

---

## 2. Distillation to Our Architecture

### What We Already Have (~75%)

| MUSE Component | Our Equivalent | Location | Status |
|----------------|----------------|----------|--------|
| Skill interface (behavioral contract) | `ConstraintPruner` trait (`is_valid`, `relevance`) | `src/pruners/mod.rs` | ✅ Production |
| Skill retrieval (which skill to use) | `BanditPruner<P>` with `BanditStrategy` | `src/pruners/bandit.rs` | ✅ Production |
| Skill refinement (improve over time) | `AbsorbCompress` — promote winning patterns | `src/pruners/absorb_compress.rs` | ✅ Production |
| Skill update (hot-swap at runtime) | `HotSwapPruner` — atomic pruner replacement | `src/pruners/hot_swap.rs` | ✅ Production |
| Cross-session persistence | Freeze/thaw — `repr(C)` disk I/O | `src/pruners/freeze.rs` | ✅ Production |
| Multi-level memory | Five-tier memory system | `src/memory/` | ✅ Production (MORE granular than MUSE's 3-tier) |
| Context compression | `AbsorbCompress` delta-gated merging | `src/pruners/absorb_compress.rs` | ✅ Production |
| Bandit feedback as skill selection | UCB1/Thompson/ε-greedy over pruner arms | `src/pruners/bandit.rs` | ✅ Production |
| Safe baseline (fallback) | `NoScreeningPruner` as conservative default | `src/pruners/mod.rs` | ✅ Production |
| Validation/proof | GOAT benchmark proof system | `katgpt-rs/.benchmarks/` | ✅ Production |

### What We're Missing (~25%) — The Gap

| MUSE Feature | What's Missing | Complexity | Impact |
|--------------|---------------|------------|--------|
| **Per-skill memory** | No accumulated experience per pruner across sessions (`.memory.md` equivalent) | Low — extend freeze/thaw to carry per-pruner edge-case notes | High — skills learn from mistakes |
| **Test-gated registration** | No unit-test gate before a pruner enters the bandit bank | Medium — WASM sandbox + test harness | High — prevents harmful skills from polluting bandit |
| **Progressive disclosure catalog** | No catalog (name+desc) injection with on-demand body load | Low — catalog struct + lazy load | Medium — saves 40-60% prompt tokens for large pruner sets |
| **Adaptive inference context compression** | No two-level (L1 per-node, L2 span merge) for long inference chains | Medium — DAG + summarizer | Medium — enables longer reasoning without context overflow |

---

## 3. Creative Fusion: Inference-Time Skill Evolution (ITSE)

### The Core Idea

**Every pruner/validator/bandit arm is a "skill" with a unified lifecycle.** Unlike MUSE (which uses LLM reasoning to create/refine skills), our skills evolve through four purely modelless mechanisms:

```
┌──────────────────────────────────────────────────────────────┐
│                   INFERENCE-TIME SKILL EVOLUTION              │
│                                                              │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────┐ │
│  │  BANDIT   │  │ PER-SKILL    │  │   WASM   │  │ FREEZE  │ │
│  │ FEEDBACK  │  │ MEMORY       │  │ TEST GATE│  │ /THAW   │ │
│  │           │  │              │  │          │  │ w/MEM   │ │
│  │ Arms that │  │ Accumulate   │  │ Skills   │  │ Skills  │ │
│  │ work get  │  │ edge cases   │  │ must     │  │ survive │ │
│  │ promoted  │  │ across       │  │ pass     │  │ w/their │ │
│  │           │  │ sessions     │  │ sandbox  │  │ exper.  │ │
│  │ (existing)│  │ (NEW)        │  │ (NEW)    │  │ (ENHNC) │ │
│  └──────────┘  └──────────────┘  └──────────┘  └─────────┘ │
│         │              │               │              │      │
│         └──────────────┴───────────────┴──────────────┘      │
│                        │                                      │
│                   SKILL LIFECYCLE                             │
│                        │                                      │
│         ┌──────────────┴───────────────┐                     │
│         │                              │                     │
│    ┌────▼─────┐                 ┌──────▼──────┐              │
│    │ PROMOTE  │                 │   DEMOTE    │              │
│    │ to stable │                │ to cold     │              │
│    │ arm set  │                 │ storage     │              │
│    └──────────┘                 └─────────────┘              │
└──────────────────────────────────────────────────────────────┘
```

### Mechanism 1: Bandit Feedback (Existing)

Already implemented. The `BanditPruner` promotes arms that achieve positive reward and demotes arms that don't. This IS MUSE's "management" stage, but modelless — no LLM needed for skill selection.

### Mechanism 2: Per-Skill Memory (NEW)

Each pruner accumulates experience across sessions in a memory file (analogous to MUSE's per-skill `.memory.md`):

```rust
/// Per-pruner memory: accumulated edge cases and lessons across sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrunerMemory {
    pub pruner_id: String,
    pub edge_cases: Vec<EdgeCase>,
    pub success_patterns: Vec<SuccessPattern>,
    pub failure_signatures: Vec<FailureSignature>,
    pub total_sessions: u32,
    pub last_updated: u64, // epoch seconds
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeCase {
    pub context_hash: [u8; 32], // blake3 of the input context
    pub expected: f64,           // expected reward
    pub actual: f64,             // actual reward
    pub deviation: f64,          // |expected - actual|
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureSignature {
    pub pattern: String,         // e.g., "high_entropy_low_reward"
    pub count: u32,
    pub last_seen: u64,
    pub recovery_action: String, // what worked to fix it
}
```

This memory is:
- **Saved** via freeze/thaw (enhanced — memory goes alongside Q-values)
- **Loaded** on pruner initialization
- **Queried** before pruner invocation (edge-case lookup table)
- **Updated** after each session with new observations

**Cost:** Negligible — blake3 hashes + small Vec. Memory file is typically <1KB per pruner.

### Mechanism 3: WASM Test Gates (NEW)

Before a new pruner enters the bandit bank, it must pass sandboxed unit tests:

```rust
/// Test gate for pruner registration.
pub trait PrunerTestGate {
    /// Run sandboxed tests; return pass/fail + coverage.
    fn test(&self, pruner: &dyn ConstraintPruner) -> TestGateResult;
}

pub struct TestGateResult {
    pub passed: bool,
    pub coverage: f64,        // fraction of test cases exercised
    pub regressions: u32,     // tests that passed before but now fail
    pub avg_reward_delta: f64, // vs existing best arm
}

pub struct WasmTestGate {
    sandbox: WasmSandbox,
    test_cases: Vec<TestCase>,
    regression_threshold: f64, // max allowed regressions (default: 0)
}
```

**Why WASM:** The existing WASM validator infrastructure (Plan 034) provides sandboxed execution. Test cases are domain-specific (riir-ai), but the gate mechanism is generic (katgpt-rs).

**MUSE's result:** ~30% of proposed skills are rejected by test gating. This prevents harmful skills from entering the bandit bank — a direct reliability improvement.

### Mechanism 4: Freeze/Thaw with Memory (ENHANCED)

Existing freeze/thaw persists Q-values and visit counts. Enhanced version also persists per-pruner memory:

```rust
/// Enhanced frozen data: bandit stats + per-pruner memory.
#[repr(C)]
pub struct FrozenSkillBank {
    pub magic: [u8; 4],
    pub version: u32,
    pub bandit_stats: FrozenBanditStats,
    pub pruner_memories: [FrozenPrunerMemory; MAX_PRUNERS],
    pub num_memories: u32,
}
```

The key enhancement: when a pruner is thawed, it remembers its past edge cases and failure signatures. This prevents repeating the same mistakes across sessions.

### Why ITSE Beats MUSE for Our Architecture

| Dimension | MUSE (LLM-based) | ITSE (Modelless) |
|-----------|-------------------|-------------------|
| Skill creation | LLM generates text (383K tokens) | Pruner already exists — no creation cost |
| Skill selection | Embedding-based retrieval | Bandit UCB1 (proven optimal) |
| Skill testing | LLM-generated unit tests | WASM sandbox (deterministic) |
| Skill refinement | LLM edits text | Bandit feedback + memory (O(1) per use) |
| Cross-session persistence | Vector DB | Freeze/thaw binary (zero-dependency) |
| Amortization | ~3 reuses (383K cost) | **Immediate** (0 creation cost) |
| Latency | Additional LLM call for skill retrieval | Bandit selection is ~O(log N) |

**The critical difference:** MUSE pays 383K tokens to CREATE a skill. Our skills already exist as pruners — we pay 0 tokens for creation. Our entire cost structure is:
- **Per-skill memory:** ~1KB binary per pruner (negligible)
- **Test gate:** WASM execution (microseconds, no LLM)
- **Freeze/thaw:** Binary I/O (microseconds, no serde)

---

## 4. GOAT Verdict

### The GOAT Insight

The MUSE bomber arena shows: **HL (+475) > LoRA+WASM (-15)**. This means WASM validators WITHOUT lifecycle management can HURT performance. But MUSE proves that adding lifecycle management (per-validator memory, test gating, refinement) flips this to **HL+WASM+Lifecycle > HL alone**.

Our architecture has HL + bandits + freeze/thaw. Adding per-pruner memory + test gates + progressive disclosure completes the lifecycle. The prediction: **HL+ITSE > HL alone**, and the margin should be comparable to MUSE's +15.21pp.

### Verdict Table

| Component | GOAT? | Default? | Reason |
|-----------|-------|----------|--------|
| Per-pruner memory | ✅ GOAT | ✅ Default | Near-zero cost, prevents repeated mistakes across sessions |
| WASM test gates | ✅ GOAT | ✅ Default | Catches ~30% harmful skills, deterministic, reusable |
| Progressive disclosure catalog | ✅ GOAT | ✅ Default | Saves 40-60% prompt tokens, lazy load is free |
| Adaptive context compression | 🔧 Feature-gated | ❌ Default-off | Needs DAG infra, only useful for very long inference chains |

### Justification

1. **Per-pruner memory** is a natural extension of existing freeze/thaw — same `repr(C)` binary, just more fields. The `EdgeCase` and `FailureSignature` types are new but trivial (~50 lines each). GOAT proof: run bomber arena with/without memory, measure win rate delta.

2. **WASM test gates** leverage existing WASM validator infra (Plan 034). The `PrunerTestGate` trait is ~30 lines. Test cases stay in riir-ai (domain-specific). GOAT proof: measure bandit bank contamination rate with/without gates.

3. **Progressive disclosure catalog** is a lazy-loading wrapper around existing pruners. The `SkillCatalog` struct is ~40 lines. GOAT proof: measure prompt token usage with/without catalog.

4. **Adaptive context compression** is more complex (DAG + L1/L2 summarizer). Defer until a concrete use case requires it (e.g., long CoT chains in riir-ai).

### Feature Gates

```toml
# katgpt-rs (MIT)
[features]
skill_memory = []     # Per-pruner memory (default-on)
skill_test_gate = []  # WASM test-gated registration (default-on)
skill_catalog = []    # Progressive disclosure (default-on)
```

```toml
# riir-ai (private)
# Per-game test cases for WASM test gates
# Per-game pruner memory tuning (edge-case thresholds, failure signatures)
# Domain-specific catalog entries
```

### GOAT Proof Plan

1. **Bench target:** Bomber arena, 1000 rounds × 3 phases (freeze → thaw → evolve)
2. **Metric:** Win rate with ITSE vs win rate without ITSE
3. **GOAT gate:** Win rate delta > 0 (any positive improvement passes)
4. **Stretch goal:** Win rate delta ≥ +5pp (comparable to MUSE's +15.21pp on easier tasks)

---

## 5. Related Research References

| Research | Connection | Overlap |
|----------|------------|---------|
| **R021 (G-Zero Self-Play)** | G-Zero discovers strategies via self-play. ITSE provides the lifecycle management that makes G-Zero's discoveries persistent and refinable. | G-Zero = skill creation; ITSE = skill lifecycle |
| **R032 (Percepta Distillation)** | HL infrastructure that ITSE builds on. `ConstraintPruner` trait, `BanditPruner`, `AbsorbCompress` all originate here. | 70% of ITSE's foundation is from R032's HL pipeline |
| **R034 (D2F / WASM)** | WASM validator infrastructure provides the sandbox for test gates. ITSE's `WasmTestGate` reuses this sandbox. | Test gate mechanism depends on WASM infra from Plan 034 |
| **R049 (G-Zero / PTRM)** | Recursive model that generates hints. ITSE's per-skill memory could store hints that worked, creating a feedback loop. | Hint storage ↔ pruner memory |
| **R092 (Freeze/Thaw)** | Cross-session persistence that ITSE enhances with per-pruner memory. `FrozenSkillBank` extends `FrozenBanditStats`. | ITSE is a direct extension of freeze/thaw |
| **R098 (PrudentBanker)** | Safe phased aggression for bandits. ITSE's test gates complement PrudentBanker's safety — gates prevent bad arms from entering, PrudentBanker limits damage from exploration. | Defense in depth: test gate (prevention) + safe bandit (containment) |
| **R105 (SkillOpt)** | Text-space skill optimization via LLM. ITSE is the modelless counterpart — skills evolve through bandit feedback instead of LLM edits. Complementary: SkillOpt for offline optimization, ITSE for online evolution. | SkillOpt = model-based refinement; ITSE = modelless refinement |
| **R111 (Data Gate / Emergent Analogical)** | Strict task-level filtering. ITSE's test gates are the skill-level analog — filter pruners by test results, not by data quality. | Data Gate filters inputs; ITSE gates filter skills |
| **R137 (Pplx Fast Viterbi)** | Fast unigram trie for routing. ITSE's catalog could use Pplx-style trie for fast skill name lookup in the progressive disclosure layer. | Trie-based skill catalog lookup |
| **R145 (SIA Harness + Weight Co-Evolution)** | Co-evolution of harness (prompts) and weights (LoRA). ITSE is the pure harness co-evolution — skills (prompts/validators) evolve without weight changes. | SIA = weight+harness; ITSE = harness-only (modelless) |
| **R163 (EoS Selective Learning)** | Curvature-based allocation of learning resources. ITSE's bandit feedback is a simpler form of selective learning — allocate exploration budget to skills with high uncertainty. | EoS = curvature signal; ITSE = bandit reward signal |
| **R194 (Adaptive CoT, planned)** | Bandit learns when to think. ITSE provides the skill lifecycle that makes adaptive CoT's "think" skills persistent and refinable across sessions. | Adaptive CoT = when to think; ITSE = how to persist+refine thinking skills |
| **R168 (Ruliology Competition)** | Exhaustive enumeration of simple programs as bandit arms. ITSE gives each enumerated program a full lifecycle — memory, test gate, freeze/thaw. The "universal winners" from ruliology become the most-promoted skills in ITSE. | Ruliology = discover skills; ITSE = manage discovered skills |

### The Unifying Thread

```
R021 (G-Zero)     → Discover strategies via self-play
R105 (SkillOpt)   → Optimize strategies offline via LLM edits
R168 (Ruliology)  → Enumerate ALL strategies exhaustively
R172 (ITSE)       → Lifecycle management for ALL of the above
R194 (Adaptive CoT) → Decide WHEN to use strategies
```

ITSE is the **glue layer** that makes every other skill discovery mechanism persistent, testable, and refinable. It's not a new discovery method — it's the infrastructure that makes discoveries durable.

---

## Open/Close Boundary

```
katgpt-rs (MIT, open)                    riir-ai (Private, closed)
─────────────────────────                ──────────────────────────
PrunerMemory struct + serialization      Per-game edge-case thresholds
PrunerTestGate trait                     Per-game WASM test cases
WasmTestGate sandbox wrapper             Domain-specific failure signatures
SkillCatalog struct + lazy loader        Catalog descriptions per game
FrozenSkillBank (extended repr(C))       Per-game memory tuning params
Feature gates: skill_memory,             Frozen memory bundles per game
  skill_test_gate, skill_catalog         

"plug socket"                            "plug"
```

---

## TL;DR

MUSE-Autoskill proves that skill lifecycle management (creation → memory → management → evaluation → refinement) delivers **+15.21pp accuracy** and skills that exceed human ceiling (87.94%). Our architecture already has ~75% of this via `ConstraintPruner` + `BanditPruner` + `AbsorbCompress` + `HotSwapPruner` + freeze/thaw + five-tier memory. The 25% gap is four concrete components:

1. **Per-pruner memory** (NEW) — accumulate edge cases across sessions via enhanced freeze/thaw
2. **WASM test gates** (NEW) — pruners must pass sandboxed tests before entering bandit bank
3. **Progressive disclosure catalog** (NEW) — inject name+desc, load full pruner on demand (saves 40-60% tokens)
4. **Adaptive context compression** (DEFERRED) — DAG-based L1/L2, feature-gated off until needed

The creative fusion ("Inference-Time Skill Evolution") treats every pruner as a skill with a unified modelless lifecycle. Unlike MUSE (383K token creation cost, amortizes after ~3 uses), our skills cost **0 tokens to create** — they already exist as pruners. The GOAT insight from MUSE's bomber arena: WASM validators WITHOUT lifecycle management HURT performance (-15), but WITH lifecycle management they should FLIP to positive. First 3 components are default-on; context compression is feature-gated off. GOAT proof: bomber arena win rate delta > 0 with ITSE vs without.
