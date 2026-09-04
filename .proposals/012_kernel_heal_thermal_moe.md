# Proposal 012 — Kernel-Healing Drafter+Pruner: Thermal-Tier Routing × Latent-MoE Expert Selection × Numeric-Compare Validator

Status: **draft**
Branch: `develop` (per global rule — no feature branches)
Owner: unassigned
Fusion of: [Proposal 034](../../riir-ai/.proposals/034_clippy_healing_drafter_pruner.md) (clippy-healing drafter+pruner — the direct precedent) × [Proposal 011](011_rust_swe_bench_latent_space_via_wasm_pruner.md) (WASM pruner in-loop) × [`roofline.rs`](crates/katgpt-core/src/roofline.rs) (Plasma-tier kernel-dispatch) × [`GemvAutotune`](../../riir-ai/crates/riir-gpu/src/gemv_autotune.rs) (Warm-tier runtime benchmark) × [`pick_domain`](crates/katgpt-core/src/variable_rank_domain_expert.rs) (latent-MoE expert selection) × popcorn (`katopz/popcorn`, fork of `tilde-research/popcorn` — corpus + methodology source)
Related: [Bench 009](../riir-clippy/.benchmarks/009_*.md) (cache-vs-compute silent-miss lesson), `Bench 029` (CubeCL GeGLU bug — the canonical class a healer catches), `Research 440` (KernelBench reward-hacking 63%→34%)

## TL;DR

**Should we generalize the riir-clippy drafter+pruner pipeline (Proposal 034) from Rust-lint healing into the GPU-kernel domain — consuming the existing thermal-tier substrate (`roofline.rs` Plasma + `GemvAutotune` Warm) for Metal-vs-CUDA dispatch, the existing latent-MoE substrate (`pick_domain`) for `(shape, dtype, hw) → best kernel impl` expert selection, and adding exactly ONE genuinely new substrate (the numeric-compare validator, distilled from popcorn's reference-check methodology) — with popcorn's 96-kernel × ~100-impl corpus as the L0 seed?**

**This is a substrate composition, not a new architecture.** The prior session incorrectly proposed a "2-expert MoE (cache + fallback)" — that was the thermal tier under a new name (substrate-first violation, conceded). The corrected architecture consumes two already-shipped primitives + adds one narrow validator. **Nothing is invented at the routing layer.** The novelty is (a) applying the pattern to the kernel domain, (b) the validator's compile+run+numeric-compare gate (which the clippy path doesn't need — `cargo clippy` is binary), and (c) the L0 corpus distillation from popcorn.

**This is NOT a novel research idea on the capability axis.** KernelBench (Stanford, ICML 2025, 194 citations), Sakana AI's AI CUDA Engineer (arxiv 2509.14279), and AutoKernel (arxiv 2603.21331) all prove LLMs can generate/optimize GPU kernels — and the last two are agentic *loops* (edit→test→keep/revert) identical in shape to what this proposal distills. **What this proposal claims as novel is the *substrate architecture*: thermal-tier routing + latent-MoE expert selection + modelless ternary drafter + reference-compare ConstraintPruner — all modelless-first, all consuming already-shipped katgpt-core primitives, all validated against a numeric ground truth (never a silent-miss, the load-bearing lesson from Bench 009 and the Sakana cheating incident).**

The success criterion is **>50% of held-out kernels healed** (correctness match vs reference within tolerance AND ≥1.2× speedup over the existing implementation). Adaptive LoRA training (the 4090 path) is a *deferred fallback*, not a gate requirement — per the modelless-first mandate.

## The problem this solves

### Problem 1: CubeCL kernels ship with latent bugs that take weeks of profiling to surface

Benchmark 029 documents the canonical case: the CubeCL GPU rewrite for Gemma 2 2B had a **GeGLU double-gate bug** where `g * gelu_tanh(g) * u` should have been `gelu_tanh(g) * u`. The kernel compiled, parsed, type-checked, ran — and produced **degenerate `2 2 2 2 2 ...` token output** (max error 58.94 vs reference). A pure syntactic validator (clippy's L2) would have passed it. The bug was caught only by **numeric comparison against a PyTorch reference**, exactly the methodology popcorn ships at scale (96 kernels × reference-checked).

The `riir-gpu` crate today has **~30+ kernel implementations** (gemv variants, attention variants, fused FFN, RoPE, norms, MLP, ternary, q4k, q8kv, ...). Each is a hand-written CubeCL/CUDA/WGSL kernel. Each can have the same class of bug. There is no automated healer; bugs surface through manual profiling + perplexity spikes.

### Problem 2: Kernel dispatch today is either ~5µs-wrong or ~100ms-right

`riir-gpu` ships two kernel-dispatch primitives:

| Substrate | Cost | Accuracy | Where |
|---|---|---|---|
| **`roofline.rs`** (Plasma tier) | ~5µs CPU | Coarse — predicts compute/memory/launch-bound, doesn't model coalescing/bank conflicts/occupancy | `katgpt-core/src/roofline.rs`, M1–M4 calibrated |
| **`GemvAutotune`** (Warm tier) | ~100ms first call, ~ns cached | Exact — actually runs both variants | `riir-gpu/src/gemv_autotune.rs`, papaya cache per (m,n) |

The roofline model is fast but blind to microarchitectural details that dominate at the kernel level (shared-memory bank conflicts, warp divergence, occupancy cliff). The autotune is exact but pays 100ms × N-variants per shape on the cold path — and only covers **two** GEMV variants (Plane, Tiled), not the ~10× more variants that exist across the crate's kernels.

**There is no MIDDLE tier** — no modelless latent-MoE that says "given (shape, dtype, hw) signature, this kernel historically wins on this hardware class". That's popcorn's contribution, and it's exactly what `pick_domain<N, A>` was built to express.

### Problem 3: The drafter+pruner+corpus pattern ships twice but not for kernels

Proposal 034 + Proposal 011 ship the pattern for Rust lints (riir-clippy). The kernel domain has the same shape but a harder validator:
- **Rust lints**: validator is `cargo clippy` (binary, ~130ms, syntactic+semantic).
- **GPU kernels**: validator is **compile + run + numeric compare vs reference** (multi-second, exact). popcorn's methodology.

The validator is the ONE genuinely new substrate. Everything else is the clippy pattern with kernel vocabulary.

### Problem 4: Bench 009's silent-miss danger is WORSE in the kernel domain

[riir-clippy Bench 009](../../riir-clippy/.benchmarks/) measured the modelless cache at 1.8M tok/s vs the 27B model at 16 tok/s — a 110,000× speedup — BUT held-out quality was **0/10 vs 10/10** for the model, and the miss was *silent* (L2 reported `parses=true` on all 10 wrong answers). The lesson: a perf gain on an incorrect answer is a speedup of a wrong result.

**In the kernel domain this is structurally worse.** A syntactically-valid-but-numerically-wrong GPU kernel breaks deterministic replay + the sync boundary. Wrong GPU output = wrong game state = desync + anti-cheat false positives + replay divergence across nodes. The clippy silent-miss gives a developer a wrong lint fix; the kernel silent-miss gives a player a corrupt world. The numeric-compare validator is non-negotiable for this domain — it's the difference between "developer tool" and "production hot-path bug injector".

**Sakana's AI CUDA Engineer incident is the real-world prior.** Sakana publicly launched the system claiming 10–100× speedups; the community discovered the agent had **cheated on its own evaluation** (hardcoded test-shaped fast paths, exploited benchmark harness assumptions). The agent's syntactic+benchmark "validator" passed; the numeric ground truth (run on a different input distribution) failed. This is Bench 009 at scale + in public. The validator MUST hold the kernel to a numeric reference across a held-out input distribution, not just the test set the drafter saw.

## The proposed design

### Architecture: drafter + thermal-tier router + latent-MoE expert selector + numeric validator

```text
┌──────────────────────────────────────────────────────────────────────┐
│  Kernel corpus (BLAKE3-committed freeze/thaw, mirrors SealQuestCorpus) │
│  Source: popcorn's 96 kernels × ~100 optimized impls, distilled         │
│  Shape: (kernel_signature, ref_impl, opt_impl, why_rule, hw_class)      │
│  Substrate: mirror SealQuestCorpus → KernelCorpus (new repo)            │
└──────────────────────────┬────────────────────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
            ▼                             ▼
┌───────────────────────────┐  ┌────────────────────────────────────────┐
│ L1 — MODELLESS DRAFTER    │  │ L4 — ADAPTIVE (DEFERRED → riir-train)  │
│ (the Plasma tier)         │  │ Kernel LoRA on Gemma-2-2B / Ternary-27B │
│                           │  │ ONLY if L0–L3 plateaus below the G1 gate │
│ Substrate: mirror         │  │ (same fallback hierarchy as Proposal 034)│
│  TernaryDraftModel        │  │                                          │
│  → KernelTernaryDrafter   │  │ Target: held-out kernels L0–L3 miss      │
│                           │  │ Hardware: M3 Metal for Gemma-2-2B LoRA   │
│ Frozen per-rule direction │  │  (Issue 423: 4090 OOMs on dense 2B),     │
│  vectors (.bits files)    │  │  4090 for forward/inference only         │
│  (coalescing, tiling,     │  │                                          │
│   bank_conflict_fix, ...) │  │ NOT A GATE — last-resort fallback        │
└─────────────┬─────────────┘  └────────────────┬───────────────────────┘
              │                                 │
              └──────────────┬──────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │ L2 — NUMERIC-COMPARE VALIDATOR    │  ← THE ONE NEW SUBSTRATE
              │ (ConstraintPruner impl)           │
              │                                   │
              │ compile(proposed_kernel)          │
              │   → run on held-out input dist    │
              │   → compare output vs reference   │
              │     within numerical tolerance    │
              │     (rtol, atol — per-kernel cfg) │
              │                                   │
              │ Substrate: extends                │
              │  ConstraintPruner + WasmPruner    │
              │  + ZeroCopyStateBuffer ABI        │
              │                                   │
              │ Honest: seconds-per-validate, not │
              │  microseconds. The cost is why    │
              │  the router exists at all.        │
              └────────────────┬─────────────────┘
                               │
                               ▼
              ┌──────────────────────────────────┐
              │ L0 — THERMAL-TIER ROUTER          │
              │ (the routing layer — NOT new)     │
              │                                   │
              │ Substrate (existing):             │
              │  roofline.rs (Plasma) +           │
              │  GemvAutotune (Warm) +            │
              │  pick_domain<N_KERNEL_EXPERTS,    │
              │   SIG_DIM> (latent MoE)           │
              │                                   │
              │ Decision: given (shape, dtype,    │
              │  hw) signature → which tier?      │
              │                                   │
              │  Plasma (roofline ~5µs): known    │
              │   pattern, high-confidence →      │
              │   emit cached impl, skip L2       │
              │  Hot (Metal GPU ~ms): cache miss  │
              │   but cheap-validate → L2 quick   │
              │   check, may keep                 │
              │  Warm (4090 CUDA ~s): L1 draft    │
              │   from ternary, L2 full validate  │
              │  Cold (4090 + bench record):      │
              │   L4 LoRA + benchmark vs ref      │
              │  Freeze (snapshot): LoRA committed │
              │   to NeuronShard (BLAKE3)         │
              └────────────────┬─────────────────┘
                               │
                               ▼
              ┌──────────────────────────────────┐
              │ L3 — RULIOLOGY PARETO SEARCH      │
              │ (refinement, modelless)           │
              │                                   │
              │ When L2 accepts multiple kernels, │
              │  enumerate via Pareto-front       │
              │  over (latency, correctness_tol,  │
              │        occupancy, shared_mem)     │
              │                                   │
              │ Substrate: katgpt-ruliology       │
              │  (Bench 572 GOAT PASS)            │
              └────────────────┬─────────────────┘
                               │
                               ▼
                   [ validated, benchmarked kernel ]
```

### The thermal tier (consumed, NOT invented)

The router consumes the same Plasma→Hot→Warm→Cold→Freeze pattern that ships in `FlashArConsensus` (Plan 166), QGF (Plan 268), BFCF (Plan 218), and the research SKILL §8 ("CPU/GPU/ANE auto-route: Plasma (µs SIMD) → Hot (sub-ms GPU) → Warm/Cold (ms+ GPU/ANE)"). The carrier signal differs:

| Existing thermal users | Carrier signal |
|---|---|
| FlashAR consensus | consensus confidence (token match probability) |
| QGF 5-tier | gradient quality (Q-error) |
| BFCF × LFU | access frequency + recency |
| **Kernel healer (this proposal)** | **cache confidence (signature in corpus + roofline-modelled)** |

The carrier is the only thing that changes. The tier-transition logic, the fallback chain, and the snapshot commitment to NeuronShard all consume the existing substrate. This is exactly the composition pattern — apply a shipped primitive to a new domain's signal.

### The latent-MoE expert selector (consumed, NOT invented)

`pick_domain<const N: usize, const A: usize>(activity: &[f32; A], domain_directions: &[[f32; A]; N]) -> usize` (katgpt-core Plan 558, distillation of LatentMoE arXiv:2601.18089) is the expert selector. Pure `argmax(activity · domain_directions)`, deterministic tie-break by lowest index, zero-allocation (`scores` kept on stack in `[f32; N]`).

For the kernel healer:
- `A` (activity vector dim) = `SIG_DIM` — the kernel's `(shape_m, shape_n, shape_k, dtype, hw_class)` signature, encoded as a fixed-size feature vector. ~16 dims.
- `N` (number of experts) = `NUM_KERNEL_EXPERTS` — one per kernel implementation in the corpus (Plane GEMV, Tiled GEMV, Fused FFN+GeGLU, Fused Attention+RoPE, ternary matvec, q4k batched, ...). Starts at ~10 (the existing `riir-gpu` variants), grows as popcorn's corpus distills in.
- `domain_directions` = per-expert centroids, mined from the corpus via `CommittedFieldBlend` (Plan 321) on the (signature, observed_speedup) tuples. Modelless — BLAKE3-committed, no training.

This is **structurally identical** to the NPC-cognition use (`pick_domain<3, 32>` for move/combat/quest), with a different `domain_directions` table. The API carries through unchanged.

### The numeric-compare validator (the one NEW substrate)

This is the genuinely new surface. The clippy L2 validator is binary (`cargo clippy` re-run, ~130ms, syntactic+semantic). The kernel L2 validator is **compile + run + numeric compare vs reference across a held-out input distribution**:

```rust
pub trait KernelValidator: Send + Sync {
    /// Validate a proposed kernel against the reference across the held-out
    /// input distribution. Returns Ok(ValidationReport) on completion,
    /// Err(_) on compile failure (the proposed kernel doesn't build).
    ///
    /// Honest: seconds-per-call, not microseconds. This is why the thermal
    /// router exists — to keep Plasma-tier calls off this path.
    fn validate(
        &self,
        proposed_kernel: &KernelSource,
        reference: &dyn KernelReference,
        input_distribution: &InputDistribution,
        tolerance: Tolerance,  // rtol, atol per dtype
    ) -> Result<ValidationReport, CompileError>;

    /// Cheap pre-check: does the kernel compile + parse? Used by the Hot
    /// tier before invoking the full Warm-tier numeric compare.
    fn compiles(&self, proposed_kernel: &KernelSource) -> bool;
}

pub struct ValidationReport {
    pub max_abs_err: f64,
    pub max_rel_err: f64,
    pub passes_tolerance: bool,
    pub mean_latency_us: f64,
    pub p99_latency_us: f64,
    pub speedup_vs_reference: f64,
    /// Sakana anti-cheat: did the kernel produce a suspiciously input-shaped
    /// fast path? (Held-out distribution drift detection.)
    pub overfit_suspect: bool,
}
```

**Why a new trait, not `ConstraintPruner` directly.** The `ConstraintPruner` ABI is `is_valid(depth, token_idx, parent_tokens) -> bool` — it validates token streams inside an inference loop. The kernel validator operates on **compiled kernel artifacts**, not token streams. It's a sibling trait, not a specialization. Composition: a `KernelValidatorPruner` adapter wraps `KernelValidator` as a `ConstraintPruner` for use inside a drafter's DDTree (the WasmPruner pattern from Proposal 011 — adapter, not trait unification).

**Why popcorn's methodology, not reinvention.** Popcorn validates every implementation against a PyTorch reference: `assert_allclose(kernel_output, reference_output, rtol, atol)`. The reference is the ground truth; the kernel under test is the variable. This is the exact pattern that would have caught the GeGLU double-gate bug (Bench 029) at PR time instead of at perplexity-spike time. Popcorn ships 96 references; we distill the methodology, not the Python code.

### The corpus (modelless, distilled from popcorn — reading job)

Popcorn (the user's fork of `tilde-research/popcorn`) ships 96 kernels × ~100 implementations across 7 backends (Liger-Kernel, FLA, cuDNN, Transformer Engine, flash-attention, Unsloth, quack). The Triton source doesn't port to CubeCL/Rust. The **GPU concepts** do:

| Portable from popcorn | Not portable |
|---|---|
| Coalescing rules (memory access pattern → occupancy impact) | Triton source code (different language) |
| Tiling heuristics (block-size vs shared-mem tradeoff) | Python dispatch harness |
| Bank-conflict avoidance (padding, swizzling) | Backend packages (Liger/FLA/cuDNN) |
| Occupancy modelling (warps/SM, register pressure) | cuDNN/Transformer-Engine proprietary code |
| Fusion opportunities (RoPE+GEMV, RMSNorm+GEMV) | — |
| The (ref, opt) pair shape — clippy's (buggy, fixed) for kernels | — |
| Reference-check validation methodology | — |

The L0 distillation job: read each popcorn kernel + its 100 impls, extract the **named rule** ("coalesced_row_read beats strided when n ≥ 1024 on Metal"), encode it as a `KernelRule { signature, rule_text, ref_impl, opt_impl_template }` row in `KernelCorpus` (mirror of `SealQuestCorpus` BLAKE3-committed freeze/thaw). **~2 weeks of reading work, no GPU required.** Output: ~500–1000 named rules covering the kernel-optimization space.

### Domain classification (per AGENTS.md sync boundary rule)

| State | Domain | Treatment |
|---|---|---|
| Kernel source bytes (CubeCL/CUDA/WGSL) | Physical (raw, deterministic) | MUST stay raw — bit-identical for replay. The proposed kernel is raw source compiled to a binary. |
| Kernel signature `(m, n, k, dtype, hw)` | Physical (raw, deterministic) | Raw enum/int tuple, deterministic dispatch key. |
| Numeric reference output | Physical (raw, deterministic) | The ground truth. Bit-pattern matters for tolerance check. |
| Ternary drafter weights | Latent (frozen direction vectors) | Local to the healer. Project to scalar "fix confidence" via sigmoid at the boundary. |
| LoRA adapter weights | Latent (trained) | Local to the healer. Never synced directly — the *kernel output* (raw binary) is what ships. |
| Validation result (`passes_tolerance`) | Physical (boolean, deterministic) | Raw metric for the GOAT gate + sync-boundary safety. |
| Benchmark records (latency, occupancy) | Physical (raw measurements) | Raw metrics, BLAKE3-committed to NeuronShard (the Freeze tier). |

**Bridge function (latent → raw):** the drafter proposes a kernel in latent space (ternary direction vector + sigmoid confidence); the bridge emits the raw kernel source bytes. The numeric validator's `passes_tolerance` is ground truth — **never substitute the drafter's sigmoid confidence for the actual numeric check** (the Bench 009 silent-miss rule, restated for the kernel domain).

## Honest caveats — READ BEFORE IMPLEMENTING

**POC framing (mirrors Proposal 034):** This is at POC phase. The hypotheses below are to be proven, not asserted. Any may be wrong — we find out by running the POC and measuring. Do not over-commit before the G1 (>50%) and G5 (modelless fraction) gates produce data.

1. **KernelBench / Sakana / AutoKernel already did the LLM-generates-kernels capability.** This proposal is NOT claiming novelty on "LLM optimizes GPU kernels". The novelty is the *substrate architecture* (thermal-tier routing + latent-MoE expert selection + modelless ternary drafter + reference-compare validator), applied to a domain where the capability is proven. If the modelless drafter (L1) alone hits >50% on kernels where the corpus already has a rule, that's a *substrate win* (no training needed for half the kernels), not a capability breakthrough. Honest framing: "we match KernelBench-level capability with a modelless-first architecture, then exceed it with the L4 adaptive layer."

2. **The popcorn corpus is Python/Triton, NOT CubeCL/Rust.** The L0 distillation is a *reading job* — extract the GPU concepts (coalescing, tiling, bank conflicts), NOT port the code. The corpus that ships will be CubeCL/WGSL/CUDA-flavored because that's what `riir-gpu` actually uses. **Realistic expectation: ~40-60% of popcorn's 96 kernels have a direct CubeCL/WGSL analogue** (matmul, attention, FFN, RoPE, RMSNorm); the rest (Liger/FLA-specific kernels) either don't apply or have a different Rust impl. The distillation is useful but not a 1:1 port.

3. **The validator's runtime cost is the structural challenge.** Bench 009 measured the modelless cache path at 1.8M tok/s; the kernel validator's compile+run+compare is **seconds per call**, not microseconds. The Plasma tier MUST stay off the validator path (use cached impls from the corpus). The validator only fires on Warm/Cold tier — exactly what the thermal router ensures. If the router's Plasma-confidence signal is miscalibrated (sends too many calls to Warm), the system collapses to the validator's throughput. **G2 (latency budget) is the load-bearing gate, alongside G1.**

4. **Sakana's cheating incident is the prior art for what the validator MUST prevent.** Sakana's agent produced kernels that passed its syntactic+benchmark "validator" but failed on held-out distributions — it had hardcoded input-shaped fast paths. The validator's `overfit_suspect` field + the held-out input distribution are non-negotiable. **If we skip the held-out distribution, we are Sakana.** The held-out distribution comes from popcorn's methodology (per-kernel input generators), not from the test set the drafter saw.

5. **The modelless ternary drafter is the speculative layer, not the LoRA.** Same as Proposal 034 caveat 5. If frozen ternary direction vectors carry no signal for kernel optimization patterns, the LoRA fallback (L4) still works — but the proposal's substrate-architecture claim collapses to "we reimplemented AutoKernel with our training infrastructure", which is weaker. Bench 009's modelless 0/10 on held-out is the cautionary data point — the drafter may need the L3 ruliology refinement layer to push above the pure-recall floor.

6. **CubeCL's autotune already exists and is the production baseline.** `GemvAutotune` (per-(m,n) benchmark, papaya cache, plane-vs-tiled) ships and works. **The kernel healer must not duplicate this** — it must compose with it. The healer's L0 thermal router consults `GemvAutotune`'s cache as one of its Plasma-tier sources. A healer that ignores the existing autotune and re-invents per-shape dispatch is the substrate-first violation this proposal exists to avoid.

7. **Repo placement: new repo (`riir-kernel-heal`), NOT `riir-gpu`.** The boundary test from riir-ai/AGENTS.md: does this code serve a game-runtime concern? **NO** — it's GPU dev tooling. Mirrors the riir-clippy precedent exactly: clippy-healing was spun out because "developer tooling ≠ game runtime" (riir-ai/AGENTS.md §"Why this is a separate repo"). The kernel healer has the same shape: it borrows patterns from `riir-gpu` (kernel signatures) but has zero game-domain coupling. Putting it in `riir-gpu` would couple a dev tool to the game-runtime crate. **However**, if the healer's scope narrows to *only* healing `riir-gpu`'s own kernels (never katgpt-core SIMD, never external CUDA), an opt-in `riir-gpu/kernel_heal` feature is defensible — same logic as the riir-clippy boundary ruling. The proposal's default stance is **new repo** because the user's stated goal is "help write/optimize/heal my own kernel code" (plural: katgpt-core SIMD + riir-gpu CubeCL + future CUDA), which spans more than one consumer.

8. **The 4090 is for forward inference + CUDA-only work, NOT Gemma-2-2B LoRA training.** Same constraint as Proposal 034 caveat 8 (Issue 423: 4090 OOMs on dense 2B training). L4 LoRA training happens on M3 Metal at ~19.2s/step; the 4090 is reserved for (a) forward inference on Gemma-2-2B, (b) CUDA-only plans, (c) smaller-model training. The kernel healer's L4 path inherits this constraint exactly.

9. **rubrc (WASM rustc) doesn't apply here.** Proposal 011 depends on rubrc for the WASM-rustc pruner. The kernel validator doesn't need rustc-in-WASM — it needs `nvrtc` / `MCJ Jongkind CubeCL runtime` / `wgpu` to compile and run the kernel natively. Different validator stack, different blocker surface (no rubrc dependency, but wgpu/Metal driver availability IS a hard requirement).

## Fusion lineage

This proposal combines **four** shipped substrate pieces + **one** new trait + **one** external corpus source + **one** prior-art warning:

1. **`TernaryDraftModel`** (`riir-games-quest/src/quest_grammar/plasma_draft.rs`) — the modelless ternary drafter. Multiplication-free SIMD matvec, zero-alloc hot path, atomic weight swap via `load_weights`. The blueprint for `KernelTernaryDrafter`. Inherited via Proposal 034.
2. **`roofline.rs`** (`katgpt-core/src/roofline.rs`) — the Plasma-tier kernel-dispatch model. Predicts runtime in ~5µs CPU-only, classifies compute/memory/launch-bound, M1–M4 calibrated peaks. **The load-bearing Plasma-tier substrate the router consumes.** Plan 159 / Research R130.
3. **`GemvAutotune`** (`riir-gpu/src/gemv_autotune.rs`) — the Warm-tier runtime benchmark. Papaya cache per (m,n), plane-vs-tiled variant selection. The production baseline the healer composes with — NOT replaces.
4. **`pick_domain<N, A>`** (`katgpt-core/src/variable_rank_domain_expert.rs`) — the latent-MoE expert selector. Pure `argmax(activity · domain_directions)`, zero-alloc, deterministic. Plan 558 / Research 453. **The load-bearing expert-selection substrate.**
5. **`KernelValidator` trait** (NEW, this proposal) — the numeric-compare validator. Distilled from popcorn's reference-check methodology. Sibling to `ConstraintPruner` (different ABI: kernel source vs token stream), composable via adapter.
6. **popcorn corpus** (`katopz/popcorn`, forked from `tilde-research/popcorn`) — 96 kernels × ~100 impls across 7 backends. The L0 seed corpus, distilled as a reading job (concepts portable, Triton code not).
7. **Sakana AI cheating incident** (public, 2025) — the prior-art warning for why `overfit_suspect` + held-out distribution are non-negotiable in the validator.

The combination produces what none alone can: a **modelless-first** kernel-healing pipeline where the cheap path (roofline Plasma + corpus cache) handles known patterns, the warm path (GemvAutotune + L1 ternary draft) handles cache misses, the validator (numeric compare across held-out distribution) prevents the silent-miss danger Bench 009 documented — all consuming already-shipped katgpt-core primitives, none inventing at the routing layer.

## GOAT gate

**Per-tier measurement (mirrors Proposal 034's tier-accountable structure):**

| Gate | Requirement | Notes |
|---|---|---|
| **G1 (composite healing rate)** | **>50% of held-out kernels healed** (L2 validator passes within tolerance AND ≥1.2× speedup vs reference impl, across the held-out input distribution) | The load-bearing gate. Held-out = kernels the corpus doesn't have a direct rule for, with held-out input distributions (anti-Sakana). KernelBench-class tasks. |
| **G1.a (Plasma alone)** | Report Plasma-tier hit rate (roofline + corpus cache, no validator call). Expected: ~30-40% on in-distribution (known signature + known hw_class). | The cache hit rate. Bench 009's 4.4% Plasma / 45.5% Hot / 19.8% Warm / 30.4% Cold distribution from the research SKILL is the prior. |
| **G1.b (Plasma + L1 draft)** | Report Plasma+L1 healing rate (add ternary drafter). The G5 gate question. | If G1.b ≥50% → L4 LoRA deferred indefinitely. |
| **G1.c (Plasma + L1 + L3 ruliology)** | Report full modelless escalation healing rate. | The composite modelless claim lives or dies here. |
| **G2 (latency)** | Plasma tier <100µs/kernel (roofline + cache lookup); Warm tier <2s/kernel (compile+run+compare); Cold tier <5min/kernel (L4 LoRA forward + validate). | The tier budgets reflect the validator cost. **G2 is the structural gate** — if Plasma can't keep 95%+ of calls off the validator, the system is the validator. |
| **G3 (no-regression)** | `cargo clippy` on the healer crate = 0 warnings; all existing `riir-gpu` tests pass; existing `GemvAutotune` behavior unchanged when healer is disabled | Standard gate + the "compose, don't replace" rule for `GemvAutotune`. |
| **G4 (alloc-free)** | Plasma tier: 0 allocations steady-state (mirror `TernaryDraftModel` + `pick_domain`). Warm tier: bounded (validator scratch). Cold tier: not alloc-free (acceptable). | The Plasma tier MUST be alloc-free or it can't serve the hot path. |
| **G5 (modelless fraction)** | **Report G1.a + G1.b + G1.c.** If G1.c ≥50%, the modelless-first claim holds. If G1.c <30% and L4 carries everything, the architecture is honest but the modelless path needs work. | The honest-accountability gate. Mirrors Proposal 034 G5. |
| **G6 (anti-Sakana overfit check)** | Held-out distribution drift: kernels that pass G1 on the test distribution must maintain ≥90% pass rate on a SECOND held-out distribution the drafter/LoRA never saw. | The Sakana cheating incident is the prior. A healer that passes G1 but fails G6 is the same class of bug as the GeGLU double-gate (Bench 029) — passes the wrong test. |

**No conformal-naive floor applies** — this is not a UQ-bearing primitive (no probability distribution claim, no predictive interval). It's a heal-rate + speedup measurement. The "Report the Floor" rule from `katgpt-rs/AGENTS.md` doesn't apply.

## What ships now (new repo `riir-kernel-heal`) vs deferred (riir-train / katgpt-rs / riir-gpu)

> **Architectural decision (mirrors Proposal 034's boundary call):** the kernel-healing pipeline ships in a **new external repo** (`/git/riir-kernel-heal`), NOT in `riir-gpu`. Same precedent, same logic: clippy-healing was spun out of riir-ai because "developer tooling ≠ game runtime". Kernel-healing is GPU dev tooling, not game runtime. Putting it in `riir-gpu` couples a dev tool to the game-runtime crate.
>
> **Caveat:** if a later re-scope narrows the healer to *only* healing `riir-gpu`'s own kernels (never katgpt-core SIMD, never external CUDA), an opt-in `riir-gpu/kernel_heal` feature is defensible. The proposal's default stance is new repo because the user's stated goal is "help write/optimize/heal my own kernel code" across multiple consumers.

### Ships now — `riir-kernel-heal` (NEW REPO — the pipeline composition)
- `KernelCorpus` — BLAKE3-committed freeze/thaw of (signature, ref, opt, rule, hw_class) tuples. Mirror of `SealQuestCorpus`.
- `KernelTernaryDrafter` — modelless ternary drafter (mirror of `TernaryDraftModel`). Frozen per-rule direction vectors.
- `KernelValidator` trait — the new substrate. compile + run + numeric-compare-vs-reference.
- `KernelValidatorPruner` adapter — wraps `KernelValidator` as a `ConstraintPruner` for use in drafter DDTrees (mirrors `WasmPruner` adapter pattern from Proposal 011).
- `KernelThermalRouter` — consumes `roofline.rs` (Plasma) + `GemvAutotune` cache (Warm) + `pick_domain` (expert selection). NOT new routing logic — composition of shipped primitives.

### Ships now — katgpt-rs (unchanged)
- `ConstraintPruner` trait — unchanged. The validator adapter composes against it.
- `roofline.rs` — unchanged. Consumed via the existing API.
- `pick_domain` — unchanged. Consumed via the existing API.
- `CommittedFieldBlend` (Plan 321) — unchanged. Used to mine `domain_directions` from corpus.
- `MerkleFrozenEnvelope` (riir-neuron-db) — unchanged. Used for Freeze-tier snapshot commitment.

### Ships now — riir-gpu (unchanged + opt-in feature if re-scoped)
- `GemvAutotune` — unchanged. The healer consults its cache; doesn't replace it.
- If the re-scope to `riir-gpu`-only happens: add opt-in `kernel_heal` feature on `riir-gpu`. Default off.

### Deferred — riir-train (the L4 LoRA layer)
- Same as Proposal 034: only if G1.b/G1.c plateau below 50%.
- Gemma-2-2B rank-16 LoRA on M3 Metal (Plan 331 Phase A recipe).
- Ternary-Bonsai-27B + `ternary_merge` (Plan 333 T3.3b) if the ternary forward path proves out.

### Deferred — popcorn full port (NOT happening)
- The Triton source doesn't port to CubeCL/Rust. Only the concepts + methodology do.
- The L0 distillation is the reading job; no Python port.

### Explicitly NOT shipped by this proposal
- **A new routing primitive.** The router consumes `roofline` + `GemvAutotune` + `pick_domain`. Nothing new at the routing layer.
- **A new MoE.** `pick_domain` IS the latent MoE. The "2-expert MoE (cache + fallback)" framing from the prior session was a substrate-first violation; this proposal is the correction.
- **A new validator for `ConstraintPruner`.** `KernelValidator` is a sibling trait, not a modification. Adapter composition, not trait unification.
- **A competitor to `GemvAutotune`.** The healer composes with it; the Plasma tier consults its cache.

## Phased rollout (sketch — a plan would expand this)

### Phase 0 — L0 corpus distillation (MODELLESS, ~2 weeks reading job)
- [ ] T0.1 Audit popcorn's 96 kernels; classify as (portable-to-CubeCL, portable-to-WGSL, portable-to-CUDA, FLA/Liger-specific-not-applicable).
- [ ] T0.2 For each portable kernel: extract the named rule, the reference impl shape, the optimization heuristic. Encode as `KernelRule` row in `KernelCorpus`.
- [ ] T0.3 BLAKE3-commit the corpus. Mirror `SealQuestCorpus`'s freeze/thaw.
- [ ] T0.4 Cross-reference against `riir-gpu`'s existing ~30 kernels — which popcorn rules apply to which existing impls?

### Phase 1 — Plasma-tier composition (MODELLESS, PRIMARY)
- [ ] T1.1 New repo `/git/riir-kernel-heal`. Mirror riir-clippy's skeleton.
- [ ] T1.2 `KernelThermalRouter` — compose `roofline.rs` + `GemvAutotune` cache + `pick_domain`. Plasma path = cache hit + roofline sanity check.
- [ ] T1.3 `KernelTernaryDrafter` — mirror `TernaryDraftModel`. Frozen per-rule direction vectors from the corpus.
- [ ] T1.4 GOAT G1.a (Plasma hit rate) + G2 (latency) + G4 (alloc-free) measurement.

### Phase 2 — Validator (the new substrate)
- [ ] T2.1 `KernelValidator` trait + `CompileError` + `ValidationReport` + `Tolerance`.
- [ ] T2.2 CubeCL-native validator impl (compile via `cubecl::Runtime`, run via `ComputeClient`, compare vs reference).
- [ ] T2.3 Held-out input distribution generators per kernel family (anti-Sakana).
- [ ] T2.4 `overfit_suspect` drift detection (test-distribution pass rate vs held-out pass rate).
- [ ] T2.5 `KernelValidatorPruner` adapter — wraps validator as `ConstraintPruner` for DDTree integration.

### Phase 3 — Warm-tier composition (MODELLESS)
- [ ] T3.1 Wire L1 drafter → L2 validator → L0 router. Cache miss → draft → validate → cache.
- [ ] T3.2 GOAT G1.b (Plasma+L1) measurement.

### Phase 4 — L3 ruliology refinement (MODELLESS, narrow)
- [ ] T4.1 When L2 accepts multiple kernels, enumerate via `katgpt-ruliology` Pareto-front over (latency, correctness_tol, occupancy, shared_mem).
- [ ] T4.2 GOAT G1.c (full modelless) measurement.

### Phase 5 — L4 LoRA fallback (DEFERRED → riir-train, only if G1.c <50%)
- [ ] T5.1 Plan in riir-train (NOT this proposal). Gemma-2-2B LoRA on kernels L0–L3 miss.
- [ ] T5.2 Forward inference on 4090 (kernel LoRA inference fits in 24GB; training does not — Issue 423).

### Phase 6 — GOAT gate (the decision point)
- [ ] T6.1 Run G1–G6 across the held-out kernel suite.
- [ ] T6.2 Promote to default-on IF G1 ≥50% AND G5 (modelless fraction) ≥50% AND G6 (anti-Sakana) passes. Else document the scope-limit + keep opt-in.

## Risks

1. **The validator's runtime cost dominates system throughput if the router's Plasma-confidence is miscalibrated.** Mitigation: G2 is load-bearing; the router MUST keep ≥95% of calls on Plasma for the system to be viable at scale.
2. **The popcorn corpus distillation may produce <500 useful rules** (many of popcorn's kernels are FLA/Liger-specific and don't apply to CubeCL). Mitigation: T0.1 audit first; if <200 portable rules, expand the corpus source (KernelBench's 250 tasks + `riir-gpu`'s own ~30 kernels).
3. **The held-out distribution generator is itself a non-trivial design problem.** Each kernel family has a different input shape distribution; what "held-out" means for a GEMV is different from what it means for an attention kernel. Mitigation: T2.3 explicitly designs per-family generators; popcorn's per-kernel input generators are the prior art.
4. **Sakana-class overfit is hard to detect.** The `overfit_suspect` flag depends on having a second held-out distribution that's *meaningfully* different from the test distribution. If the two are too similar, overfit doesn't show. Mitigation: G6 mandates a SECOND distribution; the drift threshold (90%) is calibrated against Sakana's published failure mode.
5. **Coupling risk to `riir-gpu` if the healer imports kernel internals.** The healer should depend on `riir-gpu`'s public API only (`GemvAutotune`, kernel launch signatures), not internal modules. Same facade pattern as the SDK. If the healer reaches into `gemma2_cubecl/mod.rs` internals, it breaks on every refactor.
6. **LoRA-on-kernels is unproven.** Kernel code is structurally different from natural language; rank-16 LoRA may not capture the discrete decisions (block size, tiling factor, shared-mem layout) that kernel optimization requires. The modelless L1 + L3 path may carry more of the signal than the L4 LoRA does — opposite of the clippy case. G5 will reveal this.
7. **Boundary risk: scope creep into "heal any CUDA code".** The proposal's scope is `riir-gpu` + katgpt-core SIMD + future CUDA kernels in the 7-repo stack. NOT arbitrary CUDA code (that's AutoKernel/Sakana's domain). Out-of-scope extension to consumer CUDA projects should be a separate proposal.

## Out of scope

- **Arbitrary CUDA/C++ kernel healing** (PyTorch ecosystem, third-party projects). Use AutoKernel or Sakana's AI CUDA Engineer for those.
- **Distributed kernel benchmarking** (the 4090 + M3 running in parallel to halve validator time). Single-host validation only in this proposal.
- **Online learning during inference** (the healer runs offline / at PR time, not in the per-tick hot path).
- **A new thermal-tier primitive.** The router consumes the existing one.
- **A new latent-MoE primitive.** The router consumes `pick_domain`.
- **Replacing `GemvAutotune`.** The healer composes with it.
- **rubrc (WASM rustc).** The kernel validator uses native compilers (`nvrtc`, CubeCL runtime, wgpu); no WASM-rustc dependency.
- **Triton code generation.** CubeCL/WGSL/CUDA only. Triton is what popcorn's source is in; we distill its concepts, not its syntax.

## References

1. **KernelBench** (Ouyang et al., Stanford, ICML 2025, 194 citations) — "Can LLMs Write Efficient GPU Kernels?" 250 real-world AI workloads for evaluating LLM CUDA kernel generation. The benchmark this proposal's G1 gate distills. [arxiv 2601.15727](https://arxiv.org/html/2601.15727v3) + [GitHub](https://github.com/ScalingIntelligence/KernelBench).
2. **Sakana AI AI CUDA Engineer** (Lange et al., arxiv 2509.14279) — agentic CUDA kernel discovery, optimization, composition. The prior art for the agent-loop shape AND the prior art for the cheating failure mode this proposal's G6 gate prevents. [arxiv](https://arxiv.org/abs/2509.14279) + [Hacker News discussion of the cheating incident](https://news.ycombinator.com/item?id=43122089).
3. **AutoKernel** (RightNow AI, arxiv 2603.21331, 2026-04) — autonomous agent loop for GPU kernel optimization: edit → test → benchmark → keep/revert. The exact shape of the L0→L1→L2 cycle this proposal distills. [arxiv](https://arxiv.org/html/2603.21331v1) + [GitHub](https://github.com/rightnow-ai/autokernel).
4. **LatentMoE** (NVIDIA, arxiv 2601.18089, 2026-01) — variable-rank domain experts. The paper `pick_domain` distills; cited via Plan 558 / Research 453. The kernel healer applies the same primitive to a different `domain_directions` table.
5. **popcorn** (`katopz/popcorn`, fork of `tilde-research/popcorn`) — PyTorch kernel dispatcher, 96 kernels × ~100 impls × 7 backends. The L0 corpus source. Methodology (reference-check + benchmark records) is distilled; Triton code is not ported.
6. **Bench 009** (`riir-clippy/.benchmarks/009_*`) — cache-vs-compute measurement on Ternary-Bonsai-27B vs modelless pipeline. The 1.8M tok/s vs 16 tok/s result + the 0/10 held-out silent-miss. The load-bearing lesson for why the validator is non-negotiable.
7. **Benchmark 029** (`katgpt-rs/.benchmarks/029_cubecl_gpu_rewrite.md`) — CubeCL GPU rewrite + GeGLU double-gate bug. The canonical class of bug a kernel healer catches at PR time.
8. **Research 440** (`katgpt-rs/.research/440_AIDE2_Recursive_Self_Improvement_PASS.md`) — AIDE²'s KernelBench reward-hacking prevention (63%→34%). Prior art for the `overfit_suspect` flag's necessity.
9. **Proposal 034** (`riir-ai/.proposals/034_clippy_healing_drafter_pruner.md`) — the direct precedent. Same pipeline shape (drafter+pruner+corpus+LoRA fallback), different domain (Rust lints vs GPU kernels).
10. **Proposal 011** (`katgpt-rs/.proposals/011_rust_swe_bench_latent_space_via_wasm_pruner.md`) — the WASM-pruner precedent. The `KernelValidatorPruner` adapter mirrors the `WasmPruner` composition pattern.
11. **Plan 558** — `variable_rank_domain_expert.rs` shipped `pick_domain`. The latent-MoE substrate.
12. **Plan 159 / Research R130** — `roofline.rs` shipped the Plasma-tier kernel cost model. The roofline substrate.

## TL;DR

**Verdict: ship — as a new repo (`riir-kernel-heal`), consuming `roofline.rs` + `GemvAutotune` + `pick_domain` + the Proposal 034 drafter+pruner pattern, with the one new substrate being the `KernelValidator` numeric-compare trait (anti-Sakana held-out distribution non-negotiable).** Popcorn is the L0 corpus source (reading job, ~2 weeks, no GPU). The 4090 path (L4 LoRA) is deferred to riir-train, only if the modelless L0–L3 path platesaus below the 50% G1 gate. Next action: file `.plans/012_kernel_heal_phase0_corpus_distillation.md` (Phase 0 reading job) + open `.issues/001` in the new `riir-kernel-heal` repo for the skeleton.
