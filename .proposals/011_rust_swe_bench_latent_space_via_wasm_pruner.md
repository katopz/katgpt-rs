# Proposal 011 — Rust-SWE-bench as a Latent-Space Benchmark via WASM Constraint Pruner

## TL;DR

**Should we use Rust-SWE-bench (500 real-world Rust SWE tasks from 34 popular repos) as a latent-space validation target, by compiling each task's test suite to WASM and loading it as a `ConstraintPruner` inside the model's inference loop?**

This is the **moka pattern applied to SWE**: Benchmark 205 brought the Go game INTO Moka's native policy/value heads via PUCT search (98% win) instead of treating Moka as an external black box. This proposal brings the Rust test suite INTO the inference loop as a WASM-compiled symbolic pruner, instead of running `cargo test` externally.

**The enabling substrate already ships:**
- `WasmPruner` / `BomberWasmPruner` — loads WASM modules as `ConstraintPruner` impls with a `is_valid(depth, action_idx, state_ptr, state_len) -> i32` ABI, papaya instance pool, fuel-limited sandboxed execution, zero-copy state buffer, batch API.
- `HotSwapPruner` — BLAKE3-hash-detected runtime reload of WASM pruners from disk.
- `SpecAsPruner` — compiles NL specs into symbolic bitmap rules (4400× smaller than LoRA, O(1) per token, zero training, exact verification).
- `WasmTestGate` — validates pruner skills against WASM-sandboxed state checks.
- [rubrc](https://github.com/oligamiq/rubrc) — a port of `rustc` to WebAssembly (WIP, external). This is the compiler that would turn a Rust-SWE-bench task's test logic into a WASM pruner module.

**The fusion this enables:**
- **Proposal 010** (Non-Hidden-State Canonical Construction) — extracts source features (AST histogram, Clippy fingerprint, ownership graph) from the buggy + fixed code → defines canonical "fix directions" in source-feature space. Rust-SWE-bench provides the probe corpus (500 (buggy→fixed) pairs from 34 real repos).
- **Proposal 009** (Canonical Intent Space) — projects the fix direction into any model's latent space via `ModelAdapter`.
- **Proposal 032** (riir-ai, Kimi-K3 Native Support) — the model being validated. Phase 6 GOAT gate currently only tests "logits match PyTorch ref on a fixed prompt" — a weak numerical-correctness test. Rust-SWE-bench as a WASM pruner would provide a **functional correctness test** that exercises real Rust semantics inside the inference loop.

**Honest verdict: SPECULATIVE on Layer 3 (rubrc blocker), but MODELLESS-VALIDATED on Layer 4 (trajectory freeze/thaw uses shipped DEFAULT-ON substrate).** The proposal originally had three layers; a fourth was added after the modelless reframe (2026-08-01): **the test-feedback-driven inference loop's latent trajectory is itself a freezable artifact.** CUCG (Plan 333) already proved the Super-GOAT headline that "trajectory compaction and shard freeze are the same primitive" (G7 isomorphism). `committed_field_blend` (FAME, Plan 321) already computes BLAKE3-committed weights from a trajectory summary. `tf_loop` (Plan 136, DEFAULT-ON) already runs the recursive forward pass. So even if Layer 3 is blocked by rubrc AND the 0.40B model can't resolve any task, Layer 4 still produces measurable signal: the trajectory of *failed* attempts has geometry (curvature, drift, oscillation vs committed-wrong) that can be frozen and compared across snapshots. **The R463 caveat ("storage format ≠ capability") FLIPS under this reframe** — geometry is always measurable, even with zero passing patches.

**riir-train fallback is explicitly allowed.** Fine-tuning/LoRA on passing patches (if any exist) is a valid fallback path — riir-train ships adapter training. The modelless Layer 4 path is the PRIMARY (no backprop, uses shipped substrate); riir-train is the FALLBACK if modelless signal extraction proves insufficient. Per the modelless-first mandate, Layer 4 must be exhausted before deferring to riir-train.

## The problem this solves

### Problem 1: Proposal 032's Phase 6 GOAT gate is too weak

P032 Phase 6 currently tests: "load real 0.40B weights, run a forward pass, compare logits against the reference PyTorch implementation on a fixed prompt." This is a **numerical correctness** test — it verifies the forward pass math is right, but it does NOT verify the model produces **semantically meaningful** outputs on real Rust code.

The moka lesson (Benchmark 204, negative result): "blind heuristics cannot improve on a trained policy within the policy's training distribution." For Kimi-K3, the question is: does the 0.40B distillation carry enough Rust knowledge to produce coherent representations on real Rust repos? A single fixed-prompt logits comparison cannot answer this.

### Problem 2: Proposal 010 needs a probe corpus

P010 Phase 3 T3.1 needs paired `{(source_features(code_i), model_activations(code_i))}` samples for the ridge-regression `SourceFeatureAdapter` fit. The current plan says "Curate probe corpus (Rust code samples with known style properties)" — a hand-curated corpus that may be biased or too small.

Rust-SWE-bench provides 500 tasks from 34 popular Rust repos (ripgrep, bevy, tokio, clap, serde, nushell, axum, bytes, tracing, burn...). Each task is a controlled semantic transformation: (buggy code, fixed code, issue description, test patch). This is a **ready-made, diverse, ground-truth-labeled probe corpus** — far better than anything hand-curated.

### Problem 3: SWE benchmarks evaluate via external cargo test — slow + non-latent

The entire SWE-bench ecosystem (SWE-bench, Multi-SWE-bench, Rust-SWE-bench, RustForger) evaluates resolution by: LLM agent reads issue → writes patch → runs `cargo test` → checks pass/fail. This is:
- **Slow** — cargo test takes seconds to minutes per task.
- **External** — the validation happens outside the model's inference loop.
- **Non-latent** — it measures the OUTPUT (patch text), not the model's INTERNAL representations.

The moka pattern (Benchmark 205) showed that bringing the benchmark INTO the model's native inference path extracts dramatically more signal. For Go, this meant PUCT search using the policy/value heads natively. For SWE, the analogous move is: **compile the test suite to WASM, load it as a `ConstraintPruner`, and let the model's inference loop validate proposed patches in-sandbox.**

## The proposed design

### Architecture: the WASM-compiled test suite as a ConstraintPruner

```text
┌─────────────────────────────────────────────────────────────────┐
│  Rust-SWE-bench task                                            │
│  (issue description + repo snapshot + test patch + fix patch)   │
└───────────────┬─────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────┐
│  rubrc (WASM rustc, external WIP dependency)      │
│  Compiles the task's test logic to a WASM module  │
│  with the WasmPruner ABI:                         │
│    is_valid(patch_bytes_ptr, patch_len) -> i32    │
└───────────────┬───────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────┐
│  WasmPruner / HotSwapPruner (existing substrate)  │
│  Loads the WASM module as a ConstraintPruner.     │
│  Fuel-limited, sandboxed, papaya instance pool.   │
│  Runs INSIDE the model's inference loop.          │
└───────────────┬───────────────────────────────────┘
                │
         ┌──────┴──────┐
         ▼             ▼
┌─────────────────┐  ┌──────────────────────────────────┐
│ Kimi-K3 (P032)  │  │ Source features (P010)            │
│ proposes patch  │  │ AST histogram of buggy + fixed   │
│ in latent space │  │ → canonical "fix direction"       │
│ via P009 adapter│  │ → projected into latent space     │
└────────┬────────┘  └──────────────┬───────────────────┘
         │                          │
         ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  WASM ConstraintPruner validates the patch IN-LOOP              │
│  is_valid() runs the compiled test logic against the patch.     │
│  Result feeds back into the inference loop (prune invalid       │
│  branches, steer toward valid ones) — the PUCT analogy.         │
└─────────────────────────────────────────────────────────────────┘
```

### The three layers

**Layer 1 — Corpus (Rust-SWE-bench as P010 probe corpus):**
- Each task provides `(buggy_code, fixed_code)` — a controlled semantic transformation.
- P010's `ast_histogram(buggy)` and `ast_histogram(fixed)` produce source-feature vectors.
- The "fix direction" = `normalize(features(fixed) - features(buggy))` — a canonical direction in source-feature space.
- P010's `SourceFeatureAdapter` ridge-regression fit uses these pairs as the training corpus.
- This layer is **concrete and actionable today** — it just needs P010 Phase 1 (the `syn`-based AST histogram extractor) to ship.

**Layer 2 — Functional test (Rust-SWE-bench as P032 Phase 6 gate):**
- Run Kimi-K3's forward pass on Rust-SWE-bench task inputs (issue + relevant code context).
- Check that the model's latent representations are coherent:
  - Does attention highlight issue-relevant code regions?
  - Does MoE routing assign issue tokens to consistent experts?
  - Does KDA recurrent state differentiate buggy vs fixed code?
- This is **mechanistic interpretability for code models** — measuring whether the model's internals "light up" correctly on real Rust code.
- This layer is **partially actionable** — needs P032 Phase 5 (safetensors loader) to complete, then the latent-extraction harness.

**Layer 3 — In-loop validation (the wild idea — WASM pruner):**
- Compile the task's test logic to WASM via rubrc.
- Load as a `ConstraintPruner` via `WasmPruner`.
- The model's inference loop proposes patches; the WASM pruner validates them in-sandbox.
- The PUCT analogy: just as PUCT search used Moka's policy/value heads to explore + evaluate moves, the inference loop uses the WASM pruner to explore + validate patches.
- This layer is **HIGHLY SPECULATIVE** — depends on rubrc maturity, WASM compilability of real Rust test suites, and the 0.40B model's capability.

**Layer 4 — Trajectory freeze/thaw (the modelless reframe — uses SHIPPED DEFAULT-ON substrate):**

The key insight: **even if Layer 3 is blocked (rubrc) and the model proposes zero valid patches, the inference loop's trajectory through patch-space is itself a freezable artifact.** The trajectory has geometry — which patches were tried, in what order, how the latent state evolved, curvature, drift, oscillation vs commitment. That geometry can be frozen and compared across snapshots. This layer is NOT speculative — it composes already-shipped, GOAT-proven, DEFAULT-ON primitives:

| Step | Primitive | What it does | Status |
|---|---|---|---|
| 1. Run recursive forward pass | `tf_loop` (Plan 136) | ODE-motivated damped sub-stepping, K-stage RK, cache size independent of K | **DEFAULT-ON**, GOAT 034 PASS |
| 2. Measure trajectory geometry | `latent_trajectory_geometry` (Plan 342) | Length, curvature, drift, bifurcation ratio on latent state chains | G1-G5 ALL PASS |
| 3. Detect compaction points | `closed_unit_compaction` (CUCG, Plan 333) | Fires at structurally-safe moments (closed-unit ∧ summarizable ∧ progress ∧ ¬stuck). A test pass = a compaction candidate. | **DEFAULT-ON**, G7: trajectory compaction = shard freeze (isomorphism proven) |
| 4. Freeze trajectory summary | `committed_field_blend` (FAME, Plan 321) | Frozen sigmoid blend computed ONCE from trajectory summary + BLAKE3-committed. Sampling-invariant (FAME Prop. 3). | **DEFAULT-ON**, G1-G5 ALL PASS |
| (alt 4) Ridge fit from trajectory | `KarcShard` (Plan 308) | `Wout` bit-reproducible from trajectory; freeze = commit `(basis_config, k, λ, A, B)` | Plan 308 shipped |
| 5. Store frozen artifact | `MerkleFrozenEnvelope` (riir-neuron-db) | BLAKE3-checked atomic snapshot | shipped |
| 6. Self-healing swap | `reestimation.rs` (riir-ai latent_functor) | Coherence-driven re-estimation when coherence < tau_reest. Drift-triggered swap. | shipped |

**The CUCG G7 clincher:** Plan 333 already proved "trajectory compaction and shard freeze are the same primitive" — `can_freeze` isomorphism across all 4 combos. This is *literally* the claim that "loop and save the trajectory" = "freeze/thaw". The isomorphism is proven by construction, not speculated.

**The flipped R463 caveat:** The original R463 lesson ("storage format ≠ capability") said: if the model can't propose valid patches, the WASM pruner rejects everything, and there's no signal. Under the Layer 4 reframe, this FLIPS: the trajectory geometry is **always measurable**, even with zero passing patches. A trajectory of failed attempts still has shape — oscillation (high curvature, model can't commit), drift (rotating through wrong answers), committed-wrong (low curvature but wrong). That geometry is freezable and comparable across snapshots. This is strictly more information than the pass/fail-only signal.

**riir-train fallback (Layer 4b):** If modelless Layer 4 proves insufficient (e.g., trajectory geometry has no discriminative signal across snapshots), the explicit fallback is riir-train: LoRA fine-tune on whatever passing patches exist. riir-train ships adapter training. This is allowed — the modelless-first mandate says exhaust modelless paths FIRST, not "never train". Layer 4 is the modelless exhaustion step; Layer 4b is the documented deferral.

### The WasmPruner ABI extension

The existing `BomberWasmPruner` ABI is:
```text
is_valid(depth, action_idx, state_ptr, state_len) -> i32
```

For SWE validation, the ABI would extend to:
```text
is_patch_valid(patch_bytes_ptr, patch_len, test_state_ptr, test_state_len) -> i32
```

Where:
- `patch_bytes` = the model's proposed patch (as a unified diff or token sequence).
- `test_state` = the serialized test runner state (compiled test functions + fixtures).
- Returns 1 if the patch passes the tests, 0 otherwise.

This is a straightforward extension of the existing zero-copy state buffer pattern (`ZeroCopyStateBuffer` in `bomber/wasm_state.rs`). The WASM module applies the patch to the in-memory codebase representation, runs the test functions, and returns pass/fail — all sandboxed with fuel limits.

### Why this might work where external cargo test doesn't

The moka lesson (Benchmark 205): **bringing the evaluator into the model's native inference path extracts more signal.** For Go, PUCT search with the policy/value heads natively achieved 98% win vs 74% for external alpha-beta. The mechanism: the evaluator's feedback steers the search in real-time, pruning bad branches early.

For SWE, the analogous mechanism:
- The model proposes a patch (in latent space / as token drafts).
- The WASM pruner immediately validates it (no cargo build delay).
- Invalid patches are pruned from the DDTree before expansion.
- Valid patches get higher relevance scores, steering the search.
- The feedback loop is **microseconds** (WASM sandbox) not **seconds** (cargo test).

This is the SpeculativeGenerator + ConstraintPruner pattern (katgpt-rs's core architecture) applied to SWE: the model generates draft patches, the WASM pruner filters them, the DDTree explores valid branches.

## Honest caveats — READ BEFORE IMPLEMENTING

1. **rubrc is WIP + has hard dependency/proc-macro limitations.** Per the [rubrc README](https://github.com/oligamiq/rubrc) (verified 2026-08-01): rubrc runs `rustc` + `cargo` + `clang` + `llvm` + `rust-analyzer` as in-browser WASM modules via `wasi_virt_layer` (no OS subprocesses — a good fit for direct dispatch). Supported targets: only `wasm32-wasip1` + `x86_64-unknown-linux-musl`. **CRITICAL:** "external dependencies and procedural macros are currently unsupported" for Cargo. This is a near-total blocker for Rust-SWE-bench — the 34 repos (bevy, tokio, clap, serde, axum, tracing, burn...) ALL have external deps + many use proc-macros (serde derives, clap derives, tokio macros, bevy reflection). The realistic rubrc-compilable subset of the 500 tasks is **near-zero today**. **This is the #1 risk, upgraded from "WIP" to "hard blocker".** Mitigation paths: (a) wait for rubrc to add dependency/proc-macro support (timeline unknown); (b) bypass rubrc entirely — hand-extract the test assertion logic + hand-compile minimal WASM modules with the `is_patch_valid` ABI (Phase 3 T3.1 already proposes this); (c) target only the simplest tasks (single-file crates with no deps — likely <10 of the 500). Layer 1 (probe corpus) and Layer 2 (functional test) do NOT depend on rubrc — they work today once P010/P032 ship.

2. **Rust-SWE-bench repos are large.** Average: 993 files, 128K lines. The largest (bevy) is 15K files, 753K lines. Compiling these to WASM may be intractable. Mitigation: the WASM pruner only needs the **test logic + the code under test**, not the entire repo. A subset extraction step (identify which files the test patch touches, compile only those) would reduce scope.

3. **The 0.40B model may not have enough capability (DOWNGRADED — no longer a hard blocker).** RustForger (with Claude-Sonnet-3.7, a much larger model) achieves only 28.6% resolution. A 0.40B distillation will be dramatically weaker. **Under the original 3-layer design, this was the #2 risk.** Under the 4-layer modelless reframe (Layer 4), this is DOWNGRADED: even if the model proposes zero valid patches, the trajectory geometry is still measurable + freezable (see Layer 4 above). The research value holds regardless of resolution rate. **The remaining risk is whether trajectory geometry has discriminative signal across snapshots** — i.e., do different models/snapshots produce measurably different failure-trajectory shapes? This is a POC question (Plan 566 Phase 1), not a blocker. Mitigation if no signal: riir-train fallback (Layer 4b — LoRA fine-tune on whatever passing patches exist).

4. **WASM compilation of test suites with external dependencies may fail.** Many Rust-SWE-bench repos use crates that don't compile to `wasm32-unknown-unknown` (e.g., crates using `std::process`, file I/O, networking). Mitigation: filter tasks to those whose test suites are WASM-compatible (no `std::process`, no networking, no filesystem). This likely reduces the 500-task corpus significantly, but even 50 WASM-compatible tasks would be a useful POC.

5. **The "resolution in latent space" claim (Thread C) is unproven — but Layer 4 reframes it.** SWE resolution has no closed-form win condition like Go's territory score. The honest version of Layer 3 is "in-loop validation via WASM pruner" (the pruner gives pass/fail feedback), NOT "latent-space resolution" (the latent state magically contains the fix). The R463 lesson applies to Layer 3: "storage format ≠ capability" → "WASM in-loop validation ≠ resolution capability." **BUT Layer 4 flips the R463 caveat**: the trajectory geometry is always measurable, even on failed attempts. The claim is NOT "the trajectory contains the fix" — it's "the trajectory has shape that distinguishes models/snapshots, and that shape is freezable." This is a weaker but honest claim that doesn't require resolution.

6. **This is architecturally adjacent to Proposal 010's speculation level.** P010's verdict: "HIGHLY SPECULATIVE... exists because it's the ONLY remaining path, not because it's likely to succeed." This proposal inherits that risk. If P010's G5 (cross-arch agreement) fails, the source-feature directions are meaningless, and Layer 1's "fix directions" are noise. Layer 3 can still work as a pure WASM-pruner POC (independent of P010), but the fusion value collapses.

7. **`syn` dependency weight.** Same as P010 — adding a full Rust parser to the canon crate's feature surface. Mitigation: same as P010 (behind `canon_source_features` feature flag).

## Fusion lineage

This proposal combines **five** existing substrate pieces + **one** external dependency + **one** benchmark dataset + **the Layer 4 modelless reframe substrate (six additional shipped primitives)**:

1. **`WasmPruner` / `BomberWasmPruner`** (`katgpt-pruners/src/hot_swap.rs`, `src/pruners/bomber/wasm_pruner.rs`) — the WASM-module-as-ConstraintPruner substrate. Fuel-limited, sandboxed, papaya instance pool, zero-copy state buffer, batch API. This is the load-bearing substrate for Layer 3.
2. **`SpecAsPruner`** (`katgpt-pruners/src/spec_compile/`) — compiles NL specs into symbolic bitmap rules. Layer 1's "fix direction" is a symbolic rule compiled from source features, not a neural adapter. This is the ideological ancestor.
3. **`HotSwapPruner`** (`katgpt-pruners/src/hot_swap.rs`) — BLAKE3-hash-detected runtime reload. Enables swapping the WASM pruner when the Rust-SWE-bench task changes.
4. **Proposal 010** (`katgpt-rs/.proposals/010_non_hidden_state_canonical_construction.md`) — `SourceFeatureDirection` (AST histogram, Clippy fingerprint, ownership graph) → canonical directions from source code. Rust-SWE-bench is its probe corpus.
5. **Proposal 009** (`katgpt-rs/.proposals/009_canonical_intent_space.md`) — `CanonicalIntent` + `ModelAdapter` (Procrustes, Subspace, Mask). Projects source-feature directions into model latent space.
6. **Proposal 032** (`riir-ai/.proposals/032_kimi_k3_native_support.md`) — Kimi-K3 native support (MLA + MoE + KDA + SiTU). The model being validated.
7. **rubrc** ([github.com/oligamiq/rubrc](https://github.com/oligamiq/rubrc), MIT OR Apache-2.0) — WASM-hosted Rust toolchain running in a browser worker via `wasi_virt_layer`. Embeds `rustc_opt.wasm` + `cargo_opt.wasm` + `llvm_opt.wasm` + `lsp_opt.wasm`. Supported targets: `wasm32-wasip1` + `x86_64-unknown-linux-musl` only. **Hard limitation (verified 2026-08-01):** external dependencies + procedural macros are unsupported in Cargo. Serialized execution via `CARGO_RUN_LOCK` / `RUSTC_RUN_LOCK`. The in-process module-dispatch architecture (no OS subprocesses) is a good fit for the WASM-pruner pattern, but the dependency/proc-macro blocker makes the rubrc path near-viable for real Rust-SWE-bench tasks today. The compiler that *would* turn Rust test suites into WASM pruner modules — once the dependency limitation lifts.
8. **Rust-SWE-bench** ([arXiv:2602.22764](https://arxiv.org/abs/2602.22764), Xiang et al. ICSE '26) — 500 real-world Rust SWE tasks from 34 repos. The benchmark dataset.

The combination produces what none alone can: a path to evaluate SWE capability **inside the model's inference loop** using the existing WASM-pruner substrate, validated against a real-world Rust benchmark, with source-feature-based canonical directions connecting the code structure to the model's latent space.

## GOAT gate

This proposal does NOT request default-on promotion. It requests **research validation** behind an opt-in feature flag. The gates differ per layer:

| Layer | Feature flag | G1 correctness | G2 perf | G3 no-reg | G4 alloc | G5 (decisive) |
|---|---|---|---|---|---|---|
| 1 (corpus) | `swe_bench_corpus` | source-feature extraction deterministic (BLAKE3) | AST histogram on 10K-line crate < 1s (same as P010) | opt-in, no default impact | projection apply zero-alloc | **fix directions correlate with ground-truth patches** (threshold TBD) |
| 2 (functional test) | `kimi_k3_swe_probe` | latent extraction matches PyTorch ref attention pattern | forward pass on 500 tasks < 60s total | existing P032 tests pass | per-task latent extraction alloc-free | **attention highlights issue-relevant code regions** (measurable via attention-rollout correlation with ground-truth fix locations) |
| 3 (WASM pruner) | `swe_bench_wasm_pruner` | WASM module produces same pass/fail as cargo test | WASM validation < 100ms per patch (vs seconds for cargo) | existing WasmPruner tests pass | sandbox state buffer reused | **in-loop validation prunes more invalid patches than no-validation baseline, improving resolution rate on a WASM-compatible subset** |
| **4 (trajectory freeze)** | `swe_trajectory_freeze` | trajectory geometry extraction deterministic (BLAKE3 on summary) | geometry measurement < 5µs per trajectory (matches Bench 342) | opt-in, no default impact; composes shipped primitives only | freeze/thaw zero-alloc hot path (CUCG G4) | **trajectory geometry discriminates across snapshots/models** — different models produce measurably different failure-trajectory shapes (curvature, drift, oscillation). **The load-bearing POC**: even with zero passing patches, the geometry is freezable + comparable. |
| **4b (riir-train fallback)** | (riir-train) | LoRA training reproducible | training run completes | N/A (separate repo) | N/A | **LoRA fine-tune on passing patches improves resolution rate** vs no-fine-tune baseline. Only if Layer 4 G5 shows insufficient signal. |

**No "Report the Floor" rule applies** — this is not a UQ-bearing primitive (no probability distribution / coverage claim).

**Promotion criterion:** Layer 3 G5 is the load-bearing gate. If it passes on even a 50-task WASM-compatible subset, the pattern is proven. If it fails, Layer 1 + Layer 2 still ship as research validation (probe corpus + functional test) — they don't depend on Layer 3.

## What ships now (katgpt-rs) vs deferred

### Ships now — katgpt-rs (if validated)
- `RustSweBenchTask` struct (task ID, repo, issue text, buggy commit, fixed commit, test patch path).
- `RustSweBenchCorpus` loader (reads the 500-task dataset, filters by WASM-compatibility heuristic).
- `source_features` extraction adapter (P010 integration — uses P010's AST histogram on buggy + fixed code to compute fix directions).
- `SweBenchWasmPruner` (Layer 3 — extends `WasmPruner` with the `is_patch_valid` ABI). Behind `swe_bench_wasm_pruner` feature.
- **`SweTrajectoryFreezer`** (Layer 4 — composes `tf_loop` + `latent_trajectory_geometry` + `committed_field_blend` + `MerkleFrozenEnvelope`). Behind `swe_trajectory_freeze` feature. The modelless path that works even when Layer 3 is blocked.
- G1/G2/G4 gates on extraction + WASM validation + trajectory freeze.
- **G5 runs in riir-ai** (needs Kimi-K3 loaded for Layers 2/3; Layer 4 can POC on synthetic trajectories first).

### Deferred — riir-ai
- Kimi-K3 latent extraction on Rust-SWE-bench tasks (Layer 2 functional test).
- In-loop validation wiring (Layer 3 — the WASM pruner integrated into Kimi-K3's DDTree inference loop).
- Real-model GOAT gate (P032 Phase 6 extension).

### Deferred — external dependency
- **rubrc maturity** — the entire Layer 3 depends on rubrc being able to compile real Rust test suites. If rubrc cannot, Layer 3 is blocked and only Layer 1 + Layer 2 ship.

### Explicitly NOT shipped by this proposal
- **A competing SWE agent** — this is NOT "build our own RustForger." The goal is latent-space validation, not agent resolution rate competition.
- **Default-on promotion** — this is research validation. Promotion (if G5 passes) requires a separate proposal.
- **Full 500-task WASM compilation** — even if rubrc works, compiling all 500 tasks' test suites is a massive effort. The POC targets a WASM-compatible subset (estimated 50-100 tasks after filtering).

### riir-train fallback (Layer 4b — ALLOWED, not blocked)
The original draft said "Training on Rust-SWE-bench — modelless-first mandate." That was too absolute. **Fine-tuning/LoRA via riir-train is an explicit allowed fallback** (riir-train ships adapter training). The modelless-first mandate says exhaust modelless paths FIRST (Layer 4), not "never train." If Layer 4's G5 shows trajectory geometry has insufficient discriminative signal, the documented deferral is: LoRA fine-tune on whatever passing patches exist (Layer 4b). This is the §3.5 modelless-unblock protocol applied honestly — Layer 4 is the exhaustion step; Layer 4b is the deferral with explicit documentation of why modelless was insufficient.

## Phased rollout (sketch — a plan would expand this)

### Phase 1 — Corpus loader + source-feature extraction (Layer 1)
- [ ] T1.1 Download Rust-SWE-bench dataset (500 tasks from [GitHub](https://github.com/GhabiX/Rust-SWE-Bench))
- [ ] T1.2 `RustSweBenchCorpus` loader struct (task ID, repo path, commit hashes, patch paths)
- [ ] T1.3 P010 integration: `ast_histogram` on buggy + fixed code → fix direction computation
- [ ] T1.4 WASM-compatibility filter (heuristic: reject tasks whose test deps use `std::process` / networking / filesystem)
- [ ] T1.5 G1 correctness: deterministic extraction (BLAKE3 commitment)
- [ ] T1.6 G2 perf: AST histogram on the largest repo (bevy subset) < 5s

### Phase 2 — Functional test harness (Layer 2, gated on P032 Phase 5)
- [ ] T2.1 Latent extraction: run Kimi-K3 forward pass on task inputs, extract MLA attention + MoE routing + KDA state
- [ ] T2.2 Attention-rollout correlation: does attention correlate with ground-truth fix locations?
- [ ] T2.3 MoE routing consistency: do issue-relevant tokens route to consistent experts?
- [ ] T2.4 G5: measurable signal on a 50-task subset

### Phase 3 — WASM pruner POC (Layer 3, gated on rubrc maturity)
- [ ] T3.1 Manual WASM compilation of 1 task's test suite (hand-compiled, not via rubrc — proves the ABI works)
- [ ] T3.2 `SweBenchWasmPruner` impl (extends `WasmPruner` with `is_patch_valid`)
- [ ] T3.3 Integration with DDTree: invalid patches pruned before expansion
- [ ] T3.4 rubrc evaluation: can it compile a real Rust-SWE-bench test suite?
- [ ] T3.5 G5: in-loop validation improves resolution rate vs no-validation baseline on the WASM-compatible subset

### Phase 4 — Scale + fusion (only if Phase 3 G5 passes)
- [ ] T4.1 Scale to 50+ WASM-compatible tasks
- [ ] T4.2 P009 adapter: project P010 fix directions into Kimi-K3 latent space, steer toward valid patches
- [ ] T4.3 PUCT-style search: combine WASM pruner feedback + policy prior + value estimate
- [ ] T4.4 Full GOAT gate + honest negative result if it fails

### Phase 5 — Layer 4 trajectory freeze (MODELLESS — runnable independently of Phase 3)

> **This phase does NOT depend on rubrc.** It composes shipped DEFAULT-ON primitives. It can run on synthetic trajectories first (no Kimi-K3 needed), then on real model trajectories once P032 Phase 5 ships. This is the cheapest validation path.
>
> **T5.1–T5.3 ran 2026-08-02 (Issue 569).** Verdict: Partial Gain — T5.1 + T5.2
> PASS (geometry discriminates failure modes + CUCG fires on test-pass events);
> T5.3 FAIL (FAME commit is deterministic but produces degenerate blends from
> random direction vectors — concentration-of-measure artifact; real freezer
> needs data-derived directions, not random). See
> `Issue 569` for the
> full verdict table + design-constraint analysis.
>
> **T5.3b ran 2026-08-02 (Issue 570).** Verdict: **PASS — Layer 4 validated on
> synthetic data.** Data-derived directions from cluster centroids of
> geometry-encoded summaries produce non-degenerate FAME blends (100% probe
> accuracy, all matching gates > 0.6). The dual-strategy test isolated two
> compounding causes of T5.3's failure: (1) random directions (concentration
> of measure) + (2) summary encoding mismatch (endpoint position doesn't
> encode failure-mode geometry). Both fixed: data-derived directions +
> geometry-aware summary encoder. **Design constraint for T5.5: summary
> encoder MUST capture failure-mode-discriminative geometry (length, curvature,
> step-to-step cosine), not just raw latent position.**
>
> **T5.6e ran 2026-08-02 (Bench 018).** Verdict: **POSITIVE — sequence trajectory
> overcomes the per-token information floor.** The SEQUENCE trajectory (64 final
> hidden states across a prompt's tokens with growing KV cache — NEVER tested
> in bench 012-017, which all used the DEPTH trajectory of 9 per-layer states)
> achieves 100% per-prompt accuracy at σ≥0.1 via the SeqStateStats encoder
> (aggregate L2 norm statistics: mean/std/max/min state norm, initial/final
> norm, norm ratio, mean cosine). d_Mahalanobis = 14.526 at σ=0.5 — **50× the
> depth trajectory's best d_M=0.285**. The expected √(64/9)≈2.67× boost is
> exceeded by 19× because each sequence step (full forward pass) carries far
> more weight-dependent signal than each depth step (single layer transform).
> The discriminative signal lives in state MAGNITUDE (weight-determined
> activation scale), not displacement (input-dependent delta). Discrimination
> floor: σ≈0.03-0.05 (3-5% relative weight noise). Implication: Layer 4
> per-attempt freezing is VALIDATED for value-level discrimination — two model
> snapshots differing by >5% relative weights can be discriminated per-attempt.
> Follow-up: the SweTrajectoryFreezer needs a SeqStateStats-like encoder (not
> the shipped GeometrySummaryEncoder) for production use. See
> `.benchmarks/018_sequence_trajectory.md`.

- [x] T5.1 **POC (synthetic): does trajectory geometry discriminate?** Construct synthetic pass/fail trajectories (mimicking SWE attempt patterns: oscillation, drift, committed-wrong). Run `latent_trajectory_geometry` on them. Question: do different failure modes produce measurably different geometry (curvature, drift angle)? Even a NEGATIVE result (no discrimination) is valuable — it means trajectory geometry alone is insufficient signal. **RESULT (Issue 569): PASS — 5 modes produce 5 distinct `(curvature, length)` signatures; oscillation hits exactly π (the ping-pong signature).**
- [x] T5.2 **POC (synthetic): does CUCG evaluate() fire on test-pass events?** Construct synthetic test-pass sequences. Run CUCG `evaluate()`. Question: does a test pass qualify as a closed unit (closed-unit ∧ summarizable ∧ progress ∧ ¬stuck)? **RESULT (Issue 569): PASS — fires Compress at exactly the test-pass events, nowhere else.**
- [x] T5.3 **POC (synthetic): committed_field_blend from failure trajectory.** Run FAME on a synthetic all-fail trajectory summary. Question: does it produce a stable, BLAKE3-committable, sampling-invariant blend even with zero positive examples? **RESULT (Issue 569): CONDITIONAL FAIL — FAME is deterministic + stable, but random direction vectors produce near-zero dots → degenerate blend.** **UPGRADED TO PASS by T5.3b (Issue 570): data-derived directions from geometry-encoded summaries produce non-degenerate blends (100% accuracy).**
- [x] T5.3b **POC (synthetic, Issue 570): data-derived directions fix the concentration-of-measure failure.** Dual-strategy test: (A) geometry-encoded summary + data-derived directions → PASS (100% accuracy, all matching gates > 0.6); (B) endpoint-position summary + data-derived directions → FAIL (17% accuracy, gates near 0.5 — endpoint positions don't cluster by failure mode). **Contrast documents the design constraint for T5.5: summary encoder MUST capture failure-mode-discriminative geometry, not just position.**
- [-] T5.4 **POC (real model, Bench 012): trajectory geometry on real Kimi-K3 depth trajectories.** Option 2 (bypass tf_loop — architecturally incompatible with Kimi-K3's hybrid MLA/KDA/MoE/attn-res layer type): added `kimi_k3_forward_token_traced` to capture per-layer hidden states; ran on REAL `model.safetensors` at D=1024, 8 layers. **RESULT: PARTIAL — G1+G2+G4 PASS, G3 FAIL (29% distinct, threshold 30%).** The substrate is numerically stable + non-degenerate at production scale, but the per-token DEPTH trajectory (embed → 8 post-layer states, 9 steps) is NOT strongly discriminative across tokens even with real weights — depth geometry is dominated by the layer weight structure (which is the same across tokens), not the input. The original P011 design assumed `tf_loop`'s iterative refinement (100-step trajectories), which is fundamentally different from a 9-step depth trajectory. **Design implication: the discriminative signal in Layer 4 likely lives in the ITERATIVE refinement trajectory (repeated forward passes on evolving patch proposals), not in the depth trajectory of a single forward pass.** This unblocks T5.5 (freezer impl with geometry-summary encoder) but reframes T5.4 as needing the iterative loop substrate (which requires either porting tf_loop to Kimi-K3 or a different trajectory extraction strategy). See `.benchmarks/012_kimi_k3_trajectory_geometry.md`.
- [x] T5.5 `SweTrajectoryFreezer` impl — composes tf_loop + latent_trajectory_geometry + committed_field_blend + MerkleFrozenEnvelope. Behind `swe_trajectory_freeze` feature. **Design constraint (from T5.3b): (1) archetype directions MUST be data-derived from clustering real failure trajectories; (2) summary encoder MUST capture failure-mode-discriminative geometry (length, curvature, step-to-step cosine), NOT just raw latent position.** **RESULT (Bench 013): ALL 4 GOAT gates PASS at substrate level.** Shipped `crates/katgpt-core/src/swe_trajectory_freeze.rs` — `GeometrySummaryEncoder` (extracts T5.3b's geometry-summary encoder, generic over D) + `derive_directions` (nearest-centroid classifier direction, generic over N+M+D) + `SweTrajectoryFreezer<N, D>` (two-stage pipeline: fit + freeze) + `TrajectoryFreezeEnvelope` (local BLAKE3 envelope matching the `MerkleFrozenEnvelope` pattern, no cross-repo dep) + `FrozenAttempt<N, D>` (the committed characterization). G1 directions non-degenerate PASS; G2 latency 4582 ns/call < 5000 ns PASS; G3 cross-mode discrimination 100% accuracy (oscillation 0.98 gates, committed_wrong 0.71, converged_correct 0.72) PASS; G4 2 allocs/call PASS (honest: both inherited from `from_states` substrate's per-call Vec allocation; the freeze pipeline itself is zero-alloc — follow-up: add `from_states_into` variant). Stays opt-in — research-validation primitive; promotion requires T5.6 G5 to pass on real-model trajectories. See `.benchmarks/013_swe_trajectory_freezer_goat.md`.
- [x] T5.6 Layer 4 G5 gate: trajectory geometry discriminates across snapshots/models. **RESULT (Bench 014): G5 PASS — 100% accuracy on cross-model discrimination (real Kimi-K3 vs random weights, 40/40 held-out trajectories classified correctly).** Shipped `bench_014_swe_trajectory_freezer_g5.rs` + two substrate improvements: (1) `from_states_into` zero-alloc variant (eliminates the 2 inherited allocs/call — `freeze_attempt_into` achieves true 0 allocs); (2) `derive_directions_and_centroid` + `SweTrajectoryFreezer::fit` constructor + mean-centering in `freeze_attempt_into` (the critical fix — without mean-centering, non-centered summaries produce dot products all on the same side of 0, yielding 50% accuracy; with mean-centering, the sigmoid threshold at 0 aligns with the natural decision boundary, yielding 100%). The T5.4 finding is reframed: depth trajectories ARE discriminative across MODELS (weight-dependent signal is strong) even though they're weakly discriminative across TOKENS (input-dependent signal is weak at 29%) — depth geometry is model-determined, not input-determined, which is exactly what G5 needs. Stays opt-in — promotion needs cross-snapshot + failure-mode discrimination on real SWE-bench attempts (separate proposal). See `.benchmarks/014_swe_trajectory_freezer_g5.md`.
- [-] T5.7 If G5 FAILS → document why modelless was insufficient, file Layer 4b (riir-train LoRA fallback) with explicit §3.5 documentation. **N/A — G5 PASSED.** Layer 4b remains an ALLOWED fallback if future cross-snapshot or failure-mode discrimination tests show insufficient signal, but the primary G5 gate passed.
- [x] T5.8 If G5 PASSES → the modelless path is validated. Layer 3 (WASM pruner) becomes an enhancement, not a dependency. **DONE — G5 PASSED on real Kimi-K3 depth trajectories (Bench 014).** The modelless Layer 4 path is validated for snapshot/model discrimination. Layer 3 (WASM pruner) remains blocked on rubrc maturity but is now an enhancement (in-loop patch validation), not a dependency for the trajectory-freeze substrate.
- [-] T5.6b **Follow-up — perturbation sensitivity (cross-snapshot proxy, Bench 015).** Probed how robust the G5 signal is to subtle weight changes by perturbing real Kimi-K3 weights at σ ∈ {0, 0.001, 0.01, 0.05, 0.1, 0.5} relative noise + measuring discrimination accuracy. **RESULT: NEGATIVE — accuracy stays at ~50% (coin flip) across ALL σ levels, even at σ=0.5 (50% relative noise).** The centroid separation at σ=0.5 is 0.048 vs Bench 014's ~0.54+ for real-vs-random — **10× too small** for the freezer to discriminate. Root cause: the `GeometrySummaryEncoder` captures SHAPE features (length, curvature, cosine) that are INVARIANT to value perturbation but sensitive to structural change. Additive perturbation changes values without destroying structure; the geometry features don't detect it. **Implication:** cross-snapshot discrimination via depth trajectory geometry is UNLIKELY — two checkpoints of the same architecture produce similar-shaped depth trajectories regardless of weight value differences. To discriminate value-level differences, either (a) a value-sensitive encoder (mean activation magnitude, variance spectrum) or (b) the iterative refinement trajectory (T5.4 path 2) would be needed. This does NOT invalidate T5.6 G5 PASS (structural discrimination is valid) or T5.1 failure-mode discrimination (shape-based). The limitation is specifically for fine-grained value-level discrimination. See `.benchmarks/015_swe_trajectory_perturbation_sensitivity.md`.
- [-] T5.6c **Follow-up — value-sensitive encoder probe (Bench 016).** Tested 4 value-sensitive encoders (DispNorms, DispStats, StateNorms, DispRatios) that capture per-layer displacement statistics instead of aggregate trajectory shape. **RESULT: NEGATIVE for per-token classification, but CORRECTS bench_015's root cause.** The value-sensitive features DO change significantly with perturbation (centroid distances 100-200× larger than geometry: DispNorms centroid_dist=11.17 vs Geometry's 0.048 at σ=0.5). But per-token accuracy stays at ~50% because **SNR ≈ 1.0** — token-to-token variance is comparable to the perturbation signal. At moderate σ (0.05-0.1), centroid-of-test-tokens classification succeeds at 100% (proving the signal exists), but individual tokens scatter too widely for per-token nearest-centroid classification. **This is a RESOLUTION FLOOR for per-token trajectory classification, not an information deficit.** The signal exists at the aggregate level but is insufficient for single-trajectory decisions. Overcoming it would require multi-token averaging (√N SNR boost, but needs ~16 samples — not applicable to per-attempt SWE-bench freezing) or a covariance-aware classifier (Mahalanobis/LDA, needs ~2D training samples per class). Neither fits the per-token use case. The iterative refinement trajectory (T5.4 path 2) remains the necessary substrate for fine-grained discrimination. See `.benchmarks/016_value_sensitive_encoder.md`.
- [-] T5.6d **Follow-up — covariance-aware classifier probe (Bench 017).** Tested whether Mahalanobis/LDA (Ledoit-Wolf shrunk covariance) can overcome the bench_016 SNR floor. 128 tokens (96 train + 32 test per class), 4 value-sensitive encoders at natural dimensionality, 3 classifiers (Euclidean / Diagonal / Full Mahalanobis) + Bayes-optimal ceiling estimate Φ(d_M/2). **RESULT: NEGATIVE — the per-token SNR floor is FUNDAMENTAL, not classifier-specific.** Mahalanobis DOES improve over Euclidean (+7-12pp: DispNorms 43.8%→56.2%), confirming the covariance structure is non-trivial. But the Bayes-optimal ceiling itself is only ~54-56% — the Mahalanobis centroid distance d_M ≈ 0.2-0.3 at σ=0.5 is ~10× below the d_M > 2 threshold needed for 80% accuracy. Root cause: the perturbation signal aligns with HIGH-VARIANCE directions of the token covariance (d_Euclidean/d_M ≈ 18-47×); whitening suppresses signal and noise equally. Actual Mahalanobis accuracy ≈ Bayes-optimal, confirming the classifier is working correctly — the limitation is information content, not classifier quality. **This definitively closes the per-token classification question.** See `.benchmarks/017_covariance_aware_classifier.md`.
- [x] T5.6e **Follow-up — sequence trajectory discrimination (Bench 018).** The CRITICAL unexplored direction after T5.6b/c/d: ALL prior benches (012-017) extracted the DEPTH trajectory (9 per-layer states per token, with `reset()` between tokens). NONE tested the SEQUENCE trajectory — the sequence of final hidden states across a prompt's tokens with growing KV cache. This is fundamentally different: (1) 64 steps vs 9 → √N SNR boost; (2) each step is a FULL forward pass through all 8 layers (not one layer transform); (3) the trajectory captures the model's processing dynamics. Tested 32 prompts × 64 tokens with 4 encoders (SeqDispStats, SeqStateStats, SeqFullProfile, Geometry) + 3 classifiers. **RESULT: POSITIVE — SeqStateStats achieves 100% per-prompt accuracy at σ≥0.1.** The aggregate state L2 norm statistics (mean/std/max/min norm, initial/final norm, norm ratio, mean cosine) are extremely weight-sensitive: d_Mahalanobis = 14.526 at σ=0.5 — **50× the depth trajectory's best d_M=0.285** (expected √(64/9)≈2.67×, actual 50× — each sequence step carries 19× more signal than each depth step). Discrimination floor σ≈0.03-0.05. The signal lives in state MAGNITUDE (weight-determined activation scale), NOT displacement (input-dependent delta) — SeqDispStats stays at 53%. **This validates Layer 4 per-attempt freezing for value-level discrimination.** See `.benchmarks/018_sequence_trajectory.md`.
- [x] T5.6f **Substrate port — StateMagnitudeEncoder (Bench 019, Issue 571).** Ported bench_018's `encode_seq_state_stats` to the substrate as `StateMagnitudeEncoder` (d=8, single-pass Welford, zero-alloc). Added `FrozenValueAttempt<N,D>` + `freeze_attempt_value` / `freeze_attempt_value_into` methods to `SweTrajectoryFreezer`. The freezer now supports BOTH paths: `freeze_attempt` (geometry, structural discrimination — bench_014) + `freeze_attempt_value` (state magnitude, value discrimination — bench_018). GOAT G1-G5 ALL PASS: G1 correctness (hand-computed values match), G2 perf (51.8µs vs geometry 100.7µs = 0.52x — faster due to single-pass), G3 no-regression (geometry path unaffected, 1851 tests), G4 tamper-evidence, G5 value discrimination (100% on synthetic scale+variance-shift). Stays opt-in (`swe_trajectory_freeze`); promotion deferred (synthetic G5 + no consumer). See `.benchmarks/019_state_magnitude_encoder_substrate_goat.md`.
- [x] T5.6g **Generation trajectory discrimination (Bench 020).** Tests whether the GENERATION trajectory (hidden states during greedy argmax decoding) discriminates as well as the PROCESSING trajectory (bench_018's fixed-token processing). 32 prompts × (16-token prefix + 48 generated tokens). Uses the substrate `StateMagnitudeEncoder` (bench_019) for both regimes. **RESULT: POSITIVE — generation trajectory achieves 100% at σ≥0.1, matching processing.** At σ=0.05, generation is actually BETTER (100% vs 81.2%) — the model's argmax choices amplify the weight perturbation's effect (a slightly different weight matrix flips argmax near decision boundaries, cascading through KV cache). d_Euclid is smaller for generation (0.762 vs 0.921 at σ=0.5) but within-class scatter is also smaller, preserving discriminability. **This validates the substrate for the full SWE-bench use case (patch generation).** See `.benchmarks/020_generation_trajectory.md`.

## Risks

1. **rubrc dependency/proc-macro blocker (highest, upgraded).** Verified 2026-08-01 via the rubrc README: external dependencies + procedural macros are unsupported. ALL 34 Rust-SWE-bench repos have external deps + most use proc-macros (serde/clap/tokio/bevy derives). The rubrc-compilable subset is near-zero today. Layer 3 via rubrc is blocked until rubrc adds dependency support (timeline unknown). Mitigation: Phase 3 T3.1 (hand-compiled WASM module) proves the ABI without rubrc. If the hand-compiled POC shows the pattern works, the question becomes "when does rubrc unblock deps?" not "does the architecture work?".

2. **WASM-compatibility filtering reduces corpus severely.** Many Rust-SWE-bench tasks have test suites with external deps (tokio, filesystem, networking) that don't compile to wasm32. The WASM-compatible subset may be < 50 tasks. Mitigation: even a small subset proves the pattern; scaling is a follow-up.

3. **0.40B model capability ceiling (DOWNGRADED — Layer 4 reframes this).** The model may not propose any valid patches, making the WASM pruner reject everything. Under the original 3-layer design this was the #2 risk. Under the 4-layer modelless reframe, this is DOWNGRADED: trajectory geometry is measurable even on failed attempts (Layer 4). The remaining risk is whether trajectory geometry has discriminative signal across snapshots — a POC question (Phase 5 T5.1), not a blocker. Mitigation: riir-train LoRA fallback (Layer 4b) if modelless proves insufficient.

4. **Source features too coarse (inherited from P010).** If P010's G5 (cross-arch agreement) fails, the fix directions are noise. Mitigation: Layer 3 (WASM pruner) works independently of P010 — it validates patches against real test logic, regardless of whether the source-feature direction is meaningful.

5. **Scope creep into "build a SWE agent."** This proposal is about latent-space validation, not agent competition. Mitigation: the GOAT gate measures internal coherence (Layer 2) + in-loop validation pattern (Layer 3), NOT resolution rate vs RustForger. Comparing to RustForger's 28.6% is a category error — they're different goals.

6. **Dataset licensing.** Rust-SWE-bench is CC-BY 4.0 (academic). Using it as a probe corpus in a commercial product needs license review. Mitigation: the corpus is used for adapter fitting (setup-time, not shipped), and the WASM pruner modules are derived artifacts. Legal review before any production use.

7. **Layer 4 trajectory geometry may have no discriminative signal (NEW).** The core Layer 4 hypothesis is that different models/snapshots produce measurably different failure-trajectory shapes (curvature, drift, oscillation vs committed-wrong). This may be FALSE — all failure trajectories may look the same (high entropy, no structure). Mitigation: Phase 5 T5.1 tests this on synthetic data first (cheapest). If the synthetic POC shows discrimination is possible in principle, T5.4 tests on real model trajectories. If T5.1 shows NO discrimination even in principle, Layer 4 is demoted and the proposal falls back to Layer 1 + Layer 2. **A negative result here is valuable** — it documents that trajectory geometry alone is insufficient, motivating the riir-train LoRA path (Layer 4b).

## Out of scope

- **Building a competing SWE agent.** This is latent-space validation, not "our RustForger."
- **Cross-language SWE benchmarks.** Rust-only (the source features + rubrc are Rust-specific).
- **Full 500-task WASM compilation.** POC on a WASM-compatible subset first.
- **Default-on promotion.** Research validation only.
- **Production steering runtime.** The adapter fit + WASM validation is the substrate; runtime integration (riir-ai NPC cognition, seal consumer) is a separate plan.

> **Note (2026-08-01):** "Training on Rust-SWE-bench" was listed here as out-of-scope. **Corrected**: fine-tuning/LoRA via riir-train is an ALLOWED fallback (Layer 4b). The modelless-first mandate requires exhausting Layer 4 first, but does not forbid training if modelless proves insufficient. See the "riir-train fallback" subsection above.

## References

1. **Rust-SWE-bench / RustForger** — Xiang, He, Wang, Tian, Zhang (SUSTech / Ant Group, ICSE '26). [arXiv:2602.22764](https://arxiv.org/abs/2602.22764). 500 real-world Rust SWE tasks from 34 repos + RustForger agent (proc-macro AST Trace command, 28.6% resolution). Used here as the probe corpus + functional test target. The RustForger agent itself is PASS (LLM-dependent semantic code generation); the benchmark dataset is Gain.
2. **Proposal 010** — [katgpt-rs/.proposals/010_non_hidden_state_canonical_construction.md](../.proposals/010_non_hidden_state_canonical_construction.md). Source-feature directions (AST histogram, Clippy fingerprint, ownership graph). The enabling substrate for Layer 1.
3. **Proposal 009** — [katgpt-rs/.proposals/009_canonical_intent_space.md](../.proposals/009_canonical_intent_space.md). Canonical Intent Space + ModelAdapter. The enabling substrate for projecting fix directions into model latent space.
4. **Proposal 032** — [riir-ai/.proposals/032_kimi_k3_native_support.md](../../riir-ai/.proposals/032_kimi_k3_native_support.md). Kimi-K3 native support (MLA + MoE + KDA + SiTU). The model being validated (Phase 6 GOAT gate).
5. **Benchmark 205** — [katgpt-rs/.benchmarks/205_puct_search_vs_moka_win.md](../.benchmarks/205_puct_search_vs_moka_win.md). PUCT search vs Moka (98% win). The moka precedent: bringing the benchmark INTO the model's native inference path.
6. **Research 463** — [katgpt-rs/.research/463_moka_freeze_thaw_lever_audit.md](../.research/463_moka_freeze_thaw_lever_audit.md). The "storage format ≠ capability" caveat. Applies to Layer 3: WASM in-loop validation ≠ resolution capability.
7. **rubrc** — [github.com/oligamiq/rubrc](https://github.com/oligamiq/rubrc). WASM-compiled rustc (WIP). External dependency for Layer 3.
8. **`WasmPruner` / `BomberWasmPruner`** — `katgpt-rs/crates/katgpt-pruners/src/hot_swap.rs`, `katgpt-rs/src/pruners/bomber/wasm_pruner.rs`. The existing WASM-module-as-ConstraintPruner substrate.
9. **`SpecAsPruner`** — `katgpt-rs/crates/katgpt-pruners/src/spec_compile/`. Compiles NL specs into symbolic bitmap rules. The ideological ancestor (symbolic rules, not neural adapters).
10. **code2vec** — Alon et al. (ICLR 2019, [arXiv:1803.09473](https://arxiv.org/abs/1803.09473)). AST path embeddings. P010 borrows the AST→fixed-vector idea (deterministic, not learned).
11. **`tf_loop`** (Plan 136, DEFAULT-ON, GOAT 034) — Training-Free Loop, recursive forward pass, ODE-motivated damped sub-stepping. **Layer 4 step 1.** This IS the "t-pass / recursive forward pass / loop transformer" — the modelless inference loop.
12. **`latent_trajectory_geometry`** (Plan 342, Research 324, arxiv 2606.09287) — measures trajectory length / curvature / drift / bifurcation. **Layer 4 step 2.** G1-G5 ALL PASS.
13. **`closed_unit_compaction` (CUCG)** (Plan 333, DEFAULT-ON) — trajectory compaction at structurally-safe moments. **Layer 4 step 3.** G7 Super-GOAT: "trajectory compaction and shard freeze are the same primitive" — `can_freeze` isomorphism proven. **The load-bearing prior art for Layer 4.**
14. **`committed_field_blend` (FAME)** (Plan 321, DEFAULT-ON) — frozen sigmoid blend computed ONCE from trajectory summary + BLAKE3-committed. Sampling-invariant (FAME Prop. 3). **Layer 4 step 4.** This IS the modelless "fine-tune weight" — frozen trajectory summary, not GD-updated weights.
15. **`KarcShard`** (Plan 308) — Delay-Basis-Ridge Forecaster. `Wout` bit-reproducible from trajectory. **Layer 4 alternative step 4.**
16. **`MerkleFrozenEnvelope`** (riir-neuron-db `freeze.rs`) — BLAKE3-checked atomic snapshot. **Layer 4 step 5.**
17. **`reestimation.rs`** (riir-ai `latent_functor/`) — coherence-driven re-estimation when coherence < tau_reest. **Layer 4 step 6 (self-healing).** Maps DiPOD's "interleave self-distillation when ELBO drifts."

## TL;DR

**Verdict: SPECULATIVE on Layer 3 (rubrc blocker); MODELLESS-VALIDATED on Layer 4 (trajectory freeze composes shipped DEFAULT-ON substrate — validated on synthetic data via T5.1+T5.2+T5.3b, Issues 569+570, AND on real Kimi-K3 via T5.6 G5 + T5.6e value-level discrimination, Benches 014+018).** This proposal sketches four paths to bring Rust-SWE-bench INTO the model's inference loop:
- **Layer 1** (probe corpus) — concrete + actionable today (P010 AST histogram on buggy/fixed pairs).
- **Layer 2** (functional test) — gated on P032 Phase 5 (Kimi-K3 loaded).
- **Layer 3** (WASM pruner) — SPECULATIVE, gated on rubrc maturity + WASM-compilability.
- **Layer 4** (trajectory freeze) — MODELLESS-VALIDATED. Structural discrimination (real vs random): T5.6 G5 PASS, 100% accuracy (Bench 014). Value-level discrimination (perturbation σ≥0.1): T5.6e PASS, 100% accuracy via sequence trajectory SeqStateStats encoder (Bench 018). The depth trajectory (T5.4, bench 012-017) fails value-level discrimination (Bayes-optimal ceiling ~55%), but the SEQUENCE trajectory (final hidden states across prompt tokens with growing KV cache) overcomes it with d_M=14.526 (50× the depth trajectory). Composes shipped DEFAULT-ON primitives (`tf_loop` + `latent_trajectory_geometry` + `committed_field_blend` + `MerkleFrozenEnvelope`). The CUCG G7 isomorphism (trajectory compaction = shard freeze) is the load-bearing prior art. Works even when the model proposes zero valid patches (trajectory geometry is always measurable). **Design constraint (T5.3b): the freezer's summary encoder MUST capture failure-mode-discriminative geometry — raw latent position is insufficient. Design constraint (T5.6e): for value-level discrimination, use the SEQUENCE trajectory with state-magnitude features (SeqStateStats), not the depth trajectory or displacement features.**
- **Layer 4b** (riir-train fallback) — LoRA fine-tune on passing patches if Layer 4 G5 shows insufficient signal.

**The proposal exists to be evaluated and potentially rejected with reasoning.** Next action: Phase 5 T5.1 (synthetic trajectory geometry POC) is the cheapest first step — it tests the core hypothesis (trajectory geometry discriminates) without needing Rust-SWE-bench, rubrc, or Kimi-K3. If T5.1 shows no discrimination, Layer 4 is demoted and the focus shifts to Layer 1 + Layer 2.
