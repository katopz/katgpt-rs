# Research 629: Depth Memory — Exact RLS Across Depth (Linear Depth Memory)

**Status:** GAIN — note + katgpt-rs `.plans/630` (modelless `rls_depth_mem` primitive) + riir-train `.plans/467` (training track) + riir-infer `.issues/049` (delta-roll journal). No code this session. **Delta 2026-10-20 (4090 idle lane): the E2 riir-infer half LANDED** — `gdn_delta_roll` opt-in in riir-infer `9fc4b05` (rank-1 unwind + per-head decay-product conditioning gate, G1 drift 6.3e-6 worst admitted cell, G4 zero-alloc, ~6× cheaper recovery than re-forward; Bench 885; the recorded-u form needs NO `(1 − β‖k‖²)` inversion; issue file closed+removed, seams named for a consumer). **Delta 2026-10-21 (4090 idle lane): E1 LANDED in full — plan 630 CLOSED with GOAT G1–G4 ALL PASS** (Phase 1 `c573d51f3` exact-RLS state + batch-equivalence dev 9.4e-7; Phase 2 `c4d7524f0` WindowedReader, windowed 2.1× on the E=4 sweep; Phase 3 `4586a3348` G2 1.17× delta-rule p50 interleaved + G4 0 allocs + E4 ordering REPLICATES RLS<delta<outer + E5 surprise pinned to its closed forms). **No consumer with a measured gap → stays opt-in per the pre-registered Phase-4 rule** (§2.5/§2.6 both re-confirmed gapless); the opt-in flag + GOAT record are the re-open asset.

> **Source:** "Depth Memory" — David González-Martínez*, Philipp Nazari*, T. Konstantin Rusch, Shiwei Liu (ELLIS Institute Tübingen / MPI Tübingen / Tübingen AI Center / ETH Zürich / Liquid AI), [arXiv:2610.18809](https://arxiv.org/abs/2610.18809), v1 2026-10-13. Code + checkpoints released; full PDF fetched and algebra re-derived (every formula below quoted from paper text).
> **Date:** distilled 2026-10-20.
> **Related Research:** **626 (InfiLoop — the direct predecessor in this distillation line: same author group, the LOOP axis; this paper is the DEPTH axis)**, 624 (associative memory capacity), 538 (GDN W4A4 — "Ternary-Bonsai is scalar-decay GDN"), 070 (GDN-2 erase/write), 006 (Raven — dense overwrite destroys old facts), 024 (δ-Mem — low δ → don't overwrite stable knowledge), 620 (Mamba-3 propagation-operator axis).
> **Related Issues/Plans:** katgpt-rs `.plans/630` (`rls_depth_mem`), riir-train `.plans/467` (LDM training lane), riir-infer `.issues/049` (delta-roll journal).
> **Classification:** Public.

---

## TL;DR

Depth mixing — how representations propagate across transformer **layers** — has been the frozen axis of architecture design while sequence and channel mixing raced ahead. The paper unifies PreLN residuals, AttnRes (softmax attention over a growing cache of layer outputs), and mHC (Sinkhorn-constrained expanded residual stream) as **streams with Read/Write ops**, then instantiates the framework as **Linear Depth Memory (LDM)**: per token, three streams — the residual `x`, a matrix memory `M`, and an inverse covariance `P` — where the Write is **exact recursive least squares** (Sherman–Morrison): `κ = Pk/(λ + kᵀPk)`, `u = v − Mk`, `M ← M + uκᵀ`, `P ← (P − κkᵀP)/λ`. This exactly solves a weighted-ridge depth regression with per-layer forgetting. The systems insight that licenses everything: **the sequence axis needs parallel scans (why DeltaNet/MesaNet approximate), but the depth axis is naturally sequential — the exact solver is free there**. Two hardware tricks (state unwinding `M_{l−1} = M_l − u_lκ_lᵀ`; windowed writes `M_j·x = M_b·x + Σu_i(κ_iᵀx)`), lowest ppl vs tuned PreLN/mHC/AttnRes at 400M and 1.5B.

**Distilled for katgpt-rs (modelless, inference-time):**
The workspace ships an associative-memory **ladder** with a hole in it. Shipped: the outer-product write (Raven-class Hebbian), the **delta-rule online rung** (`katgpt-core/src/delta_mem/state.rs::write` — `S'[i,:] = (1−β)·S − β·pred·k + β·v·k`, one gradient step, no covariance), the **batch ridge apex** (`hebbian_kernel_memory.rs` whitened closed-form; `linalg/ridge_solve.rs`), and once already GOAT-passed the **never-materialize-intermediates algebra** (`gdn_tree_verify` WS₀ folding). Missing: the **exact online rung — RLS**. Zero grep hits for the vocabulary `recursive_least|sherman|morrison|inverse_cov|least_squares` across `*.rs` in katgpt-rs/crates, riir-ai/crates, riir-infer, riir-neuron-db/src, riir-train/crates (targets excluded) outside three non-coverage hits: `depth_invariance.rs` (scalar slope), `katgpt-attn-match` (batch value fitting), `set_admission.rs` (Sherman–Morrison for admission log-det — different consumer, see §2.3). The extraction is pure algebra: `RlsMemoryState` (rank-8 square: M[64] + P[64] + scratch) with a G1 gate the approximation rungs can never claim — **the online state provably equals the batch solve of the decayed-ridge objective RLS minimizes** (`Σ λ^{N−i}‖vᵢ − Mkᵢ‖² + (λ^N/α)‖M‖²`, built as a weighted Gram + cross-cov in f64 and handed to the shipped `linalg::ridge_solve_direct_f64`, which takes precomputed `gram_reg`/`cov`). The paper's own ablation ladder (RLS > delta > outer, Δppl 0.00/0.06/0.07 nats) says our shipped online rung is the measured loser to this extraction.

---

## 1. Paper core findings

1. **The DM framework.** Depth propagation = streams `S_l = (S⁽¹⁾,…,S⁽ᴾ⁾)` with per-layer, per-token `Read(S_{l−1}, q_{l−1})` and `Write(S_{l−1}, k_l, v_l)`. PreLN: one stream, additive. AttnRes: growing KV cache, softmax-attend (explicit mixer). mHC: fixed-size expanded stream, `X_l = X_{l−1}H^res + v_l k_lᵀ` — queries and keys generated from the state itself (implicit mixer; formally a linear associative memory over depth).
2. **LDM.** Streams `(x_l, M_l, P_l)`; per group (G=16, D_k=8, D_v = D/G): Read `x + γ⊙M(g⊙q)`; Write = exact RLS above with per-layer learned forgetting `λ = σ(logit)` (init 0.99), importance `ω = 2σ(·)`, fixed ridge α=1, L2-normed content-derived queries/keys from shared `SiLU(Aff(RMSNorm(x)))` latents (H=32–64). Addresses ~1% param overhead.
3. **The axis law.** Sequence-axis exact RLS (MesaLayer, von Oswald 2024) is sequential → training-unaffordable → the field adopted delta-rule/DeltaProduct/CG approximations. Depth is *already* sequential → exact RLS costs nothing extra. Same math, different systems constraint.
4. **Results.** 400M (50B FineWeb-Edu) and 1.5B (100B): lowest held-out/WikiText-2/LAMBADA ppl among LR-swept PreLN/AttnRes/mHC; competitive zero-shot. Throughput 70–90% of PreLN (CuTe kernels, B200) — beats Full AttnRes everywhere, mHC in most settings. Ablations (400M@15B, Δ held-out nats vs LDM): memory-only layer input +0.131, kill residual stream +0.063–0.087, **delta +0.004 / outer +0.005 (RLS wins)**, store x_l not δ_l +0.024, static queries +0.014, groups G=8 slightly beats G=16 (−0.003), gates/forgetting ≈ 0 at this scale.
5. **Hardware tricks.** (a) **Unwinding**: the rank-1 update is exactly invertible — `M_{l−1} = M_l − u_lκ_lᵀ` — so backprop reconstructs states from the final state + saved per-step contexts (u, κ, k, q̃, λ, P's symmetric triangle) instead of storing every M_l. (b) **Windowed writes** (E=4): inside a window, reads compute from the window-base state + rank-1 corrections — intermediate states never materialize. Same trick class as ReplaySSM (sequence axis, inference) applied across depth in training too.

## 2. Distillation

### 2.1 The ladder this completes (the fusion finding)

| Rung | Update | Signal consumed | Status in workspace |
|---|---|---|---|
| Outer product / Hebbian | `M ← λM + vkᵀ` | (k,v) raw | Shipped family (Raven consolidation class; the paper's measured worst rung) |
| Delta rule | `M ← λM + (v − λMk)kᵀ` | (k,v) + prediction error | **Shipped**: `delta_mem/state.rs` (Plan 008 substrate, feature-hashed keys) |
| N/D gated summary | `h_t = N_t/D_t`, `γ = σ(s_t − ln βD)` | (z, content score) | **Filed**: Research 626 / katgpt-rs Issue 949 (loop axis) |
| **Exact RLS** | `M ← M + uκᵀ; P ← (P − κkᵀP)/λ` | (k,v) + full covariance | **MISSING — this note** |
| Batch ridge apex | `M = B·A⁻¹` (Cholesky) | full window | Shipped: `hebbian_kernel_memory.rs`, `linalg/ridge_solve.rs` |

The paper's measured ordering (RLS > delta > outer at fixed everything else) says the missing rung is the *online* optimum, and its SPD-by-construction property (`P₀ = αI`, SPD downdate + positive rescale ⇒ `λ + kᵀPk > 0` always) is a *stronger* numerical contract than the shipped Cholesky path's documented near-singular-Gram hazard.

### 2.2 Extractions (Path 0 inventory — merged from both advocates)

| # | Paper quantity | Modelless? | Host | Gate |
|---|---|---|---|---|
| E1 | RLS write rule (κ, u, M, P) | ✅ pure algebra | `katgpt-core/src/delta_mem/rls.rs` (new `RlsMemoryState` beside `DeltaMemoryState`, feature `rls_depth_mem`) | G1 = **batch-equivalence against the decayed-ridge objective** (weighted Gram `Σλ^{N−i}kkᵀ + (λ^N/α)I` + cross-cov built f64 → `ridge_solve_direct_f64`; pinned f32 tol) + SPD at every step (symmetric-by-construction P update required — see Plan 630 T1.2); G2 rank-8 write+read vs `DeltaMemoryState`; G3 default byte-identity; G4 zero-alloc scratch |
| E2 | Unwinding `M_{l−1} = M_l − uκᵀ` | ✅ exact algebra, f32 drift measurable → pin; **M-only** — P is SAVED (triangle in the ctx), never unwound (inverse-SM P-restore divides by λ, unstable; the paper's own backward pass saves P instead) | `delta_mem/rls.rs::unwind_step` + `p_restore`; **riir-infer GDN states** (`.issues/049` — gated form needs the α-division conditioning gate); ndb freeze+journal class | G1 round-trip: W writes → W unwinds → (with `p_restore`) M and P both match, pin measured ε; M-only = read-rollback-only, doc-named |
| E3 | Windowed writes `M_j x = M_b x + Σu_i(κ_iᵀx)` | ✅ exact associativity | `delta_mem/rls.rs::WindowedReader` (corrections summed in pinned write order) | G1 windowed == materialized; G2 traffic at E=4; G4 |
| E4 | Ordering laws (RLS>delta>outer; δ_l>x_l; content queries>static) | ✅ testable predictions | GOAT benches re-test on our workloads; honest negatives are findings | G1/G2 measured, not asserted |
| E5 | Depth-regression residual `u = v − Mk` as surprise scalar | ✅ closed-form fit at runtime | composes with shipped `delta_mem` surprise gate (`DEFAULT_THETA_SURPRISE`); latent_functor bridge pattern *if* a game consumer materializes | G1 signal sanity; G3 inert-boot hash pin |
| E6 | Learned λ, ω, g, γ | ⚠️ substituted: fixed schedules (λ=0.99 const, ω=g=γ=1) — the mechanism ships, not the tuned values | `rls.rs` config | G1 SPD under schedule + degenerate-gate equivalence |
| E7 | Groups G | complexity without measured need at our scales (paper: −0.003 nats) — **discarded** | — | — |
| E8 | LDM layer inserted into frozen GGUF serving models | **discarded** — architecture-level co-adaptation from scratch; frozen-trunk retrofit ceiling documented; laya G5 bit-parity excludes additive lanes. Training track owns the legitimate path (`.plans/467`) | — | — |

### 2.3 Signal-diffs (§3.6, one read each — coverage dismissals audited)

- **`delta_mem/state.rs::write`** (verified): consumes `(k, v)` with per-row coupled decay β and prediction-error gating — the same observation signal as RLS but a *first-order* solver; no P stream, no exact-solution property. Partial coverage: the ladder's bottom online rung, measured below RLS in the paper's own ablation. Does NOT kill E1 — it is E1's baseline.
- **`set_admission.rs::sherman_morrison_rank1`** (verified): maintains `(I+G)⁻¹` for marginal log-det gain in greedy set admission. Same rank-1 inverse-maintenance *kernel*, entirely different consumer — no associative memory read, no regression, no forgetting. Kernel precedent (the O(d²) rank-1 update idiom is house vocabulary), not coverage.
- **`hebbian_kernel_memory.rs`** (advocate-verified): batch closed-form ridge-whitened construction — the same objective at the *batch* apex; its own doc names `delta_mem` as "the online analog of this primitive's batch construction". E1 is the exact-online middle the doc's framing implies exists.
- **`gdn_tree_verify` WS₀ folding** (advocate-verified): "the S₀ contribution pre-multiplied into the RHS, so W is never materialized" — the same never-materialize algebra class as E3, already GOAT-passed on a different axis. Precedent, not blocker; E3 is the second instance of a proven house trick.
- **Research 626 (InfiLoop)**: same author group (Shiwei Liu/ELLIS Tübingen), adjacent *loop* axis, first-order normalized summary (convex-hull bounded, `γ = σ(s_t − ln βD)` sigmoid identity). LDM is second-order (regression with covariance). Does not cover: different axis, different solver class, different state shape. The two notes jointly establish a workspace **axis law**: *any naturally-sequential axis (depth, loop iterations, pipeline stages) admits the exact solver the sequence axis cannot afford*.

### 2.4 Latent-space reframe

Per-entity RLS memory across pipeline depth: the memory + covariance streams are think-brain latent state (never synced — sync-boundary law); the read `γ⊙M(g⊙q)` is a sigmoid-gated latent→latent op; only derived scalars (e.g. the E5 surprise ‖u‖ through `exact_sigmoid`) would ever cross. The paper is sigmoid-native end to end (`2σ` gates, `σ(logit)` forgetting) — zero conflict with the sigmoid-not-softmax law; `exact_sigmoid` is the drop-in.

### 2.5 Game-context reframe (written per §1 step 4; no speculative filing)

The depth axis maps onto the cognition pipeline's sequential stages (perception → belief → drives → salience → FSM) and onto consolidation cycles: a per-entity RLS memory of (stage-output key → stage-output value) associations would let later stages query earlier-stage structure with exact retrieval and covariance-weighted forgetting — the matrix-memory generalization of the shipped scalar `sigmoid(−λ·Δt)` belief-decay law (old observations lose *precision*, not just weight). Two-brain compliant: memory local to the entity, never synced. **Honest status: design-law level, no measured consumer problem** — filed nowhere; revisit only if a real pipeline-stage retrieval gap surfaces.

### 2.6 Consumer-context reframe — the healer (priority #2, same evidence bar)

The healer's pipeline (retrieve → draft → prune → verify) is short-depth and naturally sequential — the axis law applies verbatim. An online RLS memory over (span-embed key → rule/outcome value) could serve as a latent, self-updating selection signal (self-adaptive track). **Honest status: overlaps shipped selection surfaces** (rule_embed_poc, basis_cert measured NEGATIVE on formatting axes; Elo/BetaPosterior + Warm-tier trajectory store own outcome memory) and no documented retrieval gap maps to it (retrieval floors pinned and healthy). Noted here, not filed.

## 3. Verdict

**GAIN** — per-track, never pooled:

- **Track (a) modelless inference**: GAIN → `.plans/630` (`rls_depth_mem`: E1+E2+E3 behind one feature flag, G1 batch-equivalence as the headline gate). New-to-stack (grep-verified: zero RLS in-tree), force-multiplier (≥2 pillars: delta_mem family + GDN lane + freeze/journal class), but no end-to-end consumer gain measured yet → not GOAT/Super-GOAT until the plan's gates run. The G1 provable-equivalence property is the strongest correctness claim available to any primitive in this family — that is the note's one sentence of genuine excitement, and it is testable, not vibes.
- **Track (b) self-adaptive runtime**: recorded fusion ideas only (E5 surprise scalar; §2.6 healer signal) — no filing; both overlap shipped surfaces without a measured gap.
- **Track (c) model-based trained weights**: GAIN → riir-train `.plans/467` (Path 0.5 satisfied: tiered lane T0 probe ~1 GPU-h → R retrofit on the on-box kimi-0.4B trunk 4–8 GPU-h → A mechanism ~90M/1B tokens 20–60 GPU-h with STOP clause → B defensible ~170M/4B 80–180 GPU-h; paper-faithful 700–1500 out of scope). Prices are FLOP-model projections anchored to in-box records — T0 exists to convert them to measurements. Execution is owner-gated per the standing Bonsai-pivot priority; the plan exists so the training task is neither deferred-by-default nor launched unpriced.

**MOAT gate (katgpt-rs):** fundamental/primitive axis — depth mixing is a transformer-stack primitive (layers slot in the per-stack ledger); no game/chain/shard IP in the extraction. Fits. **Not** routed to riir-infer (its `deltanet/` gains only the journal, filed as its own issue), not to riir-ai (no runtime consumer filed).

**Tier: not Super-GOAT** — Q1 in-world prior art is the paper itself (we claim first-workspace-consumer, never mechanism novelty; same posture as Research 626); Q2 new-capability-class is arguable at the primitive level (provable exactness) but no system behavior demonstrated; Q3 product selling point: none yet; Q4 force multiplier: yes.

**Advocate panel record:** No-GD advocate returned GO with a 10-row inventory and three in-repo feasibility precedents (delta_mem's learned-projection→hashing substitution, hebbian_kernel_memory's closed-form house style, gdn_tree_verify's GOAT-passed folding). Model-based advocate returned TRAIN with the tiered pricing above, the γ=0-init LoRA-shaped retrofit analysis, and the honest caveats (L=8 retrofit caveat pre-registered: a null there does NOT falsify the paper; laya excluded by G5; eager-kernel penalty on Ada disclosed). Discards audited: E7 groups (−0.003 nats, NotBenchable-class micro-effect), E8 frozen-GGUF insertion (co-adaptation requirement), npc_brain consumer (brain depth too small — deferred consumer, recorded in `.plans/467`).

**Prior-art searches (§4, run before verdict):** (1) headline verbatim — only the paper itself + its cited lineage (Test-Time Regression JMLR'26, MesaLayer/MesaNet, RLS-matrix-forgetting control theory, MoR); (2) landscape — HC/ICLR'25, mHC/DeepSeek ICML'26, xHC (2607.14530), AttnRes (2603.15031), mHC-lite (2601.05732); (3) RLS-in-deep-learning — training-time RLS for CNN training/pruning (different surface), Preconditioned DeltaNet (sequence-axis curvature preconditioner, 2604.21100); (4) internal-first grep: `2610.18809|depth memory` across all 16 `.research/` + `.plans/` folders → one false positive (riir-train/424 mentions DeltaNet). No competing depth-axis-RLS follow-up in the paper's first week.

**PoC status:** none this session — no quality claim is made that requires one (§3.6); the plan's Phase-1 G1 batch-equivalence test IS the first PoC, and it is a proof-shaped test rather than a head-to-head. The paper's ppl claims are explicitly NOT claimed modelless (contamination of the claim; §3 of the No-GD advocate's honesty clause).

## Summary

**(1) Original task:** Distill arXiv:2610.18809 "Depth Memory" per the research skill — classify, fuse, verdict, file.

**(2) Accomplished:** Full pre-flight (16 `.research/` folders derived + listed; 6 src-tree vocab greps; internal-first arXiv grep — one false positive; 3-track adversarial panel run in the same batch as §4 searches; all product-set AGENTS.md read). Distilled LDM into the workspace's associative-memory ladder; identified the missing exact-online rung (grep-verified zero RLS in-tree; Sherman–Morrison kernel precedent in `set_admission.rs` is a different consumer); ran §3.6 signal-diffs on five cousins (delta_mem, set_admission, hebbian_kernel_memory, gdn_tree_verify, Research 626). Verdict **GAIN** per-track. Files: `katgpt-rs/.research/629` (this note), `katgpt-rs/.plans/630` (`rls_depth_mem`), `riir-train/.plans/467` (tiered LDM training lane), `riir-infer/.issues/049` (delta-roll journal). Highwaters bumped (628→629, 629→630, 466→467, 048-local→049-local); dual_allocation_gate green on all three repos before allocation.

**(3) What remains:** Everything in the three filed artifacts — none executed this session (note + plans + issue only, per the 626 precedent). Plan 630 Phase 1 (RlsMemoryState + G1) is the next actionable unit; riir-train 467 T0/R are owner-gated on the Bonsai-pivot priority; riir-infer 049 is unclaimed.

**(4) Active plan state:** `katgpt-rs/.plans/630_rls_depth_memory.md` — Active, Phase 1 not started. `riir-train/.plans/467_ldm_depth_memory_training_lane.md` — Active/owner-gated, Tier 0 not started. `riir-infer/.issues/049_gdn_delta_roll_journal.md` — Open, unclaimed.
