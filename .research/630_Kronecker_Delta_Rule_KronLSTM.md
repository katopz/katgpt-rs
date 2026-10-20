# Research 630: Kronecker Delta Rule — Scalar-Target Associative Memory (KronLSTM)

**Status:** GAIN — note + katgpt-rs `.issues/954` (modelless `kron_memory` primitive) + riir-train `.plans/468` (E0 training probe). No code this session. Revised once after the §5 verdict round (erase-exclusivity scoped by the G0 baseline gate; independence/rank caveats on the analytic gauges; undo/identity fp tolerances; Plan 468 repriced).

> **Source:** "Kronecker Delta Rule for Long Context Linear Attention" — Anamaria-Roberta Hartl*, Levente Zólyomi*, Michael List, Günter Klambauer, Sebastian Böck, Sepp Hochreiter (JKU Linz / NX-AI), [arXiv:2610.26834](https://arxiv.org/abs/2610.26834), v1 2026-10-19. Code + 1.5B checkpoints open: github.com/NX-AI/KronLSTM (MIT, FLA snapshot pinned `78a39922` — kernels, layer, configs public), HF NX-AI/kronlstm. Full PDF fetched; update algebra re-derived and quoted below.
> **Date:** distilled 2026-10-20 (panel: prior-art agent + No-GD advocate + model-based advocate; §5 verdict ping-pong gate before commit).
> **Related Research:** **629 (Depth Memory exact-RLS — the ladder's other new rung, landed the same day; the critical signal-diff)**, 070 (GDN-2 erase/write — the paper's strongest baseline, our shipped `gdn2_attention`), 624 (associative memory capacity), 538 (GDN W4A4 — "Ternary-Bonsai is scalar-decay GDN"), 024 (δ-Mem), 006 (Raven), 455 (Hebbian kernel memory), 482 (drift-segmented linear attention).
> **Related Issues/Plans/Benches:** katgpt-rs `.issues/954` (this extraction), `.plans/630` (`rls_depth_mem`, 629's rung), `.plans/105` (GDN2), `.plans/619` + `.docs/03_memory/lattice_memory.md` (LatticeMemory delta-rule cells), Benches 919/938 (delta-rule capacity + measured erosion laws), Bench 308 (Kronecker solve discipline), Issue/Bench 839 (`kron_apply`), riir-train `.plans/468` (E0 probe).
> **Classification:** Public.

---

## TL;DR

KronLSTM's Kronecker Delta Rule is the **scalar-target member** of the delta-rule family our stack already runs at inference: where DeltaNet/GDN/KDA/GDN-2 correct a **vector** readout (`Sk_t → v_t`, imposing d constraints per pair, interference gated by key overlap alone), the Kronecker rule corrects the **single scalar bilinear readout** `v_tᵀS k_t` toward a target `c` along the pair's own outer product `v_t k_tᵀ`:

```
S ← S − β · (vᵀSk − c) · v kᵀ          (erase the stored scalar, write c)
```

Vectorized (`a = k ⊗ v`, `s = vec(S)`) this is **Widrow–Hoff LMS on the Kronecker feature with a scalar target** — a relaxed Kaczmarz projection onto `{S : ⟨vkᵀ, S⟩_F = c}`. Three consequences matter for us:

1. **No covariance stream needed.** Each observation imposes exactly ONE constraint, so the Newton step `β* = 1/(‖k‖²‖v‖²)` is **closed-form from the feature norm alone**. Exact RLS (Research 629's rung) on the vectorized d²-state would need `P ∈ R^{d²×d²}` — infeasible. The scalar-target member is precisely the rung where optimal-per-observation stepping is free. **This is why the rule matters, not merely that it is new.**
2. **Doubly-gated interference (the paper's cleanest un-pre-empted property).** The damage to another stored pair `(kᵢ, vᵢ)` is `−β·residual·(vᵢᵀv)(kᵀkᵢ)` — the PRODUCT of value overlap and key overlap. Orthogonality in EITHER factor fully protects a stored association (so slot protection is a property of DESIGNED, near-orthogonal addresses — not unconditional); DeltaNet's interference is gated by key overlap alone. And `c = 0` with the Newton step **deletes one binding surgically** — with the collateral on every other pair computable BEFORE the write. **Honest scoping (verdict-round fix):** erase-with-audit is not Kron-exclusive — a vector-target delta erase (write `v = 0` at the Newton step, `M ← M − (Mk)kᵀ/‖k‖²`) also admits closed-form pre-write collateral, and with one-hot/orthogonal address codes a Kron cell degenerates to a plain table where unlearning is trivial. What ONLY the scalar-target member adds is the **doubly-gated** erase (collateral requires overlap in BOTH factors). Issue 954's G0 baseline gate makes Kron beat a vector-delta-erase-with-audit and a one-hot table at matched state bytes on distributed keys — or not promote.
3. **Analytic stability for free.** The memory is a Markov chain whose stationary mean is EXACTLY the least-squares memory (no O(β) bias), covariance O(β), and `E‖S‖²_F ≤ 4E[c²]/λ_min(Ā)` uniformly in t — the bounded envelope our LatticeMemory sizer currently MEASURES (Bench 938) becomes predictable. **Independence caveat (verdict-round fix):** the two-small-marginals form `λ_min(Ā) = λ_min(K̄)λ_min(V̄)` and the `K̄⁻¹BV̄⁻ᵀ` solve hold only under INDEPENDENT keys/values — associative streams are correlated by construction (the correlation IS the association), so the extraction ships these as lower-bound heuristics with a measured-violation arm and a ridge/replay fallback for the general case. Retention horizon ~d× longer than DeltaNet's under the same assumption.

**Distilled for katgpt-rs (modelless):** a new cell type beside the shipped vector-target rungs — the **pair-addressed scalar table** (`d_k·d_v` scalar slots per cell, protected where the address factors are near-orthogonal — designed addresses, not unconditional) with write/rewrite/**surgical erase (audited, G0-baselined)**, analytic health gauges (norm-budget alarm, retention half-life), LS-snap consolidation, undo-last-write (tolerance-pinned by default; snapshot variant when bit-exactness is required), and a write-conflict predicate enabling order-independent parallel writes. Consumers by the priority ladder: game runtime (grudge/relation association weights — priority #1), healer rule-binding trust memory (priority #2), consolidation-replay/ternary WY kernel (priority #3). Training track: E0 mechanism probe filed as riir-train Plan 468 — the paper's own architecture is a SECONDARY track (no KronLSTM checkpoints exist in the fleet; see §3 track split).

---

## 1. Paper core findings

1. **The rule** (main Eq. 3 / App. A1): `S ← S − β_t (v_tᵀS k_t − c_t) v_t k_tᵀ`, with the decoupled form `c̃ = c/β` splitting erase (β_t alone scales how much of the stored scalar is erased) from write (c_t alone scales what is written). KronLSTM gates all three: diagonal key-side forget `F_t` (sigmoid^α, Mamba parameterization), sigmoid erase `β_t`, sigmoid write `c_t`, sigmoid output gate; ℓ2-normalized q/k/v; short causal conv; per-head RMSNorm readout. **No normalizer state, no stabilizer state** (unlike mLSTM) — the boundedness is a theorem, not a mechanism.
2. **Property ladder** (App. A3, proofs verified by the authors; AI-use statement discloses Claude-assisted derivations, independently checked):
   - Error contraction iff `0 < β < 2/(‖k‖²‖v‖²)`; Newton step `β* = 1/(‖k‖²‖v‖²)` meets the target in ONE update; Fejér monotonicity — every update moves CLOSER to every memory consistent with the written association.
   - Interference on `(kᵢ, vᵢ)`: `Δ(vᵢᵀSkᵢ) = −β·residual·(vᵢᵀv_t)(k_tᵀkᵢ)` — product of overlaps; either factor ≈ 0 ⇒ protected. Writes whose readout is already correct cause ZERO interference regardless of target.
   - Capacity: `d_k·d_v` scalar associations per cell (vs d value vectors for DeltaNet) — a **pair-addressed scalar table**: write by rank-1, read by bilinear probe `vᵀSk`, rewrite by re-presenting the pair, **delete via c = 0**.
   - Distinguished targets: `c = vᵀk` (correlation) makes `S = I` a universal fixed point — S then parameterizes learned deviations from the inner product; `c = ‖v‖` (norm target) makes every exact retrieval memory a fixed point.
   - **Markov chain convergence** (Thm 1): stationary distribution π_β exists, unique, geometrically approached in W₂; stationary mean = least-squares memory `S_LS = K̄⁻¹BV̄⁻ᵀ` EXACTLY (unbiased); covariance O(β) via a Lyapunov equation; misadjustment `E[L] − L_min = β/2·E[ρ²/‖a‖²] + O(β²)` (classical LMS trade-off).
   - **Norm bounds** (Thm 2): `(E[c²]−E[ρ²])/λ_max ≤ ‖S_LS‖²_F ≤ (E[c²]−E[ρ²])/λ_min`; running iterates `E‖S_t‖²_F ≤ (1+q^{t/2}_β)·E[c²]/λ_min(Ā) ≤ 4·E[c²]/λ_min(Ā)` from step one, decaying geometrically to the steady-state constant.
   - **Retention** (App. A5): for independent keys/values `Ā = K̄ ⊗ V̄`, so `λ_min(Ā) = λ_min(K̄)·λ_min(V̄)` — retention horizon `1/(β·λ_min(Ā)) ≥ d × DeltaNet's`. Retention is BOUGHT from adaptation (the same `Mᵗ` governs both terms) — the honest trade.
   - **Superposition** (Prop. A3): with `S₀ = 0` the map target-sequence → final memory is LINEAR — memories written with different targets over the same pairs superpose exactly. (Consequence: undo of the last write is exact O(1); undo of an arbitrary suffix is replay-based, not claimed further.)
   - **Chunkwise WY** (App. A8): the solve matrix is `N = diag(β)·(G_kk ⊙ G_vv)` — Hadamard product of a decay-weighted key Gram and the value Gram; the `d_k·d_v` features are NEVER formed (Kronecker–Gram identity `(kᵢ⊗vᵢ)ᵀ(kⱼ⊗vⱼ) = (kᵢᵀkⱼ)(vᵢᵀvⱼ)`); unit-lower-triangular solve; pseudo-values are scalar multiples `u_r v_r` of the values; incoming memory enters through C scalars, not C vectors. Verified to machine precision by the authors across chunk lengths/initial memories/β ranges.
3. **Results:** 1.5B/100B and 400M/40B on FineWeb-Edu — beats GDN, KDA, Mamba-3, **GDN-2** (our shipped mixer, Research 070/Plan 105) on Avg-6 commonsense + LAMBADA ppl, as linear AND SWA-hybrid; strongest on long-context recall (RULER NIAH: best-or-second on every natural-text task, margin GROWS with haystack size — the interference prediction landing); 32k extension keeps the lead over GDN-2 on all natural-text tasks from 16k; state tracking via `β·‖a‖ ∈ (1,2)` negative eigenvalues (Grazzi et al. mechanism, inherited); H100 throughput 36.8–44.5K tok/s = 11–18% faster than GDN-2, within 4–8% of KDA. Weakness disclosed: S1 (repeated-noise-sentence NIAH) underperforms — residual accumulation on recurring keys.

---

## 2. Distillation

### 2.1 The ladder + the new rung (the fusion finding)

Research 629 (landed this same day) mapped the associative-memory ladder and filed the exact-RLS rung. The Kronecker rule is not a BETTER rung on that axis — it is a **different target contract** on the same ladder:

| Rung | Update | Target | State | Covariance | Workspace |
|---|---|---|---|---|---|
| Outer product / Hebbian | `M ← λM + vkᵀ` | vector (implicit) | d×d | none | Shipped (Raven class) |
| Delta rule | `M ← λM + (v − λMk)kᵀ` | **vector** `Mk ≈ v` | d×d | none (1 gradient step) | Shipped: `delta_mem/state.rs` |
| Gated summary (N/D) | `h = N_t/D_t` | (z, content score) | 2×d | none | Filed: Research 626 / Issue 949 |
| **Scalar-target Kronecker delta (THIS NOTE)** | `S ← S − β(vᵀSk − c)vkᵀ` | **scalar** `vᵀSk ≈ c` | d_k×d_v | **none needed — Newton step closed-form** | **MISSING → Issue 954** |
| Exact RLS (629) | `M ← M + uκᵀ; P ← (P−κkᵀP)/λ` | vector | d×d | full `P ∈ R^{d×d}` | Filed: `.plans/630` (`rls_depth_mem`) |
| Batch ridge apex | `M = B·A⁻¹` | vector | — | full window | Shipped: `hebbian_kernel_memory.rs`, `linalg/ridge_solve.rs` |

**Why the scalar-target rung cannot be served by the RLS rung (the signal that makes it a distinct primitive, not a duplicate):** RLS regresses the VECTOR `v` on key `k`. The Kronecker rule regresses the SCALAR `c` on the rank-1 feature `a = k⊗v`. Running RLS on the vectorized memory to impose `aᵀs = c` exactly would carry `P ∈ R^{d²×d²}` — d⁴ state, dead on arrival at any scale. The scalar-target member gets the optimal per-observation step (`β*`) for FREE because the feature is rank-1 and the loss is 1-D per observation. Capacity trade is the flip side: d² scalar slots (mutually protected in BOTH axes) vs d vector slots — complementary, not competing. **Cell-type slots, no incumbent demoted** (the `kron_memory` flag adds a cell; it does not replace `delta_mem` or LatticeMemory cells).

### 2.2 Prior-art landscape (§4 census — what is pre-empted, what is not)

The prior-art agent ran ~12 searches + 4 full-text fetches (cutoff 2026-10-19). Honest scorecard, because our extraction inherits the paper's weaknesses:

| Claim | Status | Pre-empting work |
|---|---|---|
| The exact rule (scalar bilinear target, pair-outer-product correction) | **Novel** — no published instance, deep-era or classical | — (classical lineage uses the VECTOR target: Widrow–Hoff 1960, Kohonen 1972, Kaczmarz 1937, RLS-with-forgetting 1982) |
| Kaczmarz / relaxed-projection lens | **Pre-empted** | TTR 2501.12352 (JMLR 2026) states the delta rule AS a Kaczmarz projection (vector-target, full-text verified); KLA 2605.08587 imports Kaczmarz into GDN (2026-05) |
| Steady-state / λ_min norm-bound SHAPE | **Class-level** | TTR App A (λ_min(KᵀK) bound), RLS-with-forgetting convergence (Johnstone 1982); the scalar-feature instance is a derivation |
| d² capacity table | Class-level-ish | Classical scalar-vs-retrieval capacity separation exists in cognitive literature (Bogacz & Brown 2003 per the paper) |
| **Doubly-gated interference (key×value overlap product)** | **Un-pre-empted** | Nothing found — KDA/GDN-2 gate channels, not overlap products |
| Chunkwise WY via Hadamard Gram | No direct prior found; mathematically derivative of Yang et al. 2024 WY machinery | — |
| "Targeted editability / memory editing" framing | **Pre-empted (18 days)** | 2609.33093 "How Linear Attention Remembers" (analysis, no rule); GDN-2's own press framing |
| Closest class-level threat | Q-Delta 2606.08804 (ICML 2026) — query-conditioned mixed readout errors in the write, same analysis+chunkwise packaging; NOT the scalar pair-target | GSA2 2610.02816 — two-sided (Oja + DeltaRule-2) correction |

Zero citations of KronLSTM pre-cutoff; no scalar-target variant anywhere before 2026-10-19. **For our note this means:** the modelless extraction's novelty rests on (a) the audited-surgical-erase OPERATION (ours, fused), (b) the analytic health laws applied to OUR sizers (ours), not on the rule or the Kaczmarz lens (the paper's). Cite Q-Delta/TTR/KLA/2609.33093 wherever the framing leans on theirs.

### 2.3 Extractions (Path 0 inventory — merged from the No-GD advocate, coverage-audited)

| # | Quantity | Modelless? | Host | Gate | Coverage verdict |
|---|---|---|---|---|---|
| E1 | **KronCell core**: `write_newton` (β* = 1/(‖k‖²‖v‖²), one dot + one rank-1 axpy), contraction governor (`0 < β‖a‖ < 2` enforced, divergent arm as negative test), read `(value, analytic CI)` | ✅ pure algebra | `katgpt-core/src/kron_memory/` (new cell type, feature `kron_memory`, fixed-size `[[f32; DV]; DK]` Pod — 16 KB @ 64×64, zero-alloc, BLAKE3-committable through the freeze-envelope pattern) | G1 property tests from the theorem statements; G2 ns bar; G3 inert-boot hash pin; G4 counting allocator (bench_919/938 precedent) | **NEW cell type** — `delta_mem/state.rs::write` is the vector-target baseline (signal-diff §2.4); not coverage |
| E2 | **Interference kernel + two-factor protection**: pre-write collateral audit `Δᵢ = −β·residual·(vᵢᵀv)(kᵀkᵢ)` over the cell's enumerable pairs; same-key/different-value writes FREE | ✅ two dot products per pair, cell-local | `KronCell::collateral_audit`; address-design law documented | G1: kernel equality to fp tolerance; orthogonal-v and orthogonal-k → zero interference | **NEW** — no shipped memory computes a pre-write damage bound |
| E3 | **Audited surgical unlearning** (HEADLINE, scoped by G0): `c = 0` + Newton step deletes one binding; collateral ≤ τ else refuse-or-repair (write back each collided pair's old readout — closed-form targets, one extra delta write each) | ✅ | `KronCell::unlearn_audited` → `CollateralReport` | **G0 first**: Kron-erase must beat vector-delta-erase-with-audit AND a one-hot table at matched state bytes on distributed keys (with orthogonal codes the table wins and the flag stays opt-in forever); then G1: unlearn one pair → all other readouts move ≤ τ; repair ≤ ε | **NEW on our memory stack, falsifiably**: decay memories forget globally and vector-target erase reproduces the audit — the Kron-exclusive part is the DOUBLE gating, which is exactly what G0 measures |
| E4 | Identity fixed point / write elision: `c = vᵀk` stream inert at S = I; `residual ≈ 0` writes are no-ops | ✅ | warm-start prior; write elision in the hot path | G1: hash pin — N inert writes leave S bit-identical | **NEW (small)** — a free no-op detector |
| E5 | **Analytic health** (lower-bound heuristics — verdict-round caveats): norm-budget alarm `‖S‖²_F > θ·4·Ê[c²]/λ̂_min` (regime-shift detector — i.i.d.-bound breaks ⇒ the stream LEFT its regime); retention half-life `H = ln2/(β·λ̂_min(Ā))` from streaming EMA marginals — `λ̂_min` tolerance-thresholded on the OBSERVED subspace (rank-deficient game streams otherwise send H → ∞); the `Ā = K̄⊗V̄` factorization is exact only under key/value independence, correlated streams get a measured-violation arm and the gauge ships as a heuristic, never a guarantee | ✅ via EMA marginals + shipped `jacobi_eigen` (Bench 308 Kronecker-solve discipline) | `KronCell::health` → feeds sigmoid drives (never hard edges); uncertainty claims face the conformal floor (G5) | G1: 0 false alarms on i.i.d. stream, fires within n steps on injected shift; rank-deficient arm keeps gauges finite; measured decay within band of predicted H on OUR streams (E8) | **NEW** — Bench 919/938 MEASURE these envelopes; nothing PREDICTS them per-cell at runtime |
| E6 | LS-snap consolidation: `S := K̄⁻¹BV̄⁻ᵀ` resets accumulated O(β) jitter — **exact only under independent keys/values (designed addresses)**; the general correlated case falls back to the shipped replay/ridge path (`linalg::ridge_solve`) with ridge regularization for rank-deficient streams; steady state = recency-weighted LS under the forget gate | ✅ | `KronCell::consolidate_ls`; sleep-cycle hook (riir-ai) / offline healer pass | G1: `‖mean(S_tail) − S*‖/‖S*‖ ≤ ε(β)` on a seeded stream + the independence-violation arm measures the Kronecker-solve error against the ridge path | **NEW** — the paper's steady-state theorem as an operation. Costs disclosed: the cross-moment B (S-sized, one rank-1 EMA per write) or one replay pass; the independence assumption is on the face of the API |
| E7 | Superposition + undo: linear map target-sequence → memory (S₀ = 0); `undo_last` **tolerance-pinned by default** (fp round-trip is not bit-exact — pinned ε), `undo_last_snap` bit-exact via a touched-rows/columns snapshot | ✅ | `KronCell::undo_last` / `undo_last_snap`; party belief pooling `S_pool = Σ S_i` (slot-ordered reduction — determinism law) | G1: superposition identity; undo within pinned ε (snap variant bit-exact) | **NEW** — a misfired perception is removable; disclosed: arbitrary-suffix undo is replay-based |
| E8 | Capacity/retention law RECONCILIATION: analytic `d²`-scalar capacity + λ_min(K̄)λ_min(V̄) retention vs Bench 938's measured erosion and Bench 919's first-order bound (0.35·d_k) | ✅ offline analysis | LatticeMemory sizer docs + a reconciliation bench | If analytic matches measured within calibrated band → analytic predictor becomes the sizing model for NEW configs without re-benching; divergence is ALSO a finding | Partial coverage (Bench 938 measured the VECTOR-target erosion) — this is the scalar-target + analytic-predictor gap |
| E9 | Write-conflict predicate: `\|(v₁ᵀv₂)(k₁ᵀk₂)\| ≤ τ` ⇒ order-independent ⇒ parallel apply (color the tiny conflict graph; rayon over color classes; conflicts serialized in pinned order) | ✅ | batch-write helper for tick loops | G1: either-order ΔS ≤ τ·bound; G2: parallel speedup | **NEW** — hot-tick parallelism from an interference identity |
| E10 | **WY chunk kernel** (`kron_chunk_wy`): N = diag(β)(G_kk ⊙ G_vv), unit-LT solve; features never formed; **ternary keys/values ⇒ G_kk, G_vv exact INTEGER via the popcount-GEMM family; Hadamard exact in ℤ; only the L×L solve is fp** | ✅ (inference-side: consolidation replay; training-side: Plan 468) | `katgpt-core` primitive + riir-infer kernel lane if a serving consumer materializes | G1: chunk == sequential within pinned-order tolerance; G2 GEMM throughput vs sequential | **NEW** — the never-materialize algebra exists in the house (`gdn_tree_verify` WS₀ folding, `kron_apply` Bench 839) but never for this solve |
| E11 | KronLSTM recurrent decode arm (katgpt-attn, beside `gdn2`) for FUTURE KronLSTM-class checkpoints | n/a (forward code, no training) | `crates/katgpt-attn/src/kron/` — RECORDED, not built | none until a checkpoint exists (Research 070's "when to implement" posture) | **Not actionable** — no KronLSTM model in the fleet; discard-as-now, keep as trigger |
| E12 | Training track: architecture trial — gate parameterization, init hygiene, hybrid placement, chunkwise kernel, state-tracking β-range | ❌ needs GD | **riir-train `.plans/468`** (E0 mechanism probe, ~1–1.5 days on the 4090) | `kron_vs_gdn_goat` spec in the plan | **Filed** — see the plan; SECONDARY track per serving-envelope fit |

**Audited discards (each with its reason):**

- **Learned gates** (β_t, c_t, f_t, o_t as trained projections, the α log-exponent): the modelless extraction substitutes FIXED schedules (constant β, unit or audit-chosen targets, fixed forget). Reason: no runtime consumer for learned per-token projections outside a trained mixer; the mechanism ships (E1), the tuned values belong to the architecture track (E12).
- **mLSTM-comparison / normalizer-free serving**: no KronLSTM checkpoints exist; the decode arm is trigger-gated (E11). Same posture as Research 070's adoption table.
- **32k context-extension curriculum**: our league serving window is ~2–4k; parked in Plan 468's deferred section, revisit on roadmap change.
- **Pure-gain homogeneity / per-step reparameterization (A4)**: algebraic identities with no runtime consumer beyond E1's tests; folded into the G1 property-test suite, not a separate primitive.

### 2.4 Signal-diffs (§3.6 — one read each, coverage dismissals audited)

- **`delta_mem/state.rs::write`** (verified via Research 629's read): consumes `(k, v)`, per-row coupled decay, prediction error on the VECTOR readout — a first-order solver for the vector-target contract. It is E1's measured baseline, not coverage: no scalar target, no closed-form Newton step, no surgical delete, no interference audit.
- **LatticeMemory cells** (`.plans/619`, `write_delta`): E2LSH-addressed VECTOR-target delta cells with auto-radius + measured erosion sizer. Host candidate for E1 (a lattice of Kron cells = "pair-addressed scalar table lattice"), and the consumer of E8's analytic reconciliation. Not coverage: different target contract; Bench 938's erosion identity is key-recursion accounting, not value-side surgical deletion.
- **Research 629 / Plan 630 (`rls_depth_mem`)**: vector-target RLS with P covariance, depth-axis. E1's sibling rung (table §2.1); the d⁴ argument is why neither subsumes the other.
- **`gdn2_attention`** (katgpt-attn, Plan 105, GOAT 14/14, default-on): the SERVING arm of the vector-target gated family. KronLSTM beats it on the paper's long-context benchmarks — but we serve GDN-family checkpoints; adoption is a training-track act (E12), not a katgpt-rs edit.
- **`game_traits::grudge_field`** (riir-games-shared): per-entity grudge weights — today updated by bespoke decay/blends. The E3/E5 consumer: association strengths with audited rewrite/delete and analytic half-life. Not coverage: grudge_field stores designed scalars; it has no associative cell, no interference guarantee, no bounded-envelope theorem.
- **`set_admission.rs::sherman_morrison_rank1`** (629 already adjudicated): rank-1 inverse-maintenance kernel precedent, different consumer. Kernel vocabulary, not coverage.

### 2.5 Game-context and consumer-context reframes (step 4, both axes)

**Game (priority #1):** an NPC's semantic memory becomes a pair-addressed scalar table — `read(entity, relation) → weight` (the analytic CI is gated on Issue 954's G5 conformal floor — ship it only if it beats the floor); "this NPC forgives THIS specific thing and can prove nothing else moved" (E3's receipt, G0-baselined) is a behavior no decay-memory competitor can state; the half-life gauge (E5) and the norm-budget alarm (E5 — "the NPC's social stream left its stationary regime" = free novelty signal) feed the existing FSM as sigmoid drives, never hard edges; party belief pooling (E7) merges member cells by slot-ordered superposition with only scalar projections crossing sync.

**Healer (priority #2):** rule-binding trust memory keyed `(context-hash, rule-embedding) → trust scalar`; the interference kernel (E2) is a computed rule-jurisdiction overlap — high-κ rule pairs are mining-queue merge/split candidates, corpus hygiene from arithmetic at corpus-build time; audited rule retirement (E3) gives the `Withdrawn` evidence tier a memory-state transition with a computed collateral bound; analytic half-life (E5) replaces ad-hoc decay constants in rescue-tally schedules. Elo/BetaPosterior stay the scoring authority — the cell is the associative layer beside them, and E8's reconciliation bench is the falsifiable claim.

---

## 3. Verdict

**GOAT — Gain. Deliberately NOT Super-GOAT.** The four-question gate, honestly scored:

| Q | Answer | Reasoning |
|---|---|---|
| Q1 — no prior art (on stack)? | **YES** | Internal-first sweep clean (zero KronLSTM/2610.26834 hits). The scalar-target cell + audited erase + analytic per-cell health do not ship anywhere in the workspace; nearest cousins are vector-target rungs (§2.4). Global novelty for the RULE belongs to the paper (1 day old) — we distill, and the framing pre-emptions (§2.2) are inherited, not ours. |
| Q2 — new behavior class? | **NO (the honest call)** | The audited unlearning is a new OPERATION on an existing class (delta-rule associative cells) — the same relation Bench 938's erosion-aware removals had to LatticeMemory: an issue/bench-grade addition, not a new system class. Super-GOAT Q2s in this workspace founded capability classes (e.g. Research 311's non-blocking GPU cascade). |
| Q3 — product selling point? | **MODERATE** | "NPCs forgive one specific thing and can prove nothing else moved" is finishable and real, but it strengthens the memory stack rather than founding a pillar; the healer fusion is plausible but unproven until E8 reconciles. |
| Q4 — force multiplier? | **YES** | Connects lattice_memory addressing, the delta_mem family, Research 629's ladder, game_traits::grudge_field, the healer corpus/trajectory tiers, and the Kronecker-solve discipline (Bench 308 / Issue 839). |

Q2 failing the strength bar caps the tier at GOAT. Outputs this session: this note + `.issues/954` (the primitive spec — implementation lands when the issue is picked up, feature `kron_memory`, GOAT gates pre-registered there, **led by the G0 baseline arms that make the headline claim falsifiable**: Kron-erase vs vector-delta-erase-with-audit vs one-hot table, matched state bytes, distributed keys) + riir-train `.plans/468` (training track). **Track priority (serving-envelope fit):** the modelless primitive is PRIMARY (runs inside hot paths: NPC ticks, healer corpus passes); the architecture trial is SECONDARY and says so — a from-scratch mixer has no production consumer while the league serves frozen GDN checkpoints (Bonsai-27B), and the paper's own edge is measured at 16–32k against our ~2–4k window. Plan 468's E0 probe exists precisely to measure whether the advantage survives at our window before anyone bets days of 4090 time.

## 4. Latent vs raw boundary

- `KronCell` is LATENT, local to its entity/process: per-NPC association tables, healer rule trust. Never synced as a matrix.
- What crosses the sync boundary: the SCALAR readouts (and E3's audit verdicts) — the 5-scalar law's pattern: project at the boundary, never ship the cell.
- Pooled cells (E7) merge as fixed-size Pods with SLOT-ORDERED reduction if a result is ever committed — pinned fold order per the determinism law (Issue 940 `dot_f32_ordered` discipline); anything replay-compared uses the sequential form, never the WY chunk form (different rounding).
- BLAKE3-commit through the existing freeze-envelope pattern (fixed-size Pod); `E[c²]/λ̂_min` budget is metadata, committable.

## 5. What NOT to do

1. **Don't train KronLSTM in katgpt-rs** (Research 070's law, unchanged): training belongs to riir-train / riir-ai GPU lanes.
2. **Don't quote the i.i.d. bounds as guarantees** on game/healer streams (correlated, non-stationary): ship them as reference dynamics + ALARMS — the alarm IS the regime-shift detector. The ~d× retention claim is the paper's; measure Bench-938-style on OUR streams before quoting it in any doc (E8).
3. **Never materialize the d² features** — the Kronecker–Gram identity is the whole point of E10; forming `k⊗v` Grams is the naive quadratic-in-state trap.
4. **Sequential is canonical, WY is an optimization**: chunked rounding ≠ sequential rounding; anything synced/replay-compared stays sequential; chunk order pinned for the local consolidation lane; never mix tiers silently.
5. **Audits are cell-local** — the interference enumeration requires pair access; global ("whole-memory") audits don't exist and must not be claimed.
6. **One-shot exactness degrades with feature conditioning** — the health gauge is the guard, not a footnote; a cell past its budget refuses or snaps (E6), never silently drifts.
7. **Don't claim erase-with-audit as Kron-exclusive** (verdict-round fix): a vector-target delta erase admits the same closed-form audit, and orthogonal address codes make any cell a trivial table — the Kron-exclusive part is the double gating, and G0 pins the claim to exactly that.
8. **Never compute the residual through two reduction orders**: the identity fixed point and write elision need the residual computed through ONE ordered-dot path (`dot_f32_ordered`) — two different reduction orders can disagree about `residual == 0` and break the hash pin (Issue 940 discipline).

## 6. Relationship to existing research

- **Research 629 (exact-RLS rung, same-day):** sibling rung on the ladder (§2.1); Plan 630 owns the vector-target exact solver, Issue 954 owns the scalar-target member. Read together they complete the "online associative solvers" spread: delta → Kron-scalar → RLS → batch ridge.
- **Research 070 (GDN-2) / Plan 105:** the serving-side family; KronLSTM is its measured successor on long-context benchmarks — E11/E12 own the adoption question, this note owns the modelless extraction.
- **Benches 919/938 + Plan 619:** the measured-capacity substrate E8 reconciles against; LatticeMemory is the first lattice host candidate for Kron cells.
- **Bench 308 / Issue 839:** the Kronecker-solve and `kron_apply` precedents E5/E10 consume.
- **Research 624 (associative memory capacity), 024 (δ-Mem), 006 (Raven):** the capacity/overwrite-axis lineage the scalar table refines (Raven's dense-overwrite failure mode is exactly what the interference kernel formalizes).

## 7. References

- Hartl, Zólyomi, List, Klambauer, Böck, Hochreiter — [arXiv:2610.26834](https://arxiv.org/abs/2610.26834) (2026-10-19); code NX-AI/KronLSTM (MIT, FLA snapshot `78a39922`), HF NX-AI/kronlstm.
- Prior-art census (§2.2): Q-Delta [2606.08804] (ICML 2026) · GSA2 [2610.02816] · TTR Shi & Fox [2501.12352] (JMLR 2026) · KLA [2605.08587] · "How Linear Attention Remembers" [2609.33093] · Longhorn [2407.14207] · Titans [2501.00663] · Yang et al. DeltaNet [2406.06484] · GDN [2412.06464] · GDN-2 [2605.22791] · KDA [2510.26692] · RWKV-7 [2503.14456] · classical: Widrow–Hoff 1960 · Kohonen 1972 · Kaczmarz 1937 · Tanabe 1971 · Strohmer–Vershynin 2009 · Johnstone et al. 1982 · Bogacz–Brown 2003.
- Workspace: Research 629/Plan 630 (RLS rung) · Research 070/Plan 105 (GDN-2) · Research 624 · 024 · 006 · 455 · 482 · Plan 619 + `.docs/03_memory/lattice_memory.md` · Benches 919, 938, 308, 839 · katgpt-rs Issue 954 · riir-train Plan 468.

*(External repo NOT cloned: the paper's Appendix A8 ships the full chunkwise algorithm + verification protocol; no mining batch is planned this session — §0.5 decision recorded.)*
