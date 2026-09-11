# Issue 747: ASEntmax modelless mining — entmax damping schedule, derived support controller, theorem-backed KV eviction window

**Status:** In progress — P0 DONE (GOAT PASS, opt-in; Bench 713). P1–P3 open. P0.7 (forward-path wiring + promotion re-gate) tracks the default-on decision.
**Research:** [katgpt-rs/.research/549](../.research/549_ASEntmax_Length_Adaptive_Entmax_Attention.md) · **Source:** [arXiv:2506.16640](https://arxiv.org/abs/2506.16640) (ICLR 2026) · **Training arm:** riir-train Plan 396 (secondary)

arXiv:2506.16640 derives what our stack is missing on the entmax side of a law we already exploit on the softmax side: softmax needs sharpening ∝ log n (SSMax, Plan 411, shipped), α-entmax needs **damping ∝ (log n)^{−0.5}** (their Eq 10, closed form). Our `entmax_1p5` (DashAttention routing, default-on) has **no length term at all** — as the scored candidate set grows, logit range grows `2σ√(2 log n)`, the threshold eats the support, and routing over-sparsifies (the paper's Copy-table failure mode: fixed-α entmax 28.5% vs softmax 99.4% OOD).

Execution tracker for the modelless rows of Research 549 §2.3. Feature flags per row; GOAT gates per the house rule; Bench-032 harness pattern (NIAH at growing chunk counts) is the G2 axis.

## Tasks

### P0 — `asentmax_schedule`: derived damping mode for entmax routing — **DONE 2026-09-11 (Bench 713)**

**Deviation from T0.1 as written:** the schedule ships in a new `dash_attn/asentmax.rs` module (not inline in `entmax.rs`) — mirrors `ssmax.rs`'s standalone status; the transform and the schedule are distinct concerns. **Amended G1 expectations (measured truth):** for pure IID-Gaussian scores the exact stationarity claims are the scaled RANGE (n-invariant by construction) and the σ-AXIS (support σ-invariant — σ cancels in σ·β); the damped support grows mildly ∝ ln n on the n-axis (analytic k* ≈ 4·ln n), and the collapse counterfactual is real on the σ axis (raw support → 1 at σ=10, scheduled holds). The issue's original "stationary ±ε / → 1 without it" wording was pre-measurement; see Bench 713 for the corrected claims.

- [x] **T0.1** `AsentmaxSchedule` (None/Derived/Generalized) + `apply_asentmax_inplace` + `RollingSigmaEstimator` (lock-free EMA, Kamath range law `σ̂ = range/(2√(2 ln n))`) in `katgpt-attn/src/dash_attn/asentmax.rs`; routing socket `score_blocks_entmax_with_schedule_into` (routing.rs, shared-body refactor — bit-identical at `None`, unit-pinned); feature `asentmax_schedule = ["dash_attn"]` + root shim + root re-exports. Zero-alloc: one `powf` + one mul per head per step.
- [x] **T0.2 G1 PASS** (tests/asentmax_g1_stationarity.rs, 4/4): scaled range pinned 0.83–0.90 over n=512→512k (ratio 1.09, unscheduled 1.58); support σ-invariant (max/min < 2 across σ ∈ {1,3,10}); collapse counterfactual at σ=10 (raw ≤ 4, scheduled ≥ 4×); simplex + EXACT zeros; estimator self-consistency (range-fed σ̂ pins scaled range to exactly 1.0 — finite-n EV corrections cancel by construction, beating the oracle-σ arm's 0.91).
- [x] **T0.3 G2 PASS** (bench_747_asentmax_goat): graded-relevance planted set (k=8, noise-jittered) recall **0.81–0.91 scheduled vs 0.14–0.31 raw** at n_c 256→16k × σ 1→8; raw support collapses to 1–2.5 chunks, scheduled holds 7–8; planted mass ≈ 1.0 both arms.
- [x] **T0.4 G3 PASS**: single-needle (032 T23 anchor) retrieval parity 100%/100% at 256 chunks; scheduled support 12–13 sits INSIDE Bench 032's shipped envelope (avg 21.5, range 4–40+); needle mass ≥ 0.61.
- [x] **T0.5 G4 PASS**: 0 allocs / 1000 steady-state cycles (CountingAllocator + liveness canary; cross-crate `#[path]` include of katgpt-core's shared test common). Latency ≈ 9.7–18.6 µs/step at n=16k (row scan + multiply — same order as the RollingDeltaEstimator precedent).
- [x] **T0.6 GOAT verdict:** 🟢 PASS, **stays opt-in** — the primitive is not yet on a production hot path (`forward.rs` routing calls the unscheduled variants); default-on now would be an unwired state (feature-gate-audit discipline). `.docs/09_feature_catalog/opt_in_features.md` row added; Bench 713 records the full evidence.
- [ ] **T0.7 (new)** Wire `score_blocks_entmax_with_schedule_into` into the forward-path routing call site (config-gated, e.g. `DashAttnConfig` schedule field or a forward-local flag), re-run G2/G3 on the real prefill path, then re-evaluate default-on promotion (demote fixed-α on the routing path if it wins there too).

### P1 — Lemma-2 support controller `k̂ = 4/Δ̂²`

- [ ] **T1.1** Derived budget in `dash_attn/adaptive_k.rs` (beside the sigmoid path): `k̂ = ((α−1)·Δ̂)^{−1/(α−1)}` from the shipped `RollingDeltaEstimator` (max−mean proxy), clamped `[k_min, k_max]`.
- [ ] **T1.2** **G1:** two-level synthetic logits (gap Δ, k top) — realized support == k̂ for ALL n ∈ {1k..1M} (**length-independence is the test**; the sigmoid arm has no such law).
- [ ] **T1.3** **G2/G3:** head-to-head vs `sigmoid(w·var+b)` budget on the T0.3 sweep; no-reg at 256 chunks.

### P2 — Prop 6 eviction window (theorem-backed, bit-identity gate)

- [ ] **T2.1** Window calc `d_max = ⌊(z_max − z_min + 1)/m_h⌋ + 1` for ALiBi-biased entmax heads (katgpt-attn primitive); wire as a KV-retention predicate in the riir-ai engine KV path (consumer-side follow-up filed there if needed).
- [ ] **T2.2** **G1 (the cleanest gate in this batch — correctness is bit-identity by theorem):** attention beyond `d_max` is *exactly* zero ⇒ evicted-vs-full run must be bit-identical under index-ordered accumulation; brute-check support ⊆ window.
- [ ] **T2.3** **G2:** KV bytes evicted at n = 1M + decode tok/s delta.

### P3 — Lemma 1 incremental decode entmax

- [ ] **T3.1** Incremental variant beside `entmax_1p5`: below-τ score additions leave existing probs exactly unchanged — recompute only on support-entry events; reused sort scratch.
- [ ] **T3.2** **G1:** incremental == full-resort, bit-exact, incl. threshold-brushing adversarial streams. **G2:** decode-step cost O(candidates) vs O(n log n) resort at n = 512k.

### P4 — stretch (only after P0–P2 land)

- [ ] **T4.1** `SsmaxMode::HoldConcentration{c,k}` — analytic `θ*(n) = Δ̂/ln((n−k)c/(k(1−c)))` from Lemma 2's softmax side (upgrades shipped SSMax with the exact coefficient).
- [ ] **T4.2** Kamath range-law detector `ρ = Δ̂/(2σ̂√(2 log n))` (Gaussian vs spiked regime) + normalized-entropy dispersion diagnostic `H(p)/log n` (O(s) over support) — `katgpt-core` estimator module beside ssmax.
- [ ] **T4.3** Offline per-head (β,γ) grid sweep (frozen table, freeze/thaw-versioned) — only if the derived γ=−0.5 default passes P0 and a swept fit demonstrably beats it; harvested constants may also arrive from riir-train Plan 396 Ph2.
- [ ] **T4.4** Prop 7/8 RoPE cutoff spectrum — retention must respect periodic re-entry windows (window union, not first cutoff).

## Close-out

Close this issue when P0–P3 are gated (P4 stretch may defer to follow-ups). Record the GOAT verdicts + bench file refs here, then remove per the noise-reduction rule (git history keeps the record).
