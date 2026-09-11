# Issue 747: ASEntmax modelless mining — entmax damping schedule, derived support controller, theorem-backed KV eviction window

**Status:** Open
**Research:** [katgpt-rs/.research/549](../.research/549_ASEntmax_Length_Adaptive_Entmax_Attention.md) · **Source:** [arXiv:2506.16640](https://arxiv.org/abs/2506.16640) (ICLR 2026) · **Training arm:** riir-train Plan 396 (secondary)

arXiv:2506.16640 derives what our stack is missing on the entmax side of a law we already exploit on the softmax side: softmax needs sharpening ∝ log n (SSMax, Plan 411, shipped), α-entmax needs **damping ∝ (log n)^{−0.5}** (their Eq 10, closed form). Our `entmax_1p5` (DashAttention routing, default-on) has **no length term at all** — as the scored candidate set grows, logit range grows `2σ√(2 log n)`, the threshold eats the support, and routing over-sparsifies (the paper's Copy-table failure mode: fixed-α entmax 28.5% vs softmax 99.4% OOD).

Execution tracker for the modelless rows of Research 549 §2.3. Feature flags per row; GOAT gates per the house rule; Bench-032 harness pattern (NIAH at growing chunk counts) is the G2 axis.

## Tasks

### P0 — `asentmax_schedule`: derived damping mode for entmax routing

- [ ] **T0.1** Add `AsentmaxSchedule` mode to `katgpt-attn/dash_attn/entmax.rs` (socket-mirrors `SsmaxMode`): pre-scale scores by `(δ + β·(log n_c)^γ)` with derived defaults `δ = 0, γ = −0.5`, `β = 1/(2σ̂√2)` from a rolling σ̂; zero-alloc, one `powf` + one mul per head per step. Feature `asentmax_schedule`, opt-in.
- [ ] **T0.2** **G1 (stationarity — the paper's failure mode as an assertion):** IID-Gaussian scores, n_c = 512 → 512k; mean support size stationary ±ε with the schedule, → 1 (collapse) without it. Simplex + exact-zero properties preserved.
- [ ] **T0.3** **G2 (routing quality):** Bench-032-pattern NIAH sweep at growing chunk counts (256 → 4k+): scheduled routing ≥ fixed-α routing coverage/retrieval; record per-arm active-chunk histograms.
- [ ] **T0.4** **G3 (no-regression):** at Bench-032's original 256-chunk scale, scheduled ≈ fixed-α within noise (adaptive_k floor unchanged).
- [ ] **T0.5** **G4:** CountingAllocator — 0 steady-state allocs.
- [ ] **T0.6** GOAT verdict → promote to default (demote fixed-α on the routing path if it wins) or record negative; update `.docs/09_feature_catalog/` + Bench note.

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
