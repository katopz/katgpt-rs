# Research 682: Symmetric-Kernel Fourier Latent Embeddings

> **Source:** "Symmetry in language statistics shapes the geometry of model representations" — Karkada, Korchinski, Nava, Wyart, Bahri — [arXiv:2602.15029](https://arxiv.org/abs/2602.15029) (ICML 2026, v3)
> **Date:** 2026-11-19
> **Status:** Active
> **Related Research:** 307 (Fourier continuation — Gibbs suppression for non-periodic belief windows), 305 (phase-rotation subspace mix), 246 (manifold power-iteration router), 564 (zone-attention triad §2.2)
> **Related Issues:** katgpt-rs Issue 1234 (open-primitive POC), riir-ai Issue 1245 (game fusion POC)
> **Classification:** Public

---

## TL;DR

The paper proves WHY learned representations form circles (months), open 1D loops (years), and smooth spatial eigenmodes (geography): when pairwise co-occurrence statistics carry **translation symmetry** (`P_ij = P_i·P_j·C̃(dist(x_i, x_j))`), the normalized-PMI matrix becomes circulant, its eigenvectors are Fourier modes, and word2vec-class models provably learn embeddings `w_i = Σ_n a_n·(cos k_n x_i, sin k_n x_i)` with amplitudes `a_n` given analytically by the kernel's spectrum. It further proves the geometry **survives statistical ablation** (zeroing all month-month co-occurrences still yields the circle) because a shared latent variable modulates MANY items' statistics — a collective-eigenvalue effect bounded by Weyl/Davis–Kahan.

**Distilled for katgpt-rs (modelless, inference-time):** a closed-form, deterministic toolkit for **cyclic and ordinal latent coordinates** — construct Fourier-feature embeddings FROM the world's own symmetric statistics (frequencies AND amplitudes derived, not hand-set), select embedding width from the kernel spectrum, read the latent coordinate back by phase (`atan2` over the conjugate eigen-pair), and discover ring-shaped latent structure topologically via the shipped DEC harmonic projector. Zero training, zero allocation, bit-identical — the embeddings are a pure function of world statistics, exactly the freeze/thaw-friendly shape.

---

## 1. Paper Core Findings

1. **Spectral view of embeddings** (Karkada et al. 2025, their App B): word2vec-class training ≈ eigendecomposition of `M⋆` (normalized PMI, ≈ log P_ij/P_iP_j); embeddings learn the top eigenmodes `W = Φ√|Λ|`.
2. **Assumption 3.1 (translation symmetry):** `M⋆(ij) = C(dist(x_i, x_j))` on a semantic continuum subset S — month co-occurrence depends only on month *difference* (empirically validated, their Figs 5–6: periodized exponential kernel fits Wikipedia).
3. **Prop 1 (periodic BC):** a circulant (block-)matrix is diagonalized by the DFT ⇒ PCA-projected embeddings are exactly Fourier features: `w_i ∝ (a_1 cos k_1 x_i, a_1 sin k_1 x_i, a_2 cos k_2 x_i, …)`, `k_n = πn`, `a_n` = kernel DFT magnitude. Degenerate sin/cos pairs ⇒ the top-2 projection is a **circle**.
4. **Cor 2 (periodized exponential kernel):** `a_n → sqrt(2σ/(1+σ²k_n²))` — a Lorentzian spectrum; amplitudes decay ~1/k ⇒ top PCs carry the long-wavelength modes (the circle), higher PCs the "ripples" (overtones; their Fig 8 "calendar Pringles").
5. **Prop 3 (open BC):** eigenfunctions are Sturm–Liouville modes with self-consistent quantization `tan k = −σk` (odd) / `tan k = k/(1+σ(1+σ)k²)` (even) ⇒ **open loops** and Lissajous curves in any 2-PC projection (Green's-function/Youla diagonalization of `exp(−|x−y|/σ)`).
6. **Prop 4 (linear decode):** rank-r Fourier embeddings decode the latent coordinate linearly with error `ε² ≲ r^(−1/D)` — the width–resolution law.
7. **§4 (collective robustness):** a latent variable `t` modulating many words (`P(i|t) = P(i)(1+g(t−t_i))`) makes the PMI circulant with large eigenvalues; ablations touching O(1) entries cannot rotate top eigenvectors (Weyl/Davis–Kahan vs O(N) eigengaps). Seasonal helper words reconstruct the month ring (error ~ 1/√H in helper count); a word's seasonality = phase of its complex affinity over the recovered cycle.
8. **App D (joint model):** continuous-latent (Fourier/circulant) and binary-attribute (Walsh/Kronecker) signals occupy **orthogonal eigenspaces** of the joint PMI — `PMI = K_t⊗J + J⊗K_attr + const`.

---

## 2. Distillation — the modelless primitive family

All closed-form, deterministic, zero-alloc; composes with shipped DFT / `fourier_continuation` / DEC substrate:

1. **Kernel-derived Fourier embeddings** — given latent coordinates `x_i` (ring: periodic BC; ladder/line: open BC) and a kernel (periodized/open exponential), emit `(cos k_n x_i, sin k_n x_i)` scaled by analytic `a_n`. **The delta vs published classes:** RFF (Rahimi & Recht 2007) draws RANDOM frequencies to approximate an arbitrary kernel; Transformer sinusoidal PEs use a hand-set integer ladder with NO kernel; this construction derives frequencies and amplitudes from the world's OWN measured symmetric statistics — deterministic, exact for translation-invariant kernels, and honoring the determinism law (BLAKE3/bit-identical posture).
2. **Width selector** — smallest `w` with kernel-spectrum tail energy < τ (Lorentzian makes this geometric decay, closed form); decode-resolution bound `ε² ≲ r^(−1/D)` prices "how many dims do I need to resolve phase to δ" before building.
3. **Phase readout** — `atan2(⟨w_i, sin-mode⟩, ⟨w_i, cos-mode⟩)` recovers the latent coordinate from an embedding family: a latent→scalar readout at the bridge boundary (sigmoid-gated where it drives behaviour; raw where it crosses sync).
4. **Ring/symmetry discovery** — two independent detectors: (a) Gram-eigenvalue conjugate-pairing signature (the paper's own spectral fingerprint of translation symmetry); (b) **DEC harmonic projector / `betti_numbers`** on the adjacency complex — β₁ > 0 = a 1-cycle = a ring latent, with harmonic phase ordering the ring (topological, d ≤ 3 per the manifold-perf caveat).
5. **Robustness bound** — the §4 theorem is a usable guard: collective latent-modulated spectra make top eigenvectors immune to O(1)-row perturbations (Davis–Kahan, eigengap-scaled). As a report-only readout: *how many corpus rows can be demoted before retrieval geometry shifts more than τ* (rulewright follow-on candidate, not filed — see §5).

---

## 3. Path 0 inventory (value = math, no training loop)

| Component | Analog ships? | Extractable modellessly? | Action |
|---|---|---|---|
| Fourier embedding construction from symmetric kernel | Partial — sinusoidal PEs (fixed ladder, no kernel), RFF (random, approximate), Research 307 `fourier_continuation` (signal EXTENSION for FFT preprocessing, not embedding construction) | YES — circulant eigendecomposition is closed-form | katgpt-rs Issue 1234 |
| Width selection from kernel spectrum | Partial — `k_selector.rs` is a LEARNED UCB1 bandit per relation (feedback signal), `spectral_flatness` is an init-quality diagnostic; no analytic kernel-spectrum width law | YES — Lorentzian tail + Prop-4 bound | Issue 1234 |
| Phase readout of latent coordinate | Partial — `phase_rotation_bridge` mixes two subspaces by a sigmoid-derived angle (a gate, not a coordinate readout) | YES — atan2 over the conjugate pair | Issue 1234 |
| Symmetry-class / ring discovery | Partial — DEC `betti_numbers`/`harmonic_projector` (topological, graph side); Gram conjugate-pairing test NOT shipped | YES — both detectors closed-form | Issue 1234 (T4) + riir-ai Issue 1245 |
| Robustness theorem (collective latent → Davis–Kahan) | No analog (no corpus-ablation geometry bound anywhere) | YES — the theorem IS closed-form (Weyl/Davis–Kahan) | Note-only candidate (§5) |
| Prop-4 decode-error law | No | YES | folded into width selector |

Strongest class per the skill: rows with no coverage + yes-extraction → the robustness bound and decode law. The construction row is "partial" — the delta (deterministic kernel-derived frequencies/amplitudes) is real but the class is published; graded accordingly in §6.

---

## 4. Closest cousins — signal-diff (read this session)

- **`katgpt-core/src/spectral/continuation.rs`** (Research 307, Plan 323): consumes a scalar belief window, emits an extended periodic signal so FFT sees no Gibbs. Signal: time series. Output: signal. The paper's primitive: kernel over a latent coordinate → embedding family. **No overlap; compositional neighbor only.**
- **`phase_rotation_bridge`** (Research 305, Plan 322): `cos α ⊙ a + sin α ⊙ b` with `α = sigmoid(⟨state, dir⟩)·π/2` — a two-subspace interpolation gate. Once a cyclic latent HAS an embedding, this is the natural mixing operator ON the ring; it does not construct embeddings or read phase. Complementary.
- **`manifold_router_probe`** (Research 246, Plan 279): MoE router conditioning by power iteration on expert Grams — "manifold" = expert Gram geometry. Unrelated mechanism.
- **`dual_gram_pca_bridge` / `procrustes_bridge`** (Issue 372): eigensolver shortcut + orthogonal Procrustes alignment. Procrustes is useful HERE: the paper's helper-reconstruction metric (their Fig 17) is Procrustes error — the POC can measure reconstruction with the shipped `orthogonal_procrustes`.
- **DEC `hodge.rs`** (`hodge_decompose`, `betti_numbers`, `harmonic_projector`, `hodge_spectrum`): topological cycle machinery on `CellComplex` — the discovery leg's substrate; the paper's detection is Gram-spectral, not DEC — the pairing of the two is the fusion (§5-A).
- **`zone_attention_triad.rs`** (Issue 966, Plan 599): zone attention = `sigmoid(dot(pref, emb))` × grounding × spread — the consumer exists with evidence-grounded embeddings; analytic ring/hour embeddings would FEED it, unchanged.
- **`k_selector.rs`** (Plan 318): learned bandit over rank-k per relation. The paper's width law is the ANALYTIC counterpart when a kernel exists — different signal (feedback vs spectrum), candidate compose, not coverage.

---

## 5. Fusion + context reframes

**A (primary) — paper × DEC:** harmonic-form ring discovery feeds kernel-derived embedding construction. Discover a world's cyclic latents topologically (zone-adjacency β₁ = 1), then embed them spectrally along the recovered cycle. Neither leg ships the other; DEC gives discovery, the paper gives geometry.
**B — paper × zone attention / world clock:** the consumer `sigmoid(dot(pref, emb))` already exists; hour-of-day (seal-clock, a genuine 24-tick periodic latent) and ring-zone position become structured latents with ZERO trained weights. Nocturnal/activity drives gate through the drive pipeline (never a hard `if hour < 6` edge beside the FSM).
**C — robustness bound × rulewright corpus:** active-set demotion safety readout (report-only): Davis–Kahan over the retrieval corpus's top modes → "demoting these N rules moves retrieval geometry ≤ τ". Candidate only; not filed (ungoat discipline — no measured consumer yet).
**D — App D orthogonality × refine dispatch:** continuous-latent (Fourier) and categorical (Walsh) signals provably occupy orthogonal eigenspaces — design validation for `span_shape_dispatch` (categorical) coexisting with latent retrieval (continuous) in one fan-out. No action.
**Game reframe:** rings (zone adjacency), the 24-tick world clock, ordinal quest-step lattices (open BC) — all latent continuums whose embeddings this toolkit constructs analytically; per-NPC behaviour reads them via existing sigmoid dot-gates. → riir-ai Issue 1245.
**Consumer reframe:** the decision stack's hashed embeddings (reflex `embed.rs`) are content-hash bags — no ordinal/cyclic structure by construction; a suite-latent phase probe would be a harness diagnostic, deferred (report-only, lowest priority).

---

## 6. Verdict

**PoC 2026-11-19 (Issue 1234 CLOSED, [Bench 989](../.benchmarks/989_kernel_fourier_features_goat.md)):** construction, width law (as a conservative bound), phase readout and pairing fingerprint all PASS their pre-registered gates. The width claim stands (no G1c mismatch). Spec correction: the paper-form even quantization belongs to the double-centered kernel; raw = `tan k = 1/(σk)`. Noise robustness (§4) not measured.

**Gain.** One-line reasoning: the mechanism class is published (deterministic Fourier features for a latent coordinate = sinusoidal PEs; kernel→features = RFF; the phenomenon = Engels et al. 2024/2025; the proof = this paper) — no new capability class; the DELTA is the fusion (DEC topological discovery × deterministic kernel-derived construction × game latents × the robustness bound as a corpus guard) plus packaging the width law and phase readout as modelless primitives.

Novelty-gate scoring: Q1 prior art — YES published (class), the fusion delta is unclaimed but class-level novelty fails. Q2 new behavior class — marginal in-stack (zero-training structured latents for clocks/rings is new HERE, not in the literature). Q3 selling point — yes ("our NPCs gate on time-of-day and ring position with zero trained weights; the geometry is derived from world statistics"). Q4 force multiplier — yes (DEC + zone attention + corpus/retrieval). Not all-4 ⇒ **Gain**, fusion idea + POC issues (no Super-GOAT guide required).

**MOAT gate:** `katgpt-rs` — generic closed-form primitive family in the spectral lane, beside `fourier_continuation`; public-safe math (no game semantics). Game wiring → `riir-ai` (Issue 1245). Rulewright diagnostics → note-only candidate (§5-C). No riir-train deferral (nothing trains).

**Proof path (pre-registered, the gates are the falsifiers):**
- katgpt-rs Issue 1234 — G1: at full width the embedding Gram ≍ centered kernel (synthetic periodized exponential, N=24; max abs diff < 1e-5) — a round-trip pin only (the Fourier basis exactly diagonalizes the circulant Gram at full width for any σ, so G1 cannot fail); **G1c: at truncated width w < N/2 the relative reconstruction error `‖K − Φ_w Φ_wᵀ‖_F / ‖K − mean‖_F ≤ 1.5 × sqrt(Σ_{n>w} a_n² / Σ_{n≥1} a_n²)` (dropped-tail bound, 1.5× headroom pinned pre-run), per σ — the width law's actual falsifiable test**; G1b: phase readout recovers `x_i` to < 1/(2w) at the truncated width; G2: O(N·w) construction; G4: zero-alloc. **Falsifier:** G1c mismatch at any (σ, w) = the kernel-spectrum law wrong there ⇒ demote the width claim, keep the construction; a phase-readout failure ⇒ demote T3.
- riir-ai Issue 1245 — synthetic ring-zone fixture: DEC β₁ = 1 detected, harmonic phase orders the ring bit-identically across runs; consumer `sigmoid(dot(pref, emb))` unchanged; feature-off byte-identical; behavioural test asserts PROGRESS through the drive pipeline (brain+FSM law), never a hard hour edge.

**PASS-Redirects (synthesis):** Engels et al. [arXiv:2405.14860 "Not All Language Model Features Are Linear"] — the phenomenon (circular features in LLMs) is published; cite, don't re-derive. Rahimi & Recht 2007 (RFF) — the random-frequency kernel-approximation class; our delta is deterministic kernel-derived frequencies. Transformer sinusoidal PEs — the fixed-ladder deterministic class; our delta is statistics-derived frequencies/amplitudes.
