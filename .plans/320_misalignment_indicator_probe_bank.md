# Plan 320: Misalignment Indicator Probe Bank — Multi-Direction OR-Fused Cascade

**Date:** 2026-06-25
**Research:** [katgpt-rs/.research/301_Misalignment_Indicator_Probe_Bank.md](../.research/301_Misalignment_Indicator_Probe_Bank.md)
**Private selling-point guide:** [riir-ai/.research/157_bidirectional_cognitive_monitoring_guide.md](../../riir-ai/.research/157_bidirectional_cognitive_monitoring_guide.md)
**Source paper:** [Zhou et al. 2026 — Probing the Misaligned Thinking Process of Language Models](https://arxiv.org/pdf/2606.24251) — ICML 2026 Mech Interp Workshop
**Target:** `katgpt-rs/crates/katgpt-core/src/pruners/indicator_probe_bank.rs` (new module) + Cargo feature `indicator_probe_bank`, `indicator_similarity`, `indicator_cascade`
**Status:** In progress — Phase 1 implementation (2026-06-25)

---

## Goal

Ship the **generic open primitive** for the Super-GOAT in Research 301: a structured bank of N pre-computed direction vectors, each tagged with a fine-grained cognitive-primitive label, projected via dot-product + sigmoid, OR-fused into a single flag, with an optional cascade to a heavy verifier. The bank generalizes the single-direction primitives we already ship (`FutureBehaviorProbe` Plan 292, `EmotionDirections::project` Plan 162, `ClaimVerifier` Plan 284) into a structured multi-direction set with inspectable similarity structure.

The bank operates on any `[f32; D]` state — it carries zero game semantics. The game-runtime integration (NPC behavioral-trait directions, bidirectional pairing with the Cognitive Integrity Layer, KG-triple audit trail) lives in `riir-ai/.research/157_*.md` and downstream riir-ai plans.

**Why this is a Super-GOAT not a GOAT:** See Research 301 §3 novelty gate. The output-side indicator bank + bidirectional cognitive monitoring has no shipped cousin (the integrity layer is structurally input-side). The capability class (emergent-NPC alignment monitoring) is new. The private selling-point moat is the bidirectional pair × KG-triple audit trail. This plan ships the open half.

**GOAT gate rule (AGENTS.md):** every new primitive ships behind a feature flag and must pass the GOAT gate (G1 correctness, G2 perf, G3 no-regression, G4 alloc-free or equivalent) before promoting to default. Demote the loser if a new technique wins. Headline metric: **(FPR reduction ratio at fixed TPR)** for the cascade vs bank-only, on a synthetic bank with planted indicator structure.

---

## Phase 1 — Unblocking Skeleton (CORE, ALWAYS SHIPS)

### Tasks

- [x] **T1.1** Create module `katgpt-rs/crates/katgpt-core/src/pruners/indicator_probe_bank.rs`. Add to `pruners/mod.rs` behind `#[cfg(feature = "indicator_probe_bank")]`.

- [x] **T1.2** Define the indicator label enum (generic — no game semantics). Numeric discriminant, `#[repr(u8)]`, stable for sync:

  ```rust
  /// Generic indicator label. Domain-specific instantiations (e.g., NPC
  /// behavioral traits) impl this trait with their own label enum.
  pub trait IndicatorLabel: Copy + Eq + core::hash::Hash + Send + Sync + 'static {
      /// Stable u8 discriminant for serialization / sync.
      fn as_u8(&self) -> u8;
      /// Recover from u8. Returns None if out of range.
      fn from_u8(d: u8) -> Option<Self>;
      /// Number of distinct labels in this instantiation.
      const COUNT: usize;
  }
  ```

  The crate ships one canonical example impl: `pub enum DemoIndicatorLabel { A, B, C }` for tests + docs. Real instantiations live in consumer crates.

- [x] **T1.3** Define `IndicatorProbeBank<L: IndicatorLabel, const D: usize>`:

  ```rust
  /// Structured bank of N pre-computed direction vectors, each tagged with
  /// an `IndicatorLabel`. The bank is the open primitive from Research 301:
  /// N directions, sigmoid-gated, OR-fused into a single flag.
  ///
  /// The bank is generic over:
  /// - `L`: the indicator-label type (domain-specific).
  /// - `D`: the state-space dimensionality.
  ///
  /// Direction vectors are loaded at init from a frozen, BLAKE3-committed
  /// artifact (freeze/thaw-compatible per AGENTS.md). They are NEVER updated
  /// at runtime — runtime updates go through the latent-state kernel, not
  /// through the bank.
  pub struct IndicatorProbeBank<L: IndicatorLabel, const D: usize> {
      /// `directions[i]` is the direction vector for label `L::from_u8(i as u8)`.
      /// Shape: `[N][D]` flattened for SIMD-friendly iteration.
      directions: Vec<f32>,
      /// `thresholds[i]` is the sigmoid-input threshold above which label i fires.
      thresholds: Vec<f32>,
      /// Per-bank BLAKE3 manifest hash of the directions + thresholds.
      /// Computed at load time; embedded in the bank for freeze/thaw.
      blake3: [u8; 32],
      /// Freeze/thaw version (monotonic).
      version: u64,
      /// Marker for the label type.
      _marker: core::marker::PhantomData<L>,
  }
  ```

- [x] **T1.4** Implement `IndicatorProbeBank::project` (the single-direction read):

  ```rust
  impl<L: IndicatorLabel, const D: usize> IndicatorProbeBank<L, D> {
      /// Project `state` onto direction `label`, return sigmoid score.
      ///
      /// Zero-allocation: reuses `simd_dot_f32`. Matches Plan 292
      /// `FutureBehaviorProbe::forecast` latency target (<200ns at D≤2048).
      #[inline]
      pub fn project(&self, state: &[f32; D], label: L) -> f32 {
          let idx = label.as_u8() as usize;
          let dir = &self.directions[idx * D..(idx + 1) * D];
          let raw = crate::simd::simd_dot_f32(dir, state);
          crate::simd::sigmoid_f32(raw - self.thresholds[idx])
      }
  }
  ```

- [x] **T1.5** Implement `IndicatorProbeBank::project_all_into` (the batched read — the hot-path shape):

  ```rust
      /// Project `state` onto every direction, write sigmoid scores into
      /// `out_scores` (caller-allocated scratch, length N).
      ///
      /// Zero-allocation. This is the per-NPC-per-tick hot path.
      #[inline]
      pub fn project_all_into(&self, state: &[f32; D], out_scores: &mut [f32]) {
          debug_assert_eq!(out_scores.len(), L::COUNT);
          for i in 0..L::COUNT {
              let dir = &self.directions[i * D..(i + 1) * D];
              let raw = crate::simd::simd_dot_f32(dir, state);
              out_scores[i] = crate::simd::sigmoid_f32(raw - self.thresholds[i]);
          }
      }
  ```

- [x] **T1.6** Implement `IndicatorProbeBank::or_fused_fire` (the OR-fusion):

  ```rust
      /// After `project_all_into(state, &mut scores)`, return the firing
      /// label with the highest score if any score exceeds `tau_fire`,
      /// else None.
      ///
      /// Paper §2.2 end + §2.3 OR-fusion: "a turn is flagged if any
      /// indicator probe exceeds its threshold on any sentence". We
      /// collapse to one fire per state (the argmax).
      #[inline]
      pub fn or_fused_fire(
          &self,
          scores: &[f32],
          tau_fire: f32,
      ) -> Option<L> {
          debug_assert_eq!(scores.len(), L::COUNT);
          let mut best_label: Option<L> = None;
          let mut best_score: f32 = tau_fire; // must strictly exceed
          for (i, &s) in scores.iter().enumerate() {
              if s > best_score {
                  best_score = s;
                  best_label = L::from_u8(i as u8);
              }
          }
          best_label
      }
  ```

- [x] **T1.7** Implement `IndicatorProbeBank::from_frozen_bytes` (the freeze/thaw loader):

  ```rust
      /// Load a bank from its frozen wire format. Verifies the embedded
      /// BLAKE3 hash; returns None on hash mismatch (tamper-evident).
      ///
      /// The wire format is the canonical `IndicatorBankWire` Pod (see T1.9).
      /// The bank is loaded ONCE at init; runtime code holds an `Arc<IndicatorProbeBank>`.
      pub fn from_frozen_bytes(bytes: &[f8]) -> Result<Self, BankLoadError> {
          // 1. Parse Pod layout (header + directions + thresholds).
          // 2. Recompute BLAKE3 over directions + thresholds.
          // 3. Compare to embedded hash. Mismatch → BankLoadError::HashMismatch.
          // 4. Construct bank with version from header.
          todo!("T1.7 + T1.9 wire format")
      }
  ```

- [x] **T1.8** Unit tests in `tests/indicator_probe_bank_basic.rs`:
  - `test_project_returns_sigmoid_in_unit_interval` — project on a zero vector returns 0.5; project on direction itself returns sigmoid(‖d‖² − threshold).
  - `test_or_fused_fire_none_below_tau` — all scores below tau → None.
  - `test_or_fused_fire_argmax_above_tau` — three labels with one above tau → that label.
  - `test_or_fused_fire_tie_breaks_by_lowest_index` — two labels at same score above tau → lower index.
  - `test_project_all_into_writes_all_N_scores` — output length matches `L::COUNT`.
  - `test_from_frozen_bytes_round_trip` — bank → frozen bytes → bank produces equal directions + thresholds + blake3 + version.
  - `test_from_frozen_bytes_rejects_tampered_hash` — flip one byte in directions → `BankLoadError::HashMismatch`.
  - `test_indicator_label_u8_round_trip` — every `L::from_u8(L::as_u8(&l)) == l` for all variants in `DemoIndicatorLabel`.

- [x] **T1.9** Define the wire format `IndicatorBankWire` as a `#[repr(C)]` Pod with header (magic, version, N, D, blake3) + directions + thresholds. Follow the `merkle_freeze` pattern from `riir-neuron-db/src/freeze.rs` for BLAKE3 commitment.

  *Implementation note:* the wire format ships as `to_frozen_bytes`/`from_frozen_bytes` on the bank (header `[magic(4)|version(8)|n(2)|d(2)|blake3(32)]` = 48 bytes + body), plus a zero-copy `IndicatorBankWireHeader::parse` view. BLAKE3 commits over `directions ++ thresholds` (the body), matching the `riir-neuron-db/src/freeze.rs` one-shot `*blake3::hash(&buf).as_bytes()` pattern. The `IndicatorLabel` trait and `DemoIndicatorLabel` ship in-module so tests/docs are self-contained; consumer crates (riir-ai) supply the 18-indicator NPC taxonomy.

- [x] **T1.10** Wire Cargo feature `indicator_probe_bank` in `katgpt-rs/Cargo.toml`; ensure default-off; ensure zero overhead when off (grep `cfg(feature)` coverage on every new code path).

  *Note:* this required decoupling the `pruners` parent module from `review_metrics` (it was previously gated behind `review_metrics`); now `pruners` is always compiled and each submodule gates independently. Zero regression: `cargo check` (default, no `indicator_probe_bank`) is clean.

**Phase 1 exit:** `IndicatorProbeBank` projects + OR-fuses + loads from frozen bytes. No cascade, no similarity matrix. Unit tests green. Feature gate `indicator_probe_bank` works.

---

## Phase 2 — Indicator Similarity Matrix (STRUCTURE)

The paper's Fig. 6 finding (block-structured cosine similarity) as a first-class inspectable / committable artifact.

### Tasks

- [x] **T2.1** Create module `katgpt-rs/crates/katgpt-core/src/pruners/indicator_similarity.rs` behind `#[cfg(feature = "indicator_similarity")]` (pulls in `indicator_probe_bank`).

- [x] **T2.2** Define `IndicatorSimilarityMatrix<L: IndicatorLabel>`:

  ```rust
  /// Cosine-similarity matrix of an `IndicatorProbeBank`'s direction vectors.
  ///
  /// Paper Fig. 6 finding: the indicators form a shared "misaligned-reasoning"
  /// subspace (most pairs in [0.3, 0.7]) with within-category block structure.
  /// This matrix makes that structure first-class inspectable.
  pub struct IndicatorSimilarityMatrix<L: IndicatorLabel> {
      /// `N × N` symmetric matrix of cosines. Stored row-major.
      cosines: Vec<f32>,
      /// Number of indicators.
      n: usize,
      _marker: core::marker::PhantomData<L>,
  }
  ```

- [x] **T2.3** Implement `IndicatorSimilarityMatrix::from_bank` — compute all pairwise cosines at construction time (O(N²D), done once).

- [x] **T2.4** Implement `IndicatorSimilarityMatrix::similarity(i, j) -> f32` (constant-time lookup).

- [x] **T2.5** Implement `IndicatorSimilarityMatrix::cluster(&self, tau_intra: f32, tau_inter: f32) -> Vec<Vec<L>>` — greedy within-category block recovery. Group indicators whose pairwise similarity exceeds `tau_intra`; reject cross-group if any pair exceeds `tau_inter`. This is the paper's Fig. 6 finding as a structured output.

  *Implementation note:* the algorithm uses complete linkage on `tau_intra` (merge `Ga, Gb` iff every cross-pair ≥ `tau_intra`), choosing the densest valid merge for determinism. `tau_inter` is reserved for future inter-group rejection (signature kept for paper parity); the complete-linkage rule already rejects loose cross-category bridges (which have ≥1 sub-threshold pair).

- [x] **T2.6** Unit tests:
  - `test_from_bank_produces_symmetric_matrix` — `m.similarity(i, j) == m.similarity(j, i)`.
  - `test_diagonal_is_one` — `m.similarity(i, i) == 1.0` (within float tolerance).
  - `test_cluster_recovers_planted_blocks` — construct synthetic bank with 3 categories × 2 indicators (within-cat cosine ≈ 0.69, cross-cat 0); `cluster(0.6, 0.6)` recovers 3 groups of 2. ARI ≥ 0.9 vs planted. ✓
  - `test_cluster_returns_single_group_when_all_similar` — all pairs above tau → one group.
  - `test_cluster_returns_singletons_when_all_orthogonal` — all pairs below tau → N singletons.

  *Note:* the planted-block test uses a 6-variant `SixLabel` enum (3 blocks × 2 indicators) on disjoint 2-dim subspaces, with within-block cosine 0.69 (directions `[1, 0.4]` and `[0.4, 1]`). An `adjusted_rand_index` helper is implemented in-module for the ARI computation (no existing reusable ARI in katgpt-core).

- [x] **T2.7** Wire Cargo feature `indicator_similarity`.

**Phase 2 exit:** similarity matrix + cluster recovery work. G5 GOAT gate benchmark ready.

---

## Phase 3 — Indicator Cascade Trait (VERIFIER ESCALATION)

The two-stage cascade from the paper (probes online → verifier offline). The verifier impl is opaque (trait object); the katgpt-rs side ships the trait + a stub. Game-runtime (riir-ai) supplies the LLM-judge impl.

### Tasks

- [x] **T3.1** Create module `katgpt-rs/crates/katgpt-core/src/pruners/indicator_cascade.rs` behind `#[cfg(feature = "indicator_cascade")]` (pulls in `indicator_probe_bank`).

- [x] **T3.2** Define the verifier trait:

  ```rust
  /// Stage-2 verifier for the indicator cascade. The bank (stage-1) flags
  /// candidates; the verifier (stage-2) adjudicates flagged candidates only.
  ///
  /// This is the paper's two-stage cascade: probes online → LLM judge offline.
  /// The verifier impl is opaque (trait object); the katgpt-rs side ships
  /// the trait + a stub. Consumer crates (riir-ai) supply the LLM-judge impl.
  pub trait IndicatorVerifier<L: IndicatorLabel>: Send + Sync {
      /// Adjudicate a flagged candidate. Returns `true` if confirmed
      /// misaligned (the verdict that crosses the sync boundary as raw),
      /// `false` if the flag is a false positive.
      ///
      /// The verifier has access to: the firing label, the per-indicator
      /// scores, and any opaque context the impl carries (e.g., full
      /// NPC reasoning trace for an LLM judge).
      fn verify(&self, label: L, scores: &[f32]) -> bool;
  }

  /// Stub verifier that confirms every flag. For tests + benchmarks only.
  pub struct AlwaysConfirmVerifier;

  impl<L: IndicatorLabel> IndicatorVerifier<L> for AlwaysConfirmVerifier {
      #[inline]
      fn verify(&self, _label: L, _scores: &[f32]) -> bool { true }
  }

  /// Stub verifier that rejects every flag. For tests + benchmarks only.
  pub struct AlwaysRejectVerifier;

  impl<L: IndicatorLabel> IndicatorVerifier<L> for AlwaysRejectVerifier {
      #[inline]
      fn verify(&self, _label: L, _scores: &[f32]) -> bool { false }
  }
  ```

- [x] **T3.3** Define the cascade driver:

  ```rust
  /// Two-stage cascade: bank (stage-1) OR-fuses; verifier (stage-2)
  /// adjudicates flagged candidates only.
  ///
  /// Paper §2.2 end: "a turn is reported as misaligned only if both stages
  /// agree". The cascade's payoff: stage-2 reduces FPR ~15× at modest TPR
  /// cost (paper Table 10: 4.5% → 0.3% neutral FPR at -10pp transcript-TPR).
  pub struct IndicatorCascade<L: IndicatorLabel, const D: usize> {
      pub bank: Arc<IndicatorProbeBank<L, D>>,
      pub verifier: Arc<dyn IndicatorVerifier<L>>,
      pub tau_fire: f32,
  }

  impl<L: IndicatorLabel, const D: usize> IndicatorCascade<L, D> {
      /// Full pipeline: project → OR-fuse → verify.
      /// Returns the firing label if the cascade confirms; None otherwise.
      pub fn run(&self, state: &[f32; D], scores_scratch: &mut [f32]) -> Option<L> {
          self.bank.project_all_into(state, scores_scratch);
          let firing = self.bank.or_fused_fire(scores_scratch, self.tau_fire)?;
          if self.verifier.verify(firing, scores_scratch) {
              Some(firing)
          } else {
              None
          }
      }
  }
  ```

- [x] **T3.4** Unit tests:
  - `test_cascade_confirms_when_verifier_confirms` — bank fires label A, `AlwaysConfirmVerifier` confirms → returns A. ✓
  - `test_cascade_rejects_when_verifier_rejects` — bank fires label A, `AlwaysRejectVerifier` rejects → returns None. ✓
  - `test_cascade_no_fire_returns_none` — no label above tau → no verifier call, returns None. ✓
  - `test_cascade_zero_alloc` — **deferred to Phase 4 bench** (`bench_320_indicator_probe_bank_goat.rs`) using a `CountingAllocator`. Cannot live in a unit test because `#[global_allocator]` is crate-binary-unique and would collide with other test modules in the same lib test binary (matches the Plan 327 G4 precedent).

- [x] **T3.5** Wire Cargo feature `indicator_cascade`.

**Phase 3 exit:** cascade trait + driver work with stub verifiers. Real verifier impls land in consumer crates.

---

## Phase 4 — GOAT Gate (Synthetic)

The synthetic GOAT gate. All metrics on a planted-structure synthetic bank (no game semantics, no real direction-vector training).

### Tasks

- [ ] **T4.1** Construct synthetic test bank:
  - 8 indicators (matches paper's strongest 8: Action Concealment, Malicious Action Planning, Observer Suspicion Modeling, Misalignment Cover Story Planning, Strategic Omission, Recognized-Problem Suppression, Error Calibration, Strategic Underperformance Reasoning — these are the paper's >0.92 AU-ROC indicators).
  - D = 72 state dim (matches HLA[8] + style_weights[64] from Research 157).
  - Plant 4 within-category blocks of 2 indicators each (sabotage pair, deception-observer pair, sycophancy pair, sandbagging pair); within-block cosine 0.7, cross-block 0.3.
  - Plant 1000 states per indicator (500 positive, 500 negative), generated as `direction_i + noise` for positives, `random` for negatives.

- [ ] **T4.2** G1 — Indicator-level AU-ROC. Compute per-indicator AU-ROC on held-out states. **Pass:** all 8 indicators ≥ 0.85.

- [ ] **T4.3** G2 — OR-fusion transcript-TPR / turn-FPR. Plant 50 states (out of 1000) with one of the 8 indicators firing; rest benign. Sweep `tau_fire`. **Pass:** transcript-TPR ≥ 0.85 at turn-FPR ≤ 0.05.

- [ ] **T4.4** G3 — Cascade FPR reduction. Same as G2 plus a stub verifier that confirms only if the second-highest score is also above `tau_fire / 2` (label-coherence check). **Pass:** stage-2 reduces turn-FPR by ≥5× at transcript-TPR cost ≤10pp.

- [ ] **T4.5** G4 — Hot-path latency. Benchmark `IndicatorProbeBank::project_all_into` + `or_fused_fire` for N=18, D=72, over 1M iterations after warmup. **Pass:** <200ns per call (matches Plan 292 `FutureBehaviorProbe::forecast` baseline).

- [ ] **T4.6** G5 — Similarity block recovery. Run `IndicatorSimilarityMatrix::cluster(0.6, 0.4)` on the synthetic bank. Compute Adjusted Rand Index vs planted blocks. **Pass:** ARI ≥ 0.9.

- [ ] **T4.7** G6 — Feature-off zero-overhead. Build with `--no-default-features`. Run existing pruners + integrity tests. **Pass:** zero regressions; zero new code compiled in.

- [ ] **T4.8** G7 — Wire-format integrity. Tamper with one byte in the frozen-bytes direction blob; verify `from_frozen_bytes` returns `BankLoadError::HashMismatch`. **Pass:** tamper-evident.

**Phase 4 exit:** GOAT gate green. Promote/demote decision per Phase 5.

---

## Phase 5 — Promotion / Demotion

### Tasks

- [ ] **T5.1** If G1–G7 all pass: promote `indicator_probe_bank` and `indicator_similarity` to **default-on** (they're pure read-side primitives with zero overhead when no bank is loaded — matches Plan 292's `FutureBehaviorProbe` precedent). Keep `indicator_cascade` **opt-in** (it implies a stage-2 verifier impl, which is consumer-crate territory).

- [ ] **T5.2** If G3 (cascade FPR reduction) fails: demote `indicator_cascade` to documentation-only. The bank + similarity matrix still ship as Phase 1+2 output. Note the failure mode in the plan and the research note.

- [ ] **T5.3** If G5 (similarity block recovery) fails: keep `indicator_similarity` opt-in. The cluster API may need a better algorithm than greedy; that's a follow-up issue.

- [ ] **T5.4** Document the GOAT gate results in `.benchmarks/320_indicator_probe_bank_goat.md` with the same format as `.benchmarks/292_fpcg_goat.md` (real measurements, no fabricated numbers).

- [ ] **T5.5** Update `katgpt-rs/README.md` Feature Showcase section with the new primitive (link to research note + plan).

- [ ] **T5.6** Update `katgpt-rs/.docs/15_paper_feature_comparison.md` with the source paper's feature mapping.

**Phase 5 exit:** primitive promoted (or demoted with documented rationale). GOAT gate results committed.

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Real direction-vector training (for the game-runtime half, P1+ in riir-ai) requires data we don't yet have | Medium | The open primitive (this plan) ships with synthetic test banks only. Real training is riir-ai P3 work. §3.5 path-2 modelless check: IRLS logistic regression is closed-form-ish, NOT backprop — it stays modelless. |
| Per-indicator AU-ROC on the low-dim (D=72) NPC state is lower than the paper's D=4096 residual stream | Low–Medium | The paper's hypothesis is that low-dim is *more* separable (single-feature criterion easier to satisfy). G1 will confirm or falsify. Fallback: drop weakest indicators, ship the strong subset. |
| Cascade FPR reduction on the synthetic stub is artificially high (stub verifier is too smart) | Medium | The stub verifier is intentionally simple (label-coherence check, not a real LLM judge). Real-world FPR reduction will be measured in riir-ai Phase 2. |
| `IndicatorLabel` trait adds dynamic-dispatch overhead | Low | Trait is monomorphized per instantiation (`L` is a type parameter, not `dyn`). Zero-cost. |
| BLAKE3 hash of directions + thresholds adds load-time latency | Low | Computed once at init. Runtime holds `Arc<IndicatorProbeBank>`; no per-tick hash. |

---

## Out of Scope (deferred to riir-ai)

- The 18-indicator NPC taxonomy translation (private game-IP).
- Per-NPC indicator sweep loop at 20Hz tick batching.
- Real cascade verifier LLM-judge impl.
- KG-triple emission templates per indicator.
- Trained direction vectors for the 18 NPC indicators.
- Integration with existing integrity layer (`ViolationType::CognitiveMisalignmentViolation` extension, `IntegrityAuditSnapshot` profile extension).
- Cross-persona robustness benchmark (G7 in riir-ai's validation protocol).

These are tracked in `riir-ai/.research/157_bidirectional_cognitive_monitoring_guide.md` §Implementation Priority Table.

---

## TL;DR

Ship a generic `IndicatorProbeBank<L, D>` primitive in `katgpt-rs/crates/katgpt-core/src/pruners/indicator_probe_bank.rs` (Phase 1, always ships), an `IndicatorSimilarityMatrix<L>` artifact (Phase 2), and an `IndicatorCascade<L, D>` trait + driver (Phase 3, opt-in). The bank holds N pre-computed, BLAKE3-committed, freeze/thaw-versioned direction vectors, projects them via dot-product + sigmoid, OR-fuses into a single firing label. The cascade escalates flagged candidates to an opaque verifier (stage-2). All three modules are generic over `L: IndicatorLabel` and `const D: usize` — zero game semantics. Phase 4 ships the synthetic GOAT gate (G1 indicator AU-ROC, G2 OR-fusion TPR/FPR, G3 cascade FPR reduction, G4 zero-alloc hot path, G5 similarity block recovery, G6 feature-off regression, G7 wire-format integrity). Phase 5 promotes `indicator_probe_bank` + `indicator_similarity` to default-on (pure read-side, zero overhead when no bank loaded); keeps `indicator_cascade` opt-in. **The private selling-point moat** (bidirectional cognitive monitoring for emergent NPC alignment, 18-indicator NPC taxonomy, KG-triple audit trail) **lives in `riir-ai/.research/157_*.md` and downstream riir-ai plans** — out of scope for this open plan.
