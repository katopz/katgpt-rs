# Plan 358: Causal Head-Importance Calibration & Scale-Normalized Heterogeneous Fusion

**Date:** 2026-07-02
**Research:** [katgpt-rs/.research/362_HydraHead_Causal_Head_Importance_Hybrid_Attention.md](../.research/362_HydraHead_Causal_Head_Importance_Hybrid_Attention.md)
**Source paper:** [arXiv:2606.20097](https://arxiv.org/abs/2606.20097) — Tan et al., HydraHead, Alibaba, Jun 2026
**Target:** `katgpt-rs/crates/katgpt-core/src/causal_head_importance/` (new module) + Cargo feature `causal_head_importance` (opt-in); RTPurbo wiring in `katgpt-rs/src/rt_turbo/calibration.rs`
**Status:** Active — Phase 1 (skeleton)

---

## Goal

Ship two modelless, inference-time primitives distilled from HydraHead (arXiv:2606.20097):

1. **`CausalHeadImportance`** — a forward-pass-only causal-intervention scorer that ranks attention heads by their *necessity* for a target capability. Produces a head-importance ranking usable as an alternative calibration source for RTPurbo (Plan 126). Strictly stronger than RTPurbo's observational attention-mass scoring: catches *correlated bystander* heads (attend strongly to the needle but are overridden downstream) that attention-mass wrongly promotes.

2. **`ScaleNormalizedFusion`** — a small primitive that fuses outputs from two heterogeneous attention branches (e.g. FA + GDN) via independent RMSNorm per branch + a learnable per-head scalar γ (paper Eq 13–14). Currently unused (Plan 182 mixes layer-wise, not head-wise) but ships ready for any future head-mixing runtime. Static γ beats dynamic softmax gate per paper Table 5.

**Modelless discipline:** zero training, zero backprop through base weights. Both primitives are pure functions / offline calibration tools. The head-importance score runs as forward passes only (the paper itself emphasizes "lightweight and one-shot, requiring only a few forward passes over a small calibration set"). The architecture HydraHead trains (head-wise FA/LA mixing + three-stage transfer) is **out of scope** — noted "→ riir-train" in Research 362 §3.5.

**GOAT gate (the calibration-slot competition):** this primitive enters the **RTPurbo calibration slot** alongside the existing `attention_mass` mode (Plan 126, R086). Two modes compete. GOAT gate decides:
- If `causal_necessity` produces a measurably more accurate partition (G2 synthetic bystander test) **and** calibration latency is acceptable (G3) → promote `causal_necessity` to default calibration mode, demote `attention_mass` to opt-in fallback.
- If `causal_necessity` agrees with `attention_mass` on most workloads (no quality gain) → keep `attention_mass` default, leave `causal_necessity` opt-in for the long-context-extreme regime where bystander heads matter.

**What this plan does NOT do (→ riir-train):**
- Train a hybrid model (head-wise FA/LA mixing as architecture).
- Implement the three-stage transfer pipeline (Stage 1 MSE alignment, Stage 2 KL distillation, Stage 3 long-context NTP).
- Implement branch-specific architecture refinements (FA NoPE+scale+gate, GDN RoPE+MHA expansion, query decomposition).
- Run the 15B-token scaling experiment.

---

## Phase 1 — Skeleton (CORE)

### Tasks

- [ ] **T1.1** Create module directory `katgpt-rs/crates/katgpt-core/src/causal_head_importance/` with `mod.rs`, `readout.rs`, `patching.rs`, `scorer.rs`, `fusion.rs`.
- [ ] **T1.2** Add feature flag `causal_head_importance` to `katgpt-rs/crates/katgpt-core/Cargo.toml` (default-off). No new heavy deps — the primitive operates on `&[f32]` slices and callbacks. Wire into `katgpt-core/src/lib.rs` behind `#[cfg(feature = "causal_head_importance")]`.
- [ ] **T1.3** Define `SpanLogitDiffReadout` in `readout.rs` (paper Eq 9):
  ```rust
  /// Span-level logit-difference readout with exponential decay (paper Eq 9).
  ///
  /// `m(x) = (1/Z) Σ_{j∈A} λ^j · (z_j[a+_j] − z_j[a-_j])`, `Z = Σ λ^j`, `λ = 0.9`.
  ///
  /// Aggregates per-position logit differences between correct (`a+`) and
  /// counterfactual (`a-`) answer tokens across the answer span `A`, weighting
  /// earlier (more informative) positions more heavily via exponential decay.
  ///
  /// Why logit-diff not probability: approximately linear in the residual stream
  /// and monotone in the underlying capability, avoiding softmax-saturation and
  /// probability measurement-floor effects (paper §4.1, Zhang & Nanda 2024).
  #[derive(Clone, Copy, Debug)]
  pub struct SpanLogitDiffReadout {
      /// Exponential decay per answer position. Paper default: 0.9.
      pub lambda: f32,
  }
  
  impl Default for SpanLogitDiffReadout {
      fn default() -> Self { Self { lambda: 0.9 } }
  }
  
  impl SpanLogitDiffReadout {
      /// Compute the readout `m(x)` from per-position `(logit_correct, logit_counterfactual)` pairs.
      ///
      /// `per_position`: `[(z_j[a+_j], z_j[a-_j]); |A|]` in answer order (earliest first).
      /// Returns `m(x) ∈ ℝ`. Larger = stronger capability expression.
      pub fn readout(&self, per_position: &[(f32, f32)]) -> f32 {
          // m(x) = (1/Z) Σ λ^j (z_j[a+] − z_j[a-_j]), Z = Σ λ^j
          let mut numer = 0.0f32;
          let mut denom = 0.0f32;
          let mut w = 1.0f32;
          for &(correct, counterfactual) in per_position {
              numer += w * (correct - counterfactual);
              denom += w;
              w *= self.lambda;
          }
          if denom > 0.0 { numer / denom } else { 0.0 }
      }
  }
  ```
  Unit tests: single position → returns `correct − counterfactual`; two equal-weight positions with λ=1.0 → mean; λ=0.0 → first-position only; empty span → 0.0 (no division by zero).
- [ ] **T1.4** Define the activation-patching direct-effect (receiver) score in `patching.rs` (paper Eq 10):
  ```rust
  /// Normalized causal-importance score for a single head (or any patchable unit)
  /// via activation patching (paper Eq 10).
  ///
  /// `IE = (m(x) − m(x; O ← O(x'))) / (m(x) − m(x'))   ∈ [0, 1]`
  ///
  /// - `m_clean`: readout on the clean input.
  /// - `m_corrupt`: readout on the corrupted input (answer replaced by distractor).
  /// - `m_patched`: readout when the head's output is replaced by its corrupted-run value
  ///   while all other components remain at their clean state.
  ///
  /// `IE ≈ 0` → head is dispensable (safely convertible to a cheaper mechanism);
  /// `IE ≈ 1` → head alone collapses the capability (load-bearing).
  ///
  /// The caller is responsible for the "freeze downstream attention to clean values"
  /// refinement (paper §4.1 / Appendix C.1) — i.e. the patched forward pass must
  /// route the patched signal only through the residual stream + MLPs, not through
  /// downstream attention that could compensate. This probe is the *measurement*;
  /// the patched forward pass is supplied by the caller via a closure.
  #[inline]
  pub fn direct_effect_importance(
      m_clean: f32,
      m_corrupt: f32,
      m_patched: f32,
  ) -> f32 {
      let denom = m_clean - m_corrupt;
      if denom.abs() < f32::EPSILON {
          // m_clean ≈ m_corrupt: the corruption itself doesn't move the readout,
          // so per-head necessity is undefined. Return 0 (treat as not load-bearing
          // for this capability; the capability isn't expressed in this input pair).
          return 0.0;
      }
      let ie = (m_clean - m_patched) / denom;
      // Numerical guard: IE should be in [0, 1] by construction when the patched
      // forward pass is a true substitution. Clamp to handle fp noise.
      ie.clamp(0.0, 1.0)
  }
  ```
  Unit tests: `m_patched = m_clean` → IE = 0 (dispensable); `m_patched = m_corrupt` → IE = 1 (load-bearing); `m_clean = m_corrupt` → IE = 0 (undefined, safe default); intermediate → linear interpolation.
- [ ] **T1.5** Define the path-patching indirect-effect (sender) score in `patching.rs` (paper Eq 11 sender component):
  ```rust
  /// One-step-back indirect-effect (sender) score via path patching.
  ///
  /// For an upstream head `u`, run the corrupted input and record the activations
  /// it sends to a receiver head `r`. Then run an otherwise-clean forward pass
  /// substituting only those recorded activations at `r`'s input. The normalized
  /// drop in the readout is the indirect contribution of `u` through `r`:
  ///
  /// `IE_send(u, r) = (m_clean − m_path_patched(u→r)) / (m_clean − m_corrupt)`
  ///
  /// A head can be causally important without writing the signal directly — by
  /// feeding a receiver. Iterating this (promoting senders to receivers, repeating)
  /// traces the shallow circuit; paper notes long-context retrieval converges in
  /// ~2 rounds.
  ///
  /// `direct_effect_importance` and `indirect_effect_importance` share the same
  /// formula structure; the difference is *what* is patched (head output vs the
  /// pathway into a downstream receiver). Both callers supply `m_patched` from
  /// their own forward-pass machinery.
  #[inline]
  pub fn indirect_effect_importance(
      m_clean: f32,
      m_corrupt: f32,
      m_path_patched: f32,
  ) -> f32 {
      // Same normalization as direct_effect_importance; the semantic difference
      // is in how m_path_patched is produced (path patching vs activation patching).
      direct_effect_importance(m_clean, m_corrupt, m_path_patched)
  }
  ```
- [ ] **T1.6** Define the per-capability score + cross-capability fusion in `scorer.rs` (paper Eq 11–12):
  ```rust
  /// Per-capability head score (paper Eq 11).
  ///
  /// `s_h^(c) = max(IE_recv_h, IE_send_h) · κ_h^(c)`
  ///
  /// where `κ_h^(c)` is task-consistency: the fraction of sub-probes in which
  /// the head exceeds the importance threshold (default 0.01) in its strongest role.
  /// Down-weights heads that score high on a single sub-probe but are negligible
  /// on the rest — favors heads whose contribution is stable across tasks.
  #[inline]
  pub fn per_capability_score(
      ie_receiver: f32,
      ie_sender: f32,
      task_consistency: f32,
  ) -> f32 {
      ie_receiver.max(ie_sender) * task_consistency
  }
  
  /// Fuse per-capability scores into a single head ranking (paper Eq 12).
  ///
  /// Min-max normalizes each capability's scores to [0,1] (per-capability drops
  /// differ in scale), then takes the weighted mean across capabilities with
  /// equal weights by default (equal prior over capabilities when no task pref).
  ///
  /// `per_capability_scores[h]` = `Vec<(capability_weight, raw_score)>` per head.
  /// Returns `Vec<f32>` of length `n_heads`, one fused score per head.
  ///
  /// Min-max normalization is per-capability across heads: for each capability c,
  /// `ŝ_h^(c) = (s_h^(c) − min_h s_h^(c)) / (max_h s_h^(c) − min_h s_h^(c))`.
  pub fn fuse_across_capabilities(
      per_head_per_capability: &[Vec<(f32, f32)>], // [n_heads] of [(weight, raw_score); n_capabilities]
  ) -> Vec<f32> {
      let n_heads = per_head_per_capability.len();
      if n_heads == 0 { return Vec::new(); }
      let n_caps = per_head_per_capability[0].len();
      if n_caps == 0 { return vec![0.0; n_heads]; }
  
      // Per-capability min-max normalization across heads.
      let mut normalized: Vec<Vec<f32>> = (0..n_heads).map(|_| Vec::with_capacity(n_caps)).collect();
      for c in 0..n_caps {
          let mut mn = f32::INFINITY;
          let mut mx = f32::NEG_INFINITY;
          for h in 0..n_heads {
              let s = per_head_per_capability[h][c].1;
              if s < mn { mn = s; }
              if s > mx { mx = s; }
          }
          let range = mx - mn;
          for h in 0..n_heads {
              let s = per_head_per_capability[h][c].1;
              normalized[h].push(if range.abs() < f32::EPSILON { 0.0 } else { (s - mn) / range });
          }
      }
  
      // Weighted-mean fusion across capabilities (normalize weights to sum 1).
      let mut out = vec![0.0f32; n_heads];
      for h in 0..n_heads {
          let mut total_w = 0.0f32;
          let mut acc = 0.0f32;
          for c in 0..n_caps {
              let w = per_head_per_capability[h][c].0;
              acc += w * normalized[h][c];
              total_w += w;
          }
          out[h] = if total_w > 0.0 { acc / total_w } else { 0.0 };
      }
      out
  }
  ```
  Unit tests: single capability + single head → normalized to 0 (range=0 guard); two capabilities equal weight → mean; min-max normalizes per-capability correctly.
- [ ] **T1.7** Define the head-importance ranking → partition helper in `scorer.rs` (mirrors RTPurbo's `calibrate_from_scores`):
  ```rust
  /// Rank heads by causal-importance score and partition into critical vs
  /// convertible sets, mirroring RTPurbo's `HeadCalibration` shape.
  ///
  /// `critical_ratio` is the fraction of heads to retain (paper default: 0.25
  /// for FA in the hybrid; RTPurbo default: 0.15 for retrieval heads).
  /// `min_one_per_layer` (paper "Constrained Global Screening" §5.6): if Some,
  /// guarantee at least one critical head per layer (caller supplies layer ids).
  ///
  /// Returns `(critical_set, convertible_set)` as sorted `Vec<usize>` of head indices.
  pub fn partition_by_causal_score(
      scores: &[f32],
      critical_ratio: f32,
      layer_ids: Option<&[usize]>,
      min_one_per_layer: bool,
  ) -> (Vec<usize>, Vec<usize>) {
      // ... sort by score desc, take top ceil(n * critical_ratio),
      //     if min_one_per_layer: for each layer not yet represented, promote
      //     its highest-scoring head into the critical set.
  }
  ```
  Unit tests: empty → empty; single head → critical; `min_one_per_layer` rescues an unrepresented layer; ties broken by index.
- [ ] **T1.8** Define `ScaleNormalizedFusion` in `fusion.rs` (paper Eq 13–14):
  ```rust
  /// Scale-normalized fusion of heterogeneous attention branch outputs
  /// (paper Eq 13–14). Independent RMSNorm per branch + index-preserving
  /// concatenation + learnable per-head scalar γ.
  ///
  /// Why: FA softmax produces sharp low-entropy distributions dominated by query
  /// norm; GDN normalization cancels query norm → smoother high-entropy outputs.
  /// Naive concatenation destabilizes (paper Table 5: -10% RULER Single w/o Norm).
  /// Independent RMSNorm per branch unifies feature scales; per-head γ lets the
  /// model adaptively recalibrate each head's contribution. Static γ beats
  /// dynamic softmax gate (paper Table 5).
  ///
  /// Generic over any two branches identified by a per-head `BranchKind` tag.
  #[derive(Clone, Debug)]
  pub struct ScaleNormalizedFusion {
      /// Per-head learnable scalar γ. Length = n_heads. Default 1.0 (identity).
      pub gamma: Vec<f32>,
      /// RMSNorm epsilon.
      pub eps: f32,
  }
  
  impl ScaleNormalizedFusion {
      pub fn new(n_heads: usize, eps: f32) -> Self {
          Self { gamma: vec![1.0; n_heads], eps }
      }
  
      /// Fuse per-head outputs in-place into `out` (length `n_heads * head_dim`,
      /// row-major). `per_head_outputs[h]` is the raw output of head h (already
      /// routed from its branch — caller does the FA-vs-GDN dispatch). Each head's
      /// output is independently RMSNormed, then multiplied by `gamma[h]`, then
      /// written into `out` at the head's index-preserving slot.
      ///
      /// Zero-allocation: scratch `norm_buf` is caller-supplied (length `head_dim`).
      #[inline]
      pub fn fuse_into(
          &self,
          per_head_outputs: &[&[f32]], // [n_heads][head_dim]
          head_dim: usize,
          out: &mut [f32],             // [n_heads * head_dim]
          norm_scratch: &mut [f32],    // [head_dim]
      ) {
          let n_heads = per_head_outputs.len();
          debug_assert_eq!(out.len(), n_heads * head_dim);
          debug_assert_eq!(norm_scratch.len(), head_dim);
          for h in 0..n_heads {
              let src = per_head_outputs[h];
              debug_assert_eq!(src.len(), head_dim);
              // RMSNorm in-place into scratch
              let mut sum_sq = 0.0f32;
              for v in src { sum_sq += v * v; }
              let rms = (sum_sq / head_dim as f32 + self.eps).sqrt();
              let inv_rms = 1.0 / rms;
              for j in 0..head_dim {
                  norm_scratch[j] = src[j] * inv_rms;
              }
              // γ-scale + write into index-preserving slot
              let g = self.gamma[h];
              let slot = h * head_dim;
              for j in 0..head_dim {
                  out[slot + j] = g * norm_scratch[j];
              }
          }
      }
  }
  ```
  Unit tests: identity γ=1 → output is RMSNorm of input; γ=0 → zeros; γ=2 → 2× RMSNormed; mixed branches (synthetic FA-sharp + GDN-flat inputs) → both normalized to same scale.
- [ ] **T1.9** Re-export public API from `mod.rs` behind the feature flag.
- [ ] **T1.10** Verify standalone compile: `CARGO_TARGET_DIR=/tmp/katgpt_358 cargo check -p katgpt-core --features causal_head_importance` (per AGENTS.md rule).

**Phase 1 exit criterion:** module compiles standalone; all unit tests pass; both primitives are instantiable in isolation.

---

## Phase 2 — GOAT Gate (G1 + G3 + G4)

### Tasks

- [ ] **T2.1 (G1 — correctness)** Unit tests in `tests/causal_head_importance_g1.rs`:
  - Synthetic FA head harness: `n_heads` heads, of which a known `k_load_bearing` subset writes the signal into the readout, and `n_heads − k_load_bearing` are *correlated bystanders* (attend to the needle but project to zero in the readout direction).
  - Compute IE scores via the closure-based patched-forward-pass (mock the forward pass as a linear map so `m_patched` is exactly computable).
  - Assert: load-bearing heads all have IE > threshold (0.01); bystanders all have IE < threshold; ranking puts load-bearing above bystanders (Spearman ρ = 1.0 on this clean synthetic).
  - **Knockout faithfulness** (paper Fig 9b reproduction): ablating heads by `−IE` collapses the synthetic capability after the top-k; random ablation controls stay high. Assert: top-k-knockout readout < 0.2 × baseline; random-k-knockout readout > 0.8 × baseline.
- [ ] **T2.2 (G3 — calibration latency)** Benchmark `partition_by_causal_score` (the *offline* calibration step, after patched forward passes are done) against RTPurbo's `calibrate_from_scores`. Target: ≤ 2× of attention-mass calibration for the partition step itself (the forward passes are a separate, amortized cost — paper emphasizes ~6 samples suffices). Use `criterion` at `benches/causal_head_importance_g3.rs`. Head counts: 16, 64, 144.
- [ ] **T2.3 (G4 — zero-alloc hot path)** The `direct_effect_importance` and `indirect_effect_importance` functions are `#[inline]` and allocate nothing. The `ScaleNormalizedFusion::fuse_into` takes caller-supplied scratch and writes in-place. Verify with a `#[cfg(test)]` alloc-tracking test or by inspection + criterion memory profile.
- [ ] **T2.4** Run full crate test suite: `CARGO_TARGET_DIR=/tmp/katgpt_358 cargo test -p katgpt-core --features causal_head_importance --lib`. No regressions.

**Phase 2 exit criterion:** G1 + G3 + G4 green. Feature remains opt-in.

---

## Phase 3 — G2 (Causal vs attention-mass discrimination)

The paper's strongest quality claim for the causal score is that it filters *correlated bystanders* — heads that observational attention-mass wrongly promotes. G2 reproduces this discrimination on a synthetic harness.

### Tasks

- [ ] **T3.1** Build the bystander harness in `tests/causal_head_importance_g2.rs`:
  - Generate `n_heads = 16` synthetic heads. For each head, define (a) an attention pattern (where it attends — needle vs local), (b) an output projection (does its output actually move the readout).
  - **K load-bearing** heads: attend to needle AND project into the readout direction.
  - **M correlated-bystander** heads: attend to needle (high attention-mass score!) but project to zero (output orthogonal to readout).
  - **N local** heads: attend locally, project to zero.
- [ ] **T3.2** Compute two partitions:
  - **Attention-mass** (RTPurbo-style): rank by needle attention mass → top-K includes bystanders.
  - **Causal necessity** (this plan): rank by IE score → top-K excludes bystanders (their IE = 0 because patching them doesn't move the readout).
- [ ] **T3.3** Assert the discrimination: causal partition's top-K set is *exactly* the load-bearing set (Jaccard = 1.0); attention-mass partition's top-K set includes ≥ 1 bystander (Jaccard < 1.0). Compute the head-set difference: `|causal_K ∩ load_bearing| / K = 1.0` and `|attention_mass_K ∩ load_bearing| / K < 1.0`.
- [ ] **T3.4** Vary the bystander fraction (0%, 25%, 50%) and show causal partition is invariant to bystander fraction while attention-mass partition degrades. Plot or table.

**Phase 3 exit criterion:** G2 demonstrates causal > attention-mass on the bystander workload. Promote/demote decision per the §Goal tracking rule.

---

## Phase 4 — RTPurbo wiring + promote/demote decision

### Tasks

- [ ] **T4.1** Add `calibrate_from_causal_scores` to `katgpt-rs/src/rt_turbo/calibration.rs` as a sibling to the existing `calibrate_from_scores` (attention-mass). Same output type (`HeadCalibration`), different input score semantics. Document the semantic difference (causal necessity vs observational mass) in the docstring.
- [ ] **T4.2** Add a `CalibrationMode` enum to `katgpt-rs/src/types.rs` (or `rt_turbo/types.rs`): `AttentionMass` (current default) | `CausalNecessity` (this plan, requires `causal_head_importance` feature). Wire into `RtTurboConfig`.
- [ ] **T4.3** Update `examples/rt_turbo_01_calibration.rs` to demonstrate both modes side-by-side on the synthetic harness from Phase 3.
- [ ] **T4.4** **Promote/demote decision (per §Goal):**
  - If G2 shows causal strictly dominates (bystander exclusion is real and matters at production head counts) **AND** G3 latency is acceptable → promote `CausalNecessity` to default `CalibrationMode`, demote `AttentionMass` to opt-in fallback. Update `RtTurboConfig::default()`.
  - If G2 shows they agree on most workloads (bystanders rare in practice) → keep `AttentionMass` default, leave `CausalNecessity` opt-in for the long-context-extreme regime. Document the regime boundary.
  - Record the decision in `.benchmarks/358_causal_head_importance_goat.md` with the G1/G2/G3/G4 numbers and the promote/demote verdict.
- [ ] **T4.5** Update `katgpt-rs/README.md` Feature Showcase with a short entry (mirroring the RTPurbo entry format) once Phase 4 decision is made.
- [ ] **T4.6** Tag release per AGENTS.md commit convention: `feat(calibration): causal head-importance scorer + scale-normalized fusion (Plan 358, Research 362, arXiv:2606.20097)`.

**Phase 4 exit criterion:** RTPurbo accepts both calibration modes; promote/demote decision recorded with evidence; README updated.

---

## Phase 5 — Docs + cross-refs

### Tasks

- [ ] **T5.1** Add `.docs/causal_head_importance.md` (mirroring `.docs/faithfulness_probe.md` format) documenting the primitive, the calibration-slot competition, and the promote/demote outcome.
- [ ] **T5.2** Cross-ref from `.research/086_RTPurbo_Retrieval_Head_Sparse_Attention.md` (add a "See also: Research 362 — causal alternative to attention-mass calibration" note in the Distillation section).
- [ ] **T5.3** Cross-ref from `.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md` (add a "See also: Research 362 — path-patching indirect-effect extension to direct-effect FaithfulnessProbe" note).
- [ ] **T5.4** Note the deferred riir-ai / riir-neuron-db follow-ups in their respective `.research/` folders (or `.issues/` if those repos prefer): HLA direction-vector importance, NeuronShard dendritic-branch importance. **Do NOT implement** — just note the cross-ref for when those repos scope the work.

**Phase 5 exit criterion:** docs land; cross-refs in place; private-repo follow-ups noted.

---

## Risks & limitations

1. **Closure-based patched forward pass is caller-supplied.** The primitive measures IE given `m_clean`, `m_corrupt`, `m_patched`; it does *not* implement the patched forward pass itself (that requires a full transformer forward with selective substitution + downstream-attention freezing, which is riir-engine / riir-games territory). The katgpt-rs primitive is the *scorer*; the *patched forward pass* is the caller's responsibility. This keeps katgpt-rs leaf-clean (no transformer dep) and matches the FaithfulnessProbe pattern (probe is generic, consumer supplies the behavior metric).
2. **Calibration latency.** Causal patching is `O(n_heads × n_calibration_samples)` forward passes vs RTPurbo's `O(1)` forward pass + per-head mass scan. ~10–100× slower calibration. Acceptable since calibration is offline and amortized (one-time per model). The partition step itself (T2.2) is the latency-sensitive comparison and should be competitive.
3. **Bystander heads may be rare in production models.** The paper's bystander effect is documented on Qwen3-1.7B; whether real game-AI models exhibit enough bystanders to justify the calibration cost is an open question. G2 (Phase 3) is synthetic; T4.4 promote/demote may keep `AttentionMass` default if bystanders are rare in practice.
4. **No real-model validation in this plan.** G1/G2 are synthetic. Real-model validation (running the scorer against an actual transformer on actual NIAH tasks) is riir-engine work — defer to a riir-ai plan if the GOAT gate passes and we want production evidence. The synthetic G1/G2 is sufficient for the promote/demote decision at the primitive level.
5. **Path patching iteration count.** The paper notes long-context retrieval converges in ~2 rounds; we ship only one-step-back (T1.5). Multi-round iteration is a caller-side loop calling `indirect_effect_importance` repeatedly with promoted senders. Document but don't implement the loop in katgpt-rs (keeps the primitive single-step).

---

## Out of scope (→ riir-train / riir-engine / riir-ai / riir-neuron-db)

- **→ riir-train**: head-wise FA/LA mixing architecture; three-stage transfer pipeline; branch-specific architecture refinements (FA NoPE+scale+gate, GDN RoPE+MHA, query decomposition); 15B-token scaling run.
- **→ riir-engine**: the patched-forward-pass implementation (selective head-output substitution + downstream-attention freezing) needed to actually compute `m_patched` on a real transformer.
- **→ riir-ai**: HLA direction-vector causal importance (Research 362 §2.5(a)) — applies the open `CausalHeadImportance` primitive to HLA's 8-dim affect space.
- **→ riir-neuron-db**: NeuronShard dendritic-branch causal importance (Research 362 §2.5(e)) — applies the primitive to `dendritic_lora` branch views for selective branch freeze/thaw.

---

## TL;DR

Ship two modelless primitives distilled from HydraHead (arXiv:2606.20097): **`CausalHeadImportance`** (activation-patching + path-patching + span-level logit-diff readout, paper Eq 9–12) and **`ScaleNormalizedFusion`** (independent RMSNorm per branch + learnable per-head γ, paper Eq 13–14). The causal scorer is strictly stronger than RTPurbo's observational attention-mass calibration (catches correlated-bystander heads); the fusion primitive is shipped ready for any future head-mixing runtime (currently unused — Plan 182 is layer-wise). Wire causal scoring as an alternative `CalibrationMode` in RTPurbo (Plan 126). Architecture itself → riir-train. **GOAT gate decides promote-to-default vs `attention_mass`** based on whether the bystander-exclusion effect (G2 synthetic) is real and production-relevant. Feature `causal_head_importance` ships opt-in; G1/G2/G3/G4 required; promote/demote recorded in `.benchmarks/358_*.md`.
