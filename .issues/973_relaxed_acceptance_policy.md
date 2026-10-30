# Issue 973 — Relaxed acceptance policy for speculative verification (arXiv:2610.26693 adoption)

**Status:** OPEN — POC/optimization task; modelless track of riir-train Research 484; no training dependency; quality-calibration half lives in the riir-ai twin (ai-1142).

## The finding

The PLM paper (arXiv:2610.26693, "Parallel Language Models via Variational Decoding",
Draxler et al. 2026) runs speculative verification with a **relaxed acceptance rule**
(Algorithm 1). Roles, stated precisely: the DRAFT side proposes `x̂_j` (decoder argmax
under the drawn auxiliaries); the VERIFY side decodes `x̃_j = f^Voronoi(u_j; q_j)` — the
auxiliary decoded under the CURRENT verifier law `q_j = q(x_j | x_<j)`. The rule:

```
accept(x̂_j) ⇔ [x̂_j == x̃_j] ∨ [x̂_j in ρ-nucleus of q_j] ∨ [q_j(x̂_j) ≥ ε]
```

On rejection the emitted token is **`x̃_j` — the verifier-side decode — with NO residual
correction**. That emission choice is what makes the corner exact: at ρ=0, ε=1 every
emitted token (accepted proposal when it matches, else the decode) is
`f^Voronoi(u; q)` under a uniform draw — a verifier q-sample by the paper's Corollary 1
(mass-matched cells: uniform-u sampling reproduces q exactly, up to cell rounding).

Relaxed acceptance is deliberately lossy: the nucleus/threshold terms admit merely
plausible tokens the exact rollout would have rejected. Measured: **+46.5% accepted
tokens/call at ≤1% MAUVE loss** (ε=2e-3 pretrained, ρ=0.94 post-trained — constants are
posture-fitted; the RULE is the shippable object).

**Stack formulation for OUR drafters** (no Voronoi machinery needed): draw a
verifier-side sample `x̃ ~ q` per position (one U(0,1) draw + inverse-CDF walk — the
same role the paper's Voronoi decode plays); accept a drafted token `d` iff:

```
accept(d) ⇔ [d == x̃] ∨ [d in ρ-nucleus of q] ∨ [q(d) ≥ ε]
```

On rejection the emitted token is **`x̃` — the verifier-side q-draw — with NO residual
correction**. This mirrors the paper exactly: the first term compares against a
verifier-side DRAW, never the argmax. That is what makes the corner exact: at ρ=0,
ε=1 every emitted token is `x̃ ~ q` whether accepted or emitted as fallback — the
output is exactly q-distributed (Corollary 1). ⛔ Do NOT replace the first term with
`[d == argmax q]`: at the corner that always accepts the argmax proposal, emitting it
with probability 1 instead of q(argmax) — precisely the `PrefixMatch`
distribution-biasing failure the d2f_verifier doc records. Cost note: exactness at
the corner prices one extra uniform draw per position; under a **greedy-verify
posture** the fallback x̃ = argmax (deterministic decode) and the policy reduces to
`PrefixMatch` — the argmax bias then belongs to the posture choice, not the policy.

Signal-diff against the shipped cousins (research skill §3.6):

- `DraftAcceptPolicy::TruncatedArgmax` (`crates/katgpt-forward/src/d2f_verifier.rs`) is
  approximate but **rejection-sampling shaped** — it tests `u ≤ p̃(d)` against a
  truncated target law and needs the uniform draw + residual correction; it preserves a
  (modified) distribution. The relaxed rule is a **deterministic likelihood
  accept-gate with no residual correction** — deliberately lossy, calibrated to a
  distributional quality budget. Different shape.
- `entropy_bounded` commit gates (`crates/katgpt-speculative/src/entropy_bounded.rs`)
  consume the **drafter's own entropy** (`Σ H − max H ≤ γ`) — a proposal-side pruning
  gate. The relaxed rule consumes the **verifier's likelihood of the proposal** — an
  acceptance-side gate. Different signal.

So the deterministic nucleus/ε accept-gate does not ship. Prior art is published (the
PLM paper; adjacent relaxed-verification work — EMNLP 2025 Alignment-Augmented Spec
Decoding citing Bachmann et al. 2025): **no literature novelty claimed**; this issue is
internal adoption.

## Tasks (katgpt-rs scope — modelless only)

- [ ] `NucleusAccept { rho, eps }` variant on `DraftAcceptPolicy`
      (`crates/katgpt-forward/src/d2f_verifier.rs`) + the equivalent arm in the
      `verifier_trait`/tree verify paths it composes with; zero-alloc, single pass over
      the preallocated top-k buffer, pinned first-index tie rule (Issue-940
      determinism law). Document the rejection emission (verifier draw; argmax under
      greedy postures) on the variant. **Seed independence (verdict round 3 note):**
      the uniform draw behind `x̃` MUST come from a separate seed stream (e.g. a
      different splitmix domain tag) than the drafter's randomness — if a sampled
      drafter reuses the same u, `d == x̃` holds by construction, every draft is
      accepted at the corner, and G1a fails. One doc-comment line; the G1a seeded
      fixtures are the net that catches it.
- [ ] **G1a corner-DISTRIBUTIONAL test:** at ρ=0, ε=1 the emitted token distribution
      equals the verifier's law q (TV/χ² over seeded streams vs direct q-sampling).
      Mechanism, stated precisely: under sampled verification the corner accepts `d`
      with probability q(d) — `d` matches an independent q-draw `x̃`; `SoftmaxArgmax`
      ALSO accepts with probability q(d), through its threshold test `u ≤ q(d)`. Same
      acceptance probabilities, same output distribution; **only the per-draw
      decisions differ** (inverse-CDF matching vs threshold test) — which is why no
      draw-level bit-identity claim against `SoftmaxArgmax` may be written.
- [ ] **G1b greedy-posture bit test vs `PrefixMatch`:** under the greedy-verify
      posture (fallback x̃ = argmax q), the corner is bit-identical to `PrefixMatch`
      on **every fixture draw — accepts AND rejects** (the reject arm is the
      discriminating half; a restricted accepts-only fixture set cannot fail).
- [ ] **G1c monotonicity property test:** loosening (ρ↑ or ε↓) never decreases the
      accepted fraction, on fixtures.
- [ ] Planted-violation arm: a miscalibrated nucleus (off-by-one mass, overlapping
      bands) must red the property test.
- [ ] Throughput A/B on fixtures: accepted/call strict vs relaxed (no model-quality
      metric — that half is ai-1142); `--release`; box-state PROVENANCE line beside any
      latency figure.
- [ ] Scope extensions (separate tasks, same seam): quadratic **salvage refill**
      (reuse the proposal matrix's off-diagonal entries after a rejection instead of
      prefix-restart — needs a drafter exposing its K×K proposal matrix; `dd_tree` /
      `progressive_mcgs` are the candidates); **round-width sweep law** (interior
      K_pred optimum + break-even arithmetic E[accepted]/W — pin OUR constant, the
      paper's 7 is GPU-specific).
- [ ] On gates pass: record the bench, promote per flag discipline if a consumer lane
      adopts it; the corner-exact posture ships regardless (it anchors the tests).

## Out of scope here (deliberately — modelless boundary)

The **quality half** of the calibration — MAUVE-budget fitting of (ρ, ε), per-family
conditional-retention gates, the ≥ +25%-accepts-at-≤1%-MAUVE floor — needs a model
embedder and lives in the consumer lane: **riir-ai Issue 1142** (the twin). katgpt-rs
ships the policy + property tests; the twin owns the quality gate. Boundary check:
katgpt-rs dep allowlist carries no transformer-embedder surface; adding MAUVE here
would be a boundary violation, not a convenience.

## References

- riir-train Research 484 (full distill + Path 0 table + signal-diffs)
- riir-train Plan 477 (the training twin; its G3 reuses ai-1142's calibration lane)
- riir-ai Issue 1142 (the quality-calibration twin)
- `crates/katgpt-forward/src/d2f_verifier.rs` (`DraftAcceptPolicy`)
- `crates/katgpt-speculative/src/entropy_bounded.rs` (the adjacent gate family)
