# Plan 627 — the `set_refine` operator: the one capability an independent option scorer structurally lacks

**Status:** T1-T3 COMPLETE 2026-10-16 — T3 GOAT verdict: MEASURED NEGATIVE (G1 fail, monotone in γ; reflex Bench 150). The lane stays opt-in katgpt-core-side (`set_refine` feature) + reflex-side (`set_refine` feature, knob default 0.0); promotion REFUSED on G1 — T4's proposal is moot (the plan's own law: a negative is a valid landing). See §5 records.
for a new katgpt-core primitive; implementation unclaimed. Downstream of reflex
Issue 090 C2 (the Jeva fusion follow-up; the issue text is the seed spec). The
promotion decision stays gated on the MEASURED GOAT (§4) — nothing here promotes
or wires by default.
**Claimed:** 4090-setrefine, 1792106760, expires 2026-10-17T23:26:00Z
**Note:** claim scope T1-T3; the claim epoch was 2026-10-15T23:26Z (format-normalized to epoch digits by set-refine-t1-m3, semantics unchanged).

## 1. Substrate-first verdict (the gate this plan exists to record)

Vocabulary translation searched 3+ name variants across `katgpt-rs/**/*.rs`,
`.plans/`, `.research/` (2026-10-16, both boxes' canonical checkout):

| Variant | Result |
|---|---|
| `set_refine` / `set-refine` / `SetRefine` | ZERO hits |
| `option_interaction` / `set_dependent` / `preference_reversal` / `cross_option` / `option_conditioned` | ZERO hits in katgpt-rs (`option_cond` exists REFLEX-side — count-table terms keyed (qid, option); state-side signal, NOT option-interaction feedback) |
| `hopfield` / query-update / state-update / set-aware / decoy | **`cp_hopfield`** (Plan 567, opt-in) — the closest prior art |

**Adjacent substrate that EXISTS and will be consumed, not duplicated:**
- `exact_sigmoid` — the gate (never softmax; the C2 `w_i = σ(dot(x,k_i)/τ)` form).
- `cp_hopfield` (Plan 567) — the same FAMILY (set-conditioned state update over a
  stored pattern set: `recall_step` aggregates the set into the state), but a
  different job: associative-memory CAPACITY on CP^(d-1) via top-eigenvector
  alignment, opt-in, GOAT'd at N∈{8,64}. NOT a decision-option scoring operator —
  cited as prior art and a design warning (its machinery is orders heavier than
  the C2 operator needs).
- `spectral_pencil` / `basis_split` — direction-vector vocabulary (the k_i keys
  are plain embedded option rows; no learned directions needed at v1).
- Reflex engine seams the consumer side already ships: `Scratch::domains`,
  `semantic_route` (Bench 149), the `--rubric-probe` lane's route-agreement
  characterization (context==state-alone 59.7% — the measured gap a
  set-conditioned term would close or consciously not close).


**AMENDED 2026-10-16 (T1 claim, second substrate pass at implementation time):** the
verdict above is INCOMPLETE — one more pass found the KERNEL substrate the table's
vocabulary missed (the vocabulary-translation trap the skill warns about: "cross-Datapoint
Set Attention" matches none of `set_refine`/`option_interaction`/`preference_reversal`/
`cross_option`/`option_conditioned`/`set_dependent`):

- **`set_attention` (Plan 354, opt-in feature `set_attention`)** ships
  `set_sigmoid_attention_into`: `h_i + (gamma/N) * sum_j sigma((q_i.k_j)/sqrt(k)*beta) *
  (v_j - h_i)` with identity projections shipped (`identity_projection`), `w_v: Option`
  (None = identity values), zero-alloc caller-owned scratch, and G1
  permutation-equivariance (bit-exact) + G4 alloc gates ALREADY SHIPPED
  (`tests/set_attention_g1_g5.rs`). The C2 operator is the single-query /
  identity-projection special case: state `x` = row 0, options `k_i` = rows 1..=N, `w_q =
  w_k = identity`, `w_v = None`, `beta = sqrt(d)/tau` => `alpha_i = sigma(dot(x, k_i)/tau)
  = w_i` exactly. Options are READ ONLY (the kernel writes only `output`) — S4
  immutability by construction.
- **Residual form (amends the section-2 formula):** the shipped form is the N-normalized
  PULL form `x' = x + (gamma/N) * sum_i w_i * (k_i - x)` rather than the raw additive
  `x + g * sum_i w_i * k_i` — the `/N` makes gamma invariant to set size (structurally
  solving the S4 blowup concern: the sum mass cannot grow with N) and the `-x` self-damp
  keeps `||x'||` near `||x||` at high option mass. Same mechanism class (sigmoid-gated
  kernel blend, options immutable, sum-symmetric). T1 ships the kernel's form; the suite
  pins whichever form ships.

**Second implementation-time discovery (T1):** the legacy kernel entry point carries a
release `assert!(d <= 16)` (a `[f32; 16]` stack accumulation scratch — an implementation
artifact, not a design boundary) and reflex's `EMBED_DIM` is 256. T1 therefore ALSO added
`set_sigmoid_attention_wide_into` to Plan 354's module: identical math, caller-owned
`sum_av` scratch (the zero-alloc law holds), no d cap; the legacy `set_sigmoid_attention_into`
is now a thin stack-scratch wrapper, bit-identical at d <= 16 (asserted by test in
`tests/set_attention_g1_g5.rs`, dense + top-k). `set_refine` consumes the wide entry.

**T1 LANDED 2026-10-16** — 2160 lib tests + Plan-354 G1-G5 (7) + CLR G8 (3) + the S-suite
(9: S1 reversal capability incl. the independent-scorer contrast arm, S1b ordinal flip,
S2 same-order bit-identity + 6-permutation drift <= 1e-6, S3 sigmoid-not-softmax
denominator-free, S4 option immutability + displacement bound, S5 gamma=0 byte-identity,
d=256 wide smoke, 100-call stability) — all green; clippy `-D warnings --all-targets`
clean at the feature posture; default-features check clean (flag-off compiles to nothing).
The S1 fixture: x=e1, A=rot(+25deg), B=rot(-35deg), D=rot(-45deg), tau=0.5, gamma=1.0
(working envelope tau 0.4-0.7 x gamma 0.8-1.5; gamma >= 2 over-refines and flips {A,B}
directly — the documented envelope, not a defect).

**REVISED VERDICT: the operator is NOT new substrate — it is a thin composition (~60-80 LOC
wrapper) over `set_sigmoid_attention_into`, feature `set_refine = ["set_attention"]`.**
The genuinely NEW part is the S1-S5 decision-plane suite (Plan 354's tests are entity-plane).

**Verdict: the operator itself is NEW, small substrate.** ~100–150 LOC + types +
probe suite + benches. File in katgpt-core (this repo — modelless inference
primitive, no riir dep, upstream of everything). No parallel-system risk: nothing
in the workspace re-implements set-conditioned option scoring under another name.

## 2. The operator (flag-gated `set_refine`, inert by default)

Their Prop 1 (arXiv:2610.18773): an independent per-option scorer cannot REVERSE
an A-vs-B preference when the candidate set changes; Jeva's fix is one
candidate-conditioned attention block. The modelless analog to price:

```text
x' = x ⊕ g · Σ_i w_i · k_i        w_i = σ(dot(x, k_i) / τ)   (never softmax)
```

- `x` — the state embedding (the engine's state-alone vector, D-dim f32).
- `k_i` — the option embeddings (immutable; options never mutated).
- `g` — the gate scale (0.0 = OFF, byte-identical posture; the G3 identity).
- Permutation invariance by construction (B2): the Σ over options is
  order-symmetric — assert to bit-identity in the suite.
- Value-blind argmax preserved (C1): the operator reads scores/keys, never a
  value vector — assert the C1 covariance identities still hold on refined
  distributions.
- **Invariance scope is NOT A1–A3**: set-dependent reversal REQUIRES option
  interaction — a working `set_refine` deliberately BREAKS reflex A2 by design
  and may move A1/A3 wherever coverage reads refined scores. The exemption and
  its own gate suite are the issue's law; do NOT "fix" the A2 red.

## 3. The set-awareness probe suite (the gate A2 cannot provide)

A suite that REDS unless the scorer can reverse set-dependence — the capability
test, run at every posture:

- [x] **S1 reversal capability**: constructible fixture where {A,B} picks A and
      {A,B,+decoys shaped like B} picks B (the Jeva Prop-1 demonstration, modelless
      edition) — the suite's existence proof; a scorer that cannot flip reds.
      → LANDED 2026-10-16 — S-suite 9/9 green (T1, katgpt-rs `0fb56b269`): the refined
      scorer flips A-to-B when a B-shaped decoy joins, decoy never wins; the
      independent-scorer CANNOT flip (asserted negative control).
- [x] **S2 permutation invariance to bit-identity**: refined scores identical
      under option permutation (the Σ symmetry), every fixture. → LANDED (S-suite).
- [x] **S3 monotone-gate check**: w_i ∈ (0,1), Σ-shaped response to dot(x,k_i)
      (sigmoid, never softmax — the denominator-free law; a softmax spelling reds).
      → LANDED (S-suite).
- [x] **S4 immutability**: k_i bit-identical before/after the call; x' bounded
      (‖x'‖ ≤ ‖x‖ + g·Σw_i·‖k_i‖ — no norm blowup). → LANDED (S-suite).
- [x] **S5 flag-off identity**: g=0 ⇒ byte-identical scores (G3's engine-side twin).
      → LANDED (S-suite; the engine-side twin is the γ=0.0 default posture,
      `seam_equals_substrate_composition` + the 0-flip controls).

## 4. The GOAT gates (all four, `--release`, box state quoted)

- [x] **G1** flag-on beats flag-off on a REAL decision suite beyond the
      conformal-naive floor (reflex harness A/B, paired, one frozen read —
      Report-the-Floor law). → **MEASURED FAIL** (Bench 150): −12.3..−14.0 pt on
      banking77/massive/ag_news at γ=1.0/τ=0.5, monotone in γ — structural (unit-
      centroid keys regress the state toward the corpus mean). Promotion refused.
- [x] **G2** latency p99 ≤ the engine's 1 ms per decision set budget with the
      operator armed. → PASS at bench scale (454 µs p99; the O(n·d²) identity-
      matmul caveat ~9 ms @ k=77 disclosed, mooted by G1).
- [x] **G3** flag-off hash-identity (byte-identical boots, the combo-lane law).
      → PASS (identity gates + the typed/prompt_injections 0-flip controls).
- [x] **G4** alloc-free on the hot path (the canary-armed counting allocator).
      → PASS (0 allocs post-warmup).
- [x] Only if ALL pass AND the gain is modelless: the promotion question goes to
      a proposal, never straight to default. Any miss ⇒ stays opt-in with the
      measured record (a negative is a valid landing — the Bench-109 shape).
      → RESOLVED BY THE MEASURED NEGATIVE: G1 missed ⇒ stays opt-in with the
      record (reflex Bench 150; T4 below carries the verdict).

## 5. Execution order (each its own session slice)

- [x] T1 katgpt-core: `set_refine` module (feature `set_refine`, opt-in) —
      operator + types.rs + S1–S5 unit suite. REVISED per the amendment: a thin composition wrapper over `set_sigmoid_attention_into` (`set_refine = ["set_attention"]`), NOT a fresh kernel — the original ~100-150 LOC estimate covered machinery Plan 354 already ships. Gate: `cargo test -p katgpt-core
      --features set_refine --lib`, clippy `-D warnings` default + all-features.
- [x] T2 reflex: engine seam (flag-gated, default-off) + the harness A/B arm —
      LANDED 2026-10-16, PUSHED 2026-10-16 (reflex develop `63ccf78`+`8546a68`+
      `9494d83` — the push was held only until this box's katgpt-rs main
      worktree synced past `a081134a8` (the plan-621 draft Cargo lift,
      `.plans/621_draft_wip_cargo.patch`); the bench GOAT gate arm `9494d83`
      was recovered from the uncommitted ring-worktree state and rides the
      push. The feature row requires katgpt-rs >= 0fb56b269, cargo validates
      dep-features eagerly — verified: shared-checkout `cargo check` green
      post-ff). The seam: per-question refinement inside `route_active` over
      `refine_state_into`, keys = the options' resolved domain DIRECTIONS (this
      engine's option model), route+head terms recomputed from x' (the
      mc_ensemble `eff_state` pattern); noul + drafter-only byte-identical
      (issue-030 legality); nb/oc/ridge token-keyed, structurally unaffected;
      engine-held scratch (G4: 0 allocs post-warmup). Birth-gate pair shipped:
      `tests/set_refine_seam_gates.rs` 7/7 incl. `seam_equals_substrate_composition`
      (engine probs == hand-composed substrate pipeline to 1e-6) + CLI
      `--set-refine-ab/--set-refine-gamma/--set-refine-tau` (loud refuse without
      the feature). Clippy `-D warnings --all-targets` clean at BOTH postures;
      default lib 427 passed; engine_gates 9/9.
- [x] T3 the GOAT run — DONE 2026-10-16 (reflex **Bench 150**): **G1 FAIL** —
      γ=1.0/τ=0.5 loses −12.3..−14.0 pt on the wide by-name suites
      (banking77/massive/ag_news), −2.8 sst5, −0.7 xnli; emotion +2.0 (lcb95
      −0.03 — not beyond noise). The γ=0.25 monotonicity probe shows damage
      scaling with γ, no reversal — the negative is STRUCTURAL: with
      unit-centroid keys the set-conditioned pull regresses the state toward
      the corpus mean, homogenizing exactly the geometry the route terms
      discriminate with (collateral: calibrator collapse, abstain
      0.526→0.974 banking77). Controls validate the seam: typed_decisions +
      prompt_injections read exactly 0 flips (byte-identical, the legality
      scope working as designed). G2 PASS at bench scale (p99 454 µs ≤ 1 ms;
      O(n·d²) identity-matmul caveat ~9 ms @ k=77 — the fast-path is a
      kernel_opt candidate, mooted by G1). G3 PASS (identity gates + the
      0-flip controls). G4 PASS (0 allocs). **Verdict: measured negative,
      promotion refused; a re-open needs a different KEY FAMILY (per-option
      fitted directions — riir-train territory), never a γ/τ retune.**
- [x] T4 — RESOLVED BY THE MEASURED NEGATIVE (no promotion question survives
      G1's failure; §4's own law: "Any miss ⇒ stays opt-in with the measured
      record"). Owner disclosure: reflex Bench 150 carries the full record.

## 6. Cross-references

- reflex `.issues/090` §C2 (the seed spec + the do-not-wire-blind law)
- `.research/` Jeva lineage: reflex Research 007 (rethink), riir-train Issue 632
- Plan 567 `cp_hopfield` (prior art; the heavy-machinery warning)
- Bench 149 `--rubric-probe` (the route-agreement characterization that motivates
  asking whether option-interaction buys anything at all)
