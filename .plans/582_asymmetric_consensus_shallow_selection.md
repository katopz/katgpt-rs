# Plan 582: Asymmetric Consensus for Shallow-Reasoning Selection (TTPO modelless extraction)

**Status:** CLOSED-NEGATIVE (2026-08-30) — the Phase 1 kill gates FAILED on the serving model; per T1.2's own rule the plan closes at Phase 1 with no flag and no wiring. Measurement: [riir-ai Bench 819](../../riir-ai/.benchmarks/819_plan582_asymmetric_consensus_measurement.md) (Bonsai-27B Metal, K=64 × 200 prompts, rollback determinism probe PASS). Headline: P(wrong|N)/P(wrong|P) = **0.744 (INVERTED)** at 25–50% vote accuracy and 0.960 at >50% (gate asked ≥1.5); <25% stratum unpopulated (attractors concentrate agreement). G2: s = Σ −log p·(1−Ĥ) AUC **0.472 — below random** (baselines: mean −logp 0.571, entropy-sum 0.197 INVERTED, disagreement 0.514). Mechanism: Bonsai's failure modes are degenerate ATTRACTORS (first-operand echo in math, always-forward in action) — consensus concentrates on the wrong answer and the MINORITY carries the corrections, the exact opposite of TTPO's asymmetry. Scope caveat: does not refute the paper (Qwen3 instruct+CoT regime); it kills the mechanism on OUR serving envelope, which is what this plan scoped. Salvage recorded in Bench 819 §caveats (entropy is anti-selective on this model class). Origin: arXiv:2608.27448 (TTPO) via [riir-train R435](../../riir-train/.research/435_TTPO_Test_Time_Policy_Optimization.md) §User-Challenge Addendum. Companion gradient plan: riir-train Plan 363 (secondary; its own G0 kill gate should weigh this negative).

## Summary

Port TTPO's **test-time decision layer** — not its gradients — into the stack's shallow-reasoning selection surfaces as the opt-in `asymmetric_consensus` feature (katgpt-speculative; optional katgpt-core selector mode). Three mechanisms:

1. **Asymmetric vote treatment with calibration law.** Consensus partition into majority P / minority N is robust in a specific measurable sense: P(wrong | minority) is high (~0.79 in the paper) and nearly **invariant to vote accuracy** — penalizing/pruning the minority stays correct even when the vote is wrong. This justifies `parallel_probe`'s shipped fixed-threshold deviant pruning with a *measured* calibration table (and re-pins the threshold), and adds the symmetric trust branch: majority members are selected, not merely kept.
2. **Confident-error branch scoring.** `s(t) = −log p(t) · (1 − Ĥ(t))` per token, accumulated per branch — a one-pass locator for "confidently wrong" branches using only logits the verify/draft path already materializes. Composable with disagreement pruning (either signal fires); on the **no-consensus stratum** (all branches disagree at stop time) it becomes the selector: pick the branch with the LOWEST confident-error mass, or escalate — the inverse of naive entropy-max triggering (spend where the model is confidently wrong, not where it is uncertain).
3. **Selector refinement.** `ConvergenceSelector::MajorityVote` (EqR, Plan 119) gains a confident-error tiebreak behind the same feature; `hint_regret` gains confident-error mass as an escalation-trigger input (composes with `learnable_band_gate`).

**Shallow-reasoning scope (deliberate):** single-pass candidate generation + selection — branch rollouts (`parallel_probe`), path selection (`ConvergenceSelector`/EqR), fix-candidate and escalation triage (`hint_regret`). This is our serving envelope (Bonsai decode, healer spans, 20 Hz game ticks). Deep multi-step deliberation and any gradient update are non-goals (see Plan 363 for the gradient track).

## Prior-art honesty (§4 — arxiv export API + known-literature audit, 2026-08-29)

| Component | Published precedent | What remains ours (the claim) |
|---|---|---|
| Disagreement ⇒ wrongness (sampled rollouts) | SelfCheckGPT (arXiv:2303.08896), SAC3 (EMNLP'23), BTProp (NAACL'25), semantic entropy (Farquhar et al., Nature'24) — dense family | Not claimed as new. We *consume* it. |
| Uncertainty/margin-gated verification budget | "Verify when Uncertain" (arXiv:2502.15845 — uncertainty-interval switching), OptPO (arXiv:2512.02882 — SPRT stopping) | Not claimed as new. |
| Negative-learning asymmetry (what X is NOT survives label noise) | NLNL (Kim et al., ICCV'19) — TTPO's own cited precedent | The principle is published; its **stratum-invariance calibration measurement on our domains** (G1) is not. |
| `s = −log p · (1 − Ĥ)` product form as a confident-error locator | SelfCheckGPT uses token-probability and consistency as separate signals; the product with normalized certainty is TTPO's formula (training-side there) | The **selection-side product form** + its per-branch accumulation in a zero-alloc probe loop. |
| In-stack fusion (vote-partition asymmetric treatment inside branch/EqR/hint_regret selection) | none found (in-stack grep: `parallel_probe.rs` ships answer-level majority + fixed-k prune, **no confidence scoring, no vote-quality calibration**; `ConvergenceSelector` ships MajorityVote/BestQ/Top1Converged/BtRank, no tiebreak) | **The GOAT claim:** novel in-stack combination + measured calibration law + no-consensus fallback. GOAT-tier, not Super-GOAT (component prior art exists). |

## Stack slot (per-stack ledger)

Speculative-selection + pruning. Feature `asymmetric_consensus` (katgpt-speculative; no new deps — logits already flow through the draft/verify path). **Opt-in until the GOAT gate passes**; promote to default only on G1–G5 PASS; demote the loser (any signal that loses to a baseline in G2 is dropped, not annotated into the blend).

## Integration constraints (grep-verified 2026-08-29)

- `BranchProbeState<A>` (parallel_probe.rs:115) is answer-level and generic over `A` — it holds no logits. The per-branch `s` accumulator therefore rides the **verify/draft path** (where per-token log-probs exist) as a side-channel fed into the controller, NOT as new fields demanding `A: Logit` (that would break the generic contract). Zero-alloc: fixed `[f32; MAX_BRANCHES]` accumulator + running sum, no per-token alloc.
- `ProbeDecision<A>` (Continue/Stop/Prune/StopAndPrune) gains no variants — confidence-scoring composes into the existing `should_prune()` inputs and the stop-time selection hook.
- Sigmoid discipline: the law's outputs are bounded-[0,1] products; no softmax anywhere.

## Phase 1 — measurement (no flag; the kill gate runs first) — DONE 2026-08-30, GATES FAILED

### Phase 1 harness decision (pinned 2026-08-29 — owner call)

**Vehicle: `Ternary-Bonsai-27B-Q2_0.gguf` on M3 Metal.** Gemma-class models are
deprecated in this workspace (owner, 2026-08-29); earlier drafts that implied a
gemma-2-2b-class measurement vehicle are void. The measurement is
inference-only, so the serving model measures its own selection layer.

- **Harness location: `../riir-ai/crates/riir-gpu/tests/`** — a katgpt-rs test
  cannot host it (katgpt-rs is UPSTREAM of riir-engine/riir-gpu; the dep arrow
  is one-directional). Shape: `#[ignore]`d, release-only, macOS-gated
  integration test, run GPU-exclusive (Bench 649 rule). The extractors
  (`RegexAnswerExtractor`, `DiscreteActionExtractor`) are consumed from
  `katgpt-speculative` through riir-ai's existing katgpt-rs dep.
- **Drive pattern (bench_768 protocol):**
  `load_qwen_deltanet_ternary_weights_gguf` → `CubeCLContext::new()` →
  `TernaryDeltanetGpuForward::new` → `reset_state()` → `prefill(tokens) ->
  Vec<f32>` (logits).
- **K=64 branching without re-prefill:** `checkpoint_speculative_gpu()` once
  after prefill; per sample `rollback_speculative_gpu(prefix_pos)` → decode
  the answer via `forward_token` (full logits → CPU temperature sampling +
  per-token log-prob + entropy). KV correctness rides the validated
  append-only contract (write-before-read, pos-bounded reads — Issue 746;
  the seam's rollback round is G1-bit-exact, Issue 717). ~0.4 ms GPU copy per
  rollback, non-blocking, zero alloc.
- **Tokenizer:** `riir_engine::tokenizer::BpeTokenizer::from_gguf` (the Bench
  695/696 protocol) for prompt encode + answer detokenize.
- **Cost shape:** game-action fixtures are L=1 per sample; math fixtures
  pinned to ≤4-token answers → whole matrix ≈ 30–40 min GPU-exclusive.
  Feature set: `--features "cubecl_runtime ternary_gemv speculative_decode"`,
  `--release`.
- **T1.4 bench doc lands in `../riir-ai/.benchmarks/`** at that repo's next
  free number (582 is taken in katgpt-rs by `582_trit_pack_goat.md`; the
  `.highwater` rule applies per-repo at write time) — link back here.

- [x] **T1.1** Fixture set: ≥200 prompts with known-correct short answers across two families (math-extraction via `RegexAnswerExtractor` ladder + a discrete game-action analog via `DiscreteActionExtractor`) — both extractor impls already ship in parallel_probe. *(200 fixtures shipped in `../riir-ai/crates/riir-gpu/tests/plan582_phase1_measurement.rs`)*
- [x] **T1.2** Stratum table: K=64 sampled answers → vote-accuracy strata {<25%, 25–50%, >50%} × {P, N} → P(wrong|stratum, branch-class). **G1 kill gate: P(wrong|minority)/P(wrong|majority) ≥ 1.5 in EVERY stratum including <25%.** If G1 fails on our domains, the asymmetric treatment is dead — record the honest negative and close the plan (cheap: no flag, no wiring). *(RAN: FAIL — 0.744 INVERTED at 25–50%, 0.960 at >50%; <25% empty. [Bench 819](../../riir-ai/.benchmarks/819_plan582_asymmetric_consensus_measurement.md))*
- [x] **T1.3** Token-level signal extraction on the same fixtures: per-branch s(t) accumulation → rollout-wrongness AUC. **G2 gate: AUC ≥ 0.70 vs baselines {mean −log p, entropy-sum, length, disagreement-only}.** Record the per-baseline table; any dominant baseline demotes that signal. *(RAN: FAIL — AUC 0.472, below random and below best baseline 0.571; entropy-sum inverted at 0.197.)*
- [x] **T1.4** Results → `../riir-ai/.benchmarks/NNN_asymmetric_consensus_measurement.md` (riir-ai's next free number at write time — see the harness decision above; GPU/quiet-box note per the measurement rules). *([Bench 819](../../riir-ai/.benchmarks/819_plan582_asymmetric_consensus_measurement.md); rollback determinism probe PASS — it caught the last-forward-logits side-channel bug on run 1)*

## Phase 2 — `asymmetric_consensus` wiring — DEAD (G1 failed; never entered)

Per T1.2's kill rule, no flag is created and nothing below ships:

- [-] **T2.1** `ConfidentErrorScorer` (katgpt-speculative): `feed(branch_id, neg_log_p, entropy_norm)` + `branch_score(branch_id) -> f32`; fixed-capacity, zero-alloc, deterministic.
- [-] **T2.2** parallel_probe integration: scorer fed from the verify/draft path; `should_prune()` composes `disagreement_streak OR confident_error_mass ≥ θ_ce`; θ_ce pinned from T1.4's ROC knee (frozen constant — the calibration law is the justification for NOT adding a vote-quality estimator).
- [-] **T2.3** No-consensus fallback: when the probe stops with no majority → select min-confident-error branch (or `Escalate` when min mass > θ_esc). **G4 gate: fallback selection beats {entropy-max, random, longest} on the no-majority stratum.**
- [-] **T2.4** `ConvergenceSelector` tiebreak mode (katgpt-core, behind the same feature): MajorityVote ties broken by lower confident-error mass; carries the G2 evidence in its doc.
- [-] **T2.5** `hint_regret` escalation input: confident-error mass as a triage signal beside the existing band gate (compositional; no behavior change without the feature).

## Phase 3 — GOAT gate + verdict — DEAD (never entered)

- [-] **T3.1 G3 prune economics:** ≥2× verify-token savings at no Maj@K loss vs shipped pruning (bit-identical answer clustering across runs; determinism gate).
- [-] **T3.2 G5 no-regression:** flag-off behavior bit-identical on all existing parallel_probe + EqR tests; count-identical lib suites both states; clippy 0.
- [-] **T3.3 G4/G-det:** zero-alloc accumulators (tracking allocator), determinism across 3 runs.
- [-] **T3.4 Verdict:** all gates PASS → promote `asymmetric_consensus` to default in the selection path (demote any losing signal); any FAIL → honest negative in `.benchmarks/582_*` + R435 addendum; the measurement half (T1.2/T1.4) stands either way as the calibration record.

## Non-goals

- No game wiring — the generic primitive ships here; riir-ai crowd-consensus consumption files as its own issue only when a consumer materializes (no-default-consumer rule).
- No gradient updates, no adapter training, no teacher conditioning at runtime (Plan 363 owns the gradient track in riir-train).
- No deep-reasoning (multi-step deliberation) claims — shallow selection only.
