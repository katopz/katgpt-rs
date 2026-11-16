# Research 677: codehygiene — a Compiled Ternary System One Model for Code Hygiene (the from-scratch WSP compiler arm)

> **Source:** brianbell-x/codehygiene — https://github.com/brianbell-x/codehygiene — "A 33 KB, weight-space programmed System One model for code readability and hygiene." Cloned to `riir-refine/.raw/codehygiene`, **PIN `423c7e1f02ef409bbaea209268bfcde36e0f1cb0`** (tip at inspection 2026-11-16; 14 commits + 23 stars per the GitHub repo page fetch — the depth-1 clone cannot verify the count). **MIT declared only at `pyproject.toml:10`; no LICENSE file in the tree.** Pure Python + NumPy (`numpy>=1.24` sole dep): `compile.py` (1663 lines, the circuit compiler) + `codehygiene/__init__.py` (460 lines, forward + 5-trits/byte unpacker + byte-grid encoder) + committed `weights.npz` (12,97 KiB).
> **Date:** 2026-11-16
> **Status:** Active — **GOAT (fusion)** — delegated-owner verdict, Claude reviewer (`katgpt-rs-07@m3`, session `7a0b3b4e-dd69-46b4-b71a-8367d9f7292f`), 3 rounds REVISE → AGREE → close, 2026-11-16
> **Related Research:** 648 (WSP parent — the obstacle-vs-ability law), 649 (probe-authored circuits — the no-LLM author posture, value-refuted for repair), 671 (dspy-wsp — decision-head closed-form edits), 676 (AHA — self-supervised WSP authoring), 664 + 655 + 659/668 (author-solver siblings + zoo hygiene), 562 + 573 (TypeSafe System One / Jev — the calibrated-decision record; the shared noul/choice/score wire shape)
> **Related Plans/Issues:** katgpt-rs `.issues/1225_ternary_circuit_compiled_wsp_poc.md` (the two-arm PoC, proof path inline) · riir-rethink `.proposals/006_compiled_system_one_lane_class.md` (the serving posture) · riir-refine `.distill/001_mining_queue_snapshot.md` (the distill verdict line)
> **Classification:** Public

---

## TL;DR

codehygiene compiles hand-written mathematical circuits into a complete 1.58-bit ternary decision network — 569,662 matrix trits + 4,103 channel scales (93.5% zeros) packed at 5 trits/byte into a 12.97 KiB `weights.npz`, executed as a ~10 ms warm CPU forward over a (lines × 128) ASCII byte grid — and ships it as a code-hygiene judge (overall + `rl_hygiene` scores, 6 ordinal sub-scores, a 5-class style archetype, 8 per-line squiggle kinds). Zero gradients, zero training data: every matrix is constructed at compile time from explicit circuits — exact integer-max-margin threshold templates, causal prefix-scan automata, a Paley-Hadamard {−1,+1} projection built from quadratic residues mod 19, and seed-pinned coordinate-descent ternary Gram fits. Its scalar-head variable is literally named `noul_probs`, and its README cites **TypeSafe System One (noul/score/choice)** as design prior — the same shared market wire shape our decision_wire serves (Research 562 records TypeSafe as external; 562 L158: "choice/score/noul is now the market's shared wire shape across TypeSafe AND OpenAI"). **This is the from-scratch COMPILER arm of the published WSP class** — 648's Exp-5/6 "write a policy from scratch", executed by a deterministic spec compiler instead of an LLM, at 30–100× the paper's from-scratch scale — the one WSP author class our family corpus did not cover. Filed **GOAT (fusion)**: the idiom kit is new-in-stack, the class is published (648's grade logic verbatim: the composition is ours, the components are published).

**Distilled for katgpt-rs (modelless, inference-time):**
The transferable primitive is the **circuit→ternary compilation idiom kit** — a deterministic front-end that emits byte-exact ternary weight programs from explicit specs, loadable by our existing packed-ternary substrate (`kpt_archive` `.cact` + the ternary matvec). Our three-track system gains a **fourth weight class**: compiled-from-circuits nets — alongside trained `.bits` (`TernaryDraftModel`/drafter), LLM-authored overlays (`weight_edit_overlay`, 648, landed `b0f1ed615`), and probe-authored repair (649, value-refuted).

---

## 1. Source core findings (read at the pin)

| Aspect | Finding |
|---|---|
| Artifact | 32 ternary {−1,0,+1} matrices (425,044 trits v0.3 → 569,662 v0.4, 1.60 b/w) + 4,103 1D scales; 93.5% zeros; 12.97 KiB on disk (10.77 KiB v0.3); cold load ~5.7 ms; warm ~10.2 / 63.6 / 410.9 ms at 12 / 65 / 391 lines (their numbers) |
| Benchmarks (**ALL SELF-GRADED** — their own canon/suites) | Diagnostic AUROC 1.000 (clean vs messy, their 54-slice canon); squiggle recall 14/14; static slop/paranoia circuits 28/28 at v0.4 (0/28 at v0.3 — the v0.4 trit growth bought exactly this family); production canon 2.05 squiggles/100 lines, 22/22 clean traps, 0 new FPs |
| Agent benchmark (their protocol) | GPT-OSS 120B built 4 subsystems + frozen tests, then refactored under their SKILL.md + `--check`: hygiene 0.352 → 0.856 mean, 54 → 0 squiggles, 64/64 tests kept green. `rl_reward` is gated by `tests_passed` (broken solutions always 0.0 — an anti-reward-hack law our compile-gated fix verification shares) |
| Forward (`_forward`, `__init__.py:186`) | Block 0: byte/column embeds + causal quote/comment masking → 52 cell channels. Block 1: k=3/7/13 convs + 40+40 SimHash stem sketches → 226 line registers. Block 2: two 8-head sink-attention passes (±32 rel offsets + unmatched-query sink logit). Block 3: sigmoid squiggle heads (`:234`) + gated mean/max pool → sigmoid scalar heads (`noul_probs`, `:242`), softmax ordinal sub-scores (4 levels, `:244`), softmax 5-class archetype (`:245`) |
| Output surface | Shares the TypeSafe System One noul/score/choice vocabulary (README cites TypeSafe + Cloudflare Clef as System One design priors). The repo cites TypeSafe as prior; it does not cite us |
| Compile-time construction | No LLM, no gradients anywhere. Three seeded `RandomState`s: `:238` (20261115, sketch bank), `:277` (0, row signs + perm), `:314` (777, QR target) — a Rust **recompile** of the weights needs MT19937 + legacy numpy `randn` ports (named blocker; the PoC imports the committed npz instead) |

## 2. Distillation — the circuit→ternary idiom kit (each with its source site, verified at the pin)

1. **Exact integer-max-margin threshold circuits** — `compile.py:_sub_filter` (:176): per character, `_stamp_char` (:169) sets BOTH 7-bit planes at (pos, plane) = 1, so the cell fires iff the byte's bit matches the template bit; then `scale = 2·out_scale`, `bias = −(2·thresh−1)·out_scale`, `thresh = 7·len(pattern)` + word-boundary (−1 on HC_ID), line-lead (+1), code/comment-context vetoes. The conv output is positive iff EVERY bit matches — an exact pattern detector with integer margin, **no exp involved**.
2. **Automaton-in-weights** — `W_causal1/2` lower-triangular prefix matrices (`__init__.py:193/197`) scan left-to-right per line through gate MLPs implementing quote-parity (16-sawtooth) and comment-opener state — a DFA compiled as prefix matmuls; detector channels veto on the string/comment planes.
3. **Paley-Hadamard {−1,+1} projections** — `_paley_uniform_proj` (:262): quadratic residues mod 19 → Jacobi symbol → Paley-type-I H₂₀ → block H₄₀, row signs + permutation from `RandomState(0)`, disjoint column picks → a (40,16) uniform-diagonal orthogonal ±1 projection. Number theory as a weight constructor.
4. **Seed-pinned ternary Gram fitting** — `_ternary_sketch_proj` (:312) + `_fit_sketch_column` (:284): QR target → ±1 sign init → 12 rounds of coordinate descent choosing trits + per-column scales minimizing Gram error; closed-form scale `sqrt(quad_num)/norm_sq`. Compile-time-only discrete optimization — deterministic, seed-pinned; NOT gradient descent and NOT runtime.
5. **SimHash stem sketches** — `_sketch_bank` (:230): 40 seeded ±1 4-gram filters, rejection-sampled for orthogonality (|dot| ≤ 8) and stopword avoidance (≤ 1 hit); clamp01 via double relu (`relu(x) − relu(x−1)`, `__init__.py:210`); L2-normalized code/comment halves.
6. **Sink attention** — `_sink_attn` (:140): 8 heads, clipped ±32 relative-offset bias table, an unmatched-query sink logit appended to keys then sliced away (the StreamingLLM sink pattern) — lets lines attend "nowhere".
7. **Calibrated ordinal readout** — sub-scores = softmax over 4 ordered levels × level values (expected level); 5-class archetype head; gated mean/max pooling into a doc vector; sigmoid scalar heads + per-line sigmoid squiggle heads.
8. **5-trits/byte packing** — `_pack_trits` (:160) base-3 digits, 0 → 0x00; unpack `((buf // 3^k) % 3 + 1) % 3 − 1` (`:265`); per-channel 1D scales folded at load. Substrate cousin of riir-infer's PTQ1_0/EXL3 lanes and katgpt-core `kpt_archive` (`.cact`).

## 3. Verdict

**Tiers: GOAT (fusion).** Q1 prior art: the WSP class is published (Takehi/Teoh/Zhang/Khattab = Research 648's exact source; iWSP Singh et al. — the repo cites both; web sweep 2026-11-16 found the class heavily covered and NO competing compiled-circuit code-hygiene judge — hence "the first public artifact we found", never a provable "first ever") → **not Super-GOAT**. Q2 new capability class for OUR stack: YES — the fourth weight class, no in-stack compiler. Q3 selling point: "our decision lanes can seat 13 KB weight-space-compiled System One models — 10 ms CPU, deterministic, reproducible from source, no GPU, no artifact store." Q4 force multiplier: YES — ternary substrate (matvec + `kpt_archive`) × modelless mandate × decision family × refine fix-lane report rider. 3/4 + published class = GOAT (fusion).

**MOAT gate:** katgpt-rs — fundamental modelless primitive via fusion ✓ (the compiler front-end for the family's existing back-end). riir-refine (consumer-first): report-only hygiene-delta rider candidate on the fix lane — stationary-surface law: hygiene is a pure function of the input file, so LEVEL readouts only, never dynamics signals. Decision family: serving posture → riir-rethink `.proposals/006` (HIGH interest, **unproven** — no eval surface, no consumer; serving adoption owner-gated).

**Distill axis (riir-refine): NO C corpus.** Zero surviving transcription candidates: the 8 squiggle classes are heuristic judgments (noisy_comment needs stem-overlap semantics — fails clippy_lints' parse-provable gate; deep_nesting ≈ existing clippy excessive_nesting coverage; swallowed_error ≈ partial let_underscore/must_use coverage; dense_line/cramped_layout = rustfmt territory); `_conv1d` active-channel pruning is a positive instance of the corpus's existing zero-skip family; double-relu clamp01 is a NumPy spelling, not a Rust defect shape. A CONSTRUCTIVE artifact, not a defect corpus.

## 4. Prior art + cousins (all read this session; non-kill deltas)

| Cousin | Why it doesn't kill |
|---|---|
| Research 648 (WSP parent) | LLM-authored weight EDITS to existing models; codehygiene needs no LLM and no base model — the from-scratch compiler arm |
| Research 649 (probe-authored, FOLDED) | probe-guided REPAIR authoring for the weight-edit lane (value refuted on the twinarm; PoC PASS stands as the identification record); codehygiene's circuits come from the SPEC, not from probes |
| Research 671 (dspy-wsp) | writer-LLM closed-form edits of decision-head scorers on TRAINED checkpoints; complementary cell (closed-form × trained checkpoint vs compiled × from-scratch) |
| Research 676 (AHA self-supervised WSP) | a self-supervised AUTHORING LOOP over the edit lane; codehygiene is static hand-authored circuits with NO learning signal — different author class entirely |
| riir-infer PTQ1_0/EXL3 + katgpt-core `kpt_archive` | substrate coverage of trit packing + packed archives — the natural emission target for a Rust compiler; the format is the cousin, the COMPILER is the delta |
| Research 562/573 (TypeSafe/Jev) | record the EXTERNAL TypeSafe System One family + the shared noul/choice/score wire shape; codehygiene aligns to that same market vocabulary (`noul_probs` naming + README citations) |

**WSP author-class map now complete:** LLM edits (648) · LP/MEMIT/Ising solves (664/659/668) · probe-guided repair (649, refuted value) · self-supervised loop (676) · writer-LLM decision heads (671) · **deterministic spec compiler (this note — new)**.

## 5. Fusion map

- **codehygiene × 648/671:** the WSP family had authors but no **deterministic spec compiler** — a front-end that emits byte-exact ternary programs from explicit specs, loadable by the back-end we already ship (`kpt_archive` + ternary matvec).
- **codehygiene × reflex/instinct/rethink:** a **compiled lane class** for the arsenal — 13 KB embedded weights, ~10 ms, no GPU, deterministic — a new cheap-leg class between A0 bags and ENC encoders, ESC-seatable, sharing the TypeSafe noul/score/choice wire shape (→ rethink `.proposals/006`).
- **codehygiene × riir-refine:** a report-only hygiene-delta rider on the fix lane (consumer-first moat; report-only per the stationary-surface law).

## 6. Routing + files created this session

- `katgpt-rs/.issues/1225_ternary_circuit_compiled_wsp_poc.md` — the two-arm PoC with the pre-registered proof path.
- `riir-rethink/.proposals/006_compiled_system_one_lane_class.md` — the serving-posture proposal (owner-gated adoption).
- `riir-refine/.distill/001_mining_queue_snapshot.md` — the distill verdict line.
- **P0:** Issue 1225 Arms A/B → verdict. **P1:** eval surface for the lane (rethink 006 T1 — the gate). **P2:** refine report-only rider IF the PoC passes AND an eval surface exists. **P3:** Paley/automaton idioms as standalone katgpt-core builders (only with a consumer).

## 7. Caveats (honest)

- **All benchmarks are SELF-GRADED** on the author's own canon/suites; AUROC 1.000 on self-authored diagnostics is construction verification, not generalization evidence. NO quality-parity claim is granted (§3.6 discipline); our own eval is a separate, unmeasured question.
- **No LICENSE file** — MIT declared only at `pyproject.toml:10`; attribution recorded in Issue 1225 (the port is Arm A's reference reimplementation, disclosed; fixtures remain original).
- Young repo: 14 commits (GitHub page), 23 stars, single author.
- **FP profile on our corpora unmeasured** (their production canon includes ripgrep/Redis/SQLite/TigerBeetle; our codebase is unlike their canon) — the refine rider, if ever wired, measures on our files first.
- **Three seeded RandomStates** block a Rust recompile of the weights (MT19937 + legacy `randn` ports) — named blocker; the PoC imports the committed npz.
- **Softmax sites:** attention softmax (`:164`) is LOAD-BEARING for the sigmoid heads (it feeds `state`); ordinal (`:244`) + style (`:245`) softmaxes are reproduction-only side outputs. House sigmoid law untouched — the port is a reproduction; softmax appears in no new-primitive surface.

Session: zed-m3-codehyg@m3, epoch 1794825290
