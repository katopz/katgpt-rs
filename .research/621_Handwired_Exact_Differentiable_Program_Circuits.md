# Research 621: Handwired — Programs as Exact Differentiable Circuits (External Repo Distill)

> **Source:** [chenglou/handwired](https://github.com/chenglou/handwired) — "Turn your regular JavaScript code into a neural net; no training, exact, yet also fully differentiable." MIT, (c) 2026 Cheng Lou. Read at pin `4ea7fcb27ebe93acf49b9f29a93e162b975bad46` (2026-10-15) via full-tree grep in `.raw/handwired` (removed post-verdict).
> **Date:** 2026-10-15
> **Status:** Done — Gain verdict, 2 issues filed
> **Related Research:** 229 (ProgramAsWeights — the controlling sibling verdict), 185 (INSIGHT neuro-symbolic)
> **Related Issues:** katgpt-rs Issue 940 (float reduction-order / tie-band determinism audit — katgpt-rs scope), riir-ai Issue 1073 (train drive constants through the authored FSM — POC), riir-ai Issue 1074 (sync-surface f32 reduction/tie audit — game-side scope)
> **Classification:** Public

---

## TL;DR

Handwired compiles hand-written numeric programs (branches, loops, state machines, table reads) into exactly four instructions — `affine` (weighted sum), `reglu` (ReLU-gated product), `boolean` (one-op exact 0/1 rounding at 0.5), `index` (table read by computed index) — collected into layered tensor programs that run **bit-exactly (FP32, `Math.fround`)** on a JS reference evaluator, a compiled evaluator, and WebGPU shaders, with the whole program differentiable (backprop; loud-throw by default at hard atoms, opt-in straight-through / `'zero'` / `'backward-difference'` gradient policies). Weights are the program — zero training. It is our modelless-first mandate externalized into an executable form, plus a set of exact-construction laws (grid compare bits, sign cascades, one-hot gathers, prefix networks, spring matrix powers, event-clocked latch banks) that transfer as discipline even where the circuit substrate itself does not.

**Distilled for katgpt-rs (modelless, inference-time):** the transferable content is (a) the exactness *contracts* — declarations (`assumeRange`/`assumeGrid`/`assumeOneHot`) + dedicated validators + the rounding-points taxonomy that says which sums a backend may fuse without changing bits; (b) the determinism laws — FP32 reduction-order grouping analysis ("why a tree gives the loop's answer"), the argmax tie-band non-transitivity, event-clocked vs frame-clocked recurrence; (c) the training recipes with committed negative controls — annealed train-soft-then-harden with band-condition acceptance (20/20 seeds), event-clocked one-step supervision (no BPTT), few-scalars-in-frozen-structure. Not distilled: porting the circuit runtime itself (blocked by the Research 229 symbolic-first verdict — see §5).

---

## 1. Repo Core (what ships)

- **Builder + runtime** (`src/runtime/numeric-ui-core.ts`): `Circuit` class, builder macros (`emitCompare`, `emitArgmax`, `emitFirstActive`, `emitApplySpringMatrix`, …), compiler (affine composition, constant folding, CSE, dead-value removal, sparse weights, depth layering, `recomputeDepths`), two evaluators (reference + compiled, `validateTensorEvaluation` compares with `===`).
- **Exactness system**: declarations are promises (`assumeRange`, `assumeGrid`, `assumeOneHot`, `assumeBit`) checked by `validateGrids`/`validateIndexReads`/`validateOneHotGroups`; inferred ranges per rounded op; gathers lower one-hot masked sums only with finite FP32 bounds (the `Infinity*0 → NaN` trap named).
- **Exact compare constructions** (`docs/expressions.md`): grid-declared integer bits (2 ops) vs the general relu sign cascade (15 ops, ≤12 stages down to 2^-149); exact bit algebra (`AND = relu(sum-(N-1))`, `OR = 1-relu(1-sum)`); exact min/clamp by operand selection; the boolean atom `f(f(f(score)+8_388_609)-8_388_609)`.
- **Loop rewrites** (`docs/patterns.md`): findIndex/count+gather (195 ops/3 layers @64 rows), sort as ranks (4,096 ops/3 layers @64 items), FSM over queued events as one-hot state-table products with an **associativity contract** (897 ops/10 layers @64 events), prefix networks (Sklansky `(n/2)·log₂n` combines in `log₂n` levels; Brent-Kung `2n-2-log₂n` in `2log₂n-2`) with the precision inversion (tree grouping can be MORE accurate than the loop: 0.26 px vs 5.67 px error at 10k rows), spring semi-implicit-Euler step as a 2×2 matrix power (14 ops/3 layers for ANY step count), pointer jumping (log₂L), bounded-depth stack as one integer, virtualized filter→prefix-count→slot mapping.
- **Backend codegen** (`src/json-search/bit-emit.ts`): circuits written as WebGPU shaders; proven whole → `i32`, proven {0,1} → single bit of a 32-bit word; `relu(a-b)` over bits → `a & (b^1u)`; the 32-state search shader went 4,655 statements/945 ms → 219 statements/5.2 ms.
- **Demos as evidence**: HandwiredOS (4,646 live ops/130 layers, 25–81 µs/eval, bit-identical committed state across 10/30/60/120 fps + 500 ms stalls), JSON search (100k docs/keystroke on GPU, beats RegExp incl. no-backtracking-by-construction), learned gallery (trained coefficients inside a hand program via its own gradients).
- **Labs with committed negative controls**: generative-fields (violation-depth projection repairs 6,060/6,060 invalid interpolation points; structural parameterization makes a violation class unreachable), event-window (fixed-sharpness training stalls far / settles out-of-band — pinned failure; annealed 1/32→4 sharpness + LR decay + Adam-moments-reset = 20/20 seeds), cascade-argmax (soft-score fit then transfer to exact cascade; hard cascade gives no useful gradients), learned-interpreter (Adam ε as loss scale: ε=1e-8 → **0/20**, ε=1e-14 → **20/20**; hard-loss over complete candidates 20/20 vs soft-averaging 0/20; certify-domain ⊇ train-domain: 12 integer-trained programs, only 4 survive fractional certification).

## 2. Path 0 inventory (advocate-merged)

Two adversarial advocates ran (No-GD + model-based), read-only at the pin. Merged inventory, one row each — mechanism · home · gate:

**Modelless extractions (top 10 of 24):**

| # | Mechanism | Home | GOAT gate |
|---|---|---|---|
| M1 | Grid-declared exact decision bits (2 ops) + harden-at-the-decision law | only consumable inside compiled circuits; as *law* → seal-formula/decision seams | bit-equality vs f32 oracle over declared grid incl. ±1 ULP |
| M2 | Boolean atom (one-op exact rounding; breaks if a shader compiler constant-folds it) | circuit backends only | constant-survives-optimization canary |
| M3 | Sign cascade — exact compare of arbitrary finite FP32, ≤12 stages to 2^-149 | circuit backends; subnormal-flush is the documented deviation | exhaustive vs native compare down to 2^-149 |
| M4 | Spring step as matrix power (any step count in 3 layers; displacement-storage law) | riir-ai fast-forward/replay catchup, editor camera rig easing | bit-exact vs Euler loop under declared envelope; settle at 100k px |
| M5 | Event-clocked latch bank — committed state depends only on events; bit-identical across frame rates | riir-ai viewer/editor UI lanes (server ticks already event-free-deterministic) | replay event stream at 3 tick rates + stall injection → byte-identical |
| M6 | FSM over queued events as one-hot state-table matrix products (associativity contract; "product of clamped 0..1 matrices can't compute NOT") | riir-ai monster FSM lanes IF batch/GPU or differentiable; else WATCH | table-product == sequential match walk on all (state,event) pairs |
| M7 | Prefix networks (Sklansky/Brent-Kung) + FP32 grouping analysis + precision inversion | any f32 reduction; the determinism law is the import | error-vs-f64 bound; associativity precondition asserted |
| M8 | One-hot declarations + gather lowering (3N weights → 0; NaN trap named) | selectors reading one chosen row | gather == dense sum incl. false-declaration mutation reds |
| M9 | argmax tie-band law (non-transitive ~9e-8 band; tree ≠ loop) + the exact variant | katgpt-rs selectors/pruners; riir-ai goal salience | pick equality vs sequential argmax across tie-bracketing sweep |
| M10 | u32 word-packing lowering (proven-bit → 1 bit of a word; AND of words replaces table loops; 180× statement reduction) | riir-infer shader generators (WATCH — we handwrite kernels today) | WGSL equality vs f32 reference + must-NOT-fire probes |

**Training recipes (top 8 of 15):**

| # | Recipe | Pipeline | Cost/gate |
|---|---|---|---|
| T1 | **Annealed train-soft-then-harden + band acceptance** (sharpness 1/32→4, LR decay, Adam moments reset per phase; harden by rounding cells; accept on closed-loop bit-exactness over held-out traces; band asserted, never told) | riir-train host lane → riir-ai sigmoid-drive constants | CPU-seconds/seed; G1 = hardened bit-exactness + band-recovery-when-truth-moves (hand constants structurally cannot pass) |
| T2 | Event-clocked one-step supervision (latch-bank state; NO sequence dimension anywhere) | riir-train distill targets; riir-ai behavior models | no-BPTT; real-boot progress test per house law |
| T3 | Soft-score-then-transfer for exact selection (fit tables soft, harden, verify winners) | riir-refine self_evolve Elo/Beta cutoffs | hardened-decision agreement vs oracle on score_bench |
| T4 | Few trainable scalars inside large frozen exact structure (NC: 1–4 params <1% err vs 8,500-param PINN 7–93%) | riir-train quant-aware calibration cells | recovered-constant error <1%; frozen outputs bit-identical elsewhere |
| T5 | Loud-fail STE menu (throw by default; opt-in STE/zero/backward-difference; saturation guard at ReLU corner) | riir-train quant lanes, GRPO hard actions | nonzero-grad-at-init floor refuses wasted sweeps |
| T6 | Adam ε as loss scale (1e-8 → 0/20, 1e-14 → 20/20) + phased moments reset | any tiny-gradient fit (reflex gate fits, self_evolve) | seed-success count on pinned miss families |
| T7 | Hard-loss over complete candidates vs soft averaging (0/20 → 20/20) | refine candidate selection; GRPO already scores complete rollouts | selection accuracy + certification pass vs enumeration baseline |
| T8 | Certify domain ⊇ train domain (integer-trained 12 → 4 survive fractional certification) | quant-aware lanes; reflex cal slices | adversarial-slice certification pass rate as G1 |

**Audited discards (with reasons):** none discarded wholesale; M1–M3 are **conditional** — plain Rust native compares are already IEEE-exact, so these constructions only matter inside compiled circuits, and the Research 229 verdict (symbolic-first for spec compilation) blocks a general circuit port without a concrete consumer. Discard of "port the handwired runtime to katgpt-rs" rests on that cited verdict + the no-autodiff-by-design posture, not on missing capability.

## 3. Prior-art landscape (§4 searches, run before verdict)

- **The Neural Compiler** (arXiv:2605.22498, Sheneman, May 2026): first-order Scheme-like expressions → **frozen, differentiable PyTorch modules matching the source program to floating-point precision**; hybrid scientific ML (exact physics + 1–4 learned params). **The headline class is published.** Deltas held by handwired: imperative loops/state machines/events (NC is expression-language), bit-exact dual-backend CPU+GPU contract, declaration-driven lowering, UI-scale products, the negative-control lab corpus.
- **RASP → Tracr → Transformer Programs** (Weiss et al. 2021; Lindner et al. 2023; Friedman et al. 2023): programs → exact hand-set *transformer* weights. Same spirit, different target form (attention semantics, sequence DSL, no FP32 execution contract).
- **Differentiable interpreters** (TerpreT/Gaunt 2016; Differentiable Forth/Bošnjak 2017; Neural Libraries 1611.02109): differentiability in service of *learning* programs — handwired inverts: authored programs stay exact, gradients optional.
- **PWA theory**: ReLU networks ↔ continuous piecewise-linear functions exactly (Arora et al.; neural-complexity line) — the math is textbook; handwired's contribution is the engineering discipline on top.
- **Weight-Agnostic NNs** (2019): architecture-first, weights secondary — different axis.
- Conclusion: **Q1 (world-level no-prior-art) = NO → not Super-GOAT.** Workspace-level: no shipped analog (no autodiff anywhere by design; one bench used finite-difference SGD per Plan 286).

## 4. Signal-diffs vs closest shipped cousins

- **seal-formula** (seal-std): authored formula → exact deterministic evaluation, one grammar both sides. Cousin in *shape* (determinism by construction, authored weights/expressions). Diff: tree-walking evaluator, not collected into a flat layered program; not differentiable; not GPU-lowerable; has no bit-hardening machinery (native Rust compares are already exact). Coverage: partial, different layer.
- **Research 229 (ProgramAsWeights)**: the controlling internal verdict — for deterministic enumerable-output specs, symbolic constraints (bitmap pruner) beat neural compilation. Handwired does not contradict it: handwired's value where it exists is *arithmetic data-flow + gradients + GPU uniformity*, which spec membership tests do not need. The 229 verdict is cited as the blocker for a general circuit-builder port.
- **TernaryDraftModel / ternary SIMD matvec**: hand-frozen weights as matmul — cousin in "weights without training", but frozen FROM a model, approximate NN semantics, no program semantics, no exactness contract.
- **DEC operators (katgpt-dec)**: exact mathematical identities by construction (`d∘d=0`) — same *flavor* of construction-guaranteed correctness, different mathematics.

## 5. Verdict

**GAIN** — one-line reasoning: the class is published (NC 2605.22498 + RASP/Tracr + differentiable interpreters) so no Super-GOAT; the repo ships and exposes actionable, cheap, well-controlled improvements to our determinism discipline and drive-constant tuning → Gain with issues, no plan (a circuit-builder plan would contradict Research 229's standing verdict without a concrete consumer).

**MOAT gate per domain:**

| Domain | Fit | Call |
|---|---|---|
| katgpt-rs | Fundamental-primitive bar met by the *laws* (reduction-order grouping, tie-band, rounding-points taxonomy), not by a new primitive (229-blocked) | **Gain** — audit issue (940) + this note |
| riir-ai | Pillar-adjacent: the T1/T2 recipe is the trained-weights fulfillment of the no-magic-thresholds law; M4/M5/M7 serve replay/UI determinism | **Gain** — POC issue 1073 (filed there) |
| riir-train | Host lane for T1 (poincare_phi_train precedent) | rides riir-ai 1073 |
| riir-refine | T3/T6/T7 apply to self_evolve cutoffs | recorded here; no issue yet (no owner signal) |
| riir-infer | M10 word-packing codegen | WATCH (kernels handwritten today) |

**Per-track separation:** modelless track = the laws + audits (issues filed). Training track = T1–T8 recipes, strongest T1 — filed as the riir-ai POC with riir-train as host; a kill on the POC does not cascade to the modelless laws.

**Weakest point (named):** the flagship extraction (exact decision bits M1–M3) is consumable only inside a circuit substrate we are NOT building (229 verdict + no autodiff by design); the genuinely load-bearing imports are therefore the determinism audits and the training recipes — and T1's drive-constant tuning does not actually need circuits at all (annealed soft thresholds are plain constant training; the circuit is optional packaging). If the POC shows hand-tuned drives are already good enough, the whole training half degrades to recorded recipes.

## 6. Fusion

- **paper × Research 229 × no-autodiff mandate**: the synthesis is NOT "compile programs to weights" (229 declined) but "keep symbolic exactness, import the *contracts*" — declarations+validators and the rounding-points taxonomy as discipline imports for raw-domain and fixed-point code.
- **T1 × the FSM law** (behaviour through brain + FSM, never magic thresholds): train the sigmoid-drive constants through the authored FSM structure with annealed soft→harden; structure untouched, constants recovered into bands from replay traces — the first principled tuner for constants that today are hand-set, with a GOAT gate hand constants structurally cannot pass (band recovery when ground truth moves).
- **M7/M9 × quorum/replay**: pin f32 reduction order and exact-compare picks wherever float aggregates cross the sync boundary or feed committed state — the cheapest possible import with the highest tail risk removed.
