# Research 593: Temporal Recurrence Favors Fewer Layers — matched-compute allocation law

> **Status:** RECORD — GAIN verdict (thin-modelless), two tracks filed: allocation-law record (this note) + riir-train recipe plan (low priority, looped-training-lane re-based)
> **Source:** [arXiv:2609.12531 "Temporal Recurrence Favors Fewer Layers"](https://arxiv.org/abs/2609.12531) — Anokhin, Obando-Ceron, Rish, Risi (Mila + Sakana AI), submitted 2026-09-11. Code: `fewer-layers-with-rec`.
> **Date:** 2026-09-26
> **Related Research:** 073 (LT2 looped — the family hub; its PASS-redirect stack hosts the training-time looped-scaling studies incl. **SMELT 2609.01343**, the compute-matched cousin), 273 (ELT any-time loops), 519 (GRT), 592 (loop-growth scaling exponents — the nearest scaling-law cousin), 343 (System-1.5 depth+step shortcuts), 363 (planning budget gates), 131 (UNSL)
> **Related Plans:** 108 (LT2 default-on), 136 (tf_loop), 304 (GainCostLoopHalter), riir-train 421 (loop-growth recipe), **riir-train 422 (this paper's allocation-law recipe — filed with this note)**
> **Classification:** Public

---

## TL;DR

At matched per-step compute `W ≈ L·E·C(d)` (depth × parallel experts × width), **carrying latent state across external steps shifts the best allocation from depth to parallel capacity**: recurrent models saturate at L = 2–4 where non-recurrent models keep improving to L = 16–32 (FineWeb: recurrent L=2 val loss 3.51 ≈ best non-recurrent L=16; Sokoban: recurrent L=4 98.7% solve vs non-recurrent needing L=8–16). Parallel capacity (E, width) helps BOTH classes — recurrence lowers the COST of cutting depth, while E/width are where the reallocated compute pays. The serial-path win is large: **L=1,E=16 runs 0.108 ms/token vs 0.565 for L=16,E=1 (5.2×, 4×H100, B=16)** — same work, shorter serial path. The honest costs: recurrent training is slower (no parallel scan; 2–5× fewer tok/s) and OOMs at long carry windows.

**Distilled for the stack:** an allocation LAW for every streaming/carried-state surface in the corpus — when state is carried across steps (decode loop, belief tick, looped lane), prefer wide-shallow over deep-serial; depth is the latency-killer, not the compute-consumer.

---

## 1. Paper core findings (the load-bearing five)

1. **Allocation sweep, not architecture proposal.** Parallel-Experts probe framework: L depth levels × E experts/level × width d, top-down feedback `s⁰_{t+1} = s^L_t` (one extra projection); non-recurrent = same without the carry. Matched W ≈ L·E·C(d) across 3 budgets, Sokoban + streamed FineWeb (~1B tokens, Muon).
2. **Depth saturation with recurrence:** recurrent best at L=2–4 at every budget (Sokoban 96.2–98.7%; FineWeb loss 3.41–3.51); non-recurrent still improving at L=16–32. Probing (Bush-style agent-move recall): the L=2 model starts with less complete plans and **catches up across environment steps** — computation redistributed across time, matching task performance.
3. **Parallel capacity is class-independent:** increasing E improves both recurrent and non-recurrent at every tested depth; the depth-response curve is what recurrence changes.
4. **Latency:** Table 5 — at matched budget L·E=16, median ms/token falls monotonically as L drops (1.402 at L=16 → 0.108 at L=1). Training throughput inverts (recurrent 2–5× slower, OOM past C≈256–1024 depending on L,E — carried state + per-level KV across windows).
5. **Supporting details:** sqrt(·)-normalized sum aggregation (parameter-free, variance-preserving) consistently competitive — the default; shared KV cache across same-level experts costs ~+0.02 loss and buys large memory/throughput; full expert communication > ring radii (monotone).

## 2. Path 0 inventory

| # | Component | Track | Stack coverage (signal-diff checked) | Disposition |
|---|---|---|---|---|
| 1 | Allocation law (recurrence ⇒ shallower best L, E/width absorb the compute) | a+c | **No cousin states it as a function of TEMPORAL recurrence — but the compute-matched cousin EXISTS and points the other way: SMELT** (arXiv:2609.01343 "Scaling Laws for Compute-Matched MoE Looped Transformers", PASS-redirected in [073](073_LT2_Linear_Time_Looped_Transformers.md):14, also 097/487): loop middle-50% of layers 2×, narrow H to pay for the visit, recover params via expert count — matched on per-token FLOPs + non-embedding params + KV simultaneously (γ 0.250 vs 0.237). SMELT **adds** a depth visit and pays with width/experts under TRAINED WEIGHT TYING (within-step revisits); this paper **cuts** depth and pays into E under CARRIED STATE (across-step). The two point opposite ways, and they reconcile only if the conditioning variable is the recurrence axis — which is untested (recorded in §Caveats; Plan 422 is the natural adjudicator). 592 (2609.19107) = loop-count GROWTH schedules changing scaling exponent γ — different axis again. The 73-family PASS-redirect stack also hosts the training-time looped-scaling studies (SMELT/Ouro/Loopie/"Done Right") — this note's record belongs beside them. 131 UNSL = hyperparameter scaling laws, no recurrence conditioning. | **(a) this note = the law's record + the corpus citation; (c) riir-train plan 422** (allocation sweep for the looped-training lane's next run) |
| 2 | Top-down feedback wiring `s⁰_{t+1} = s^L_t` | c | GRT (519) fixed-anchor re-injection + full-bandwidth transformer (PASS-redirect in 73) are the trained-model analogs; looped-training lane carries state across loop passes, not across tokens | Plan 422 architecture arm |
| 3 | sqrt-sum variance-preserving aggregation | a | Shipped normalization stack (RMSNorm-family) covers the variance-preserving role; the 1/√E form is a one-line variant | Record-only — covered |
| 4 | Shared KV across parallel same-level experts | c | Our MoE experts are FFN-level (no per-expert KV); the paper's experts are full blocks | Not applicable to shipped lanes; noted for Plan 422's architecture arm |
| 5 | Streamed-window training recipe (carried state, reset at shard boundaries) | c | riir-train streaming lanes exist for fine-tune; from-scratch track RETIRED (bb90e9eb/e0019165) | Plan 422 re-based on the looped-training lane |

## 3. Game-context reframe (priority #1 — required, done)

NPC belief evolution (katgpt-sense `evolve_belief`, 8-dim affect carried across ticks) IS temporal recurrence with carried state, and the runtime is the paper's streaming setting. The law's implication: **per-tick cognition stacks should stay shallow and single-kernel; carried belief compensates for missing depth** — which is exactly the shipped design (closed-form kernel + carried state, no per-tick deep stacks). The Bush-style probing half suggests a *diagnostic* (plan-development-across-steps probes) we have no consumer for — our beliefs are projected scalars, not spatial plans. Verdict: the paper **validates the shipped game-runtime architecture** — recorded here, not actionable ("validates our design" is explicitly not-actionable per §1.55). The reframe is done and yields no new primitive; no selling point claimed.

## 4. Consumer-context reframe (healer, priority #2 — done)

Healer surfaces carry no across-step state with allocation freedom: the corpus/retrieval stack is stateless per query; fix trajectories are stored, not recomputed. No manifestation. Recorded as checked-empty.

## 5. Verdict

**GAIN (thin-modelless)** — one-line reasoning: the paper contributes an allocation LAW with latency support that the corpus's looped-family records lacked (all prior cousins fix or adapt L on frozen checkpoints; none conditions the optimal L on carried-state availability at matched compute), but every concrete consequence lands in the training lane (lowest fusion priority) or validates shipped design — so: law recorded here, recipe filed as riir-train 422 explicitly sequenced behind 421, no katgpt-rs plan, no feature flag (nothing modelless to implement; sqrt-sum and shared-KV are covered/not-applicable per the inventory).

- MOAT gate: katgpt-rs = fundamental design-law record for the looped/recurrent family ✓; riir-train = training-method recipe ✓ (Path 0.5: recipe + GPU-hours in the plan; GOAT gate = the lane's standard loss-at-matched-FLOPs comparison, shallow-recurrent arm vs incumbent looped arm).
- Not Super-GOAT: no new capability class, no product selling point, not modelless-implementable.

## 6. Fusion section (novelty TBD)

- **Allocation law × loop-growth (592/Plan 421):** growth schedules answer "how should K grow with compute"; this law answers "where should W sit at fixed compute once state is carried". A combined recipe (carry-state + shallow-L + growth-in-E) is untested anywhere in the corpus — Plan 422 carries it as the sweep's second axis, sequenced after 421's run.
- **L=1,E=16 latency shape × riir-gpu batch shapes:** the 5.2× serial-path datapoint supports wide-shallow kernel batching for any FUTURE carried-state small model serving; no current consumer (frozen checkpoints have fixed L). Recorded as a citation, not a task.

## Caveats

- **The SMELT direction conflict is open (verdict-reviewer catch):** SMELT (2609.01343, 073's stack) trades width for a loop visit under trained weight tying and reports a WIN; this paper trades depth for E under carried state. They are not in contradiction only if the recurrence axis is the conditioning variable — untested. Plan 422's sweep is where the two corpus-recorded laws get adjudicated on our hardware; until then, cite both, prefer neither.
- Probe framework, not a production architecture — the paper says so ("intended as a probe framework rather than a claim of architectural optimality"); allocations are best-observed, not compute-optimal.
- Latency numbers are JAX/4×H100, unoptimized inference; direction trustworthy, magnitudes not transferable.
- Recurrent training throughput costs (Table 6 OOM wall) are the plan-422 risk register; the looped-training lane's data-constrained regime (small corpus, multi-epoch) differs from the paper's 1B-token streamed regime — the 2.2×-style multi-epoch claims are 592's paper, not this one; do not cross-cite.
- Truncated-BPTT credit assignment is the authors' own named confounder for the depth-saturation result; the law is empirical, not mechanistic.
