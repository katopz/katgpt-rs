# Bench 933: Issue 929 T1.2b — Bonsai-2-27B loop-alignment probe, OOM fix + 3 smoke postures

**Status:** MEASURED 2026-10-11 (4090 box, 16-core i7-13700K, 32 GB RAM, AC) — smoke-scale runs, NOT the T1.3/T1.4 adjudication run. Session: 4090-idle post-handoff.

## Context

Issue 929 T1.2b landed code-only (owner: "code only or small test no full run").
This bench records the run-blocker fix + the three smoke postures executed when
the box CPU freed (GPU held by plan437_stage0_eval + plan404_production_rank_ab
throughout — all runs here are CPU-side, the bin is `riir-infer-core` substrate).

## The run-blocker: KV pre-allocation OOM (root-caused, fixed)

First smoke attempt aborted: `memory allocation of 1073741824 bytes failed`
(exactly 2^30). `RUST_BACKTRACE=1` pinned it to `KVCache::new` ←
`MultiLayerKVCache::new` ← `HybridCache::with_layer_types`. Root cause: the GGUF
loader sets `config.block_size = context_length` (128k-class on Bonsai-2), and
`MultiLayerKVCache::new` sizes `block_size × kv_dim` K and V buffers for EVERY
layer (DeltaNet layers included, which never touch KV) — per-layer demand ~2 GiB
× 64 layers ≈ >100 GiB of commit beside the ~8 GB loaded weights; the box (32 GB
RAM + 48.4 GB pagefile, ~39 GB baseline commit) exhausts after ~16 layers.
The gates tests never saw it (small synthetic configs).

Fix (riir-infer `61fc36f`): `HybridCache::with_layer_types_bounded(config,
layer_types, max_positions)` — attention layers `[max_positions × kv_dim]`,
DeltaNet layers empty; wires the EXISTING upstream bounded ctor
(`MultiLayerKVCache::new_with_per_layer_kv_dim_bounded`, the Gemma-4-262k
precedent). Bin caps at `max_prompt + answer_tokens + 16` (attention forward
indexes K/V positionally, no growth). Machinery gates 3/3 PASS after the change
(K=1 bit-identity preserved); clippy clean. Scratch target `E:\tmp\infer929`
(sccache-warm rebuild ~1 min; documented per the infer038 precedent).

## Smoke postures (all: K=4 loops, facts=12, fresh cache+scratch per query)

| # | framing | n | answer tokens | acc | headline | elapsed |
|---|---------|---|---------------|-----|----------|---------|
| 1 | completion | 8 | 8 | 0.0000 | model emits `<\|im_end\|> <think>` — wants chat scaffold | 1455 s |
| 2 | chat (`B929_CHAT=1`) | 8 | 96 | 0.0000 | think spans show CORRECT two-hop chains, truncated mid-hop-2 | 1575 s |
| 3 | chat | 4 | 384 | **0.5000** | **auroc_cos_bridge 1.0000/1.0000 at every loop; verdict line SIGNAL** | 1835 s |

Verdict line (smoke #3, verbatim): `verdict: SIGNAL — at least one (loop, split,
signal) AUROC 95% CI LB >= 0.6 — T1.3 G1 gate PASSES; Phase 2 GOAT adjudication
re-opens`.

**n=4 caveats, stated plainly:** AUROC 1.0 at n=4 is fragile (2-vs-2 classes);
this is NOT the adjudication run — it proves the label axis is MEASURED-ALIVE on
Bonsai-2 (the kimi legs' failure mode is cleared). `bridge_top1 = 0.0000`
everywhere — the argmax==bridge_first direct readout never fires;
`cos_bridge` is the live signal.

## Protocol findings for the full run (recorded BEFORE spending 17–21 h CPU)

1. **Framing + budget (measured):** full run needs `B929_CHAT=1` and answer
   tokens ≥ 384 (~420 s/query CPU → 180 queries ≈ 21 h — a window job).
2. **`contains`-scoring artifact:** smoke #3's q0 scored "correct" while the
   think span was mid-DOUBT — the model QUOTED the gold fact ("gentle ferret is
   the baker of warm rabbit") then questioned its direction. Quotation ≠ final
   answer; a `</think>`-aware or answer-line check tightens the axis.
3. **Fixture phrasing inversion (evidence = the model's own reasoning):** facts
   render `"{a} is the {r} of {b}"` but questions compose
   `"Who is the {r2} of the {r1} of {a}?"` — under standard English ("X is the R
   of Y" ⇒ R-of(Y)=X) the questions ask the INVERSE of the stated facts; the
   fixture only composes under a possessive-subject reading (R-of(a)=b). The
   model resolves hop-1 under the possessive reading, then English-semantics
   doubt visibly spirals (q0: "Wait, that says gentle ferret IS the baker, not
   that someone is the baker OF gentle ferret"). Proposed one-string repair
   (same triples, standard semantics): fact template `"The {r} of {a} is {b}."`
   — both hops then compose naturally. Suggested protocol: A/B the phrasings at
   n=8/384 tok before the full run. NOT changed unilaterally — it redefines the
   shared instrument mid-adjudication (katgpt-core
   `loop_alignment_probe::generate_two_hop_fixture`; kimi Benches 927/930 are
   0%-dead so no comparability loss, but the phrasing defines the science).

## Artifacts

- JSON outputs: `/e/tmp/b929_smoke.json`, `/e/tmp/b929_smoke_chat.json`,
  `/e/tmp/b929_smoke_chat384.json` (ephemeral box scratch; the CSV cells are
  quoted above and in Issue 929).
- Fix + knob commit: riir-infer `61fc36f`.

## Appendix (2026-10-11 ~17:00 +07): the phrasing A/B — MEASURED

The protocol note 3 above proposed A/B-ing the fact phrasing before the full
run. The M3 session's full adjudication run (Bench 934, possessive phrasing,
no-think posture) launched mid-day; this A/B ran the SAME posture on the 4090
box to adjudicate the phrasing axis independently of (and before) that run's
verdict lands.

**Instrument:** union `69f520e` + the `FactPhrase` knob (katgpt-core
`loop_alignment_probe`, landed with this bench; **default UNCHANGED =
possessive** — the shared instrument is not redefined mid-adjudication) + the
bin's `B929_PHRASE=standard` wiring + per-query JSON rows in `B929_OUT`
(riir-infer, same commit series). Posture: `B929_CHAT=1` (no-think chat,
8-tok answers), K=4, facts=12, n=16 (first 16 fixture items — 16 two-hop
in-distribution queries, seed 0x929), paired arms: the knob renders the fact
template only and consumes no RNG — identical fact windows in identical
shuffled order (pinned by the new katgpt-core unit test
`phrase_standard_preserves_labels_and_windows`).

**Binary disclosure:** both arms ran the build at `69f520e`+knob — pre
Plan-622 (`d1b9bd5`, which later threaded `layer_idx` through
`qwen_deltanet_ternary_layer_body`; plumbing at this feature set, hooks
`None`). Both arms share one binary — the pairing is internal-valid; a
re-run at HEAD re-derives nothing from these numbers.

| arm | phrasing | acc (two-hop, n=16) | verdict line | failure modes |
|---|---|---|---|---|
| A | possessive (default) | **0/16 (0.0000)** | UNDECIDABLE — every AUROC cell empty (label axis DEAD) | 11× immediate `<\|im_end\|>`, 3× think-start despite "Do not think", 2× bridge-stop (hop-1 entity answers) |
| B | standard (`The {r} of {a} is {b}.`) | **5/16 (0.3125)** | **SIGNAL** — `auroc_margin` 0.8727/**0.5818** @loop1, 0.8909/**0.6182** @loop2 (LB ≥ 0.6 bar) | 9× `<\|im_end\|>`, 2× think-start, 5 VERBATIM correct (`bold rabbit<\|im_end\|>`-class), 1 partial (`badger` ⊂ `warm badger` — honestly scored wrong), 1 wrong entity |

**Discordant pairs: 5 standard-only, 0 possessive-only → exact sign-test
p = 0.03125** (n=16 paired). The 5 correct generations are verbatim
two-hop answers with no think span to quote from — genuine composition, not
the contains-scoring artifact (that artifact rides the THINK postures; at
8-tok no-think there is no quotation window).

### Interpretation

1. **The phrasing inversion IS the binding constraint on the label axis at
   the no-think posture.** Possessive: 0/16, every AUROC cell empty — the
   label axis is dead, T1.3's gate cannot compute, exactly the kimi-leg
   failure shape but with a fixable cause. Standard: 5/16 with decodable
   margins (smoke-class SIGNAL at n=16; not the adjudication).
2. **Implication for the in-flight Bench 934 full run (possessive):** its
   two-hop label axis is expected DEAD — a KILL/UNDECIDABLE verdict there is
   **phrasing-limited, not signal-absent** and must be read beside this A/B.
   The one-hop controls (M3 smoke 4/8) are less phrasing-sensitive (single
   hop — the inversion bites on composition).
3. **`bridge_top1` trajectory (possessive arm): 0.8125 → 0.1875 across K=4**
   — the first token is usually the bridge (hop-1 stop) and loop passes
   ERODE that alignment, an inversion of the confidence-amplification shape
   seen on kimi (margin growth k=2–3). Loop dynamics interact with the
   phrasing confusion — worth a targeted read if Phase 2 re-opens.
4. **The default stays possessive** (decision recorded, not made): the
   shared instrument mid-adjudication is not redefined by a knob; whether
   the T1.3/T1.4 adjudication of record runs at `B929_PHRASE=standard`
   (~8 h no-think / ~21 h think-budget, either box, a window job) is the
   owner's call. This A/B is the evidence for that call.

Artifacts: `E:/tmp/b929_ab_poss.json`, `E:/tmp/b929_ab_std.json` + logs
(`b929_ab_{poss,std}.{log,err.log}`) — box scratch, cells quoted here.
