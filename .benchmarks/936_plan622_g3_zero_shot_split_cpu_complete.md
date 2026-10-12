# Bench 936: Plan 622 Phase 7 G3 — the zero-shot split, CPU legs complete (4090 box)

**Status:** RECORD — G3 CPU legs COMPLETE 2026-10-12 (all four arms + retrieval, `DONE-EXIT-0` 07:02:06). si / si-cal / si-pprope all FAIL the zero-shot pairing → **no promotion; `scale_invariant_attn` stays opt-in.** The GPU re-run lane (riir-infer plan 620) is the fast re-gate + the τ-sweep enabler; the trained quality claim stays riir-train Plan 449 (SECONDARY).

## What ran

The G3 SPLIT (plan 622 §Phase 7 (i)–(iii)) on the CPU forward, full protocol:
`si_zero_shot_ppl --arms base,si,si-cal,si-pprope --max-length 2048 --min-length 128
--rungs 16 --score-window 128 --tolerance 0.05 --tau 10.0 --retrieval --pp-train-len 1024`
over the league checkpoint (`Ternary-Bonsai-2-27B-PQ2_0.gguf`, layers=64 heads=24/4
rot=true) and the wikitext2-test corpus (2049 tokens, BOS incl., sink carve-out armed).

- **Box:** 4090 workstation (Windows, CPU lane — the PQ2_0 27B CPU decode ~1 tok/s
  class under load). NOT GPU-exclusive: the GPU carried another lane's job
  (`plan404_production_rank_ab`) throughout; these are CPU-forward rows, the GPU
  load is disclosed, not claimed exclusive.
- **Provenance:** clean worktree `E:\git\wt-si622` @ `9f3b27b` (the plan-622 G3
  engine lane), isolated target `E:\tmp\si622-target`, log `E:\tmp\si622-run.log`.
- **Wall:** build 7m12s + load 803.6s; σ probe 330.7s; arms: base 7387.7s ·
  si 7483.6s · si-cal 7283.9s · si-pprope 7258.8s; retrieval ~3h; total run
  18:13:44 → 07:02:07 (12h48m). First launch attempt 17:37 died `os error 3`
  (corpus path), relaunched 18:13 clean.
- **Cross-host calibration disclosure:** σ probe deterministic cross-host —
  M3 digest `1879…` vs 4090 `95ef…` (the AVX2-vs-NEON dot-product f32-LSB
  class); σ stats identical to 4 decimals (min/p25/median/p75 all 1.0000,
  max 3.8430, mean 1.3981). The 4090 digest `95ef…` is the posture for EVERY
  row below.

## G3 (i)+(ii) — the zero-shot pairing: FAIL at both postures

Pooled ppl over the 16 log-rungs' 128-token scored spans; the verdict law is
`si / si-cal PASS iff every rung's |Δppl|/ppl_base ≤ 0.05`:

| arm | pooled ppl | worst \|Δppl/ppl_base\| | verdict |
|---|---|---|---|
| base | **7.2449** | 0.0000 | reference |
| si (uncalibrated) | 10.5334 | **+1.3229** (L=1415) | **FAIL** |
| si-cal (4090 σ table) | 11.4104 | **+1.3290** (L=467) | **FAIL** |
| si-pprope (bypass, train-len 1024) | 11.0333 | +2.0606 (L=1415) | FAIL — EXPECTED, pre-recorded (G3 (iii)) |

Verdicts: si and si-cal FAIL the 5% tolerance **at every rung** (no rung under
+0.17); calibration did not rescue the pairing (it is a per-head σ read, not a
schedule repair). This is the honest answer to "does the schedule transfer
zero-shot to a real 27B ternary checkpoint": **it does not.** The rung ladder's
shape (worst Δ at L=1415, elevated everywhere) is consistent with the schedule's
distance-mass accumulating past the checkpoint's learned positional profile —
the trained pp-RoPE track (riir-train Plan 449) is where a compatible schedule
would be learned, not assumed.

Per-rung rows verbatim (Δ vs base, ppl_base row set from base arm):

```text
#       base pooled_ppl 7.2449 (L=128..2048: 11.2215 7.6198 8.7690 15.6446 15.6848 12.9364 8.8898 4.8511 4.4182 4.4072 3.4758 6.6939 4.5425 2.8318 12.8637 7.4890)
#         si pooled_ppl 10.5334 (13.1377 9.7870 11.1091 21.4493 18.8998 19.2844 14.5748 6.4472 7.0824 6.1360 4.7477 9.8892 6.3073 6.5779 21.5346 12.0178)
#     si-cal pooled_ppl 11.4104 (16.1691 10.1772 11.7624 20.5720 24.3831 23.6924 15.8532 11.2189 7.0048 6.1127 5.0374 11.5408 6.8947 6.5951 16.9187 10.5314)
#  si-pprope pooled_ppl 11.0333 (13.1728 9.7764 11.1037 21.4790 18.9438 20.2452 15.3244 6.6025 6.7358 6.1874 4.9922 10.2661 7.4491 8.6669 24.3399 12.0637)
```

## G3 (iii) — the pp-RoPE bypass: the pre-recorded FAIL, confirmed

The checkpoint carries NO `*.attention.si_pprope` mark (the loader verdict, loud
at startup); the `--pp-train-len 1024` measurement bypass fed a synthetic pp
table at train-len 1024. The arm ran the full ladder and FAILED as pre-recorded
(+2.0606 worst) — the untrained pp mask at a 27B unmarked checkpoint destroys
quality exactly as G3 (iii) anticipated. The quality claim lives in riir-train
Plan 449, not in zero-shot pairing.

## The retrieval probe — an asymmetry worth recording, not a gate

Needle-at-head / question-at-tail, 6 greedy tokens, one probe per cell:

| rung | base | si | si-cal | si-pprope |
|---|---|---|---|---|
| L=1024 | MISS / MISS | MISS / MISS | MISS / MISS | **zebra HIT** / MISS |
| L=2048 | MISS / MISS | **zebra HIT** / MISS | MISS / MISS | **zebra HIT** / MISS |

(base/si-cal: 0/2 zebra hits; the si lane: 3/6, si-pprope 2/2.) ⚠ Read this as a
DISCLOSURE, never a quality claim: each cell is ONE 6-token greedy draw — a
3-of-6 vs 0-of-2 split on single probes is a pattern, not a measurement. The
pattern's shape is consistent with the si transform's distance reweighting
surfacing a head-positioned needle the incumbent's positional profile buries at
long context — and it coexists with the arm's ppl FAIL (the same reweighting
that helps the needle hurts the bulk: si's worst rung is 45% worse pooled ppl).
The needle-vs-bulk trade IS the scale-invariant schedule's design question; it
gets adjudicated with real statistics (τ sweep, multi-seed probes), not here.

## What follows

- **Promotion: NO.** `scale_invariant_attn` stays opt-in (the plan's promotion
  law: G1–G4 AND zero-shot pairing tolerance; the pairing failed).
- **Phase 7's remaining legs** ride riir-infer plan 620 (Phases A–D landed
  2026-10-11/12: `4b4ff16`, `61eb815`, the retrieval-lane residue `0929fcb`,
  the si-posture gate fix `9e3c9e1`): the Phase-A flag-0 bit-identity A/B on
  the real checkpoint, then the G3 GPU re-run (~5 min/arm vs this 12.8h CPU
  run) — which is also the only practical τ-sweep lane. The sweep decides
  whether ANY τ admits a tolerance pass, or the schedule is trained-only
  (Plan 449).
- The G2 cells (pp2048/tg128 ±3%) + the League rows remain owner-adjacent
  after 620's gate work.

Session: si620-continuation, 2026-10-12 (collection + record; the run itself
was launched by the prior session's handoff).
