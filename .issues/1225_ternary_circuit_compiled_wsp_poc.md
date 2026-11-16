# Issue 1225: ternary_circuit — compiled WSP PoC (two arms: full-forward reproduction + exact-circuit builder)

**Lease closed:** zed-ddc61e9c@amo, epoch 1794846279, closed 2026-11-17T03:30Z, expires 2026-11-18T16:00:00Z — T1 scope COMPLETE (emitter `084772e5c` + assembly/pins `c5f061677`); Arm A/B untouched by this lease.
Claim scope was: T1 (the npz→.cact transcode lane: lossless intermediate emitter + kpt_archive assembly + BLAKE3 pins, meta-driven shapes); G2-A's pre-registered QUIET-m3 posture will be re-boxed WITH disclosure at its run (m3 takes no jobs from this lane — bmo/amo quiet-window substitute, box delta recorded).
**Claimed:** zed-b9d1042a@amo, epoch 1794972000, expires 2026-11-19T03:30:00Z — T1b scope only (the Arm A forward fixtures: the pinned-source Python fixture-builder + the committed full-precision JSON pins; the Rust forward port + G1-A/B stay UNCLAIMED by this lease).
**Status:** Open — filed from [Research 677](../.research/677_codehygiene_Compiled_Ternary_System_One.md) (GOAT fusion; delegated-owner verdict AGREE 2026-11-16, reviewer `katgpt-rs-07@m3`, session `7a0b3b4e-dd69-46b4-b71a-8367d9f7292f`).

**Source attribution:** ported-from brianbell-x/codehygiene @ `423c7e1f02ef409bbaea209268bfcde36e0f1cb0` — **MIT declared only at `pyproject.toml:10`; no LICENSE file in the source tree — attribution recorded HERE.** All fixtures remain original minimal reproductions; Arm A is a disclosed reference reimplementation.

Two-arm PoC for the circuit→ternary compiler primitive (the fourth weight class — Research 677). Feature: `ternary_circuit` (opt-in, katgpt-core). **NO default promotion — consumer-gated** (first candidate consumer: the compiled lane class, riir-rethink `.proposals/006`). No quality-parity claim anywhere: the source's benchmarks are self-graded on its own canon; our G1 is exactness-vs-their-forward (reproduction), never "matches their numbers".

## Arm A — full-forward reproduction (the exactness gate)

- [x] Transcode the committed `weights.npz` (12.97 KiB; holds `trits` / `vecs` / `meta`) → `.cact` via a small converter; **BLAKE3-pin BOTH the original npz bytes AND the `.cact`**; matrix count/shapes are taken from `meta` at transcode time — never hand-typed — **LANDED `c5f061677`** (T1a emitter `084772e5c` + the Rust assembler/converter/gate; 73 layers = 32 trit + 41 vec degenerate; archive_id `43a1e308…`; digit law re-verified at the pin, zero_frac 0.93492 reproduces T1a's 93.49%)
- [ ] Port the FULL forward (`_forward` + `encode_byte_grid` + the 5-trits/byte unpacker, `__init__.py` @ the pin) onto the existing ternary matvec
- [ ] Fixtures: run the Python source at the pin on N snippets spanning ≥3 languages (Rust, Python, Go + one clean trap) → JSON-pin every scalar head + line-prob row
- [ ] **G1-A exactness, staged precisely:** (a) **bit-exact ONLY upstream of the first float operation** — the integer template/conv front (embeds are integer-indexed gathers and 0/1 constructions; the Block-0 causal prefix path on the integer template stage). This stage boundary is NAMED here so the bar cannot be read as covering the context blocks: after the Block-0 gates, `state` is float, and `doc_mean` divides by a float sum. (b) post-`exp`: **ABSOLUTE error band on probabilities** — relative error is ill-conditioned near zero: sigmoid heads + line probs |Δp| ≤ 1e-4, band pre-registered HERE. (c) **fold order pinned sequential-ascending per Issue 940.** (d) softmax sites: the attention softmax (`:164`) is LOAD-BEARING for the sigmoid heads (it feeds `state`) → covered by the same absolute band as part of the primary path; the ordinal (`:244`) and style (`:245`) softmaxes are reproduction-only side outputs — banded, never cited as exactness evidence, and recorded at each site as reference-faithful semantics (house sigmoid law untouched: softmax appears in no new-primitive surface)
- [ ] Falsification arms: (1) a single-trit flip in the `.cact` → a pinned output MUST move (vacuous-exactness catch); (2) a deliberately-perturbed fixture MUST fail the band
- [ ] G2-A: warm ≤ 10 ms/file on a QUIET m3, `scripts/bench_preflight.sh` PROVENANCE line quoted beside the number (load class + power state; box state is part of the claim)
- [ ] G4-A: alloc-free steady state (counting-allocator canary)

## Arm B — the exact-circuit builder (`ternary_circuit`)

- [ ] Builder API compiling the integer idiom kit into packed ternary: threshold template circuits (the `_sub_filter` shape — bit-plane stamping + integer max-margin scale/bias), word-boundary/line-lead/context-veto modifiers, causal prefix-scan matrices, Paley-Hadamard ±1 projections
- [ ] G1-B: bit-exact circuit outputs vs hand-computed fixtures (all integer/pre-activation — **no exp in scope**)
- [ ] Zero-allocation apply path; fixed `[T; N]` for bounded domains; `#[repr(u8)]` on fieldless enums
- [ ] OUT OF SCOPE: reproducing compile.py's seeded constructions (`RandomState(20261115)` :238, `RandomState(0)` :277, `RandomState(777)` :314) — a Rust recompile needs MT19937 + legacy numpy `randn` ports; named blocker, not attempted in this PoC

## Closing

- [ ] Record the PoC verdict in Research 677 §6; **if any gate fails, DEMOTE in the same commit** (ungoat law: grade struck, tier updated — stale grades are noise)
- [ ] The refine-side rider (report-only hygiene-delta on the fix lane) proceeds ONLY IF Arm A passes AND an eval surface exists (rethink `.proposals/006` T1) — stationary-surface law: LEVEL readouts only, never dynamics
