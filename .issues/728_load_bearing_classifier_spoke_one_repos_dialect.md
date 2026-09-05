# Issue 728 — `silent_now_load_bearing` was 0 in all 16 repos because the classifier speaks ONE repo's dialect

**Status:** OPEN — **T1 DONE** (2026-09-06, the classifier is widened and the
katgpt-rs gate now reds on 2 real defects it was blind to). **T1 + T4 DONE**
(the classifier is widened, the katgpt-rs gate now reds on 2 real defects it
was blind to, and the cross-repo sweep ratchets all 12 instances). **T2 is the
remaining fix**: arm `bridge_spec_match` + `pencil_spec_match` with
`required-features` rows and re-pin `max_load_bearing` 2 → 0. T3 (the 10
sibling-owned instances) is now reported by every sweep run, which is a better
channel than a file in each of their repos.

## The finding

The first run of `cfg_gated_target_audit.py`'s load-bearing classifier over the
whole workspace (rather than over katgpt-rs, which is all
`cfg_gated_floor_gate.py` has ever covered) returned:

    silent_now_load_bearing = 0    in every one of the 16 repos

261 SILENT-NOW targets across the workspace, and not one of them classified as
load-bearing. That reads as "nobody ships a silent load-bearing gate". It
actually meant **the classifier speaks katgpt-rs's dialect** — `goat`, `gate`,
`g<N>`, `drill`, `proof`, plus the seven tokens added by Issue 713 T4c. The
token set was measured against **2,157 katgpt-rs-era target names**; the
workspace corpus is **3,081**, and the sibling naming conventions were never in
the sample it was validated against.

This is the **fourth** instance of the failure `.docs/10_audits/cfg_gated_silent_zero_pass.md` T4c
already documents, and the first one found by widening the *population* rather
than the vocabulary. AGENTS.md's own warning is the one that applies:

> Agreement licenses the pin against a classifier **bug**; nothing licenses it
> against a classifier being congenitally **narrow**.

## What the corpus table says

Candidates were measured against ALL 3,081 workspace test/bench/example target
names — not just the SILENT-NOW ones — which is the defence AGENTS.md
prescribes. Six tokens and one compound admitted; the rest rejected with
reasons, on the existing file's own standard ("names a property the file exists
to FAIL on"):

| admitted | corpus | why |
|---|---|---|
| `parity` | 22 | ALL 22 are A-vs-B equivalence (GPU vs CPU, cudarc, tokenizer, install-copy). Zero false positives; same family as `equivalence` |
| `integrity` | 6 | all genuine integrity assertions |
| `roundtrip` | 5 | encode/decode identity |
| `monotonicity` | 4 | riir-chain slashing / domain-registry invariants |
| `reachability` | 3 | `taming_reachability`'s green IS the evidence an authored goal can be won |
| `exactness` | 2 | `ledger_exactness` — exact arithmetic |
| `spec_match` (compound) | 39 | riir-chain's conformance convention, which katgpt-rs shares |

| rejected | corpus | why |
|---|---|---|
| `e2e` | 83 | names a SCOPE, not a property — the reason `check` was rejected. The largest class, and still not a gate name |
| `match` | 49 | `attn_match_*` (attention matching), `quest_match_tui` (a game match) |
| `spec` | 49 | in THIS repo `spec_` means **speculative** decoding (`spec_reconciliation_bench`, and a `_demo`) |
| `cost` `throughput` `overhead` `latency` `scale` `growth` | 70 | name MEASUREMENTS — exactly why `calibration` was rejected |
| `persistence` | 7 | names a subsystem |
| `identity` | 6 | genuine homonym: 4 of 6 are `signer_identity` / `identity_matcher`, a domain NOUN |
| `oracle` | 3 | names a technique, and in riir-chain vocabulary a component |
| `liveness` `conformance` `idempotence` | 0 | matched nothing |

**`spec_match` is why bigram support was added.** Neither half can be admitted
alone without dragging in a homonym class, so it is an explicit adjacent-pair
compound — the same principle as `regate`, generalised to a mechanism.

## The instances: 0 → 12, and two of them are ours

| repo | targets |
|---|---|
| **katgpt-rs** | `bridge_spec_match`, `pencil_spec_match` |
| riir-ai | `gemma4_q4k_gguf_parity`, `test_kimi_k3_gpu_forward_batched_parity`, `test_kimi_k3_gpu_forward_batched_with_saved_parity` |
| riir-chain | `domain_registry_monotonicity_spec_match`, `signer_identity_spec_match`, `mnemonic_spec_match` |
| riir-train | `plan354_t3_laprop_dispatch_parity`, `plan354_t4_laprop_parity` |
| riir-dapps | `kat_grant_reachability` |
| riir-viewbridge | `net_ffi_roundtrip` |

**The katgpt-rs pair is the part that matters here**, because this repo is the
one with the gate. Both are `#![cfg]`-gated on a default-off feature with **no
`[[test]]` row at all** (auto-discovered), so a plain `cargo test` compiles each
to an empty binary and prints `ok. 0 passed`:

- `tests/bridge_spec_match.rs` — feature `action_bridge` (root `Cargo.toml`)
- `crates/katgpt-core/tests/pencil_spec_match.rs` — feature `spectral_pencil`

And the reason they were invisible is worth keeping: katgpt-rs ships **seven**
`*_spec_match` targets, and **five of them were only ever classified because
they also carry `g1`** (`kda_g1_spec_match`, `mla_g1_spec_match`,
`attn_res_g1_spec_match`, `moe_g1_spec_match`, `situ_g1_spec_match`). The
convention was always here; the classifier saw it by accident, and missed the
two that do not spell `g1`.

## Tasks

- [x] **T1** — widen the classifier (6 tokens + `LOAD_BEARING_BIGRAMS` +
  adjacent-pair matching in `is_load_bearing`), each candidate measured against
  the full 3,081-name corpus, with every rejection reasoned in the source.
  Homonym rejections pinned: `spec_reconciliation_bench/_demo`,
  `attn_match_online`, `quest_match_tui`, `checkpoint_cost`,
  `bench_578_mcts_budget_sweep`, `aggregate_delegate_propagate`.
- [ ] **T2** — arm `bridge_spec_match` (root) + `pencil_spec_match`
  (katgpt-core) with `[[test]]` + `required-features` rows; re-pin
  `max_load_bearing` **2 → 0** in the same commit. **Deliberately deferred a
  few hours, not skipped**: the Issue 513 T2 sweep is mid-flight over this
  repo's `target/` (605 rows / 370 grouped cargo invocations), and editing
  `crates/katgpt-core/Cargo.toml` invalidates lib units shared across many of
  its remaining groups. `max_load_bearing` is pinned at the **measured 2** in
  the interim, so a THIRD instance still reds the push that adds it — the
  ratchet keeps working while the backlog is visible.
- [ ] **T3** — report the 10 sibling instances to their owners. Not this
  repo's to fix; the cross-repo sweep (below) is what keeps them from growing.
- [x] **T4** — `scripts/cfg_gated_drift_sweep.py` +
  `cfg_gated_drift_floors.txt`, the fifth member of the sweep family. **Done
  now rather than after T2** — the deferral reasoning written here first ("only
  meaningful once this repo's own number is 0") was wrong, and
  `numbering_drift_floors.txt` is the precedent that refutes it: a ratchet
  pinned at each repo's MEASURED backlog is exactly how a sweep covers repos
  whose defects it does not own. Waiting for a 0 would have left the other ten
  ungated for no gain. Canaried 13 directions, each verified on the RIGHT
  assertion; the two instrument-failure directions exposed a real gap — the
  report's selftest raises a bare `AssertionError`, so the sweep died with a
  traceback where it should have returned the exit-2 "instrument untrustworthy"
  verdict. Fixed.

## Why the gate could not have caught this itself

`max_load_bearing = 0` is green over whatever `is_load_bearing` can NAME. That
is the documented hazard, and the documented defence is the corpus-wide
candidate-token table — **not** a second opinion, because two independently
built classifiers already agreed here and agreed on the wrong population. The
only thing that moved this was pointing the instrument at a bigger corpus.
