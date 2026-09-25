# Issue 885 — `laya-tetris-v3`: the real-hard-drop lane (Issue 884 path B)

**Status:** IN PROGRESS — sim + join + JS port landed locally; oracle/fixture/head/wasm/demo pending.

Owner call (2026-09-25): do path B. Per Issue 878's rule it ships as a NEW grammar
(`laya-tetris-v3`) with its own fixture and head; the pinned v2 fixture stays committed and
byte-identical everywhere.

## Measured delta (v2 → v3, seed 607)

- Same 120 boards, same pieces, same option count (2660) — the greedy ladder never picked a
  tunnelled spot, so no play-state changes.
- Exactly **3 options move**, all on the authored `holes` archetype (`arch:holes:{T,J,L}`,
  rot 3 col 8): v2 rows 15/14/16 (through the covered hole at (16,8)) → v3 rows 10/9/11 (on
  the plateau). Their sentences change; the other 2657 are byte-identical.
- v2 dump digest still reproduces: `aa07b3b4…` (the v2 default is unchanged).

## Oracle provenance decision

The laya oracle is one independent forward per option sentence. riir-reflex numerics moved
since the v2 generator (`76f4b92`, 164 commits), so a full 2660 re-run could perturb all 117
unchanged states and v3 would no longer isolate the drop fix. The join therefore re-runs the
oracle on the **3 changed states only** (102 forwards) and carries the other 117 verbatim
from v2 — allowed only when every option sentence is byte-identical (asserted). The 99
unchanged options inside the 3 fresh states are the numerics-parity read, recorded in `_meta`.

## Checklist

- [x] katgpt-rs `DropRule {DeepestFit (v2), FromTop (v3)}` + `landing_options_with`; v2 default kept
- [x] `tetris_01_state_enum --grammar laya-tetris-v3` + `--join/--carry-from` (proven: rebuilds v2 states byte-identically, both full and carried paths)
- [x] dev-dep `serde_json/float_roundtrip` — the default parser was 1 ULP off on 90/120 v2 lines
- [x] fixture drift detector + arenas follow the fixture's own grammar (`fixture_rule`)
- [x] reflex-site JS `DropRule`, site default v3; JS FROM_TOP == Rust v3 dump 2660/2660
- [ ] oracle on the 3 changed states (riir-reflex `laya_oracle_batch`, M3 Metal) + join → `tests/fixtures/tetris_oracle_laya_en_v3.jsonl`
- [ ] riir-reflex: serve the v3 head; `fixture_pins()` hashes all three embedded fixtures (pins were length-only)
- [ ] reflex-site: v3 golden sha256 pin, wasm-head re-gen (corpus blob + anchors), rebuild `arena_head.wasm`
- [ ] re-record demo walks against a local v3 engine; head parity + demo check + demo smoke
- [ ] docs (anchors, λ, agreement) + deploy + close
