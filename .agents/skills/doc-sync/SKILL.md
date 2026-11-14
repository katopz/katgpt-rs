---
name: doc-sync
description: Synchronize each repo's `.docs/` and `README.md` with recently succeeded plans/issues/benchmarks by diffing git history against the last documented entry. Use after landing a GOAT-passing plan, closing a batch of issues, or quarterly as a doc-hygiene gate. Covers every workspace repo and knows each repo's doc layout + where to record what, including each repo's root BOUNDARY.md contract (drift rows vs issue state).
---

# doc-sync — Keep `.docs/` + `README.md` in sync with landed work

This skill brings a repo's documentation up to date with the work that has
**landed in git but not yet been written up**. It is the doc equivalent of a
`cargo doc` rebuild: the code shipped, now make the narrative match.

## When to use

- After a plan closes with a GOAT/gain verdict (promote, keep-opt-in, or honest fail).
- After a batch of issues resolves (especially negative-result issues that move a
  primitive's status line).
- After a feature is promoted to default-on OR demoted to opt-in.
- Quarterly as a doc-hygiene gate.
- **Any numbered-file work session (issue 1205 fix path 3):** run that repo's numbering gate before leaving — `py ../katgpt-rs/scripts/numbering_gate.py <repo>` (exit 1 = an un-bumped `.highwater`, a collision, or a stale/malformed allocator; fix in the same commit). riir-ai's guard layer 1.39 carries the same check every guard run; this bullet is the standing lane for repos without that layer.
- **NOT** for speculative work — only landed, committed work counts.

## `BOUNDARY.md` — the 4th doc surface (added 2026-08-21)

Every repo ships a root `BOUNDARY.md`: Owns / Does not own / May depend on
(crate-granular allowlist) / Inherited (links) / **Drift ledger**. It is a
doc-sync surface because its drift ledger is a *claim about issue state*, and
claims rot:

- **Row ⟺ open issue.** A `fixable` / `owner-call` row REQUIRES an existing
  issue file; a `by-design` row cites a decision record instead. When an issue
  closes, its row must be removed **in the same commit** — the noise-reduction
  rule extended to BOUNDARY.md. A row whose issue is gone is the boundary
  equivalent of a stale README claim.
- **Flag row-without-issue** and **issue-without-row** (a boundary issue that
  landed with no ledger row is invisible to the guard).
- **Don't hand-verify the dep tables** — run
  `riir-ai/scripts/ci_boundary_contract.sh`. It fails on an undeclared
  cross-repo dep, a stale allowlist row, an unparseable ledger, and on the 4
  split-prep invariants. `--list-deps` prints the measured graph.
- **Numbers in a contract are measurements**, so they carry a date. If a row
  cites "N symbols" or "N packages", re-measure before trusting it in a new
  decision (`riir-ai/BOUNDARY.md` D2/D3 are the pattern).
- The contract is per-repo; cross-repo rules live in ONE canonical home
  (chain admission → `riir-chain`, dep matrix + split-prep → `riir-ai`) and
  every other repo LINKS. Never copy a cross-repo rule into a second file —
  that is the duplication doc-sync exists to catch.

## The workspace repos and their doc shapes (**derived, never counted** — de-counted 2026-09-04)

Hand-typed counts here went wrong twice within days (a header saying 14 over 12 rows; 18
over a 16-repo window — different membership, same count), and repos missing a row were
skipped silently by runs that walked the table. **Never re-type the count or trust the
census below as current — it is a dated snapshot; derive the membership:**

```bash
cd /Users/katopz/git && for d in */; do
  [ -f "$d/BOUNDARY.md" ] && [ -d "$d/.git" ] && echo "${d%/}"
done
```

Canonical count + product-vs-workspace split: `katgpt-rs/AGENTS.md` §"Repo count".

Each repo has a different doc layout. **Read the repo's `AGENTS.md` first** —
it documents the canonical layout and the numbering discipline.

| Repo | `.docs/` shape | `README.md` | Numbering highwater | Working branch |
|---|---|---|---|
| `katgpt-rs` | 10 numbered folders (`01_orientation/` … `10_audits/`), unnumbered files inside. The **public** selling-point book. | Large showcase + feature tables + getting-started. | `.plans/.highwater`, `.issues/.highwater`, `.benchmarks/.highwater`, `.research/.highwater` | `develop` |
| `riir-ai` | **12 numbered folders** (`01_orientation/` … `12_inference/`; `02_inference/` was renumbered to `12_inference/` on 2026-08-08 to resolve a `02_` prefix collision with `02_crates/`). The **private** consolidated selling-point book. | Large showcase + crate table. | same `.highwater` files | `develop` |
| `riir-chain` | **7 numbered folders** (`01_orientation/` … `07_formal_verification/`, reindexed from flat on 2026-08-08), unnumbered files inside. Canonical self-description — the chain owns the truth about the chain; `riir-ai/.docs/07_neuro_symbolic_chain/` is the consumer/fusion view and links here. Two workspace members (`riir-chain` lib + `riir-chaind` daemon). Build surface still lives in `README.md`; FV invariants in `AGENTS.md` + `.proofs/README.md`. | Build commands + feature flags + the wallet/RPC trust surface + the `merkle_root` lesson. | same | `develop` |
| `riir-neuron-db` | **11 numbered folders** (`01_orientation/` … `11_cli/`; the 11th added 2026-09-09 for the `ndb` CLI — Plan 327, Warm-tier CRUD + identity login + multi-node sync), unnumbered files inside. Covers all `src/` modules + `examples/`. Matches the `riir-ai/.docs/` format. | What the crate owns + feature gates (default-on / opt-in) + feature→chain mapping + `merkle_root`/`can_freeze` lessons. | same | `develop` |
| `riir-train` | **6 numbered folders** (`01_orientation/` … `06_cross_cutting/`, reindexed from flat on 2026-07-15), unnumbered files inside. Training-method research vault. | Role + sibling layout. | same | `develop` (**flipped from `main` 2026-09-04** — develop created from the `main` tip and made the default branch, the Issue 704 convention; `main` frozen) |
| `riir-game-sdk` | **10 numbered folders** (`01_orientation/` … `10_multiplayer_topology/`; the 10th was added for the two-binary production topology + avatar/game sync facade), unnumbered files inside. Covers all `src/` modules + `examples/`. Matches the `riir-ai/.docs/` / `riir-neuron-db/.docs/` format. | Boundary rule + leaf constraint + spatial canonical + Phase 2/3 status + feature gates. | same | `develop` |
| `riir-mmorpg-examples` | **No `.docs/` folder** — docs live in `AGENTS.md` (extensive: role, topology, plans/issues/benchmarks index, canonical-failure lessons) + `README.md` (status + build commands + env vars) + `.plans/` / `.issues/` / `.benchmarks/` files. POC consumer of `riir-game-sdk`. | Status + build commands + Plan/Issue index. | same | `develop` |
| `riir-refine` | **12 numbered folders** (`01_orientation/` … `11_domains/` + `12_ane/`; the 12th added for the Apple Neural Engine substrate knowledge — private runtime API, MIL/blob formats, M3 Max findings, the Rust-bridge negative result + working ObjC substrate — riir-ai Issue 726 T0 distillation), unnumbered files inside. The rulewright vault — corpus/drafter/pruner/verify/self-evolve/domains narrative. `AGENTS.md` carries the batch-mining progress notes (the sweep record home for cross-repo clippy-fix passes). | Status + Quick Start + Usage + feature gates. | same | `develop` |
| `riir-unity` | **RETIRED 2026-09-04 → `git/obsolete/` (owner act). Lineage only; do not route work here.** No `.docs/` folder — AGENTS.md-centric (domain boundary, Unity MCP rules, issue log) + `.benchmarks/`. The Unity host; Rust work belongs in riir-viewbridge, so doc-sync here = AGENTS.md issue-log sections + module-map freshness. | Role + boundary + sibling layout. | same | `develop` |
| `riir-viewbridge` | **1 numbered folder** (`.docs/01_orientation/` — `README.md` + `crate_role.md`) + a `.docs/README.md` index; AGENTS.md still carries the workspace layout, boundary rules (latent/raw wall, generated-bindings, catch_unwind) + issue log, and `.benchmarks/` the node GOAT. The Rust FFI side of the Unity bridge. | Role + boundary + build commands. | same | `develop` |
| `mmorpg-remake` | **1 numbered folder** (`.docs/01_orientation/` — `README.md` + `unity_host.md`, from Plan 031 Phase 5) + AGENTS.md/README.md/BOUNDARY.md. The Bevy/wasm viewer half after the Unity host split out. | Role + boundary + build/run commands + the vessel-texture + present-mode records. | same | `develop` (highwater: `.issues` 005, `.benchmarks` 002) |
| `riir-dapps` | **1 numbered folder** (`.docs/11_kat_service/`, numbered to MIRROR the kat-service group elsewhere — there is no 01–10 here, so don't read the prefix as a tenth sibling) holding dated evidence bundles (`2026-09-10_domain_swap/`: a README narrative + 5 manifest/health JSONs), plus the AGENTS.md-centric remainder (the one-way game → dapps → chain invariant, the three-test rule, tiered-durability record) + `.plans/` / `.issues/` / `.benchmarks/`. The settlement-composition layer. | Boundary + build + the `direction_gate` + kat rail status. | same | `develop` |
| `riir-dao` | **No `.docs/` folder** — AGENTS.md-centric (the KAT tokenomics agent: signals → strategy → guard → advisory → commit; the G5 advisory-only verdict) + `.plans/` / `.benchmarks/`. | Boundary + build + the direction gate. | same | `develop` |
| `riir-armageddon` | **RETIRED 2026-09-02 → `git/obsolete/` (owner act). Lineage only; do not route work here.** `.docs/` exists but is EMPTY — AGENTS.md-centric in practice (arena/game-product domain types). Added 2026-09-01 | yes | `.issues` 005, `.plans` 008 | **`main`** — not `develop`; check before branching |
| `riir-auth` | **`.docs/` exists but holds only `.highwater`** — i.e. no docs at all, AGENTS.md-centric in practice (the numbering file was created ahead of the folder's first document). Added 2026-09-01 | yes | `.issues` 002, `.benchmarks` 4, `.plans`/`.docs`/`.research` at 0 | `develop` |
| `riir-burner` | **RETIRED 2026-09-04 → `git/obsolete/` (owner act). Lineage only; do not route work here.** Flat numbered FILES, no folders — `.docs/001_model_verdict.md` … `016_*.md` (7 files; two share 016 — the numbering discipline is not enforced here). Added 2026-09-01 | yes | `.issues` 015, `.plans` 019 | `develop` |
| `riir-deployer` | **2 numbered folders** (`01_orientation/`, `02_runbooks/`) + a `.docs/README.md` index — the smallest numbered shape in the workspace. No `CLAUDE.md`. Added 2026-09-01 | yes | `.issues` 003, `.plans` 002, `.benchmarks` 001 | `develop` |
| `katgpt-web` | **No `.docs/` folder** — AGENTS.md-centric. Added 2026-09-01 | yes | none | `main` — the `feat/percepta-arch-diagrams` checkout note is history: Issue 002 (2026-09-09, `01a9c51`) merged + ff'd main to the working branch and the local feat branch is deleted; the repo sits on its trunk |
| `mmorpg-editor` | **NAMED (not numbered) `.docs/` subfolders** — `new-game-schema/`, `registry/`, plus loose `GAME_ASSETS.md`. Carries `ARCHITECTURE.md` + `DESIGN.md` alongside AGENTS/README, and `ARCHITECTURE.md` is where the internal layering lives (`BOUNDARY.md` covers only the outer edge). Added 2026-09-01 | yes | `.issues` 141, `.plans` 140 | `develop` |
| `riir-esp32` | **No `.docs/` folder** — AGENTS.md/BOUNDARY.md-centric (the ESP32 Satellite device-tier POC: `crates/riir-satellite-probe`, emulator recipes; explicitly not prod). **Row ADDED 2026-09-11** — the repo moved out of riir-chain 2026-09-06 and this census never gained a row (the same silent-skip class the de-counted header warns about). | Role + boundary + domain test. | `.issues` 110, `.proposals` 006 | `develop` |
| `riir-kat` | **No `.docs/` folder, no README** — AGENTS.md/BOUNDARY.md/HISTORY.md-centric (the KAT network CLIENT + wire-protocol plane, spun out of riir-clippy issue 088 on 2026-09-10; single crate). **Row ADDED 2026-09-11** — born 2026-09-10, censused a day late. | none — AGENTS.md §Status is the surface | `.issues` 1 | `develop` |
| `riir-shader` | **4 numbered folders** (`01_orientation/`, `02_substrate/`, `03_porting_method/`, `04_size_budget/`), unnumbered files inside — the WebGPU visual-effect substrate as Bevy plugins (ports every vgpu.sh example to wasm32+native as composable plugins; leaf, zero riir-* deps; upstream `vercel-labs/vgpu` @ 42bc4bc, MIT, attribution header per shader) + the `crates/riir-shader-graph` serde-only graph medium (Proposal 001 Phase 1, `3aa01ce`). | yes (README) | `.issues` 16 · `.plans` 002 · `.benchmarks` 3 · `.proposals` 001 | `develop` |
| `mmorpg-remaster` | **Flat numbered FILES** (`.docs/00_principal.md` … `10_quest_system.md` + `lessons_code_smell_audit_023.md` + `task_index.md`; design docs from the remaster gap analysis, `c0954dc`). **READ-ONLY to agents** (main + develop, owner rule). | yes (README) | `.issues` 30 · `.plans` 045 · `.benchmarks` 042 | `develop` (read-only) |
| `seal-remake` | **6 numbered folders** (`01_orientation/` … `06_tooling/` — 02_viewer, 03_game_core, 04_edge, 05_multiplayer_topology, 06_tooling were grown 2026-10-24 by the Step-0 code-surface enumeration from the crates' own lib docs) + a top-level `.docs/README.md` entry point. AGENTS.md remains the LANE-level narrative (feature umbrellas, boot postures, env vars) — the book never restates it. | yes (README) | `.issues` 119 · `.plans` 052 · `.benchmarks` 026 · `.proposals` 007 | `develop` |
| `riir-reflex` | **5 numbered folders** (`01_orientation` … `05_resources`) + top `README.md` — the fleet-format book its AGENTS.md documents; `03_decision_flow` mirrors to reflex-site (the mirror law). Public decision-engine serving repo (katgpt-rs Plan 603 deliverable). **Row ADDED 2026-10-24** — born post-09-14, never gained a row (the seal-remake class). | yes (README) | same `.highwater` files | `develop` |
| `riir-reflexer` | **6 numbered folders** (`01_orientation`, `02_wire_protocol`, `03_decision_flow`, `04_vessel_format`, `05_measurement_lane`, `06_resources`) + top `README.md` — the vessel FORMAT repo. **Row ADDED 2026-10-24** (post-09-14, no row). | yes (README) | same `.highwater` files | `develop` |
| `riir-instinct` | **3 numbered folders** (`01_serving`, `03_decision_flow`, `05_resources`) + top `README.md` — the open decision lane (the moat half is `riir-rethink`); the number gaps are the seed's own, not a shape to fix. **Row ADDED 2026-10-24** (post-09-14, no row). | yes (README) | same `.highwater` files | `develop` |
| `riir-rethink` | **Numbered folders with gaps** (`03_decision_flow`, `05_resources`, `06_trust_flow`, `07_moat_licensing_key_root`, `08_architecture_primer` — no 01/02/04; the seed's own numbering) — the PRIVATE moat repo, born 2026-10-03 from `riir-instinct/moat/`. **Row ADDED 2026-10-24**. | yes (README) | same `.highwater` files | `develop` |
| `riir-refab` | **1 numbered folder** (`001_orientation`) + README — the pure-Rust rigged-GLB refab lane, born 2026-10-11, zero workspace deps by contract. **Row ADDED 2026-10-24**. | yes (README) | same `.highwater` files | `develop` |
| `riir-infer` | **Flat numbered FILES** (`001_exl3_trellis_format_support.md` … `004_silero_vad_audio_lane_poc.md` + unnumbered `p4_opening_checklist.md`) + README — the LLM-inference substrate carved from riir-ai 2026-09-22 (public). **Row ADDED 2026-10-24** (post-09-14, no row). | yes (README) | same `.highwater` files | `develop` |
| `seal-game-editor` | **4 numbered folders** (`01_assets`, `02_schema`, `03_registry`, `04_signoffs`) + top `README.md` — the desktop content editor; AGENTS.md is the routing table, `ARCHITECTURE.md`/`DESIGN.md` carry the layering. **Row ADDED 2026-10-24** — predates the 09-14 census but the seal family was never in the contract set that census counted. | yes (README) | same `.highwater` files | `develop` (tickets per the `git-workflow` skill) |
| `seal-online-remaster` | **Flat numbered FILES** (`00_principal.md` … `10_quest_system.md` + `task_index.md`; the mmorpg-remaster shape class) — the game runtime + legacy data source. **DEPRECATED 2026-09-16**: migration source only, read-only upstream, its remaining crates path-dep INTO seal-remake; do not route new work here. **Row ADDED 2026-10-24**. | yes (README) | same `.highwater` files | `develop` |
| `sealm-toolkit` | **`.docs/` exists but is EMPTY** (the riir-auth class) — the Seal M asset toolkit (Python tools beside seal-game-editor). **Row ADDED 2026-10-24**. | yes (README) | same `.highwater` files | `develop` |
| `mirror-twin-lint` | **4 numbered folders** (`01_assets`, `02_schema`, `03_registry`, `04_signoffs`) + top `README.md` — the seal-game-editor shape. **Row ADDED 2026-10-24** — BOUNDARY carrier with no row. | yes (README) | same `.highwater` files | `develop` |
| `jup-tool` | **No `.docs/` folder** — born 2026-10-14 (the JUP rescue tooling: Rust binary + CF Worker, hand-rolled Solana wire primitives). **Row ADDED 2026-10-24**. | yes (README) | same `.highwater` files | `develop` |
| `reflex-site` | **No `.docs/` folder** — the public arena site front (riir-ai/reflex-site). **Row ADDED 2026-10-24**. | yes (README) | same `.highwater` files | `develop` |
| `riir-llm` | **No `.docs/` folder** — the shared LLM-transport leaf (riir-refine Issue 124). **Row ADDED 2026-10-24**. | yes (README) | same `.highwater` files | `develop` |
| `seal-std` | **No `.docs/` folder** — born 2026-10-07; the shared standard crates (leaf-per-crate, wire-is-contract law in AGENTS.md). **Row ADDED 2026-10-24**. | yes (README) | same `.highwater` files | `develop` |
| `seal-render` | **No `.docs/` folder** — born 2026-10-21 (the canvas core carved from seal-game-editor, plan 412 step 1c); AGENTS/README/BOUNDARY only. **Row ADDED 2026-10-24**. | yes (README) | same `.highwater` files | `develop` |

## The sync workflow (per repo)

### Step 0 — Enumerate the code surface (owner mandate, 2026-10-24 — do this FIRST, every run)

Baseline-diffing (Step 1) covers TEXT, never SURFACE: a book that never grew passes every
baseline range green while the code underneath tripled. seal-remake held a 2-doc book over an
8-crate / ~550-file workspace for weeks — every run green, nothing doc-shaped ever committed.
Before diffing anything:

1. **Derive the surface** — workspace `members` + `[[bin]]` rows from the root manifests, and
   the crate file counts (`find crates/*/src -name '*.rs' | wc -l`). Never trust a count typed
   in any doc (including this one).
2. **Ask what no doc row covers.** A crate/bin with zero book coverage is a GAP even when every
   existing doc is accurate — the book UNDER the code is the same rot as a stale claim, just quieter.
3. **Fill gaps from code facts**: workspace members → a crates map (role from each crate's own
   lib doc); a big uncovered crate → its own numbered folder (family format: bare slugs, a README
   index per folder, a top-level `.docs/README.md` entry point). Structural maps only — module
   roles from `//!` headers, features from Cargo.toml, bins from `[[bin]]` + bin headers. Anything
   not derivable this session is "—" or deleted, never guessed. NEVER restate the repo's
   AGENTS.md lane prose — that duplication is the drift this skill exists to catch; link it.
4. **Then run the Shape-change contract below** if the fill added top-level folders — the census
   row update is part of the fill, not an afterthought.

### Step 1 — Find the last documented commit

```sh
git --no-pager log --oneline <branch> -- ".docs/**" "README.md" | head -20
```

The most recent `docs:` commit is your baseline. Everything after it is **undocumented work**.

### Step 2 — List landed-but-undocumented work

```sh
git --no-pager log --oneline <baseline>..<branch>
```

Filter for:
- `feat:` / `fix:` commits that close a plan or issue (grep the message for `Plan NNN` / `Issue NNN`).
- `docs:` commits that close research notes or benchmarks (these may already be half-documented).
- Promotions / demotions (search for `promote`, `demote`, `default-on`, `opt-in`).

Cross-reference against the repo's `.plans/`, `.issues/`, `.benchmarks/`,
`.research/` folders — read the highwater files to know the current max number.

### Step 3 — Classify each landed item

For each undocumented plan/issue, classify it:

| Verdict | What to write |
|---|---|
| **GOAT PASS + promoted to default-on** | Add to the default-features list in README. Add/update the feature table row in `.docs/01_orientation/overview.md` (or equivalent). Mark the plan's TL;DR with the promotion date. |
| **GOAT PASS + stays opt-in** | Add to the opt-in features table in README. Update the `.docs/` feature catalog. Honest about why it stays opt-in (heavy, fusion-pending, diagnostic-only). |
| **GOAT FAIL / negative result** | Add to the negative-results section (`09_feature_catalog/negative_results.md` for katgpt-rs, equivalent elsewhere). Mark the plan with the failure mode. **Keep the entry** — negative results are load-bearing. |
| **Issue closed (investigation)** | If it changes a primitive's status (e.g. "map-fidelity hypothesis exhausted"), update that primitive's README/docs entry. If it's pure investigation with no status change, it may not need a doc writeup — judge case by case. |
| **Research note (PASS/Gain/GOAT)** | If it led to a plan, the plan entry is the writeup. If it's a standalone PASS verdict with no plan (e.g. "already shipped"), add a one-liner to the relevant `.docs/` group README. |

### Step 4 — Write the updates

Apply the repo-specific rules:

#### katgpt-rs (the public engine)
- **README.md**: feature showcase entries (one `###` section per primitive with a GOAT gate table), the opt-in features table, the default-features list, the Documentation Index.
- **`.docs/01_orientation/overview.md`**: the full feature-flag table (one row per flag).
- **`.docs/09_feature_catalog/`**: opt-in features + negative results.
- **`.docs/<group>/README.md`**: the group's fusion map + file list.
- Numbering: never reuse a plan/issue/benchmark/research number. Read the `.highwater` file, use `value + 1`, write it back.

#### riir-ai (the private runtime)
- **README.md**: crate table + feature showcase.
- **`.docs/`**: 12 numbered folders — drop new docs in the right group, add one line to the group README.
- Cross-repo: if a katgpt-rs primitive was consumed, note the fusion in the riir-ai doc AND the katgpt-rs doc (bidirectional cross-refs).

#### riir-chain (7-folder `.docs/` book, reindexed 2026-08-08)
- **README.md**: build surface, feature flags, consumers, drift notes.
- **`.docs/`**: 7 numbered folders mirroring the `riir-ai/.docs/` format — `01_orientation` (what it is + feature surface + module map + how the ledger works), `02_consensus`, `03_economics`, `04_daemon` (incl. the operator runbook), `05_wallet` (trust boundaries, SIWR, node certificates), `06_operations` (rolling upgrade across protocol versions, e2e coverage, failure scenarios), `07_formal_verification` (pointer — the invariant table stays in `AGENTS.md`). Drop new docs in the right group folder and add one line to that folder's `README.md` index table. The top-level `.docs/README.md` is the entry point.
- **Division of labour with riir-ai (set 2026-08-08):** riir-chain holds the canonical chain docs; `riir-ai/.docs/07_neuro_symbolic_chain/` is a **fusion map + feature highlights** that links here and keeps only what is riir-ai's own (the Egg/Shell raw-vs-latent boundary, latent precision realms, game-layer sync strategy, CF Workers edge topology). Do not re-centralize chain internals in riir-ai — that duplication is what drifted before. Cross-link bidirectionally.
- **Module map discipline:** `01_orientation/overview.md` claims to list every `src/` and `crates/riir-chaind/src/` subtree. If a plan adds a module, add the row — a map that silently omits modules reads as "these do not exist".
- **AGENTS.md**: the FV (Lean 4) invariant table lives here (mirrored in `.proofs/README.md`), NOT in `.docs/`. Plan 016 spec self-tests live next to each spec module under `.proofs/RiirChainProof/`.

#### riir-neuron-db (11-folder `.docs/` book; 10th added 2026-07-30, 11th `11_cli/` added 2026-09-09)
- **README.md**: build surface — feature gates (default-on / transitive / opt-in / per-feature prose sections for promoted primitives) + Formal Verification summary + License. Prose sections are reserved for promoted default-on features; opt-in features get table rows only.
- **`.docs/`**: 11 numbered folders mirroring the `riir-ai/.docs/` format. Drop new docs in the right group folder (by capability: shard substrate / freeze-thaw / consolidation / vessel / specialized / zone / examples / FV / **local-kv Warm tier** / **CLI**), add one line to that folder's `README.md` index table. The top-level `.docs/README.md` is the entry point. The `05_secure_vessel/vessel_primitive.md` doc is the restored home of the old `15_vessel.md` (corrected: riir-neuron-db is "this crate", NOT katgpt-rs per Plan 006). The `10_local_kv/` folder covers the `LocalKvStore` + `CommitLevel`/`CommitBatch` + WAL compaction + BM25 (the Warm tier substrate backing per-player state recovery in riir-mmorpg-examples Plan 013, added Issue 043). The `11_cli/` folder covers the `ndb` binary (Plan 327: Warm-tier CRUD, identity login via riir-auth `account_key`, multi-node WAL-mirror sync).
- **AGENTS.md**: the FV (Lean 4) invariant table lives here (mirrored in `.proofs/README.md`), NOT in `.docs/`. The `.docs/09_formal_verification/` folder is the narrative overview; `AGENTS.md` is the authoritative invariant table.
- Cross-repo: if a primitive was consumed by `riir-ai` or `riir-chain`, the fusion is documented bidirectionally.

#### riir-train (6-folder `.docs/` book, reindexed 2026-07-15)
- **6 numbered folders** (`01_orientation/` … `06_cross_cutting/`), unnumbered `.md` files inside — mirrors the `riir-ai/.docs/` format.
- Training-method research vault: adapter training, distillation/RL, data filtering, cross-cutting audits.
- `README.md` is minimal — role + sibling layout.
- `main` branch (no `develop`).

#### riir-game-sdk (10-folder `.docs/` book; 10th added for multiplayer topology)
- **README.md**: build surface — boundary rule, leaf constraint, spatial canonical, feature gates, Phase 2/3 status table.
- **`.docs/`**: 10 numbered folders mirroring the `riir-ai/.docs/` / `riir-neuron-db/.docs/` format. Drop new docs in the right group folder (by capability: spatial-entity / tick-world / rules-ai / game-builder / zone-living-world / gm-dashboard / examples / lessons / **multiplayer-topology**), add one line to that folder's `README.md` index table. The top-level `.docs/README.md` is the entry point. The `10_multiplayer_topology/` folder covers the two-binary authority/player production model + avatar/game sync facade (the consumer pattern for the documented C1/C2/C4 chain topologies).
- **AGENTS.md**: authoritative repo-local context (phase status, boundary rule rationale, leaf-constraint argument, the canonical-failure lessons). The `09_lessons/` folder is the narrative mirror of those lessons.
- `examples/`: showcase examples are part of the doc surface (Issue 517 rule) AND documented in `.docs/08_examples/`.
- **Leaf constraint reminder**: this crate has zero sibling path deps. Docs that reference sibling repos use relative links only — never imply a code dependency.

#### riir-mmorpg-examples (no `.docs/` folder — AGENTS.md-centric)
- POC consumer of `riir-game-sdk` (orchard multiplayer: 1000-NPC swarm + cross-target Bevy binary).
- **No `.docs/` folder** — documentation lives in:
  - `AGENTS.md` — the authoritative narrative (role, topology, plans/issues/benchmarks index, canonical-failure lessons, honest POC-grade caveats).
  - `README.md` — build surface (status, build commands, env vars, Plan/Issue index).
  - `.plans/` / `.issues/` / `.benchmarks/` — individual plan/issue/benchmark files.
- The `AGENTS.md` is large (~1000+ lines) and IS the doc surface — `doc-sync` for this repo means keeping `AGENTS.md` sections current with landed plans.

#### riir-refine (12-folder `.docs/` book)
- **`.docs/`**: 12 numbered folders mirroring the `riir-ai/.docs/` format — corpus / drafter / pruner / verify / ruliology / examples / benchmarks / lessons / self-evolve / domains / **ANE substrate knowledge** (`12_ane/`, riir-ai Issue 726 T0 distillation). Drop new docs in the right group folder, add one line to that folder's `README.md` index table.
- **`AGENTS.md`**: the batch-mining progress notes + sweep records live here (the cross-repo clippy-fix record home). A landed fix slice in a sibling repo (katgpt-rs, riir-train, riir-ai) gets its progress note in the SAME commit as the fix — a later `doc-sync` run defers to the fixing session (never write progress notes for someone else's in-flight sweep).
- **README.md**: Status + Quick Start + Usage + feature gates.

#### riir-unity — RETIRED 2026-09-04 (`git/obsolete/`), lineage only
- **`AGENTS.md`**: domain boundary (no Rust crates here; UPM package is build output; no engine substrate in C#) + the Unity MCP rules + the issue log. Doc-sync = issue-log sections for resolved issues + module-map freshness (the `Packages/com.riir.viewbridge/` population + scene wiring notes).
- The Rust side of any feature lives in `riir-viewbridge` — cross-repo arcs (e.g. Issue 004) document on BOTH sides at arc close.

#### riir-viewbridge (`.docs/01_orientation/` + an AGENTS.md-centric remainder)
- **`AGENTS.md`**: workspace layout (core/derive/abi/xtask) + boundary rules (latent/raw wall, generated-bindings rule, catch_unwind) + the issue log.
- **`.benchmarks/`**: GOAT records (e.g. Bench 002 node GOAT). Doc-sync = issue-log resolution entries + benchmark cross-refs.

#### Every repo not subsectioned above (riir-dapps, riir-dao, riir-auth, riir-deployer, riir-esp32, riir-kat, katgpt-web, mmorpg-remake, mmorpg-editor)
- Follow the census-table row — these are AGENTS.md/BOUNDARY.md-centric: doc-sync = AGENTS.md/BOUNDARY.md status sections + numbering highwater + README freshness (riir-kat has no README; its AGENTS.md is the surface). The repo's own AGENTS.md supersedes this skill. The no-docs rows added 2026-10-24 (jup-tool, reflex-site, riir-llm, seal-std, seal-render) and the empty-book `sealm-toolkit` ride the same treatment.
- `katgpt-web` checkout may sit on a feature branch (see census row) — sync the branch you find, and say which one in the run log.

### Step 5 — Verify

- **No broken links**: every `[...](.plans/NNN_*.md)` must point to a file that exists.
- **No stale numbers**: if a README entry says "ratio 0.01" but the benchmark says "0.27", the README is wrong — update it.
- **Numbering discipline**: `.highwater` files must be bumped when new plans/issues land.
- **Honesty**: a GOAT FAIL stays a GOAT FAIL in the docs. A "stays opt-in" primitive is documented as opt-in with the reason. Never upgrade a verdict in the docs without the benchmark to back it.

### Step 5.5 — Semantic drift lane (OPTIONAL stage — T4, Issue 977 / EGM `egm-drift-report`)

For each doc the quarter's sync actually touched (not the whole book — the touch set bounds the cost): verify the doc's **named symbols and checkable claims** against the CURRENT code state — does `fn name` still exist at the cited path, does the named feature still carry the cited default, does the named struct still own the cited field? Report ONLY differences (a clean sweep prints "No drift detected", never a per-doc recital). `feature-gate-audit` keeps ownership of the feature-gate CLAIM subset — this lane covers the general case: any named symbol/claim in a touched doc that the code no longer agrees with.

- The lane REPORTS, never auto-edits: a drift finding lands as a listed difference for the sync step (Step 4) to resolve, because the fix direction is a judgment (update the doc vs the code drifted).
- **Proof fixture (the zero-finding trap):** a deliberately drifted doc pair seeds the lane — a doc claiming a `fn drift_probe_fixture` at a path where it does not exist MUST be flagged; a run that reports nothing on the fixture fails the lane, never reads as clean docs (fixture seeded 2026-11-04, Issue 977; recovery via `git log -S drift_probe_fixture -- <this file>`).

### Step 6 — Commit (ONLY through the gate — never bare `git add` + `git commit`)

Per the global `AGENTS.md` rule: **always commit at task completion**. Use `docs:`
prefix. Stay on the repo's working branch (`develop` for most, `main` for
riir-train). Do not push.

**Commit through `tools/sync_commit.sh`** — the mechanical form of the
staged-index standing lesson below (the #42 incident swept a stale sibling
index through a bare `git add` + `git commit`; prose did not stop it, this
does). The gate refuses a non-empty index up front (stale sibling state —
adjudicate it first, never auto-clear), stages only the named paths, refuses
anything staged beyond them, and commits via the partial form
(`git commit -- <paths>` — a temporary index from HEAD + the named paths), so
content staged concurrently never enters the commit and the real index is
never disturbed:

```sh
sh <workspace>/katgpt-rs/.agents/skills/doc-sync/tools/sync_commit.sh <repo> \
  "docs: sync .docs + README with recent plans (NNN, NNN, NNN)" \
  .docs README.md
```

Name exactly the paths THIS run edited — the file when the edit is one file,
the smallest containing dir otherwise; never a directory you did not touch.
`.plans/ .issues/ .benchmarks/ .research/` join the list only when the run
actually wrote there.

## Cross-repo coordination

The 5-repo (now 10-repo) family shares numbering namespaces for
plans/issues/benchmarks/research **within each repo** but NOT across repos.
When a katgpt-rs primitive is consumed by riir-ai, the fusion is documented
**bidirectionally**: the katgpt-rs doc notes "consumed by riir-ai/NNN", and the
riir-ai doc notes "consumes katgpt-rs/NNN".

Formal verification (Lean 4) has its own cross-repo pattern (Research 351):
each repo's `.proofs/` instance is self-documenting via its invariant table in
`AGENTS.md`. The `doc-sync` skill does NOT cross-port Lean files between repos
(coordinator rule C4: private proofs stay private).

## Shape-change contract (when `.docs/` grows a new top-level `NNN_*` folder)

**This is the root-cause guard for skill drift.** The recurring failure mode: a
plan adds a top-level `.docs/NNN_*/` folder to a repo, lands the commit, and
nobody updates this skill file — so the next `doc-sync` run operates on a
stale folder-count assumption (canonical drifts: `riir-neuron-db` 9→10 via
Issue 043, `riir-game-sdk` 9→10 via the multiplayer-topology docs, `riir-chain`
flat→7 via the 2026-08-08 reindex, `riir-ai` 11→12 via Plan 455's orphaned
`02_inference/` folder discovered 2026-08-08). This contract makes the update a
grep-able checklist instead of an implicit expectation.

**Trigger:** any plan/issue/commit that adds a new top-level `.docs/NNN_*/`
folder to any repo that already has numbered folders — or a doc-sync run's
Step 0 surface fill doing the same (seal-remake 2026-10-24: 1 → 6 folders +
the top-level entry point, from one run's enumeration). **Or** giving a `.docs/`
to a repo with none, or a first document to an empty-`.docs/` repo — creating
the folder is itself a shape change and requires this contract. Derive the
split per run (`ls */.docs`), never from this file: past hand-typed lists here
named repos as "no docs" while their folders already shipped, which is exactly
the hiding-a-shape-change failure this contract exists to kill.

**Checklist (run in the SAME pass as the folder-adding commit):**

- [ ] **Verify ground truth.** `ls <repo>/.docs/` and count the `NNN_*`
  folders. Do not trust the skill's current number — it may already be stale.
- [ ] **Update the table row** in `## The workspace repos and their
  doc shapes` above: bump the folder count, extend the range
  (`…NN_<new-folder>/`), and add a short provenance note
  (plan/issue number + one-phrase capability description).
- [ ] **Update the Step 4 section** for that repo: change the header
  count, add the new folder's name to the capability list, and add a
  one-sentence description of what the folder covers.
- [ ] **Grep-verify zero stale counts.** After the edit, run
  `grep -nE "<old_count> (numbered|folder)" ~/.agents/skills/doc-sync/SKILL.md`
  for the repo you touched — it MUST return zero hits. (Example: after
  bumping riir-neuron-db from 9 to 10, `grep -nE "9 (numbered|folder)"`
  filtered to the neuron-db rows must be empty.)
- [ ] **Commit.** This file is NOT in a git repo (`~/.agents/` is on-disk
  only), so the update lands by saving — but the repo-side commit that adds
  the folder should reference this contract in its message (e.g.
  `docs: add .docs/10_local_kv/ (shape-change contract: doc-sync SKILL.md
  updated)`).

**Who runs this:** the agent executing the plan that adds the folder — NOT a
later `doc-sync` run. `doc-sync` is the consumer of the skill; the contract is
the producer-side obligation. A `doc-sync` run that discovers a stale count
(row says 9, disk says 10) is a SIGNAL that the producer skipped this
contract — fix the skill then, but also note the gap.

## Anti-patterns

- **Do not** write a doc entry for a plan that hasn't landed yet. Speculative docs go in `.proposals/`.
- **Do not** remove a negative-result entry when closing its issue — the negative result is load-bearing documentation.
- **Do not** upgrade a GOAT FAIL to a PASS in the docs without the benchmark file to back it.
- **Do not impose a `.docs/` shape that differs from the repo's existing convention — respect the FORMAT you find.** Numbered vs flat vs named folders, slugs vs prefixes: follow what's there. **FORMAT is not COVERAGE (owner correction, 2026-10-24, twice-paid):** respect-for-shape was once read as "the book is 1 folder, keep it 1 folder" while the code grew to 8 crates / ~550 files — the convention decides HOW docs are shaped, never WHETHER the code is covered; Step 0's surface gap overrides it, and the fill lands in the family format. The workspace shape split (numbered-folder books / no-`.docs/` AGENTS-centric / empty-`.docs/` / NAMED subfolders / flat numbered FILES) is DERIVED per run from the census trigger's one-liner — never re-typed here; every hand-typed version of that split in this file's history was wrong within days. A shape change is a deliberate, committed decision governed by the **Shape-change contract** above.
- **Do not** renumber existing docs — the numbering discipline is monotonic and never reused.
- **Do not** document trivial mechanical commits (lockfile bumps, clippy fixes) unless they close a tracked issue.


## Standing lessons (distilled from the run log — the load-bearing process rules)

- **Per-file logs + pickaxe are the reliable baseline tools.** `git log -- .docs README.md AGENTS.md` once reported `5a1330b` as game-sdk's newest doc-touching commit while `git log -- AGENTS.md` + `git log -S '<landed-string>' -- <file>` proved `1977d83` (two days newer) had touched AGENTS.md. Always re-verify a suspicious baseline per-file before declaring a range clean.
- **Narrow producer-side syncs leave holes the baseline heuristic can't see.** A narrow sync fixes its own feature and skips everything else, so the strict `baseline..HEAD` range can be empty while the book still misses older landings. Gate runs GREP the book for recently-landed headline features; don't just trust the range. The inverse gap exists too: code landing to MATCH an already-documented row (mmorpg `?transport=local`) is invisible to any baseline heuristic — nothing to fix, but don't declare a gap either.
- **Grep extraction regex must include digits.** `^[a-z_]+ =` over `[features]` silently dropped `avatar_sync_ed25519` and `game_sync_p2p` — nearly filed phantom "documented-but-dead feature" findings. Use `^[a-z0-9_]+ =`.
- **The staged-index check must GATE the commit, not precede it.** One run saw two sibling-staged files in `git diff --cached --name-only` output and then sailed past them because the check was `;`-chained ahead of `git add` + `git commit` — the sibling's WIP landed inside the run-log commit (benign outcome, luck not process). The correct form: run the check as its OWN command and read it, or use `git commit -- <paths>` (a partial commit builds a temporary index from HEAD + the named paths, leaving anyone else's staged hunks intact — the safe form on shared checkouts). `scripts/staged_set_audit.py` (katgpt-rs) reports the same signal class pre-commit. **Mechanical since the 10-26 gate unit: every doc-sync commit goes through `tools/sync_commit.sh` (refuse-nonempty-index → add named → verify staged ⊆ named → partial commit); the bare form is retired for this lane — the #42 incident proved the prose alone doesn't hold.**
- **A doc claim of determinism is a claim about a proof.** When a fix refutes the proof (the ndb Bm25 tie-truncation class), grep the book for the CLAIM, not just for coverage of the fix — the fix landed with in-source comments only and the book kept asserting the falsified invariant.
- **False positives: grep the repo's OWN `.benchmarks/` before declaring coverage.** Different repos (and even different series in one repo) reuse bench numbers — the overview's only "Bench 025" hit was a different numbering series.
- **Broken links: prose citations survive file removal, markdown links do not.** The noise-reduction rule removes record files but no link-fix pass followed, so every closed issue left `](.issues/NNN…)` links dangling. Standing tools (committed in katgpt-rs **`.agents/skills/doc-sync/tools/`** — moved out of a repo-root `tools/` that no longer exists; the 2026-09-04 landing row below still says `tools/` and is history, not a path): `python3 .agents/skills/doc-sync/tools/linkcheck_sweep.py` (census) + `python3 .agents/skills/doc-sync/tools/link_fix.py <repo>` (auto R1-repoint / R3-delink), re-sweep to verify, commit pathspec'd `.md` only. Guards baked into the tools: **absent-repo** (a link into a workspace repo not checked out on the running box is UNVERIFIABLE, not broken — `exists()` is box-relative), **backtick** (link text already code-marked is emitted unwrapped), **EOL** (the fixer reads/writes with `newline=""` since 2026-09-22 — CRLF/mixed files round-trip byte-exact; before that the read_text/write_text pair LF-normalized whole files, measured on a CRLF `.research/200`, and the old 'restore EOL to HEAD convention manually' workaround is retired), and a markdown link split across lines is invisible to one-line fix regexes (NOMATCH → manual two-line edit).
- **Never cite "Plan/Research NNN" without naming the repo** when the number exists in more than one namespace — and never link a path you haven't verified (`ls` it first; per-repo numbering namespaces collide constantly).
- **Classification against dirty trees uses `git show <rev>:` content, never the working tree.** Sync remotes first; `git show -1 <sha>` mis-parses (the `-1` overrides the sha — pass the sha alone). The reliable per-file baseline is the 4-surface set (AGENTS.md / README.md / `.docs/` / the feature file) + a since-count.
- **"0 passed" in a baseline run is the reliable gate detector** — grep on attribute FORMS lies (whole-file `#![cfg]` gates sit below doc-comment headers, mod-level `cfg(any(feature…))` compiles empty under default).
- **Baseline-diff is coverage of TEXT, never of SURFACE (owner correction, 2026-10-24).** A clean `baseline..HEAD -- .docs README.md` range says nothing about landings the book NEVER covered: seal-remake's book held 2 orientation docs (one retired) over an 8-crate workspace, and every run passed it green because nothing doc-shaped was committed. Each run ENUMERATES the code surface once (workspace members + bins from the root manifests; derive, never trust a stale count) and asks what no doc row covers; the crates map it writes is role-from-lib-doc only (facts read that session), never a restatement of AGENTS.md lanes — that duplication is the drift this gate exists to catch.

## Run log (compact — full narratives live in git history)

Each row's durable record is the named `docs:`/fix commit(s) in the repo it
touched, plus this file's own history
(`git log -p -- .agents/skills/doc-sync/SKILL.md` — rows carried full
narratives until the 2026-09-05 compaction; `git log -S '<date>' -- <this
file>` recovers any of them). **Re-compacted 2026-09-11** — the rows appended
after 09-05 had regressed to full narratives again; same recovery applies.
**Re-compacted 2026-11-10 (amo, the idle-loop lane)** — the 11-09/11-10 rows
regressed to narratives a third time (the documented cycle: rows born compact,
grown narrative, compacted); all 15 compacted to the one-line form, 62.4→50.1KB
— the row-prune lever plus the narrative-compact lever together, so the
"prune lever exhausted" notes of 11-10 are superseded by this run.
New rows append ONE compact line each (hashes + one phrase — never narratives; the 09-05 and 09-11 compactions both followed regressions to full narratives). **Maintenance rule: whenever this file exceeds 60KB, prune the run log to the newest 15 rows** (checked at any full-gate run); every removed row is recoverable via `git log -S '<date>' -- <this file>`. Prune events: 09-21 (119→15, 104→51KB — the one-line convention HELD, the ~8-rows/day cadence didn't: 67 file-touching commits in 10 days) · 09-23 (25→15, trip-preempt at 59.1KB) · 09-28 (25→15, 56.9→50.4KB, same trip-preempt) · 10-12 (17→14+new=15, 60,152→56,506 — trip at the 60KB line) · 10-26 (two lanes pruned in parallel — #31/#32/#33/#34 out across the gate unit + the ai-1127 unit; 15→13 rows, 60,851→<60KB) · 10-30 (14→10, 62.7→59.0KB, trip-preempt) · 11-12 (25→15, 62,051→57,511 bytes, trip) · 11-13 (16→byte-cap under 60KB, trip-preempt) · 11-15 (11→9+new=10, trip-preempt at the 461st unit's row) . again 11-15 (11->9+new=10, trip-preempt at the 463rd unit's row)

| Date | Scope | Verdict | Record (primary fix commits) |
| 2026-11-15 | delta unit, ZERO writes (m3, the sprint!053-landing housekeeping unit — post-overflow continuation) | windows since the 464th/465th coverage verified SELF-documenting: seal-sprints 380fbdc/e196ca5/459e571 are the tracker plans' own records (029 calibration v3 + the 057 branch-side gate-timing record — the fixing lane's close-out docs ride its merge, live; 033 step-0 measured row); seal-remake d826616b IS the sprint!055 HISTORY row (its fix commits' canonical record, no second narrative owed); riir-ai cd3784d7e (497 T1.2 trial-step) is plan-497-write-side in-commit (read-only here — riir-train routing restriction); riir-refine af5dd5f3 bench-151-is-the-doc (i237 live lease exp 11-17T03:00Z — close-out rows defer to that lane); riir-reflex/sealm-toolkit claims+merge self-documenting; seal-sprints numbering NOT re-run (no numbered files written this unit; the 463rd fresh PASS stands); mining GATED fresh (plan 375 lease re-verified live exp 2026-11-21T12:00Z); benches NOT owed (md-only unit; the box carries a sibling release build — Plan-337 zero-contention posture held); 4090 gitsync fired via Tailscale (LAN refused, IP-rotated), mid-unit poll in progress | katgpt-rs (this row, this commit) |
| 2026-11-15 | delta unit, ZERO writes (amo, the 633-tvspeech housekeeping lane zed-d937a94f) | riir-infer windows since the 445th/456th coverage verified SELF-documenting or DEFERRED: 35d9ef77 (the i1181 ICB-replay tranche-1 machinery) is a LIVE sibling lane (lease amo-idle-i1181 exp 11-16T14:00Z; tranche-1 claimer zed-a0615aba exp 11-17) — its AGENTS lane-table row is the fixing session's at close, deferred per the fixing-session rule; the 633 wave self-documenting (plan 633 + the bench 1220 record 09ddab2c + research 029's verdict commit f96f148c); i1218/1219/1220 issue/bench-is-the-doc; numbering gate PASSED fresh (exit 0); benches NOT owed (tracked tree clean; the quiet-gate tvspeech cell running — never bench a BUSY rig); mining GATED (B364 live lease exp 11-21T12:00Z); ZERO writes | katgpt-rs (this row, this commit) |
| 2026-11-15 | delta unit, ZERO writes (m3, the seal triage lane zed-f30cc31c — housekeeping re-prompt unit) | windows since the 463rd/0aae22bb0 coverage verified SELF-documenting: seal-sprints 57ff5b1 the sprint!053 tracker's own record (gate lane moved to amo + the merge ref re-cut; the rig bare-sha worktree SETUP trap recorded in-commit — a sprint!058-class sibling, not re-filed here); katgpt-rs 24c19bc0c IS the i1211 consumer-note doc commit; seal-remake tip unchanged 45a5acb5 (the tree's untracked sibling self-symlinks grew to 6 + Cargo.lock dirty — the mirror-machinery signature as-found, the owning build lane's teardown); mining GATED fresh (riir-refine plan 375 lease verified live exp 2026-11-21T12:00Z); benches NOT owed (md-only unit, zero .rs touched) | katgpt-rs (this row, this commit) |
| 2026-11-15 | delta unit, ZERO writes (m3, the sprint!055-close housekeeping unit) | the seal-sprints window (93c2d12..2d1b726) verified SELF-documenting: c6e0209 IS the close record (HISTORY row + TASKS line + the issue-file removal per the noise-reduction rule), the 057/058 filings + bf56acc citations are issue-is-the-doc, 2d1b726 the sprint!053 lane's own tracker (close-out rides the in-flight bmo merge); seal-remake unchanged from the 462nd coverage (45a5acb5 chore md); numbering gate PASSED fresh on seal-sprints; benches NOT owed (md-only unit; seal-remake dirty bot_fight.rs+Cargo.lock = the live 177 lane WIP, as-found); mining GATED fresh (plan 375 exp 11-21T12:00Z); i218 stands discharged upstream (459th ffaaeb9e5, no re-poll owed); run log pruned 11->9+new=10 (60KB trip-preempt at this row); katgpt-rs ff-synced behind-5 to c1e515563 in passing (WIP overlap verified empty pre-pull) | katgpt-rs (this row, this commit) |
| 2026-11-15 | delta unit, ZERO writes (m3, the seal triage lane zed-f30cc31c — the bmo-job-pickup overflow continuation) | the ONE new window is sprint!055's CLOSE (seal-sprints c6e0209, landed post-462nd) — SELF-documenting: the HISTORY close record names every fix commit (seal-std a4e8aad · riir-ai 5ab57aa50 · seal-remake d1b152a1+cf259af0) + the full gate ledger (01a1dba6/gate-1115-0302 GATE PASS all seven arms), so NO seal-remake local HISTORY row is owed (the fix commits' canonical record home is the issue's own repo — a second narrative would be the duplication drift); the 177 pad window (047f64b6/45a5acb5) self-documenting in its retained .issues/177 file (pad posture + the party-sleep fsm find + DUEL_PAD env in-commit; the AGENTS harness row's env-agnostic verdict respected per the chaos-lane precedent); seal-sprints 2d1b726's residual row names riir-ai issue 1225 (the 053 lane's own routing record); seal-remake's untracked katgpt-rs/riir-ai entries are a live build lane's sibling SYMLINKS (01:59 today, path-dep joins) — not repo pollution, left untouched; mining GATED unchanged (plan 375 lease exp 11-21T12:00Z); i218 stays discharged | katgpt-rs (this row, this commit) |
| 2026-11-15 | delta unit, ZERO writes (m3, cycle 2 — the immediate next activation) | the ONE new window is the sprint!053 lane's own tracker record (seal-sprints 93c2d12: .issues/053 verdict rows + TASKS.md — plan-is-the-doc, the lane live mid-gate at bmo 01a1db9c-f910, its close-out docs ride the merge); every other origin tip re-fetched and verified unchanged from the 461st's coverage; mining GATED unchanged (plan 375 lease exp 11-21T12:00Z); i218 discharged upstream ffaaeb9e5 | katgpt-rs (this row, this commit) |
| 2026-11-15 | delta unit, ZERO writes (m3, the post-overflow housekeeping continuation lane) | windows since the 460th verified SELF-documenting: seal-remake 1d5e71ca's record is its own .issues/177 file (md+shell, the DUEL_PAD env documented in-commit; the AGENTS harness row stays env-agnostic per the chaos-lane precedent); seal-sprints 87edd3c/4c86637 tracker rows are their own records (sprint!055 close-out docs stay the fix sibling zed-a8f6f7d7's, live lease); riir-ai's wave docs landed upstream (dcb458f58 IS the HISTORY + riir-net AGENTS row per the 11-15 sibling unit; 5ab57aa50 is covered by sprint!055's seal-sprints record); seal-std leaf commits self-documenting (a4e8aad seal-item origin-shard, in-code docs only, no book); sge 9964efc2e its own record (456th covered); sprint!044 branch-side unmerged — its docs ride the merge; run log pruned 11→9+new=10 (60KB trip-preempt at this row); mining GATED (plan 375 B364 live lease zed-m3-inferenceforge-distill@m3 exp 11-21T12:00Z); i218 poll DISCHARGED today upstream ffaaeb9e5 | katgpt-rs (this row, this commit) |
| 2026-11-15 | delta unit + TWO book rows (m3, the sprint!055 doc-sync lane zed-1c70d8c6 — the yielded-fix lane's housekeeping unit) | riir-ai 60846be55 (the relay refusal trace, sprint!055 instrumentation) carried NO book row — HISTORY dated entry + the riir-net AGENTS row landed dcb458f58 (in a detached worktree at the origin tip: local develop was stale-diverged against the force-pushed 60846be55 with the fix sibling's WIP in-tree — the worktree route avoided both); sdk e109216 (the IXAB sender-gate row, seal-remake issue 200's measured root) carried no HISTORY row — landed 483eb89 through sync_commit.sh (partial form over 4 foreign-dirty files); seal-remake windows verified self-documenting (issue 200 retained per the pre-11-09 convention = its own record; the 089024af TRACE-forward has no AGENTS surface — the chaos lane's env plumbing is not AGENTS-documented); seal-sprints skipped (the fix sibling's dirty .issues/055); run log pruned 15→14+new=15 (61.4KB trip-preempt); sprint!055's fix rows deferred to the fix lane's own close (live claim zed-a8f6f7d7) | riir-ai `dcb458f58` + riir-game-sdk `483eb89` + katgpt-rs (this row, this commit) |
| 2026-11-14 | delta unit + Batch 363 LANDED (amo, the idle-lane zed-amo-b363-hotpath — the same chain the sibling 375-row saw as CLAIMED c1344e5d) | plan 374 executed END-TO-END this unit: claim → T0 live-page figure re-verification (all faithful to Research 407) → T1 dedup neighbors pinned → T2 fixtures 8/8 standalone pre-entry (3 model defects caught) → T3 entries_79 4 rules → T4 two-run stable preflight → T5 gates (clippy -D ×3, floors 29/0 HELD, lib 3909/0, needle gate 1/1) → T6 pins_manifest --write + the count web (AGENTS/README/.docs heads + 10 pin sites re-pinned, manifest+prose AGREE) → T7 research 407 → batch-363-landed + snapshot row → T8 landing a81a5914 pushed (rebased over sibling plan-378 docs ×2 + their 5b427e4e ane rows — disjoint write-sets, the count web verified fresh post-rebase); book coverage is IN-COMMIT per the batch convention (plan-is-the-doc + the snapshot row; the README/AGENTS/.docs rust_perf prose heads carry the 342 count themselves — no separate doc-sync edits owed); i218 5TH consecutive poll UNCHANGED (PR #3025 still open at 7f297fb, updated_at 11-12T04:27Z, mergeable clean — record stands, outcome poller 3–5 open); benches NOT owed (corpus-only delta, zero clippy_lints surface); boundary 457th row same unit (renumbered on the sibling-456th discovery); lease folded into **Lease closed:** per discipline | riir-refine `7509d651`+`a81a5914` + katgpt-rs (this row, this commit) |
| 2026-11-14 | delta unit + ONE gap CLOSED (amo, the idle housekeeping lane amo-idle-hk-1114b) | the riir-refine B362/377 wave verified: B362 SELF-documenting (counts web in-commit); the Plan-377 ane landing 486978a2 carried AGENTS + HISTORY in-commit but LEFT TWO BOOK GAPS — no README domain-matrix row, no .docs/11_domains/ane.md — BOTH CLOSED this unit (5b427e4e via sync_commit.sh, rebased over the sibling's 8c8b5240 that landed mid-unit; the 11_domains header's stale "Three domains" prose repaired to the derived 8-domain set + the feature-gate table ane/ane_pregate rows); sge sprint#032/#043/#010 + the desk fix self-documenting; i218 4th poll UNCHANGED; i224 lease live (exp 11-15T20:56Z) — yield; mining GATED (B363 CLAIMED c1344e5d, m3 lease exp 11-16T16:12Z); benches NOT owed; boundary 456th row same unit | riir-refine `5b427e4e` + katgpt-rs (this row, this commit) |
| 2026-11-14 | delta unit, ZERO writes (amo, the riir-refine re-fire lane zed-c06ba6e0) | both windows SELF-documenting (riir-refine b6f65045 = issue 237's own anchor line; riir-ai 0f76191fa IS the supergoat candidate doc) — no book gaps; i218 re-polled THIRD consecutive unchanged: PR #3025 OPEN mergeable-clean at head 7f297fb, updated_at still 2026-11-12T04:27:11Z, ZERO comments since 11-12T05:00Z — record stands; mining GATED fresh (B362 mid-flight under the live m3 lease exp 11-16T07:58Z, EXPECTED-RED pins_manifest mid-flight); benches NOT owed; boundary 455th row same unit; inbox EMPTY; ZERO commits | katgpt-rs (this row, this commit) |
| 2026-11-14 | delta unit, ZERO writes (amo, the i1208-closure housekeeping lane zed-1df3fef0) | riir-refine windows since the 453rd verified SELF-documenting (B361 landing carries its AGENTS.md 337/361 row + plan 372 in-commit; the B362 authoring is plan-is-the-doc with the snapshot row riding the pending T8 landing per the batch convention; i234 steer_retry AGENTS.md row in-commit; i237 + the ThreeRT distill their own records) — no book gaps; i218 re-polled: the 11-13 zed-f0fa869b@m3 CI-watch row CONFIRMED UNCHANGED (PR #3025 still OPEN unmerged at 7f297fb, PR updated_at still 11-12T04:27Z, no maintainer event since the 11-12 replies — record stands, NO edit owed; outcome poller steps 3–5 stay open); i224 foreign lease LIVE (zed-amo-b352-lone exp 11-15T20:56Z) — yield; mining GATED this activation: Batch 362 authored+claimed by the live m3 lease (exp 11-16T07:58Z) with EXPECTED-RED pins_manifest mid-flight — the same corpus-surface collision class the 453rd/i1210 rows named, box contention secondary (a sibling exclusive job on amo); benches NOT owed (riir-refine tree clean post-ff, no perf claim this unit); boundary 454th row same unit (scoped exit 0, 20 edges); riir-refine exits this unit with ZERO commits | katgpt-rs (this row, this commit) |
| 2026-11-14 | delta unit + Batch 363 LANDED (amo, the idle-lane zed-amo-b363-hotpath — the same chain the sibling 375-row saw as CLAIMED c1344e5d) |
## TL;DR

Diff git history against the last `docs:` commit. For each landed plan/issue,
write the matching README/docs entry using the repo's existing shape and the
verdict from its benchmark file. Commit with `docs:` prefix on the working
branch. Never upgrade a verdict without proof; never delete a negative result.
