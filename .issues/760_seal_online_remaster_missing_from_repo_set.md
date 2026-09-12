# Issue 760 — `seal-online-remaster` is a new contract repo missing from `scripts/repo_set.txt` (workspace is 20, not 19)

**Status:** OPEN — katgpt-rs-side drift FIXED 2026-09-12 (repo_set.txt + AGENTS.md §Repo
count + research/substrate-first SKILL.md prose, one commit); the M3-side task remains
(clone the repo, rerun the gate — expect 17/17).

## The finding (2026-09-12, docs_gate run on the 4090 box)

A workstation docs-gate run on the 4090 (`E:\git`, 14 live contract repos) surfaced 3 red
checks. One was REAL drift, two were the same drift amplified by box topology:

- `skill_repo_set_gate.py`: "repo_set.txt is stale vs the live workspace — **missing
  ['seal-online-remaster']** gone ['katgpt-web', 'riir-dao', 'riir-deployer',
  'riir-esp32', 'riir-shader', 'riir-viewbridge']"
- `population_sync_gate.py`: same only-in-file/missing-from-file disagreement.
- `issue_citation_gate.py`: "derived 14 contract repos < floor 15" (topology-blind on a
  partial clone — 6 canonical repos are not checked out on this box).

Root cause: **`seal-online-remaster/` is a NEW contract repo** (own root `BOUNDARY.md`
titled "seal-online-remaster — boundary contract" + real `.git` dir at
`E:\git\seal-online-remaster\`) that never made it into the canonical
`scripts/repo_set.txt` (derived on the M3). It is NOT a rename — `seal-remake/` still
exists alongside it. Corroborating evidence: riir-ai's AGENTS.md boundary table already
references `seal-online-remaster` ("Product ≠ engine", grouped with
riir-mmorpg-examples).

En-route prose drift also fixed (all stale on the same event): AGENTS.md §Repo count said
"19 repos"; research SKILL.md said "18 contract repos" twice (its "rest of the workspace"
enumeration was missing BOTH `riir-shader` AND `seal-online-remaster`); substrate-first
SKILL.md said "18 contract repos" twice (one derived "11 of the 18" → "13 of the 20").

## Fixed (this commit)

- [x] `scripts/repo_set.txt` +`seal-online-remaster` (evidence-backed typed line — NOT a
      regeneration from this box; a regeneration here would DROP the 6 repos absent from
      this box. Authoritative regeneration stays an M3-side act.)
- [x] `AGENTS.md` §Repo count 19 → 20 + name added to the add-list (agents_repo_set_gate
      PASSED at 20 post-edit: "membership + both declared counts agree").
- [x] research SKILL.md: "20 contract repos" (L19), "the other 13" header + restored
      bullet structure, `seal-online-remaster` joined the product-consumers bullet,
      `riir-shader` given a minimal honest entry (no invented routing), constraint #5
      enumeration + count fixed.
- [x] substrate-first SKILL.md: both "18" → "20" (the derived "11 of the 18" → "13 of
      the 20", arithmetic 20−7=13 holds).

## Remaining (M3-side)

- [ ] Clone `seal-online-remaster` onto the M3 workspace (`/Users/katopz/git` or wherever
      the full set lives), pull this commit, rerun `./scripts/docs_gate.sh` — expected
      17/17 green (the "gone 6" topology failures on the 4090 box are partial-clone
      artifacts; with the full 20-repo walk the floors clear).
- [ ] If the M3 walk finds MORE unlisted contract repos (the 4090 walk can only see its
      own 14), regenerate `repo_set.txt` there per the file's docstring and re-check the
      AGENTS.md count — this box cannot validate that axis.

## Notes

- The 4090 box's docs-gate baseline after this fix: 14/17 green + 3 red, ALL of the red
  being the documented topology class (6 canonical repos not checked out here; citation
  gate floor 15 vs 14 live). Verify with:
  `PYTHONUTF8=1 PATH=/tmp/py3bin:$PATH bash scripts/docs_gate.sh` (python3 ≥ 3.11 via the
  uv-managed cpython-3.12.11; `PYTHONUTF8=1` is REQUIRED on this box — the console
  codepage is cp874 and Python crashes printing ✓/✗ marks without it).

## Related

- `scripts/repo_set.txt` docstring (the regeneration command + workstation rule)
- katgpt-rs `AGENTS.md` §Repo count (the canonical count paragraph this issue pins)
- riir-ai `AGENTS.md` §Domain Boundary table (early adopter of the new repo name)
