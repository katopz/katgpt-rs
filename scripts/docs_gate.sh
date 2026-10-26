#!/usr/bin/env bash
# Docs gate — the manifest/doc/skill drift assertions, as one command.
# The count is deliberately NOT written here: CHECKS below is the list, and a
# prose count beside a list is the drift this repo keeps rediscovering.
#
# Why this file exists: all three checks already existed, and NOTHING ran any of
# them. Measured 2026-09-01, two of the three were RED on `develop`:
#
#   count_features.py       green, but only because it had just been fixed; it
#                           checked ONE README site out of five and two of 29
#                           manifests.
#   bench_doc_audit.py      exit 1 — a false positive on a doc that correctly
#                           recorded "opt-in ... promoted to DEFAULT-ON".
#   cargo_comment_audit.py  exit 1 — a false positive from a case-SENSITIVE
#                           "Opt-in" regex that missed the repo's 32 "OPT-IN"
#                           comment lines.
#
# An assertion nobody invokes is decoration, and a red one nobody invokes is
# worse: it trains the next reader to assume the tool is broken.
#
# Cost: ~3s total. That is what makes per-push affordable here, in deliberate
# contrast to scripts/full_gate.sh (>13 min, weekly). It was ~556s before the
# manifest walk was pruned — `rglob("Cargo.toml")` descended into target/
# (117 GB, ~1.3M entries) and filtered afterwards, four times per run.
#
# Unlike the full gate this is platform-INDEPENDENT: pure Python over manifests
# and markdown, no cfg(target_os) surface, so ubuntu is correct and macOS would
# only cost more. Don't "fix" it to macos-latest.
#
# WORKSTATION REQUIREMENTS (measured on the partial 4090 box 2026-09-04, where
# 5 of 8 checks red for environment, not drift): (1) python3 >= 3.11 on PATH —
# cfg_gated_target_audit.py imports tomllib (3.10 lacks it; the Windows Store
# python3 alias also shadows real installs); (2) PYTHONIOENCODING=utf-8 — a
# cp874/cp1252 console cannot print the gates' checkmark output and the failure
# masquerades as a gate failure; (3) a FULL workspace checkout —
# skill_repo_set_gate.py re-derives the live repo set and FAILS on repos the
# box simply has not cloned (10 of 16 here, incl. riir-mmorpg-examples/riir-dao)
# — regenerating repo_set.txt on a partial box would corrupt the canonical set;
# the M3 is the canonical full workstation for that half. Issue 765 (2026-09-13)
# added the explicit marker for known-partial boxes: DOCS_GATE_PARTIAL_CLONE=1
#
# UPDATE 2026-09-13 (4090 session): (1) and (2) are now SELF-SUPPLIED — the
# gate resolves its interpreter by EXECUTION probing python3/python/py and
# prefers the HIGHEST version (a working `python` 3.10 was shadowing `py`
# 3.14; the stub python3 is skipped), warns loudly below 3.11, and exports
# PYTHONIOENCODING=utf-8 unless the caller set their own. skill_repo_set_gate
# reads UTF-8 with an explicit encoding (a cp874-default read_text() crashed
# the scan). First 17/17 green on the 4090 at that commit; (3) remains the
# one M3-canonical requirement (the deferral rides each population check's
# final line under the marker).
# makes the three population checks (skill_repo_set_gate,
# population_sync_gate, issue_citation_gate) print a loud deferral on the
# population axis instead of a remedy that invites the corruption. Marker-gated,
# never auto-detected — a genuinely-missing sibling still refuses.
#
# skill_repo_set_gate.py (added 2026-09-01, Issue 703) has a second axis the
# other three do not: it reads SIBLING repos, which CI does not have. It does
# NOT skip there — it separates its VOCABULARY (committed snapshot,
# scripts/repo_set.txt) from its POPULATION (the SKILL.md it can actually see),
# prints both, and the workstation run re-derives the snapshot and FAILS on
# drift. So CI checks this repo's 8 skills against all 18 repo names, and says
# out loud that it saw 8 of 12. A gate that skipped instead would be the
# vacuous green it exists to catch.
#
# cfg_gated_floor_gate.py (added 2026-09-03, Issue 713) is katgpt-rs-SCOPED on
# purpose, unlike the sibling-reading check above it. Its instrument
# (cfg_gated_target_audit.py) audits any repo, but CI has a single checkout, so
# a cross-repo version would derive an empty population and print a confident
# green over zero repos — the same defect it exists to catch, which is also why
# docs_drift_sweep.py is deliberately absent from CHECKS. Sibling coverage is
# Issue 713 T3, an owner call per repo. Its pins are two-sided (two ceilings +
# two blindness floors) because a ceiling cannot fail once the auditor goes
# blind and reports zero; see scripts/cfg_gated_floors.txt.
#
# orphaned_attr_gate.py (added 2026-09-03) is pinned at ZERO offenders: the
# shape it forbids -- an OUTER #[cfg] separated from its item by a blank line,
# which Rust still binds to that item -- measured zero sites across every
# contract repo at the fix (19 then; re-measured 2026-09-04 over the live 16,
# still zero everywhere). It exists because that shape sat in katgpt-pruners for
# two days and broke every RELEASE build of `sdar_gate` (26d055c6 -> a08376a0),
# while the commit that introduced it validated in debug and reported 597/0.
#
# It DOES have floors, added 2026-09-04, and the paragraph above is why: a zero
# ceiling cannot fail once the walk goes blind. The reasoning was written here
# for docs_drift_sweep.py and not applied to the check three lines below it.
# Two population floors (.rs files, outer-#[cfg] sites), katgpt-rs-scoped --
# riir-viewbridge has 24 .rs files against this repo's 2,418, so a shared floor
# would red every small sibling forever. The same commit stopped its PASS line
# printing "measured 0 across 19 repos", a cross-repo claim no single-repo run
# had made, two lines under the repo-set gate correctly saying 16.
#
# Runs every check even after one fails — the same reason full_gate.sh passes
# --keep-going: stopping at the first failure under-reports the drift.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CHECKS=(
    "scripts/count_features.py:flag counts in README + examples/README vs every manifest"
    "scripts/bench_doc_audit.py:(default-on|opt-in) labels in .benchmarks + .docs vs Cargo defaults — plus two blindness floors and BlindRead, exit 2: an OSError on a file the walk just listed means the tree is unreadable, and a PARTIAL manifest read fabricates mismatches ABOVE any floor (Issue 790 F9)"
    "scripts/cargo_comment_audit.py:inline Cargo.toml comments vs the default closure"
    "scripts/skill_repo_set_gate.py:hand-typed repo sets in SKILL.md command blocks (Issue 703)"
    "scripts/agents_repo_set_gate.py:AGENTS.md §Repo count membership vs scripts/repo_set.txt"
    "scripts/cfg_gated_floor_gate.py:#![cfg]-gated targets that report a green 0-pass (Issue 713)"
    "scripts/orphaned_attr_gate.py:a #[cfg] separated from its item by a blank line (a08376a0)"
    "scripts/percentile_floor_gate.py:a percentile index that lands on n-1 and so reports the MAX"
    "scripts/numbering_gate.py:a .plans/.issues/.research/.proposals number allocated twice, or a stale/malformed .highwater (Issues 724, 725) — including the majority case where BOTH holders have CLOSED and been removed, so nothing is on disk and the tracked check reads clean: walled by membership above the era boundary, ratcheted below it (Issue 795)"
    "scripts/dual_allocation_gate.py:this checkout and its upstream both allocated a numbered document since their merge base — TWIN (same stem, the rebased own line) annotates exit-neutral, INDEPENDENT (two documents claiming one number) exits 1 naming both sides' adding commits (Issue 796) COUNTER exits non-zero too: the first two verdicts compare DOCUMENTS, and a number allocated and CLOSED in one commit never has a file in any tree, so the counters are the only thing that sees it. A one-sided bump stays green by construction; armed on every push with the reader injected and by a real two-repo fixture under --prove-fires (Issue 850)"
    "scripts/docs_gate_paths_sync.py:docs_gate.yml's two hand-duplicated trigger paths lists stay identical (Issue 724 T4b)"
    "scripts/required_features_static_gate.py:a required-features row naming a feature its package cannot enable (Issue 513)"
    "scripts/cfg_row_implication_gate.py:a required-features row that BUILDS and compiles its target to NOTHING (Issue 513)"
    "scripts/population_sync_gate.py:the ten independent contract-repo predicates must agree, and the registry that lists them must be COMPLETE (else an instrument audits a different set and still prints green)"
    "scripts/trap_sentinel_gate.py:a shell gate whose set -u abort would report exit 0 — this repo's own two, by membership (Issue 734)"
    "scripts/issue_citation_gate.py:a cross-repo Issue/Plan/Bench citation naming no repo — it rebinds to the WRONG doc once the number is allocated locally (Issue 749)"
    "scripts/markdown_fence_gate.py:a fenced code block never closed — everything after it renders as code, and a fence scanner mis-phases on it (Issue 756)"
    "scripts/platform_dead_code_floor_gate.py:an item declared ungated whose every use sits behind a platform cfg — dead code on a platform no automatic lane compiles (Issue 775)"
    "scripts/subprocess_encoding_gate.py:a subprocess call that decodes with the SYSTEM locale — silent mojibake, or stdout=None with the returncode intact (Issue 778)"
    "scripts/instrument_reachability_gate.py:a tracked scripts/*.py no root and no documented instrument names — invisible to the census that would find it (Issue 787)"
    "scripts/sweep_advisory_membership_gate.py:a *_drift_sweep.py that does not call a FAMILY-WIDE MECHANISM — the Issue-797 worktree advisory (findings and floors then describe whatever the working tree happened to say) or the Issue-815/821 known-extra exemption (the sweep hard-reds on a repo the contract does not claim, with zero content findings); a REGISTRY, per mechanism and never pooled, because this gate governed one mechanism by name and watched Issue 821 miss 3 of 19 beside it (Issues 797 T5, 824)"
    "scripts/locale_io_gate.py:text I/O that decodes/encodes with the SYSTEM locale - Path.read_text/write_text/open() in text mode with no encoding=; the FILE seam under subprocess_encoding_gate's PIPE seam, found when a cp874 box silently mangled a selftest FIXTURE and made its arms pass for the wrong reason (Issue 829)"
    "scripts/console_encoding_gate.py:a tracked scripts/*.py that prints a non-ASCII glyph and defends neither stream — on a non-UTF-8 console it dies with NO verdict and its findings go unread; docs_gate.sh's PYTHONIOENCODING only covers runs that go through the wrapper (Issue 804)"
    "scripts/global_rng_gate.py:a free-function global-fastrand draw with no pin row — the unseeded thread-local global made a shipped pruner non-deterministic, found by executing one commit twice; membership + per-row reason, both directions, floors on the walk and the predicate (Issue 809)"
    "scripts/algebraic_op_ban_gate.py:any algebraic_div/algebraic_rem CODE occurrence in tracked *.rs — the Issue 871 T4 div/rem ban as a tracked check, not a doc line (add/mul reassociation adopted feature-gated in katgpt-attn-match/algebraic_dot; div/rem banned outright); ceiling 0, no exemption vocabulary, walk floor + planted-source predicate arms (Issue 871)"
    "scripts/shared_temp_path_gate.py:a test writing to a FIXED env::temp_dir() path — safe against sibling tests in one binary, and truncated by any concurrent PROCESS running the same test; the x86_64 matrix filed one as TRANSIENT because this class passes alone BY CONSTRUCTION (Issue 832)"
    "scripts/cross_repo_path_dep_gate.py:a \`path = \"../X\"\` dependency on a repo that is NEITHER on disk NOR in repo_set.txt — cargo resolves path deps even when \`optional = true\`, so the CITING repo stops building entirely, and the population buckets (present-unregistered, absent-registered) leave that state enumerated by nothing (Issue 835)"
    "scripts/repo_registration_gate.py:a per-repo pin file with no row for a registered, on-disk repo — registering a repo is a 22-file operation and only repo_set.txt is gated, so the other 21 are discovered one red sweep at a time; measured the day riir-llm joined, 19 of 21 sweeps red and four live findings sitting behind them (Issue 837)"
    "scripts/cross_module_attr_gate.py:a tracked scripts/*.py naming an attribute a sibling module does not define — Python has no link step, so the reference resolves at CALL time and a rename is an AttributeError the next EXECUTION finds; when the importer is a gate the failure is NO VERDICT rather than a wrong one. Measured: a commit privatised is_checkout and deleted worktree_fixture, updating the seven in-module callers and neither of the two external ones — both of them CHECKS in this array, develop red for 6h40m. --prove-fires is two-sided, 0 findings at the parent and exactly those 4 at the commit; the runtime half of the coverage stays arm_reach_gate's 157.6s workstation verdict (Issue 848)"
    "scripts/import_health_gate.py:a tracked scripts/*.py that does not IMPORT — the EXECUTION half of Issue 848, where the static half (cross_module_attr_gate) cannot reach: a circular import, a missing third-party dependency, a raise in top-level code. The instrument that already executes every module is arm_reach_gate's BASELINE-CRASH, kept out of this budget at 157.6s; this is its cheap half, one child, 0.11s of import. A per-module subprocess was measured too and is not worth 3x (10.28s vs 7.52s, identical verdicts). ⛔ The affordability measurement found its own blocker — 6.238s of 6.34s was ONE module whose whole body was top-level, guarded in the same change. MISSING-DEP is its own bucket and is never flagged: pinning it would red the gate on a box that HAS the package (Issue 848 T3)"
    "scripts/shipped_target_feature_gate.py:a SHIPPED path selecting its fast arm on a COMPILE-time target_feature — that predicate is OFF by default on x86_64, so the arm compiles to NOTHING on every ordinary build and the dispatcher silently runs its fallback. AGENTS.md documented this shape only for GATES, where the cost is an unproven claim; on a shipped path the cost is latency on every call, measured at 4.4-5.6x in a DEFAULT-ON feature and 2.4-2.5x on bf16 RNE narrowing. Three exclusions a naive grep gets wrong — wasm32/simd128 (no runtime detection there, so a compile-time gate is the only option), NEON (implied by the arch), and the runtime probe's own body, which is the same attribute doing the opposite job. The key is line-free and resolved by brace counting, the two cases needing opposite lookups — an attribute inside a body belongs to its enclosing fn, one ON an item to the NEXT. Membership + a reason per row, both directions, floors on the walk and on the attribute parse; --prove-fires is two-sided (Issue 847 T3)"
    "scripts/timed_region_guard_gate.py:a latency ceiling with no loud-zero defence — rustc + fat LTO deletes a timed loop whose result is dead, so the bar is satisfied by **absent work** and passes with MAXIMUM margin. This is the \`#![cfg]\` green-zero rule one layer down, and worse: the assertion RUNS, so the output is a plausible number rather than a zero count and no count floor can see it. ⛔ Not reasoned about — **executed**: all 34 asserting regions at n ≥ 1000 were run and **7 were satisfied by absent work** (20.6%), two of them GOAT gates, one printing \`Speedup: 8657.9×\`, one printing a well-formed \`0.00x\` because only the NUMERATOR vanished. ⚠ The static predicate Issue 855 T4 proposed is REFUTED by that run — \`let _ =\` vanished **3 of 15** against **4 of 19** for the rest, i.e. the base rate wearing a grep; the column that separates is \`black_box\`, **7 of 23** without it against **0 of 11** with. So the gate does not predict which region is broken; it gates the decidable thing — *is there a loud-zero defence at all*. ⛔ Two tiers, and the split is the honest part: a LITERAL loop bound is the population that was READ end to end (membership wall, one MEASURED number per row), while a bound needing one hop of resolution is real and **unread** (a ratchet on the derivative — pinning an unread bucket by name is Issue 785's forbidden shape). One hop is not optional: T1's own two founding specimens are \`let n = 100_000; for _ in 0..n\`, so a literal-only predicate would have shipped the class it was written for. ⚠ It does NOT claim a pinned region is safe — \`black_box\` is the weakest of three defences (result, **arguments**, **receiver**) and two arms vanished carrying one (Issue 855 T4)"
    "scripts/python_syntax_floor_gate.py:tracked *.py parses at the 3.9 grammar — a 3.10+ only file is a dead tool on the workstation's python3 3.9.6, dying at import before any verdict; two ceilings (SYNTAX + UNPARSED) over walk + parse floors (Issue 965)"
    "scripts/check_validation_gate.py:a CHECK in this array whose own arithmetic no arm asserts — including one whose arm is flag-gated and so never runs (Issue 789)"
    "scripts/docs_gate_checks_sync.py:this CHECKS array vs the AGENTS.md table documenting it — membership both ways + quantity words (Issue 750)"
    "scripts/skill_size_gate.py:a skill SKILL.md over the 80KB ceiling — the THIRD 100KB-regrowth class (doc-sync 09-05/09-11/09-21, boundary-guard 09-08/09-11/09-21: the one-line convention held, the ~8 rows/day cadence didn't) — the gate forces each file's documented prune-to-15 maintenance rule; recovery via git log -p"
)

# Resolve the interpreter by EXECUTION, never by `command -v` alone (the
# Issue-770 T4-4090 class, fixed in riir-ai's perf_rematch 2026-08-31 and
# here 2026-09-13): Windows ships App Execution Alias STUBS at
# AppData/.../WindowsApps/python3(.exe) that `command -v python3` FINDS —
# the stub prints a Microsoft Store ad instead of running code, so a
# presence-only probe green-lights an interpreter that cannot execute a
# single check (measured: all 17 checks red on the 4090 box while a real
# Python310 sat one name away as `python`). Probe function, not presence.
resolve_py() {
    # Probe by execution AND pick the HIGHEST version: `python` (3.10) can
    # shadow `py` (3.14) on a Windows box that carries both — and this gate's
    # checks need tomllib (3.11+), so a working-but-old interpreter silently
    # reds 7 checks as ImportError (measured on the 4090: py -0 lists 3.14 +
    # 3.10; the first-working probe order resolved 3.10).
    # A uv-managed interpreter joins the candidates (the workspace's Python
    # toolchain is uv): measured 2026-10-23 on amo — PATH carried only the
    # system 3.9 while uv held 3.12, and 21 of 35 checks red on import/kwarg
    # errors that a 3.12 run cleared to 4. An array, not a word list: a uv
    # path can contain spaces on Windows.
    local c best_c="" best_v="" v=""
    local -a cands=(python3 python py)
    if command -v uv >/dev/null 2>&1; then
        c="$(uv python find '>=3.11' 2>/dev/null | tr -d '\r')" || c=""
        [ -n "$c" ] && cands+=("$c")
    fi
    for c in "${cands[@]}"; do
        command -v "$c" >/dev/null 2>&1 || continue
        [ "$("$c" -c 'print("pyok")' 2>/dev/null | tr -d '\r')" = "pyok" ] || continue
        v=$("$c" -c 'import sys; print("%d" % (sys.version_info[0]*100 + sys.version_info[1]))' 2>/dev/null | tr -d '\r')
        [ -n "$v" ] || continue
        if [ -z "$best_v" ] || [ "$v" -gt "$best_v" ]; then
            best_c="$c"; best_v="$v"
        fi
    done
    [ -n "$best_c" ] || return 1
    printf '%s\n' "$best_c"
}
PY="$(resolve_py)" || { echo "✗ no WORKING python interpreter on PATH (python3/python/py probed by execution; Windows Store stubs are skipped) — docs gate cannot run"; exit 1; }
# The checks import tomllib (3.11+). A lower interpreter is allowed (the
# non-tomllib checks still run) but the limitation is named loudly, never
# discovered as seven ImportErrors.
PY_V="$("$PY" -c 'import sys; print(sys.version_info[0]*100 + sys.version_info[1])' 2>/dev/null | tr -d '\r')"
if [ -n "$PY_V" ] && [ "$PY_V" -lt 311 ]; then
    echo "⚠ resolved interpreter '$PY' is python $((PY_V / 100)).$((PY_V % 100)) < 3.11 — every check importing tomllib (3.11+) or using 3.10+ syntax/kwargs fails on import or in its self-test (measured: 17 of 35 on 3.9); install 3.11+ (\`uv python install 3.12\` is probed automatically) or expose it via the py launcher"
fi

# Self-supply documented workstation prerequisite (2), header above: a
# cp874/cp1252 Windows console cannot encode the gates' checkmark output and
# the UnicodeEncodeError masquerades as a gate failure. utf-8 unless the
# caller deliberately set their own.
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

# ── This gate times ITSELF ──────────────────────────────────────────────────
# The duration used to be hand-typed in AGENTS.md, and a hand-typed duration
# drifts exactly like a hand-typed count. It was also the wrong quantity: this
# gate's WALL time is contention-dominated (measured: 12.65s / 12.52s / 12.69s
# CPU on runs whose WALL was 128.3s / 299.1s / 15.0s — a 20x wall spread over
# 1.4% of CPU spread), and which
# check absorbs the wait moves between runs. So print BOTH — the per-check
# wall time names whichever check is blocking today, and the CPU total is the
# load-invariant figure to compare across runs. Numbers and the measured
# non-explanation live in AGENTS.md §Docs gate, not duplicated here.
# `$EPOCHREALTIME` is bash >= 5.0 and this box is 3.2.57, so the stamp goes
# through the resolved interpreter — already a hard dependency above.
now() { "$PY" -c 'import time; print("%.2f" % time.time())'; }
GATE_T0="$(now)"

failed=0
for entry in "${CHECKS[@]}"; do
    script="${entry%%:*}"
    what="${entry#*:}"
    if [ ! -f "$script" ]; then
        # A missing check is a failure, not a skip: silently dropping a check is
        # how this gate would rot back into the state that motivated it.
        echo "✗ $script — MISSING (expected: $what)"
        failed=$((failed + 1))
        continue
    fi
    echo "▸ $script — $what"
    check_t0="$(now)"
    if out="$("$PY" "$script" 2>&1)"; then
        printf '%s\n' "$out" | tail -1 | sed 's/^/    /'
    else
        failed=$((failed + 1))
        printf '%s\n' "$out" | sed 's/^/    /'
        echo "  ✗ $script FAILED"
    fi
    check_dt="$("$PY" -c "print('%.1f' % ($(now) - $check_t0))")"
    case "$check_dt" in
        # Only the slow ones are worth a line; the rest are noise at 0.0-0.9s.
        0.*) ;;
        *) echo "    ⏱  ${check_dt}s wall" ;;
    esac
done

# CPU is the load-invariant total (`times` reports this shell + its children);
# wall is what the operator experiences. A large gap means the box was busy —
# compare CPU across runs before concluding a check got slower.
gate_wall="$("$PY" -c "print('%.1f' % ($(now) - $GATE_T0))")"

# `times` must be REDIRECTED, never captured. Measured on bash 3.2.57 against
# a child that burned 0.167s of user time:
#     times                 -> 0m0.001s 0m0.002s / 0m0.167s 0m0.015s   correct
#     times > file          -> 0m0.001s 0m0.002s / 0m0.168s 0m0.024s   correct
#     times | sed 's/^/ /'  -> 0m0.000s 0m0.000s / 0m0.000s 0m0.000s   ZERO
#     $(times | tail -1)    -> 0m0.000s 0m0.000s                       ZERO
# A pipeline forks and a command substitution forks, and the fork has no
# children of ITS own, so it reports zero however much CPU the checks burned —
# even piping through `sed` purely to indent destroys the number. A
# redirection does not fork, so capture once and then format and assert from
# the file as freely as you like. The first two versions of this block piped,
# and printed a confident 0m0.000s next to a 308s run: the "inert instrument
# reports a clean number" failure the gates in this directory exist to catch,
# occurring in the code that measures them.
# No EXIT trap for the temp file ON PURPOSE — registering one would put this
# script into trap_exit_launder_audit.py's population (Issue 734), and the
# only cost of not having one is a single stray file if the gate is killed.
times_out="$(mktemp)"
times > "$times_out"
gate_cpu="$(awk 'NR==2 { t=0; for (i=1;i<=NF;i++) { split($i, p, "m"); sub("s","",p[2]); t += p[1]*60 + p[2] } printf "%.2f", t }' "$times_out")"

# ── Does `times` account for these children AT ALL? (Issue 792) ─────────────
# The ~0 guard below catches a number destroyed by a fork. It does NOT catch
# the other way this figure stops being a measurement, because that one prints
# a well-formed, plausible-looking value: on Windows/MSYS, `times` accounts for
# MSYS children and reports essentially NOTHING for native (non-MSYS) ones.
# Measured 2026-09-14 on this workspace's Windows box, one child at a time,
# each burning ~2s of CPU:
#     bash -c 'while ...'               -> children 1.796s user 0.468s sys  ✓
#     py -c '...'                       -> children 0.000s user 0.015s sys  ✗
#     <python.exe by absolute path>     -> children 0.000s user 0.045s sys  ✗
# So it is the MSYS/native boundary, NOT the `py` launcher shim — resolving a
# real executable does not recover it. The gate printed 1.26s CPU against a
# 19.7s wall and the ~0 guard stayed quiet, while AGENTS.md instructs the
# reader to cite exactly that figure.
#
# A ratio against the wall clock cannot separate the two (this gate's own
# history is 12.65s CPU on a 299.1s wall — 4%, legitimately), so the
# instrument proves itself instead: burn a KNOWN amount of CPU in a child and
# check whether `times` saw it. An unaccounted platform gets a refusal, never
# a number.
cal_burn=0.25
cal_before="$(awk 'NR==2 { t=0; for (i=1;i<=NF;i++) { split($i, p, "m"); sub("s","",p[2]); t += p[1]*60 + p[2] } printf "%.3f", t }' "$times_out")"
"$PY" -c "import time
_t = time.process_time()
while time.process_time() - _t < $cal_burn:
    pass" >/dev/null 2>&1
cal_out="$(mktemp)"
times > "$cal_out"
cal_after="$(awk 'NR==2 { t=0; for (i=1;i<=NF;i++) { split($i, p, "m"); sub("s","",p[2]); t += p[1]*60 + p[2] } printf "%.3f", t }' "$cal_out")"
cal_seen="$(awk -v a="$cal_after" -v b="$cal_before" 'BEGIN { d = a - b; printf "%.3f", (d > 0 ? d : 0) }')"
# Half the burn is the bar: scheduling noise and interpreter startup move this
# by tens of milliseconds, not by a factor of ten.
cal_ok="$(awk -v s="$cal_seen" -v b="$cal_burn" 'BEGIN { print (s >= b / 2) ? "OK" : "UNACCOUNTED" }')"
rm -f "$cal_out"

if [ "$cal_ok" = "UNACCOUNTED" ]; then
    echo "  ⏱  total ${gate_wall}s wall · CPU SUPPRESSED (Issue 792)"
    echo "     ⛔ children CPU is NOT accounted on this platform: a child that burned"
    echo "        ${cal_burn}s of CPU moved the times children total by ${cal_seen}s."
    echo "        The raw rows below are real for MSYS children and ~0 for native ones,"
    echo "        so the total is NOT the quantity AGENTS.md tells you to cite. Read"
    echo "        wall only here, and take the CPU figure from a POSIX workstation."
    sed 's/^/     /' "$times_out"
else
    echo "  ⏱  total ${gate_wall}s wall · ${gate_cpu}s CPU in the checks (rows: this shell, then the checks)"
    sed 's/^/     /' "$times_out"

    # ...and the instrument must prove itself NON-INERT, because the failure mode
    # above is a well-formed number, not an error. N python3 checks cannot burn
    # ~no CPU: if the total reads as ~0 across a multi-second run, the measurement
    # broke and the figure must not be quoted.
    if [ "$(awk -v c="$gate_cpu" -v w="$gate_wall" 'BEGIN { print (c < 0.05 && w > 5) ? "INERT" : "OK" }')" = "INERT" ]; then
        echo "  ⛔ the CPU figure above is NOT a measurement: ${gate_cpu}s of CPU across"
        echo "     ${gate_wall}s of wall is impossible for ${#CHECKS[@]} python3 checks."
        echo "     The times builtin was read from a forked context — do not quote it."
    fi
    echo "     CPU is the load-invariant figure: 12.65s / 12.52s / 12.69s measured"
    echo "     on runs whose WALL was 128.3s / 299.1s / 15.0s — 20x wall, 1.4% CPU."
    echo "     Cite the CPU number; read wall as a range, never as a baseline."
fi
rm -f "$times_out"

if [ "$failed" -ne 0 ]; then
    echo "✗ docs gate FAILED — $failed of ${#CHECKS[@]} check(s)"
    exit 1
fi
echo "✓ docs gate PASSED — ${#CHECKS[@]}/${#CHECKS[@]} checks clean"
