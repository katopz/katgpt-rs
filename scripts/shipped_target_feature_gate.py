#!/usr/bin/env python3
r"""A SHIPPED path selecting its fast arm on a COMPILE-time `target_feature`
— Issue 847 T3.

```rust
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]   // COMPILE time
{ unsafe { fast_arm(..) } return; }
#[cfg(not(any(target_arch = "aarch64",
              all(target_arch = "x86_64", target_feature = "avx2"))))]
{ scalar_arm(..) }
```

`target_feature = "avx2"` is **off by default on x86_64**, so the fast arm
compiles to nothing on every ordinary build and the dispatcher silently takes
the scalar path. AGENTS.md already documents this shape and **only for
GATES** — *"an arm gated `cfg(all(target_arch = "x86_64", target_feature =
"avx2"))` compiles to nothing without it, and the run then exercises the
scalar fallback and proves nothing."* On a shipped path the cost is latency on
every call, in the configuration everybody builds, and nothing was looking for
it there.

Measured when Issue 847 found it: a default build was giving up **4.4-5.6x**
on `dequant_dot_via_lut` — in a DEFAULT-ON feature — and **2.4-2.5x** on
`f32_to_bf16_rne_into`. The correct form needs no new machinery:
`if simd_level() == SimdLevel::Avx2 { .. }`, a cached CPUID probe
`katgpt-types` already ships, with the kernel gated on the **arch alone**
(its `#[target_feature(enable = ..)]` attribute, not the cfg, is what makes
the intrinsic body compile).

## Three exclusions, each of which a naive grep gets wrong

The task that commissioned this gate names them, and getting one wrong makes
the cries-wolf instrument AGENTS.md warns gets ignored:

- **wasm32 / `simd128`** — that target has no runtime feature detection in
  this workspace, so a compile-time gate is the ONLY option and `full_gate.sh`
  layer 2b already builds both arms. Every site correct by construction;
  the count is on the PASS line, never typed here.
- **aarch64 / NEON** — implied by the arch; there is nothing to detect.
- **the runtime probe's own body** — `#[cfg(target_feature = "avx2")] { true }`
  inside `is_avx2_fma_available()` **is** the correct pattern (a build that
  already has AVX2 need not ask CPUID). Pinned, because it is indistinguishable
  from the defect by any predicate: it is the same attribute doing the opposite
  job.
- **`#[target_feature(enable = ..)]`** is the RIGHT attribute and is never this
  class — it makes a body compile on any build, which is what pairs with a
  runtime probe. Counted on the PASS line, never typed here.

## Population: this repo. Measured, not inherited.

`check_validation_gate` T4 declined a sweep on a population of ONE and was
right; `console_encoding_gate` INHERITED that answer and was wrong by seven
repos. So it was counted, over the 17 contract repos on the workstation:
every `target_feature` cfg attribute in `src/` was in katgpt-rs and zero in
every sibling (measured 2026-09-19: 166 of 166 — a dated record, and the
only figure kept here because it is the MEASUREMENT the no-sweep decision
rests on). Re-derive it rather than trusting this paragraph:
`--workspace` prints the table.

Exemptions are pinned by MEMBERSHIP with a REASON per row
(`scripts/shipped_target_feature_expected.txt`); a reasonless row is refused,
and a row whose site is gone REDS. The key is **LINE-FREE** —
`<path>::<enclosing fn>#<ordinal>` — because a line number drifts on every
edit above it and a pin file that reds on noise is one people delete
(`arm_reach_gate`'s recorded rule).

    scripts/shipped_target_feature_gate.py              # the verdict, per push
    scripts/shipped_target_feature_gate.py --workspace  # the population, exit 0
    scripts/shipped_target_feature_gate.py --prove-fires 32056164

The arms run UNCONDITIONALLY, behind no flag: `docs_gate.sh` invokes each check
as `"$PY" "$script"` with no arguments (Issue 789's finding).
"""

from __future__ import annotations

import contextlib
import io
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import console_safe  # noqa: E402
from platform_dead_code_audit import mask_file  # noqa: E402
from tracked_walk import tracked_files  # noqa: E402

console_safe.apply()

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED = REPO_ROOT / "scripts" / "shipped_target_feature_expected.txt"

# The commit that repaired `simd_lut_dequant`; its parent carries the six
# sites that repair removed, which is what `--prove-fires` adjudicates.
PROVE_SHA = "32056164"

FEAT = re.compile(r'target_feature\s*=\s*"([^"]+)"')
ENABLE = re.compile(r"target_feature\s*\(\s*enable")
# x86_64 features this workspace can detect at RUNTIME (katgpt-types ships the
# cached CPUID probe). Anything else is compile-time by nature.
RUNTIME_DETECTABLE = re.compile(r"^(avx2|fma|avx512|sse4)")
FN = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)")

# Two floors, failing differently.
#   MIN_FILES  the WALK. A glob that matches nothing reports every path clean
#              and prints a confident green over zero of them.
#   MIN_ATTRS  the ATTRIBUTE PARSE — every `target_feature` attribute the
#              masker hands back, in ANY bucket. If `mask_file` stops
#              capturing attribute interiors (where the cfg VALUE strings
#              live) the class count goes to zero over an unchanged walk,
#              which is byte-identical to a clean repo. This is the floor
#              that fails in that direction, and it is the one worth having.
MIN_FILES = 800
MIN_ATTRS = 100


def verdict(raw: str) -> str:
    """One attribute -> its bucket.

    Order matters: `#[target_feature(enable = ..)]` also contains the string
    `target_feature`, and testing the feature VALUES first would read its
    `enable` list as a cfg predicate.
    """
    if ENABLE.search(raw):
        return "enable-attr"
    feats = FEAT.findall(raw)
    if not feats:
        return "none"
    if "simd128" in feats:
        return "wasm32"
    if any(f.startswith("neon") for f in feats):
        return "neon"
    if any(RUNTIME_DETECTABLE.match(f) for f in feats):
        return "CLASS"
    return "other"


def _fn_spans(text: str) -> list[tuple[str, int, int]]:
    """`(name, body_start, body_end)` for every `fn` with a body, by brace
    counting over the MASKED text.

    Brace counting rather than a parser because the class lives in attributes,
    which a Rust parser would give us no more accurately — and because
    `mask_file` has already removed the strings and comments that make brace
    counting wrong. A `fn` with no body (a trait signature) contributes no
    span, which is correct: an attribute cannot be inside one.
    """
    spans: list[tuple[str, int, int]] = []
    for m in FN.finditer(text):
        i = text.find("{", m.end())
        if i < 0:
            continue
        # A `where` clause or a return type can carry braces; the body is the
        # first brace that opens a balanced run reaching depth 0 again.
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    spans.append((m.group(1), i, j))
                    break
    return spans


def owner_of(text: str, pos: int, spans: list[tuple[str, int, int]]) -> str:
    """The fn a site BELONGS to — line-free, and stable under edits above it.

    Two cases, and they need opposite lookups:
    - the attribute sits INSIDE a body (a dispatcher's arm selection) -> the
      INNERMOST enclosing span;
    - the attribute sits ON an item (a kernel) -> no span contains it, so the
      owner is the NEXT `fn` declared after it.
    """
    inner = [s for s in spans if s[1] < pos < s[2]]
    if inner:
        return min(inner, key=lambda s: s[2] - s[1])[0]
    nxt = FN.search(text, pos)
    return nxt.group(1) if nxt else "<module>"


def scan(root: Path) -> tuple[list[str], dict[str, int], int]:
    """-> `(finding keys, bucket counts, files walked)`.

    Reads `mask_file`'s captured ATTRIBUTES, never raw text: a `#[cfg(...)]`
    inside a raw-string fixture is test INPUT, and three sibling instruments
    here have each reported findings inside their own fixture strings.
    """
    files, _excluded = tracked_files(root, "*.rs")
    counts = {"CLASS": 0, "enable-attr": 0, "wasm32": 0, "neon": 0,
              "other": 0, "none": 0}
    keys: list[str] = []
    walked = 0
    for f in files:
        rel = f.relative_to(root).as_posix()
        if not (rel.startswith("src/") or "/src/" in rel):
            continue          # SHIPPED path only — not tests/benches/examples
        walked += 1
        try:
            text, attrs = mask_file(f.read_text(encoding="utf-8",
                                                errors="replace"))
        except OSError:
            continue
        spans = None
        seen: dict[str, int] = {}
        for a in attrs:
            if "target_feature" not in a.raw:
                continue
            v = verdict(a.raw)
            counts[v] = counts.get(v, 0) + 1
            if v != "CLASS":
                continue
            if spans is None:
                spans = _fn_spans(text)
            owner = owner_of(text, a.start, spans)
            base = f"{rel}::{owner}"
            seen[base] = seen.get(base, 0) + 1
            keys.append(f"{base}#{seen[base]}")
    return sorted(keys), counts, walked


def read_expected(path: Path = EXPECTED) -> dict[str, str]:
    """`<key> = <reason>` rows. A reasonless row is REFUSED, not ignored: the
    reason is the adjudication (Issue 785)."""
    pins: dict[str, str] = {}
    if not path.exists():
        return pins
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            print(f"⛔ {path.name}:{i}: row without a reason — refused")
            raise SystemExit(2)
        key, reason = line.split("=", 1)
        if not reason.strip():
            print(f"⛔ {path.name}:{i}: empty reason — refused")
            raise SystemExit(2)
        pins[key.strip()] = reason.strip()
    return pins


def selftest() -> list[str]:
    """Arms over the classifier AND over this gate's own pin arithmetic, which
    a classifier arm cannot reach (Issue 775's rule)."""
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    # --- the bucket predicate ---------------------------------------------
    check(verdict('#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]')
          == "CLASS", "the defect shape was not classified CLASS")
    check(verdict('#[cfg(target_feature = "avx2")]') == "CLASS",
          "a bare avx2 cfg was not classified CLASS")
    check(verdict('#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]')
          == "wasm32", "a simd128 cfg was counted as the class")
    check(verdict('#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]')
          == "neon", "a NEON cfg was counted as the class")
    # ⛔ ORDER: this attribute also contains the string `target_feature`, and
    # a predicate that reads its enable list as a cfg would flag every correct
    # kernel in the repo — the exact cries-wolf outcome this gate must avoid.
    check(verdict('#[target_feature(enable = "avx2", enable = "fma")]')
          == "enable-attr", "the CORRECT attribute was read as the defect")
    check(verdict('#[cfg(target_arch = "x86_64")]') == "none",
          "an arch-only cfg produced a feature verdict")
    check(verdict('#[cfg(target_feature = "crt-static")]') == "other",
          "a feature with no runtime probe was counted as the class")

    # --- the LINE-FREE owner key ------------------------------------------
    INSIDE = (
        "fn dispatch(x: u8) {\n"
        '    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]\n'
        "    {\n"
        "        fast(x)\n"
        "    }\n"
        "}\n"
    )
    t, attrs = mask_file(INSIDE)
    spans = _fn_spans(t)
    check(owner_of(t, attrs[0].start, spans) == "dispatch",
          "an attribute INSIDE a body did not resolve to its enclosing fn")

    ON_ITEM = (
        "fn earlier() {\n"
        "    let _ = 1;\n"
        "}\n"
        '#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]\n'
        "unsafe fn kernel(x: u8) {\n"
        "    let _ = x;\n"
        "}\n"
    )
    t, attrs = mask_file(ON_ITEM)
    spans = _fn_spans(t)
    tf = [a for a in attrs if "target_feature" in a.raw][0]
    check(owner_of(t, tf.start, spans) == "kernel",
          "an attribute ON an item resolved to the PREVIOUS fn, not the one "
          "it attributes")

    # ⛔ Line-INVARIANCE is the property the key exists for, so it is armed:
    # padding above a site must not change its key.
    t2, attrs2 = mask_file("// pad\n// pad\n// pad\n" + ON_ITEM)
    spans2 = _fn_spans(t2)
    tf2 = [a for a in attrs2 if "target_feature" in a.raw][0]
    check(owner_of(t2, tf2.start, spans2) == owner_of(t, tf.start, spans),
          "the key moved when unrelated lines were added above it")

    # --- the walk + fixture masking, on a synthetic tree -------------------
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "src"
        src.mkdir(parents=True)
        # Path.write_text's newline= kwarg is 3.10+ (Issue 965: these
        # fixtures must run on the boxes' python3 3.9.6).
        with open(src / "good.rs", "w", encoding="utf-8", newline="") as _f:
            _f.write(
                '#[cfg(target_arch = "x86_64")]\n'
                '#[target_feature(enable = "avx2")]\n'
                "unsafe fn ok(x: u8) {\n    let _ = x;\n}\n")
        with open(src / "bad.rs", "w", encoding="utf-8", newline="") as _f:
            _f.write(
                "fn dispatch(x: u8) {\n"
                '    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]\n'
                "    {\n        let _ = x;\n    }\n}\n")
        # A fixture STRING holding the defect is test input, not a site — the
        # class three sibling instruments here have each met.
        with open(src / "fixture.rs", "w", encoding="utf-8", newline="") as _f:
            _f.write(
                "const SAMPLE: &str = r#\"\n"
                '#[cfg(all(target_arch = "x86_64", target_feature = "avx2")]\n'
                "\"#;\n")
        # tests/ is not the SHIPPED path: a gate may gate itself on +avx2.
        (root / "tests").mkdir()
        with open(root / "tests" / "t.rs", "w", encoding="utf-8",
                  newline="") as _f:
            _f.write(
                '#[cfg(all(target_arch = "x86_64", target_feature = "avx2")]\n'
                "fn t() {}\n")
        keys, counts, walked = scan(root)
        check(keys == ["src/bad.rs::dispatch#1"],
              f"the synthetic tree did not yield exactly the one site: {keys}")
        check(counts["enable-attr"] == 1,
              f"the correct attribute was not counted: {counts}")
        check(walked == 3, f"the src/ walk counted {walked}, want 3")


    # --- `_fn_spans` edge cases, armable from fixture strings alone --------
    # A signature with NO body (a trait method) must contribute no span: an
    # attribute cannot be inside one, and inventing a span for it would make
    # the NEXT function's attributes resolve to the wrong owner.
    spans = _fn_spans("trait T {\n    fn sig(&self) -> u8;\n}\n")
    check(all(n != "sig" for n, _s, _e in spans),
          f"a bodiless fn produced a span: {spans}")

    # Nested fns: the depth counter must close the OUTER body at the outer
    # brace, not at the inner one. `depth == 0` is the test that does it, and
    # flipping it silently truncates every enclosing span to its first nested
    # block — after which every dispatcher attribute resolves to <module>.
    NESTED = ("fn outer() {\n"
              "    fn inner() {\n        let _ = 1;\n    }\n"
              "    let _ = 2;\n"
              "}\n")
    spans = _fn_spans(NESTED)
    by = {n: (a, b) for n, a, b in spans}
    check("outer" in by and "inner" in by,
          f"nested fns were not both spanned: {spans}")
    o, i2 = by["outer"], by["inner"]
    check(o[0] < i2[0] and i2[1] < o[1],
          f"the outer span does not CONTAIN the inner one: {by}")

    # `owner_of` must pick the INNERMOST enclosing span, which is what the
    # width comparison decides. An attribute inside `inner` belongs to
    # `inner`, not to `outer`.
    mid = (i2[0] + i2[1]) // 2
    check(owner_of(NESTED, mid, spans) == "inner",
          f"the innermost enclosing fn was not chosen: "
          f"{owner_of(NESTED, mid, spans)}")
    outer_only = (o[0] + i2[0]) // 2
    check(owner_of(NESTED, outer_only, spans) == "outer",
          "a position inside only the OUTER body did not resolve to it")

    # --- the pin PARSER, reachable now that `read_expected` takes a path ---
    # ⛔ Four refusal branches sat behind a module-level constant, so no arm
    # could hand them a file: unreachable by construction, which is the
    # pattern AGENTS.md records three times — the EXTRACTION is the repair.
    with tempfile.TemporaryDirectory() as td:
        pin = Path(td) / "pins.txt"

        check(read_expected(pin) == {},
              "a MISSING pin file did not read as zero pins")

        # Path.write_text's newline= kwarg is 3.10+ (Issue 965: these
        # fixtures must run on the boxes' python3 3.9.6).
        with open(pin, "w", encoding="utf-8", newline="") as _p:
            _p.write("# only a comment\n\n")
        check(read_expected(pin) == {},
              "comments and blank lines were parsed as rows")

        with open(pin, "w", encoding="utf-8", newline="") as _p:
            _p.write("a::b = a real reason\n")
        check(read_expected(pin) == {"a::b": "a real reason"},
              "a well-formed row did not parse")

        # ⚠ Both refusals PRINT before they raise, and that output is the
        # arm's expected noise rather than this gate's verdict — swallowed,
        # or a green run carries two `⛔` lines above its own `✓`.
        def _refuses(text: str) -> bool:
            with open(pin, "w", encoding="utf-8", newline="") as _p:
                _p.write(text)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    read_expected(pin)
            except SystemExit:
                return True
            return False

        check(_refuses("a::b\n"),
              "a row with NO `=` was accepted — the reason IS the "
              "adjudication, and a row without one is a backlog wearing a pin")
        check(_refuses("a::b =   \n"),
              "a row with an EMPTY reason was accepted")

    # --- this gate's OWN pin arithmetic, which no classifier arm reaches ---
    check(MIN_FILES > 0 and MIN_ATTRS > 0, "a floor is non-positive")
    try:
        read_expected()
    except SystemExit as e:  # pragma: no cover - only on a malformed pin file
        fails.append(f"read_expected refused the live pin file: {e}")
    return fails


def workspace_census() -> int:
    """A REPORT (exit 0) — the population, re-derived per run.

    ⛔ **No sweep half, and it is a MEASUREMENT rather than a preference.**
    `check_validation_gate` T4 declined a sweep on a population of ONE and was
    right; `console_encoding_gate` inherited that answer and was wrong by
    seven repos. Counted over the contract repos on the workstation, every
    `target_feature` cfg attribute in `src/` is in THIS repo. Take the live
    figures from this table, never from a sentence.
    """
    import repo_alias
    from skill_repo_set_gate import derive_repos

    ws = REPO_ROOT.parent
    print(f"{'repo':<26}{'src .rs':>9}{'tf attrs':>10}{'CLASS':>7}"
          f"{'enable':>8}{'wasm32':>8}{'neon':>6}")
    tot = {"files": 0, "attrs": 0, "CLASS": 0}
    carriers = 0
    for n in sorted(derive_repos(ws)):
        keys, c, walked = scan(ws / repo_alias.disk(n))
        attrs = sum(c.values())
        tot["files"] += walked
        tot["attrs"] += attrs
        tot["CLASS"] += len(keys)
        carriers += 1 if attrs else 0
        print(f"{n:<26}{walked:>9}{attrs:>10}{len(keys):>7}"
              f"{c['enable-attr']:>8}{c['wasm32']:>8}{c['neon']:>6}")
        for k in keys:
            print(f"    X {n}: {k}")
    print(f"\n{tot['files']} src *.rs · {tot['attrs']} target_feature "
          f"attribute(s) · {tot['CLASS']} CLASS site(s) · {carriers} repo(s) "
          f"carrying any")
    return 0


def prove_fires(sha: str) -> int:
    """Extract `<sha>~1` and `<sha>` and require the count to DROP.

    A known-answer tree, in the `platform_dead_code_audit.py --prove-fires`
    idiom. `32056164` repaired `channel_aware`'s duplicated kernel and
    `simd_lut_dequant`'s three dispatchers; its parent carries the sites that
    repair removed, so the classifier must see strictly more there.
    """
    with tempfile.TemporaryDirectory() as td:
        seen = {}
        for ref in (f"{sha}~1", sha):
            tree = Path(td) / ref.replace("~", "_")
            tree.mkdir()
            ar = subprocess.run(["git", "-C", str(REPO_ROOT), "archive", ref,
                                 "crates"], capture_output=True)
            if ar.returncode != 0:
                print(f"✗ --prove-fires: cannot `git archive {ref}`")
                return 2
            tar = tree / "t.tar"
            tar.write_bytes(ar.stdout)
            subprocess.run(["tar", "-xf", str(tar), "-C", str(tree)],
                           capture_output=True)
            tar.unlink()
            keys, _c, walked = scan(tree)
            if walked < MIN_FILES:
                print(f"✗ --prove-fires: {ref} walked {walked} src file(s) — "
                      f"the arm proves nothing")
                return 2
            seen[ref] = keys
            print(f"  {ref}: {len(keys)} CLASS site(s) over {walked} src *.rs")
        before, after = seen[f"{sha}~1"], seen[sha]
        if len(before) <= len(after):
            print(f"✗ --prove-fires: the repair did not reduce the count "
                  f"({len(before)} -> {len(after)}) — the classifier cannot "
                  f"see what {sha} fixed")
            return 2
        for k in sorted(set(before) - set(after)):
            print(f"      repaired: {k}")
    print(f"✓ --prove-fires: the classifier sees {len(before)} site(s) at the "
          f"parent and {len(after)} after the repair")
    return 0


def main() -> int:
    bad = selftest()
    for f in bad:
        print(f"✗ selftest: {f}")
    if bad:
        print(f"⛔ shipped-target_feature gate SELFTEST FAILED — {len(bad)} "
              f"arm(s); the classifier cannot be read as a verdict")
        return 2

    expected = read_expected()
    keys, counts, walked = scan(REPO_ROOT)
    attrs = sum(counts.values())

    unpinned = sorted(set(keys) - set(expected))
    stale = sorted(set(expected) - set(keys))

    for k in unpinned:
        print(f"✗ {k} selects a fast arm on a COMPILE-time `target_feature`. "
              f"On x86_64 that predicate is OFF by default, so the arm "
              f"compiles to NOTHING on every ordinary build and this path "
              f"silently runs its fallback — measured at 4.4-5.6x on a "
              f"default-on feature (Issue 847). Use the runtime probe "
              f"(`simd::simd_level() == SimdLevel::Avx2`) and gate the kernel "
              f"on the ARCH alone, or pin it with a reason in "
              f"{EXPECTED.name}")
    for k in stale:
        print(f"✗ {k} — pinned but no longer present. Drop the row: a pin file "
              f"that only ever loosens stops being a wall")

    if walked < MIN_FILES:
        print(f"✗ walk FLOOR breached: {walked} tracked src *.rs < "
              f"{MIN_FILES} — the walk went blind and every count above is "
              f"vacuous")
        return 1
    if attrs < MIN_ATTRS:
        print(f"✗ attribute FLOOR breached: {attrs} target_feature "
              f"attribute(s) < {MIN_ATTRS} — the masker stopped capturing "
              f"attribute interiors, which looks exactly like a clean repo")
        return 1
    if unpinned or stale:
        return 1

    print(f"✓ shipped-target_feature gate PASSED — every compile-time "
          f"`target_feature` arm on a shipped path is pinned with a reason, "
          f"over {walked} tracked src *.rs (floor {MIN_FILES}) / {attrs} "
          f"target_feature attribute(s) (floor {MIN_ATTRS}), "
          f"{len(expected)} pinned, 0 stale. Excluded BY CONSTRUCTION and "
          f"counted: {counts['enable-attr']} `#[target_feature(enable = ..)]` "
          f"(the correct attribute), {counts['wasm32']} wasm32/simd128 (no "
          f"runtime detection on that target), {counts['neon']} NEON (implied "
          f"by the arch). ⚠ It does NOT see a dispatcher that reads "
          f"`cfg!(target_feature = ..)` as a runtime-looking boolean, nor one "
          f"built from a type alias (Issue 847 T3)")
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--workspace" in argv:
        raise SystemExit(workspace_census())
    if "--prove-fires" in argv:
        i = argv.index("--prove-fires")
        raise SystemExit(prove_fires(argv[i + 1] if i + 1 < len(argv)
                                     else PROVE_SHA))
    raise SystemExit(main())
