#!/usr/bin/env python3
"""The platform-dead_code class — an item declared UNGATED whose every use
sits behind a platform cfg the declaration lacks.

    const NEON_U8: usize = 16;                      // ungated, any platform
    #[cfg(target_arch = "aarch64")]                 // the only use
    unsafe fn neon_is_interval_closed(..) { ..(bytes, NEON_U8) }

On every platform except aarch64 this item is DEAD CODE — rustc says so on
the lanes that compile x86_64, and stays silent on the lanes that compile
aarch64. Five specimens across two repos in two days (riir-clippy intake P22,
2026-09-14), every one caught by a lane NO CI owns:

    katgpt-rs NEON_U8            (fixed ea4c2873 — interval_pruner/simd.rs)
    zed     KEYCHAIN_SERVICE     (fixed a4f70e64a7 — claude_usage.rs)
    zed     war_room_screenshot  (example, macOS-only harness)
    zed     ProcessStatus        (import, unix-only use — reliability.rs)
    zed     util/process.rs test imports (module held only a not-windows test)

katgpt-rs full_gate is macOS/aarch64 (where the aarch64 side compiles and the
item is USED); wasm32_gate is wasm32. The x86_64-native Windows lane that
actually sees the dead_code warning is a workstation lane. The defect is
therefore invisible to every automatic gate this workspace runs.

A REPORT, not a gate (always exit 0) — the population spans repos whose
owners have not taken the intake, and the classifier has honest blind spots
listed under Divergences. The gateable half (a per-repo findings ceiling of
0, the way `cfg_gated_floor_gate.py` pins its family) belongs to the repo
owners after the first sweep adjudicates the row list.

# The rule

For each module-scope declaration (const/static/fn/struct/enum/type/use, incl.
impl/trait-associated items) that is NOT bare `pub` in a library file
(externally reachable items are never dead; `pub` in bin/example/test files
still counts — no external consumer exists, the zed war_room_screenshot
shape), collect every identifier occurrence in the same crate unit (comments,
strings, and attribute interiors masked out). Resolve each occurrence's cfg
context — the conjunction of enclosing item attrs, block attrs (cfg'd
statement blocks included), file-leading `#![cfg]`, and enclosing `mod foo;`
decl gates resolved ACROSS FILES up the directory chain.

A context's PLATFORM ATOMS are the `target_*` / `unix` / `windows` conjuncts
at the top level of its predicates (nested in `all(...)` allowed). A use is
NARROWER than the declaration iff it carries a platform atom (key, value)
where the declaration's context has no atom for that key, or a different
value for it.

FINDING iff the declaration has at least one use and EVERY use is narrower —
on the platforms the declaration compiles but no use does, the item is dead.
Read that as all-uses-gated per-use (each use may gate on a different atom);
the intake's "shares a platform predicate" wording is satisfied a fortiori.

# MOD-REF is a separate bucket and is never folded into the count

A `mod name;` declaration satisfies the rule above and is NOT a rustc
finding. Measured on the first row this audit produced (2026-09-14,
`katgpt-types/src/simd/mod.rs:49`): `mod horizontal;` is ungated and all 15
references to it are `target_arch = "x86_64"`, and `cargo check -p
katgpt-types --target wasm32-unknown-unknown` emits NOTHING — every item
INSIDE that module is itself x86_64-gated, so on wasm32 the module is empty
rather than dead. Appending one ungated `fn` to the same file reproduces the
warning immediately, and it names the FN. So rustc reports dead code at the
ITEM, the items are reached by this audit independently, and a `mod` row is
an upstream observation about where a module is referenced from — a style
nit at most. Pooling it would report that nit at the severity of a dead item,
the pooled-total mistake this workspace documents in three other audits.

# Conservative by construction — the directions that only lose findings

  * a platform atom under `any(...)` or `not(...)` makes the whole context
    UNMODELLED and the use DISQUALIFIES the declaration (it may be alive on
    every platform we care about);
  * `feature = ...`, `test`, `doc`, `debug_assertions` conjuncts are not
    platform atoms — the feature-gated green-zero class is
    cfg_gated_target_audit's territory, not this one;
  * occurrences inside macro bodies/args count as uses (a macro that
    references the item keeps it alive wherever it is invoked);
  * doc-comment mentions are NOT uses (masked) — but an INLINE FORMAT ARG
    is: `println!("n = {NPC_COUNT}")` binds the identifier under Rust 2021,
    so identifiers inside a string's `{...}` groups survive the mask. A
    non-format string containing `{Foo}` therefore yields a phantom use,
    which can only hide;
  * a name declared more than once in a unit (shadowing across modules) is
    skipped outright — occurrences cannot be attributed to one decl;
  * a `pub use` re-export of the name counts as an ungated use wherever the
    re-export itself compiles (an occurrence is an occurrence).

# Divergences (documented)

⛔ The first version of this header claimed "no known direction in which this
INVENTS a finding" and that claim was FALSE on the first sweep it ever ran:
masking string literals dropped Rust 2021 inline format args, and riir-ai's
`SWEEP_COUNTS` — printed ungated at `ane_npc_goat.rs:250` and otherwise used
only under `target_os = "macos"` — read as a finding. Repaired above and
pinned by two self-test arms (format arg / plain mention). The list below is
the set of gaps that have been LOOKED for, not a proof that none remains:
a conservative-by-construction argument is a claim about the code somebody
wrote, and this one was wrong until a real corpus contradicted it.


  (a) same-name locals/params/fields elsewhere count as uses (hides);
  (b) `#[path]`-aliased modules and out-of-tree includes lose their chain
      gates (parentage unresolved -> treated as ungated, hides);
  (c) uses reachable only through macro expansion of OTHER names are
      invisible unless the macro body mentions the name literally (hides);
  (d) a `#[cfg]` on a struct FIELD, an enum VARIANT or a fn PARAMETER is not
      modelled — those are not items to this pass, so a type named only in a
      gated field reads as used from the struct's own context (hides). The
      item HEADER is modelled: a block item's segment is extended back over
      its generics, parameters and return type, which is what makes the
      `fn wait() -> ProcessStatus` import shape reachable at all;
  (e) over-marking risk is the absence of one: every attribution gap above
      points toward fewer findings, never more.

# Validation

`selftest()` pins the classifier on synthetic trees in BOTH directions and
runs on EVERY invocation (exit 2 on any miss — an untrustworthy instrument is
not the same verdict as drift, and a report that cannot classify must not be
read as `0 findings`). `--prove-fires <fix_sha>` extracts `<fix_sha>~1` and
`<fix_sha>` of THIS repo via `git archive` and requires the NEON_U8 row to
fire at the parent and be absent at the fix — the ceiling is never a pin
nobody has watched fail. The four zed specimens live on a box this script
does not run from; they are cited as provenance, never claimed as validated.

⛔ An arm is only a canary if its own perturbation REDS it. Measured on the
three arms this pass added: deleting the format-arg restore reds arm 20,
pooling MOD-REF into findings reds arm 23 — and neutering `vendored_p` red
NOTHING, because the synthetic trees have no `.git` and were taking the
filesystem-walk branch, where a redundant `"vendor"` in SKIP_DIRS was doing
the filtering instead. One exclusion, two code paths, and the arm certified
the path it was not aimed at. Removing the duplicate made it two-sided.

Standing (2026-09-14, this box's 16 of 20 contract repos): **0 findings · 1
MOD-REF** over 8694 tracked `.rs` / 3433 units / 119452 candidate decls, with
the two riir-ai rows from the first sweep repaired and compile-verified —
`note_ane_dispatch` (x86_64 Windows, `--features ane_prefill`) and
`gen_u64_bytes` (wasm32, `--features chacha20_rng`).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfg_gated_target_audit import derive_repos  # noqa: E402
from cfg_row_implication_audit import leading_inner_cfgs, split_conjuncts  # noqa: E402

# ── platform vocabulary (DATA — rustc's closed set of platform cfg keys) ────

PLATFORM_KEYS = {
    "target_arch", "target_os", "target_family", "target_env",
    "target_vendor", "target_pointer_width", "target_abi",
}
BARE_FAMILY = {"unix": ("target_family", "unix"),
               "windows": ("target_family", "windows")}
NON_PLATFORM_WORDS = {"test", "doc", "doctest", "miri", "debug_assertions",
                      "panic", "procmacro", "coverage", "feature"}

ITEM_KEYWORDS = {"fn", "const", "static", "struct", "enum", "type", "use",
                 "mod", "impl", "trait", "union", "macro_rules"}
TYPEISH_KEYWORDS = {"struct", "enum", "union", "type"}
FNISH_FOLLOWERS = {"fn", "unsafe", "async", "extern"}

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\S")
# Rust 2021 captured identifiers: `println!("n = {NPC_COUNT}")` is a USE.
FMT_GROUP = re.compile(r"\{([^{}]*)\}")

# ── cfg predicate → platform atoms ─────────────────────────────────────────


@dataclass(frozen=True)
class Ctx:
    """A cfg context: top-level platform conjuncts + an honesty flag."""
    atoms: frozenset = frozenset()          # {(key, value)}
    unmodelled: bool = False                # platform atom under any()/not()

    def and_(self, other: "Ctx") -> "Ctx":
        if self.unmodelled or other.unmodelled:
            return Ctx(frozenset(), True)
        merged = set(self.atoms) | set(other.atoms)
        per_key: dict[str, str] = {}
        for k, v in merged:
            if k in per_key and per_key[k] != v:
                return Ctx(frozenset(), True)  # contradictory conjunction
            per_key[k] = v
        return Ctx(frozenset(merged), False)

    def key_map(self) -> dict[str, str]:
        return {k: v for k, v in self.atoms}


def _conjunct_atoms(c: str) -> tuple[set, bool]:
    """Platform atoms of one top-level conjunct. (set(), True) = unmodelled."""
    c = c.strip()
    if c.startswith("all(") and c.endswith(")"):
        out, bad = set(), False
        for sub in split_conjuncts(c):
            a, b = _conjunct_atoms(sub)
            out |= a
            bad = bad or b
        return out, bad
    if c.startswith("any(") and c.endswith(")"):
        # `split_conjuncts` unwraps `all(...)` and nothing else, so the
        # wrapper has to come off HERE — handing it `any(...)` whole returns
        # the same string and recurses until the stack dies.
        out, bad = set(), False
        for sub in split_conjuncts(c[4:-1]):
            a, b = _conjunct_atoms(sub)
            out |= a
            bad = bad or b
        return set(), bool(out) or bad  # alive on a union — cannot bound
    if c.startswith("not(") and c.endswith(")"):
        inner, _ = _conjunct_atoms(c[4:-1])
        return set(), bool(inner)  # alive everywhere except — cannot bound
    m = re.match(r'^([\w:]+)\s*(?:=\s*"([^"]*)")?$', c)
    if m:
        key, val = m.group(1), m.group(2)
        if key in PLATFORM_KEYS and val is not None:
            return {(key, val)}, False
        if key == "feature":
            return set(), False
        if val is None:
            if key in BARE_FAMILY:
                return {BARE_FAMILY[key]}, False
            return set(), False  # bare flag (test etc.) or unknown — not platform
        return set(), False  # unknown key with a value: not a platform gate
    return set(), True  # shape we do not model — refuse to guess


def pred_atoms(preds) -> Ctx:
    ctx = Ctx()
    for p in preds:
        for conj in split_conjuncts(p):
            a, b = _conjunct_atoms(conj)
            ctx = ctx.and_(Ctx(frozenset(a), b))
            if ctx.unmodelled:
                return ctx
    return ctx


# ── phase 1: masking + attribute capture ───────────────────────────────────

SPECIAL = re.compile(r'//|/\*|br#*"|br"|r#*"|r"|b"|f"|"|\x27|#!\[|#\[')


@dataclass
class Attr:
    start: int
    end: int
    preds: list
    raw: str
    inner: bool = False   # `#![...]` — governs the FILE, not the next item


def mask_file(text: str):
    """Mask comments/strings/attribute interiors; capture attrs verbatim.

    Attribute interiors are captured BEFORE masking because the cfg value
    strings ("aarch64") live there. Inner `#![...]` attributes are masked and
    captured with `inner=True`: their cfg predicates are read from disk by
    leading_inner_cfgs (file-level, already in the chain context), so applying
    them a SECOND time as the next item's pending attrs would gate one item
    and not its siblings. They are kept only for the `#![allow(dead_code)]`
    whole-file suppression, which has no other reader.
    """
    n = len(text)
    masked = list(text)
    attrs: list[Attr] = []
    pos = 0
    while True:
        m = SPECIAL.search(text, pos)
        if not m:
            break
        j, tok = m.start(), m.group(0)

        def blank(a: int, b: int) -> None:
            for k in range(a, min(b, n)):
                if masked[k] != "\n":
                    masked[k] = " "

        def blank_str(a: int, b: int) -> None:
            """Blank a string literal but KEEP inline format-arg identifiers.

            `println!("Sweep sizes: {SWEEP_COUNTS:?}")` is a real use under
            Rust 2021 captured identifiers, and blanking it INVENTED a
            finding — riir-ai's `SWEEP_COUNTS` was the first row this audit
            ever produced that was wrong, and it was wrong in the one
            direction the header claims is impossible. The braces themselves
            stay blanked, so brace balance is untouched; a non-format string
            that happens to contain `{Foo}` yields a phantom USE, which can
            only hide.
            """
            b = min(b, n)
            body = text[a:b]
            blank(a, b)
            for gm in FMT_GROUP.finditer(body):
                if gm.start() and body[gm.start() - 1] == "{":
                    continue                    # `{{` is an escaped brace
                base = a + gm.start(1)
                for im in IDENT.finditer(gm.group(1)):
                    for k in range(base + im.start(), base + im.end()):
                        masked[k] = text[k]

        if tok == "//":
            e = text.find("\n", j)
            e = n if e < 0 else e
            blank(j, e)
            pos = e
        elif tok == "/*":
            e = text.find("*/", j + 2)
            e = n if e < 0 else e + 2
            blank(j, e)
            pos = e
        elif tok in ("#[", "#!["):
            k = j + len(tok)
            bd = 1
            in_str = False
            while k < n and bd:
                ch = text[k]
                if in_str:
                    if ch == "\\":
                        k += 1
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == "[":
                    bd += 1
                elif ch == "]":
                    bd -= 1
                k += 1
            raw = text[j:k]
            preds = []
            # `cfg(` ONLY — a `cfg_attr(pred, attr)` predicate selects when
            # the ATTRIBUTE applies, never whether the ITEM exists, so its
            # pred must not gate the item. Folding it gated ungated items:
            # seal-remake lighting_panel's `#[cfg_attr(target_arch =
            # "wasm32", allow(unused_mut))]` fn made its own parameter's
            # type use read wasm32-only and flagged a live import (rustc's
            # native build warns nothing there). The `allow(dead_code)` half
            # still works — `pending_allow` reads the RAW text, not preds.
            for pm in re.finditer(r"\bcfg\s*\(", raw):
                p0 = pm.end()
                pd = 1
                p = p0
                while p < len(raw) and pd:
                    ch = raw[p]
                    if ch == "(":
                        pd += 1
                    elif ch == ")":
                        pd -= 1
                        if pd == 0:
                            break   # STOP on the closer, do not swallow it —
                            # `target_arch = "aarch64")` matches no atom regex
                            # and reads UNMODELLED, i.e. a silent clean run
                    elif ch == "," and pd == 1:
                        break
                    p += 1
                arg = raw[p0:p].strip()
                if arg:
                    preds.append(arg)
            attrs.append(Attr(start=j, end=k, preds=preds, raw=raw,
                              inner=tok == "#!["))
            blank(j, k)
            pos = k
        elif tok == "'":
            # `'` is a char literal OR a lifetime, and the two must not share
            # a branch: treating a lifetime as an opening quote sends the scan
            # hunting for a close that is not there and blanks the rest of the
            # file — a masking bug reports CLEAN, never a finding.
            if j + 1 < n and text[j + 1] == "\\":
                k = j + 2
                while k < n and text[k] != "'":
                    k += 1
                blank(j, k + 1)
                pos = k + 1
            elif j + 2 < n and text[j + 2] == "'":
                blank(j, j + 3)
                pos = j + 3
            else:                       # lifetime — leave it, it is a token
                pos = j + 1
        elif tok.startswith("r") or tok.startswith("br"):
            # raw string: no escapes, closed by `"` + the same hash count
            hashes = tok.count("#")
            close = '"' + "#" * hashes
            e = text.find(close, j + len(tok))
            e = n if e < 0 else e + len(close)
            blank_str(j, e)
            pos = e
        else:                           # plain `"`, `b"`, `f"` — escape-aware
            k = j + len(tok)
            while k < n:
                if text[k] == "\\":
                    k += 2
                    continue
                if text[k] == '"':
                    break
                k += 1
            blank_str(j, k + 1)
            pos = k + 1
        pos = max(pos, m.end())
    return "".join(masked), attrs


# ── phase 2: token walk — segments, decls, mod-gates ───────────────────────

SCOPE_DECLS = {"file", "mod", "impl", "trait", "trait_impl"}
# Items declared in these scopes are never this class's finding: a trait's
# associated items are reachable through every implementor, and a trait-impl
# body's items are reachable through the trait. Both are collected (they still
# count as USES of other names) and then excluded as candidates.
NO_CANDIDATE_SCOPES = {"trait", "trait_impl"}


@dataclass
class Decl:
    file: Path
    name: str
    kind: str
    line: int
    ctx: Ctx                 # in-FILE context; the mod chain is ANDed later
    bare_pub: bool
    has_allow_dead: bool
    name_off: int
    scope: str = "file"
    stmt_start: int = 0      # use-decls: occurrences in [stmt_start, stmt_end)
    stmt_end: int = 0        #   are the import itself, not uses
    bin_file: bool = False


@dataclass
class ModGate:
    name: str
    ctx: Ctx                 # in-file ctx of the `mod name;` decl


@dataclass
class FileScan:
    path: Path
    segments: list           # [(start, end, Ctx)] incl. (0, n, Ctx())
    decls: list
    mod_gates: list
    file_allow_dead: bool
    ok: bool                 # False = unbalanced braces; reported, not hidden


def _split_top(s: str) -> list:
    """Comma-split at brace depth 0 (a `use` tree's siblings)."""
    out, cur, d = [], [], 0
    for ch in s:
        if ch == "{":
            d += 1
        elif ch == "}":
            d -= 1
        elif ch == "," and d == 0:
            out.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    tail = "".join(cur)
    if tail.strip():
        out.append(tail)
    return out


def use_leaf_names(body: str, prefix: str = "") -> list:
    """Names a `use <body>;` binds into scope.

    `use a::b::{c, d as e, self, *}` binds c, e and b — never `*` (a glob
    binds names this pass cannot enumerate, so the statement contributes no
    decl and the glob's members keep whatever decls they already had).
    """
    body = body.strip()
    if not body:
        return []
    d = 0
    brace_at = -1
    for i, ch in enumerate(body):
        if ch == "{":
            if d == 0:
                brace_at = i
                break
            d += 1
    if brace_at >= 0:
        d = 0
        end = len(body)
        for j in range(brace_at, len(body)):
            if body[j] == "{":
                d += 1
            elif body[j] == "}":
                d -= 1
                if d == 0:
                    end = j
                    break
        pre = body[:brace_at].strip().rstrip(":")
        last = pre.split("::")[-1].strip() if pre else prefix
        out = []
        for part in _split_top(body[brace_at + 1:end]):
            out.extend(use_leaf_names(part, last))
        return out
    m = re.search(r"\bas\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", body)
    if m:
        return [m.group(1)]
    seg = body.split("::")[-1].strip()
    if seg == "self":
        seg = prefix
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", seg):
        return []               # `*`, `crate`-only, or a shape we do not model
    return [seg]


def scan_file(path: Path, masked: str, attrs: list, bin_file: bool) -> FileScan:
    """One file's segments + decls + mod gates, in FILE-LOCAL context.

    The chain context (`mod` gates up the directory tree, plus the file's own
    leading `#![cfg]`) is ANDed in afterwards rather than threaded through as
    a `base`: `Ctx.and_` is associative and idempotent, so composing once at
    the end is identical to scanning per-chain — and it scans each file ONCE
    instead of once per resolution attempt.
    """
    n = len(masked)
    toks = [(m.start(), m.end(), m.group(0)) for m in TOKEN.finditer(masked)]
    outer = sorted((a for a in attrs if not a.inner), key=lambda a: a.end)
    file_allow = any("dead_code" in a.raw for a in attrs if a.inner)

    segs: list = [(0, n, Ctx())]
    decls: list = []
    mod_gates: list = []

    ctx_stack: list = [Ctx()]
    scope_stack: list = ["file"]
    open_marks: list = []
    header: dict = {}
    last_term_end = 0
    ok = True

    nl_at = [m.start() for m in re.finditer("\n", masked)]

    def line_of(off: int) -> int:
        lo, hi = 0, len(nl_at)
        while lo < hi:
            mid = (lo + hi) // 2
            if nl_at[mid] < off:
                lo = mid + 1
            else:
                hi = mid
        return lo + 1

    def closed_group_between(a_end: int, pos: int) -> bool:
        """The attribute sits inside a `(...)` group that closes before pos.

        A cfg on a fn PARAMETER (`fn f(#[cfg(wasm32)] s: S) {`) or on a
        tuple/call argument gates THAT element, never the block after the
        group. Measured: seal-remake viewer.rs `poll_and_apply`, one wasm32
        parameter attr, read the whole ~1050-line system body as wasm32-only
        and minted 5 false findings.
        """
        d = 0
        for ch in masked[a_end:pos]:
            if ch == "(":
                d += 1
            elif ch == ")":
                d -= 1
                if d < 0:
                    return True
        return False

    def pending_attr_ctx(pos: int, since: int) -> Ctx:
        ctx = Ctx()
        for a in outer:
            if a.end <= since or a.end > pos:
                continue
            if closed_group_between(a.end, pos):
                continue
            ctx = ctx.and_(pred_atoms(a.preds))
            if ctx.unmodelled:
                break
        return ctx

    def pending_allow(pos: int, since: int) -> bool:
        return any("dead_code" in a.raw for a in outer
                   if since < a.end <= pos and not closed_group_between(a.end, pos))

    def bare_pub_before(idx: int) -> bool:
        """A bare `pub` (not `pub(...)`) among this item's qualifiers."""
        k = idx - 1
        while k >= 0:
            w = toks[k][2]
            if w == "pub":
                return (toks[k + 1][2] if k + 1 < len(toks) else "") != "("
            if w in ("unsafe", "async", "extern", "default", "const", "crate"):
                k -= 1
                continue
            return False
        return False

    def impl_is_trait(idx: int) -> bool:
        """`impl Trait for T {` vs `impl T {` — a `for` before the body."""
        k, d = idx + 1, 0
        while k < len(toks):
            w = toks[k][2]
            if w == "<":
                d += 1
            elif w == ">":
                d -= 1
            elif w == "{":
                return False
            elif w == ";":
                return False
            elif w == "for" and d == 0:
                return True
            k += 1
        return False

    i, L = 0, len(toks)
    while i < L:
        s, e, w = toks[i]
        depth = len(open_marks)

        if w == "{":
            h = header.get(depth)
            ctx = ctx_stack[-1].and_(pending_attr_ctx(s, last_term_end))
            kind = "expr"
            if h is not None and not h.get("opened"):
                ctx = ctx.and_(h["attrs"])
                kind = h["scope_kind"]
                h["opened"] = True
            ctx_stack.append(ctx)
            scope_stack.append(kind)
            open_marks.append(s)
            last_term_end = e
            i += 1
            continue

        if w == "}":
            if open_marks:
                st = open_marks.pop()
                ctx = ctx_stack.pop()
                scope_stack.pop()
                h = header.get(len(open_marks))
                if h is not None and h.get("opened"):
                    # extend the segment back over the item HEADER: generics,
                    # parameters and the return type are governed by the
                    # item's own cfg, and `fn wait() -> ProcessStatus {…}`
                    # under `#[cfg(unix)]` is exactly the zed import specimen
                    st = min(st, h["start"])
                    header.pop(len(open_marks), None)
                segs.append((st, s + 1, ctx))
            else:
                ok = False                  # unbalanced: reported, not hidden
            last_term_end = e
            i += 1
            continue

        if w == ";":
            h = header.pop(depth, None)
            if h is not None and not h.get("opened"):
                ctx = ctx_stack[-1].and_(h["attrs"])
                if h["attrs"].atoms or h["attrs"].unmodelled:
                    # a BLOCKLESS item carries its own attrs over its own
                    # span: `#[cfg(unix)] const A: u8 = B;` gates the use of
                    # B, and without this segment that use reads as ungated
                    segs.append((h["start"], s + 1, ctx))
                if h["kind"] == "mod" and h.get("name"):
                    mod_gates.append(ModGate(h["name"], ctx))
                if h["kind"] == "use":
                    body = masked[h["stmt_start"]:s]
                    body = re.sub(r"^\s*use\b", "", body, count=1)
                    for nm in use_leaf_names(body):
                        decls.append(Decl(
                            file=path, name=nm, kind="use",
                            line=line_of(h["start"]), ctx=ctx,
                            bare_pub=h["pub"], has_allow_dead=h["allow"],
                            name_off=h["start"], scope=h["scope_at"],
                            stmt_start=h["start"], stmt_end=s + 1,
                            bin_file=bin_file))
            last_term_end = e
            i += 1
            continue

        if w in ITEM_KEYWORDS and scope_stack[-1] in SCOPE_DECLS:
            # `const` as a qualifier (`const fn` / `const unsafe fn`)
            if w == "const" and i + 1 < L and toks[i + 1][2] in FNISH_FOLLOWERS:
                i += 1
                continue
            attrs_ctx = pending_attr_ctx(s, last_term_end)
            h = {"kind": w, "attrs": attrs_ctx, "opened": False,
                 "pub": bare_pub_before(i),
                 "allow": pending_allow(s, last_term_end),
                 "start": s, "scope_at": scope_stack[-1]}
            header[depth] = h

            if w == "macro_rules":
                h["scope_kind"] = "expr"    # body is a token tree, not decls
                i += 1
                continue
            if w == "impl":
                h["scope_kind"] = "trait_impl" if impl_is_trait(i) else "impl"
                i += 1
                continue
            if w == "use":
                h["scope_kind"] = "mod"     # never opens a block
                h["stmt_start"] = s
                i += 1
                continue

            name_tok = None
            j = i + 1
            while j < L:
                ws_, _we_, ww_ = toks[j]
                if ww_ == "!":
                    break
                if re.match(r"^[A-Za-z_]", ww_):
                    if ww_ not in ("extern", "unsafe", "async"):
                        name_tok = (ws_, ww_)
                        break
                elif ww_ not in ("<", ">", "(", ")", "::", "*", "=", ":",
                                 ",", ".", "&", "const", "mut"):
                    break
                j += 1
            h["scope_kind"] = ("fn" if w == "fn"
                               else "mod" if w == "mod"
                               else "trait" if w == "trait"
                               else "expr")
            h["name"] = name_tok[1] if name_tok else None
            if name_tok and name_tok[1] != "_":
                decls.append(Decl(
                    file=path, name=name_tok[1], kind=w, line=line_of(s),
                    ctx=ctx_stack[-1].and_(attrs_ctx), bare_pub=h["pub"],
                    has_allow_dead=h["allow"], name_off=name_tok[0],
                    scope=scope_stack[-1], bin_file=bin_file))
            i += 1
            continue

        i += 1

    while open_marks:
        st = open_marks.pop()
        segs.append((st, n, ctx_stack.pop()))
        ok = False                          # unterminated block: same as above
    return FileScan(path, segs, decls, mod_gates, file_allow, ok)


# ── chain resolution across files ──────────────────────────────────────────


def unit_of(path: Path, repo: Path) -> tuple:
    try:
        rel = path.relative_to(repo)
    except ValueError:
        return ("root", repo)
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "crates":
        return (f"crate:{parts[1]}", repo / "crates" / parts[1])
    return ("root", repo)


def bin_file_p(path: Path, repo: Path) -> bool:
    try:
        parts = path.relative_to(repo).parts
    except ValueError:
        return False
    if parts and parts[0] in ("examples", "tests", "benches"):
        return True
    if len(parts) >= 3 and parts[0] == "crates" and parts[2] in (
            "examples", "tests", "benches"):
        return True
    return len(parts) >= 2 and parts[-2] == "bin"


def is_unit_root(path: Path, unit_dir: Path) -> bool:
    try:
        parts = path.relative_to(unit_dir).parts
    except ValueError:
        return False
    if parts in (("src", "lib.rs"), ("src", "main.rs"),
                 ("lib.rs",), ("main.rs",)):
        return True
    if parts and parts[0] in ("examples", "tests", "benches"):
        if len(parts) == 2 and parts[1].endswith(".rs"):
            return True
        if len(parts) == 3 and parts[2] == "main.rs":
            return True
    if len(parts) >= 3 and parts[0] == "src" and parts[1] == "bin":
        if len(parts) == 3 and parts[2].endswith(".rs"):
            return True
        if len(parts) == 4 and parts[3] == "main.rs":
            return True
    return False


def declaring_candidates(path: Path) -> tuple:
    """(module name, files that may carry its `mod <name>;` decl)."""
    d = path.parent
    if path.name == "mod.rs":
        mname = d.name
        return mname, [d.parent / "lib.rs", d.parent / "main.rs",
                       d.parent / "mod.rs", d.parent / f"{d.name}.rs"]
    mname = path.stem
    return mname, [d / "mod.rs", d.parent / f"{d.name}.rs",
                   d / "lib.rs", d / "main.rs"]


def resolve_chain(path: Path, unit_dir: Path, scans: dict) -> tuple:
    """(target-root file, chain Ctx) for `path`.

    Walks `mod name;` decls up the directory tree. A file whose declaring
    parent is not on disk resolves to ITSELF — divergence (b): the parentage
    is unknown, so the chain contributes nothing and the file reads as
    ungated, which can only HIDE a finding.
    """
    ctx = pred_atoms(leading_inner_cfgs(str(path)) or [])
    cur = path
    seen = {cur}
    for _ in range(32):
        if is_unit_root(cur, unit_dir):
            return cur, ctx
        mname, cands = declaring_candidates(cur)
        nxt = None
        for c in cands:
            sc = scans.get(c)
            if sc is None or c in seen:
                continue
            gate = next((g for g in sc.mod_gates if g.name == mname), None)
            if gate is None:
                continue
            ctx = ctx.and_(gate.ctx).and_(
                pred_atoms(leading_inner_cfgs(str(c)) or []))
            nxt = c
            break
        if nxt is None:
            return cur, ctx
        seen.add(nxt)
        cur = nxt
    return cur, ctx


# ── phase 3: occurrences + the finding rule ────────────────────────────────


def narrower(use_ctx: Ctx, decl_ctx: Ctx) -> set:
    """Platform atoms on the USE that the DECL's context does not carry.

    Empty set = not narrower. An unmodelled use context is never narrower —
    `any(...)` / `not(...)` may be live on the very platforms the declaration
    compiles for, so it DISQUALIFIES the declaration.
    """
    if use_ctx.unmodelled:
        return set()
    dm = decl_ctx.key_map()
    return {(k, v) for k, v in use_ctx.atoms if dm.get(k) != v}


def ctx_at_offsets(segments: list, offsets: list) -> dict:
    """Innermost segment context for each offset (one sorted sweep).

    Segments come from brace matching and are therefore properly NESTED, so
    the innermost containing segment is always the top of a stack built by
    pushing every segment whose start has been passed and popping from the
    top while its end has been passed.
    """
    out: dict = {}
    segs = sorted(segments, key=lambda t: (t[0], -t[1]))
    stack: list = []
    si = 0
    for off in offsets:
        while si < len(segs) and segs[si][0] <= off:
            stack.append(segs[si])
            si += 1
        while stack and stack[-1][1] <= off:
            stack.pop()
        out[off] = stack[-1][2] if stack else Ctx()
    return out


@dataclass
class Finding:
    repo: str
    rel: str
    line: int
    kind: str
    name: str
    uses: int
    atoms: set
    decl_ctx: Ctx


@dataclass
class RepoResult:
    repo: Path
    files: int = 0
    units: int = 0
    decls: int = 0
    candidates: int = 0
    occurrences: int = 0
    unparsed: int = 0
    vendored: int = 0
    findings: list = field(default_factory=list)
    mod_rows: list = field(default_factory=list)   # `mod` decls — see report()


# The population walk is ONE implementation, in `scripts/tracked_walk.py`
# (Issue 777). It moved out of this file because the rule it encodes — a
# population is what git TRACKS, not what the filesystem holds — had been
# landed here and in the trap audit and then NOT applied to the percentile and
# len-derived audits, where it went on to fabricate a floor and file a
# correctly-shaped finding against the wrong repository.
#
# These names are re-exported rather than inlined: this file's self-test
# perturbs `vendored_p` and asserts SKIP_DIRS' composition, and both arms must
# keep pointing at the definitions the walk actually uses.
#
# `vendor` is DELIBERATELY not in SKIP_DIRS: it belongs to `vendored_p`, and
# listing it in both made the self-test arm inert — the synthetic trees have no
# `.git`, so they took the rglob path and were filtered by this set no matter
# what `vendored_p` returned. A canary that passes under its own perturbation
# is certifying nothing.
#
# Vendored upstream code is not this workspace's to gate — the same rule and
# the same fork that forced it for the wasm32 surface audit (Issue 738 T3):
# riir-ai tracks a `wgpu-hal-30.0.0` fork whose gles backend is full of
# wasm32-only helpers, and every compile lane in that repo already derives
# its `-p` list with `grep -v '^vendor/'`. Excluded LOUDLY (the count rides
# the per-repo line) rather than silently, because a shrinking walk is how a
# report becomes a confident zero.
from tracked_walk import (  # noqa: E402  (sys.path is set at the top of this file)
    SKIP_DIRS,
    VENDOR_PARTS,
    tracked_files,
    vendored_p,
)

# Issue 804: this instrument is documented as directly invokable, and its
# verdict glyphs (✓ ✗ ⛔ ⚠) kill it on a non-UTF-8 console — no verdict at
# all, findings unread. docs_gate.sh's PYTHONIOENCODING only covers runs
# that go through the wrapper.
import console_safe  # noqa: E402

console_safe.apply()


def list_rs_files(root: Path) -> tuple:
    """(files, vendored_excluded) — tracked `*.rs` where git can answer.

    A thin seam over `tracked_walk.tracked_files` so this file's own
    perturbation arms have something to monkey-patch, and so the `.rs`
    pattern is stated once here rather than at every call site.
    """
    return tracked_files(root, "*.rs", vendored_p)


def source_of(p: Path, repo: Path, overlay: dict | None) -> str | None:
    """One file's bytes, taken from the OVERLAY when it names this path.

    Issue 822 T5b. A pin is a claim about the repo, and a repo's state is its
    COMMITS — but this walk reads a working tree that several concurrent
    sessions write into, so a finding may sit on a line no commit contains.
    `worktree_state.head_overlay()` answers `{repo-relative path: HEAD bytes}`
    for the dirty files, and this function is the ONE place a path becomes
    text, which is why the substitution belongs here and not at 1042 call
    sites inside the classifier.

    ⚠ `None` in the overlay is a VALUE, not an absence: it means "tracked but
    absent from HEAD" — a staged-but-never-committed file, Issue 822's
    measured case — and it must SKIP the file rather than fall through to the
    worktree copy, or another session's in-flight source is classified as
    committed. `in` and `.get()` say different things here; the test is `in`.
    An unreadable file gets the same answer for the same reason: there is
    nothing committed here to classify.
    """
    if overlay is not None:
        rel = str(p.relative_to(repo)).replace("\\", "/")
        if rel in overlay:
            return overlay[rel]
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def audit_repo(repo: Path, overlay: dict | None = None) -> RepoResult:
    """Classify one repo. With `overlay`, classify what HEAD would produce.

    The default is `None` — every existing caller is byte-identical, the
    `instrument_reachability_gate.reachable(read=)` precedent (Issue 822 T5a).
    """
    res = RepoResult(repo=repo)
    files, res.vendored = list_rs_files(repo)
    res.files = len(files)
    if not files:
        return res

    scans: dict = {}
    masks: dict = {}
    for p in files:
        text = source_of(p, repo, overlay)
        if text is None:
            continue
        masked, attrs = mask_file(text)
        sc = scan_file(p, masked, attrs, bin_file_p(p, repo))
        scans[p] = sc
        masks[p] = masked
        if not sc.ok:
            res.unparsed += 1

    # chain resolution → unit membership
    units: dict = defaultdict(list)
    chain: dict = {}
    for p in scans:
        _key, unit_dir = unit_of(p, repo)
        root, ctx = resolve_chain(p, unit_dir, scans)
        chain[p] = ctx
        units[root].append(p)
    res.units = len(units)

    for root, members in units.items():
        by_name: dict = defaultdict(list)
        for p in members:
            for d in scans[p].decls:
                by_name[d.name].append(d)
        res.decls += sum(len(v) for v in by_name.values())

        cands = []
        for name, ds in by_name.items():
            if len(ds) != 1:
                continue            # shadowed in-unit: unattributable, skipped
            d = ds[0]
            if d.has_allow_dead or scans[d.file].file_allow_dead:
                continue
            if d.name.startswith("_") or d.scope in NO_CANDIDATE_SCOPES:
                continue
            if d.bare_pub and not d.bin_file:
                continue            # externally reachable: never dead
            full = chain[d.file].and_(d.ctx)
            if full.unmodelled:
                continue            # cannot bound the declaration's platforms
            cands.append((d, full))
        if not cands:
            continue
        res.candidates += len(cands)

        tracked = {d.name for d, _ in cands}
        occ: dict = defaultdict(list)          # name -> [(file, off)]
        for p in members:
            masked = masks[p]
            hits = [(m.group(0), m.start()) for m in IDENT.finditer(masked)
                    if m.group(0) in tracked]
            if not hits:
                continue
            ctxs = ctx_at_offsets(scans[p].segments, [o for _, o in hits])
            pchain = chain[p]
            for nm, off in hits:
                occ[nm].append((p, off, pchain.and_(ctxs[off])))
        res.occurrences += sum(len(v) for v in occ.values())

        for d, full in cands:
            uses = [(p, off, c) for p, off, c in occ.get(d.name, [])
                    if not (p == d.file and off == d.name_off)
                    and not (p == d.file and d.stmt_end
                             and d.stmt_start <= off < d.stmt_end)]
            if not uses:
                continue            # no uses at all: plain dead_code, not ours
            atoms: set = set()
            for _p, _off, c in uses:
                nar = narrower(c, full)
                if not nar:
                    atoms = set()
                    break
                atoms |= nar
            if not atoms:
                continue
            row = Finding(
                repo=repo.name, rel=str(d.file.relative_to(repo)).replace("\\", "/"),
                line=d.line, kind=d.kind, name=d.name, uses=len(uses),
                atoms=atoms, decl_ctx=full)
            # MEASURED, not reasoned (2026-09-14, katgpt-types::simd on
            # wasm32): rustc reports dead code at the ITEM, never at the
            # `mod` declaration. `mod horizontal;` is ungated and every
            # reference to it is x86_64-gated, and a wasm32 `cargo check`
            # emits NOTHING — the module's own items are each x86_64-gated,
            # so it is EMPTY there. Appending one ungated fn to that same
            # file reproduces the warning, on the FN. A `mod` row is
            # therefore an upstream observation, never a rustc finding, and
            # folding it into the count would report a style nit at the
            # severity of a dead item.
            (res.mod_rows if d.kind == "mod" else res.findings).append(row)
    res.findings.sort(key=lambda f: (f.rel, f.line, f.name))
    res.mod_rows.sort(key=lambda f: (f.rel, f.line, f.name))
    return res


def fmt_ctx(c: Ctx) -> str:
    if c.unmodelled:
        return "unmodelled"
    if not c.atoms:
        return "ungated"
    return ", ".join(f"{k}={v}" for k, v in sorted(c.atoms))


def row_line(f: Finding, mark: str) -> str:
    return (f"      {mark} {f.rel}:{f.line}  {f.kind} {f.name}  "
            f"decl={fmt_ctx(f.decl_ctx)}  all {f.uses} use(s) under "
            f"{{{', '.join(f'{k}={v}' for k, v in sorted(f.atoms))}}}")


def report(results: list, verbose: bool) -> None:
    tot_f = tot_mod = tot_files = tot_units = tot_cand = tot_unparsed = 0
    for r in results:
        tot_files += r.files
        tot_units += r.units
        tot_cand += r.candidates
        tot_unparsed += r.unparsed
        tot_f += len(r.findings)
        tot_mod += len(r.mod_rows)
        if not r.files:
            continue
        head = (f"  {r.repo.name}: {r.files} file(s) walked · {r.units} unit(s) · "
                f"{r.candidates} candidate decl(s) · {len(r.findings)} finding(s)")
        if r.mod_rows:
            head += f" · {len(r.mod_rows)} MOD-REF"
        if r.vendored:
            head += f" · {r.vendored} vendored excluded"
        if r.unparsed:
            head += f" · {r.unparsed} UNPARSED"
        print(head)
        for f in r.findings:
            print(row_line(f, "✗"))
        for f in r.mod_rows:
            print(row_line(f, "· MOD-REF"))
        if verbose and not r.findings:
            print("      (no declaration has ALL of its uses platform-narrowed)")
    print()
    print(f"  floors — {tot_files} file(s) walked over {len(results)} repo(s); "
          f"{tot_units} compilation unit(s); {tot_cand} candidate declaration(s)")
    if tot_unparsed:
        print(f"  ⚠ {tot_unparsed} file(s) UNPARSED (unbalanced braces) — "
              f"their declarations are in the walk but their contexts are not "
              f"trustworthy; UNPARSED is the instrument admitting it cannot read")
    if tot_mod:
        print(f"  · {tot_mod} MOD-REF row(s) — a `mod` decl referenced only "
              f"from platform-gated code. NEVER folded into the finding "
              f"count: rustc reports dead code at the ITEM, and a module "
              f"whose own items are each gated is EMPTY rather than dead "
              f"(measured, katgpt-types::simd::horizontal on wasm32)")
    print(f"  {tot_f} platform-dead_code finding(s)")
    if not tot_cand:
        print("  ⛔ ZERO candidate declarations — that is an instrument "
              "failure, not a clean repo")


# ── validation ─────────────────────────────────────────────────────────────

# (label, {relpath: source}, expected finding names)
SELFTEST_CASES = [
    ("neon shape fires", {
        "src/lib.rs": '''
const NEON_U8: usize = 16;
#[cfg(target_arch = "aarch64")]
unsafe fn closed(b: &[u8]) -> bool { chunk(b, NEON_U8) }
'''}, {"NEON_U8"}),

    ("decl gated the same way is clean", {
        "src/lib.rs": '''
#[cfg(target_arch = "aarch64")]
const NEON_U8: usize = 16;
#[cfg(target_arch = "aarch64")]
unsafe fn closed(b: &[u8]) -> bool { chunk(b, NEON_U8) }
'''}, set()),

    ("one ungated use disqualifies", {
        "src/lib.rs": '''
const NEON_U8: usize = 16;
#[cfg(target_arch = "aarch64")]
unsafe fn closed(b: &[u8]) -> bool { chunk(b, NEON_U8) }
fn generic() -> usize { NEON_U8 }
'''}, set()),

    ("any(...) use is unmodelled, not narrower", {
        "src/lib.rs": '''
const WIDE: usize = 16;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn simd() -> usize { WIDE }
'''}, set()),

    ("not(...) use is unmodelled, not narrower", {
        "src/lib.rs": '''
const WIDE: usize = 16;
#[cfg(not(target_os = "windows"))]
fn posixy() -> usize { WIDE }
'''}, set()),

    ("a feature gate is not a platform atom", {
        "src/lib.rs": '''
const WIDE: usize = 16;
#[cfg(feature = "simd")]
fn simd() -> usize { WIDE }
'''}, set()),

    ("a cfg on a fn PARAMETER does not gate the body", {
        "src/lib.rs": '''
const STEP: usize = 16;
fn system(#[cfg(target_arch = "wasm32")] server: u8, n: usize) -> usize { n * STEP }
'''}, set()),

    ("an item cfg still gates the body past a parameter list", {
        "src/lib.rs": '''
const STEP: usize = 16;
#[cfg(target_arch = "wasm32")]
fn system(n: usize) -> usize { n * STEP }
'''}, {"STEP"}),

    ("bare `unix` is target_family and DOES narrow", {
        "src/lib.rs": '''
const SOCK: usize = 16;
#[cfg(unix)]
fn sock() -> usize { SOCK }
'''}, {"SOCK"}),

    ("pub in a lib file is externally reachable", {
        "src/lib.rs": '''
pub const NEON_U8: usize = 16;
#[cfg(target_arch = "aarch64")]
unsafe fn closed(b: &[u8]) -> bool { chunk(b, NEON_U8) }
'''}, set()),

    ("pub in an example file has no external consumer", {
        "examples/demo.rs": '''
pub const NEON_U8: usize = 16;
#[cfg(target_arch = "aarch64")]
unsafe fn closed(b: &[u8]) -> bool { chunk(b, NEON_U8) }
fn main() {}
'''}, {"NEON_U8"}),

    ("#[allow(dead_code)] is the author answering already", {
        "src/lib.rs": '''
#[allow(dead_code)]
const NEON_U8: usize = 16;
#[cfg(target_arch = "aarch64")]
unsafe fn closed(b: &[u8]) -> bool { chunk(b, NEON_U8) }
'''}, set()),

    ("a `mod` gate carries ACROSS files", {
        "src/lib.rs": '''
const KEYCHAIN: usize = 16;
#[cfg(target_os = "macos")]
mod plat;
''',
        "src/plat.rs": '''
fn store() -> usize { crate::KEYCHAIN }
'''}, {"KEYCHAIN"}),

    ("an UNGATED sibling module disqualifies across files", {
        "src/lib.rs": '''
const KEYCHAIN: usize = 16;
#[cfg(target_os = "macos")]
mod plat;
mod other;
''',
        "src/plat.rs": '''
fn store() -> usize { crate::KEYCHAIN }
''',
        "src/other.rs": '''
fn read() -> usize { crate::KEYCHAIN }
'''}, set()),

    ("a name declared twice in a unit is unattributable", {
        "src/lib.rs": '''
const NEON_U8: usize = 16;
mod inner;
#[cfg(target_arch = "aarch64")]
unsafe fn closed(b: &[u8]) -> bool { chunk(b, NEON_U8) }
''',
        "src/inner.rs": '''
const NEON_U8: usize = 32;
'''}, set()),

    ("an import whose only use is gated (ProcessStatus shape)", {
        "src/lib.rs": '''
use std::process::ProcessStatus;
#[cfg(unix)]
fn wait() -> ProcessStatus { todo!() }
'''}, {"ProcessStatus"}),

    ("a cfg_attr pred does NOT gate its item (seal-remake lighting_panel)", {
        "src/lib.rs": '''
use platform::FrameRateCap;
#[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
fn panel(mut cap: Option<FrameRateCap>) { let _ = &mut cap; }
#[cfg(target_arch = "wasm32")]
fn wasm_only(c: &FrameRateCap) { let _ = c; }
'''}, set()),

    ("a trait's associated items are never this finding", {
        "src/lib.rs": '''
pub trait Backend {
    fn width(&self) -> usize;
}
#[cfg(target_arch = "aarch64")]
fn probe(b: &dyn Backend) -> usize { b.width() }
'''}, set()),

    ("a BLOCKLESS item's own attr gates its uses", {
        "src/lib.rs": '''
const BASE: usize = 16;
#[cfg(target_os = "linux")]
const DOUBLE: usize = BASE * 2;
'''}, {"BASE"}),

    ("no uses at all is plain dead_code, not this class", {
        "src/lib.rs": '''
const NEVER_USED: usize = 16;
'''}, set()),

    ("a mention in a doc comment or a string is not a use", {
        "src/lib.rs": '''
const NEON_U8: usize = 16;
/// Uses NEON_U8 when it feels like it.
fn note() -> &'static str { "NEON_U8" }
#[cfg(target_arch = "aarch64")]
unsafe fn closed(b: &[u8]) -> bool { chunk(b, NEON_U8) }
'''}, {"NEON_U8"}),

    ("a different VALUE for the same key still narrows", {
        "src/lib.rs": '''
#[cfg(target_os = "linux")]
const WIDE: usize = 16;
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
fn simd() -> usize { WIDE }
'''}, {"WIDE"}),

    ("an inline format arg is an UNGATED use (riir-ai SWEEP_COUNTS)", {
        "examples/goat.rs": '''
const SWEEP_COUNTS: [usize; 3] = [10, 100, 1000];
fn main() { println!("Sweep sizes: {SWEEP_COUNTS:?}"); }
#[cfg(target_os = "macos")]
fn sweep() { for &n in &SWEEP_COUNTS { let _ = n; } }
'''}, set()),

    ("a plain string mention is still not a use", {
        "examples/goat.rs": '''
const SWEEP_COUNTS: [usize; 3] = [10, 100, 1000];
fn main() { println!("the SWEEP_COUNTS table"); }
#[cfg(target_os = "macos")]
fn sweep() { for &n in &SWEEP_COUNTS { let _ = n; } }
'''}, {"SWEEP_COUNTS"}),

    ("a vendored fork is not this workspace's to gate", {
        "vendor/wgpu-hal/src/gles/queue.rs": '''
fn extract_marker(s: &str) -> &str { s }
#[cfg(target_arch = "wasm32")]
fn draw(s: &str) -> &str { extract_marker(s) }
'''}, set()),

    ("a `mod` decl is a MOD-REF row, never a finding", {
        "src/lib.rs": '''
mod horizontal;
#[cfg(target_arch = "x86_64")]
fn widen() -> f32 { horizontal::reduce() }
''',
        "src/horizontal.rs": '''
#[cfg(target_arch = "x86_64")]
pub(super) fn reduce() -> f32 { 0.0 }
'''}, {"mod horizontal"}),

    ("a trait IMPL body's items are reachable through the trait", {
        "src/lib.rs": '''
struct T;
impl Iterator for T {
    type Item = u8;
    fn next(&mut self) -> Option<u8> { None }
}
#[cfg(unix)]
fn drive(t: &mut T) -> Option<u8> { t.next() }
'''}, set()),
]


def _write_tree(root: Path, tree: dict) -> None:
    for rel, src in tree.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
    (root / "Cargo.toml").write_text(
        '[package]\nname = "t"\nversion = "0.0.0"\n', encoding="utf-8")


def _case_tree(label: str) -> dict:
    """One `SELFTEST_CASES` tree, BY LABEL.

    The overlay arms below need a source that fires and one that does not, and
    copying them would be a second transcription of the classifier's own
    fixtures. A label that is gone RAISES rather than returning nothing, so the
    arm cannot quietly stop asserting when the table is edited.
    """
    for lab, tree, _expect in SELFTEST_CASES:
        if lab == label:
            return tree
    raise AssertionError(f"selftest case {label!r} is gone — the Issue 822 "
                         f"overlay arms anchor on it by name")


def overlay_arms() -> list[str]:
    """`source_of` / `audit_repo(overlay=)` — Issue 822 T5b's substitution.

    Every other arm in this file pins the CLASSIFIER. These pin the seam the
    drift sweep's ratchet reads through, which no classifier arm can reach:
    what a COMMIT of this repo would have produced, as opposed to what a
    working tree several sessions are editing says today.
    """
    fails: list[str] = []
    fires = _case_tree("neon shape fires")
    clean = _case_tree("decl gated the same way is clean")
    rel = "src/lib.rs"

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "ov"
        _write_tree(root, fires)
        p = root / rel

        # ── `source_of`, one answer at a time ─────────────────────────────
        if source_of(p, root, None) != fires[rel]:
            fails.append("source_of: no overlay must read the WORKTREE — the "
                         "default has to leave every existing caller alone")
        if source_of(p, root, {rel: clean[rel]}) != clean[rel]:
            fails.append("source_of: an overlay naming this path was ignored")
        # ⛔ `None` is a VALUE — tracked, absent from HEAD. Falling through to
        # the worktree copy here is exactly how another session's staged file
        # gets classified as a committed finding (Issue 822's measured case).
        if source_of(p, root, {rel: None}) is not None:
            fails.append("source_of: an overlay value of None fell through to "
                         "the worktree — `in` and `.get()` say different "
                         "things and this is the one that matters")
        if source_of(p, root, {"src/other.rs": clean[rel]}) != fires[rel]:
            fails.append("source_of: an overlay naming a DIFFERENT path "
                         "displaced this one")

        # ── and the verdict it inverts ────────────────────────────────────
        if not audit_repo(root).findings:
            fails.append("overlay arms: the firing fixture stopped firing — "
                         "every comparison below is vacuous")
        if audit_repo(root, overlay={rel: clean[rel]}).findings:
            fails.append("audit_repo(overlay=): HEAD's clean source still "
                         "produced a finding — an UNCOMMITTED row would be "
                         "adjudicated against the pin")
        # A staged-but-never-committed file contributes NOTHING to HEAD.
        if audit_repo(root, overlay={rel: None}).findings:
            fails.append("audit_repo(overlay=): a file absent from HEAD "
                         "produced a committed finding")

    with tempfile.TemporaryDirectory() as td:
        # ⛔ The SILENT direction, and it needs its own tree: HEAD carries a
        # row this worktree does not. Nothing in the worktree pass can see it,
        # so a sweep that adjudicated the worktree reports the repo clean.
        root = Path(td) / "masked"
        _write_tree(root, clean)
        if audit_repo(root).findings:
            fails.append("overlay arms: the clean fixture started firing")
        if not audit_repo(root, overlay={rel: fires[rel]}).findings:
            fails.append("audit_repo(overlay=): a MASKED row — committed, "
                         "hidden by the worktree — was not recovered")
    return fails


def selftest() -> int:
    """Pin the classifier in BOTH directions.

    Exit 2, never 1: an untrustworthy instrument is a different verdict from
    drift, and a report that cannot classify must not be read as `0 findings`.
    """
    bad = 0
    with tempfile.TemporaryDirectory() as td:
        for idx, (label, tree, expect) in enumerate(SELFTEST_CASES):
            root = Path(td) / f"c{idx}"
            _write_tree(root, tree)
            res = audit_repo(root)
            # MOD-REF rows carry a `mod ` prefix so one comparison pins the
            # bucket AND the row: an expectation that names a bare item can
            # never be satisfied by a MOD-REF sliding into the findings list.
            got = ({f.name for f in res.findings}
                   | {f"mod {f.name}" for f in res.mod_rows})
            mark = "✓" if got == expect else "✗"
            if got != expect:
                bad += 1
            print(f"  {mark} {label}")
            if got != expect:
                print(f"      expected {sorted(expect) or '[]'}, "
                      f"got {sorted(got) or '[]'}")
    # Issue 822 T5b — the HEAD-overlay seam. Reported in the same block and
    # counted into the same total, so a sweep that already refuses on
    # `audit.selftest()` picks these up with no change of its own.
    ov = overlay_arms()
    for f in ov:
        print(f"  ✗ {f}")
    bad += len(ov)
    if not ov:
        print("  ✓ the HEAD overlay substitutes, skips an absent file, and "
              "inverts a verdict in both directions")
    n_arms = len(SELFTEST_CASES) + 1
    print()
    print(f"  {n_arms - bad}/{n_arms} arm(s) pinned")
    if bad:
        print("  ⛔ classifier MISS — the report is not trustworthy")
        return 2
    return 0


def prove_fires(repo: Path, sha: str, name: str, scope: str) -> int:
    """Require the row to fire at `sha~1` and be gone at `sha`.

    A ceiling nobody has watched fail is a pin that certifies nothing, so the
    instrument is run against a tree whose answer is known independently —
    the commit that fixed the first specimen of the class.
    """
    rc = 0
    with tempfile.TemporaryDirectory() as td:
        for rev, want in ((f"{sha}~1", True), (sha, False)):
            out = Path(td) / rev.replace("~", "_").replace("^", "_")
            out.mkdir(parents=True, exist_ok=True)
            tar = out.with_suffix(".tar")
            r = subprocess.run(
                ["git", "-C", str(repo), "archive", "--format=tar",
                 "-o", str(tar), rev] + ([scope] if scope else []),
                capture_output=True)
            if r.returncode != 0:
                print(f"  ⛔ git archive {rev} failed: "
                      f"{r.stderr.decode('utf-8', 'replace').strip()}")
                return 2
            r = subprocess.run(["tar", "-xf", str(tar), "-C", str(out)],
                               capture_output=True)
            if r.returncode != 0:
                print(f"  ⛔ tar -xf failed for {rev}: "
                      f"{r.stderr.decode('utf-8', 'replace').strip()}")
                return 2
            res = audit_repo(out)
            got = {f.name for f in res.findings}
            hit = name in got
            ok = hit == want
            rc = rc or (0 if ok else 2)
            print(f"  {'✓' if ok else '✗'} {rev}: {name} "
                  f"{'PRESENT' if hit else 'absent'} "
                  f"(want {'PRESENT' if want else 'absent'}) — "
                  f"{res.files} file(s), {res.candidates} candidate(s), "
                  f"{len(res.findings)} finding(s), "
                  f"{len(res.mod_rows)} MOD-REF")
            if not res.candidates:
                print("      ⛔ zero candidates in the extracted tree — "
                      "the arm measured NOTHING")
                rc = 2
    if rc:
        print("  ⛔ --prove-fires did not reproduce the known answer")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(
        description="platform-dead_code audit: items whose every use is "
                    "behind a platform cfg the declaration lacks")
    ap.add_argument("repos", nargs="*", help="repo paths (default: derived)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--prove-fires", metavar="SHA",
                    help="require --fires-name to fire at SHA~1, not at SHA")
    ap.add_argument("--fires-name", default="NEON_U8")
    ap.add_argument("--fires-scope", default="crates/katgpt-pruners",
                    help="pathspec to extract (\"\" = the whole tree)")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent.parent

    if args.self_test:
        print("platform-dead_code audit — classifier self-test\n")
        return selftest()

    if args.prove_fires:
        print(f"platform-dead_code audit — --prove-fires {args.prove_fires}\n")
        rc = selftest()
        if rc:
            return rc
        print()
        return prove_fires(here, args.prove_fires, args.fires_name,
                           args.fires_scope)

    rc = selftest()
    if rc:
        return rc
    print()

    repos = ([Path(a).resolve() for a in args.repos] if args.repos
             else derive_repos(here.parent))
    print(f"platform-dead_code audit — {len(repos)} repo(s)\n")
    report([audit_repo(r) for r in repos], args.verbose)
    return 0                        # a REPORT: the verdict half is the owner's


if __name__ == "__main__":
    sys.exit(main())
