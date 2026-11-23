# Issue 1242 — claim_lease_check.py `--lint` crashes on Windows: the mandatory selftest's nested subprocess returns stdout=None (py3.14 reader-thread death), blinding every Windows lint call

**Status:** CLOSED 2026-11-24 — shikuwa BOX-VERIFIED the fix: pulled `580d0dd63` + `d44f05b72`, ran the exact reproducer `--lint .issues/132_....md` under BOTH py3.10 and py3.14 — both print `✓ lint ...: clean` (no traceback, no reader-thread death, the selftest preamble passes silently). The Windows pipe-reader/codec class is closed on the box that reported it. FIXED 2026-11-24 (zed-8fa4839b@amo, landed with the fix commit). Originally filed 2026-11-23 (zed-989683dc@4090, from the live reflex 132 lint calls during the A3-republish build). The FILE verdicts are unaffected (the file scans parse fine — verified: the 132 file's claim line lints clean through the file path when the selftest is bypassed); the crash is the SELFTEST half, which every `--lint` invocation runs FIRST, so **every `--lint` call on this box died before reading any file**. The lint works on amo/m3 (py3.10/3.13) — this is the Windows-py3.14 class.
**Lease closed:** zed-8fa4839b@amo, 1795470638, was expires 2026-11-25T12:00:00Z — T0–T1 landed same session (the two-sided root-cause fix + the reproduced-class probe; see the landing row below).

## The crash (reproduced every invocation)

```
py ../katgpt-rs/scripts/claim_lease_check.py --lint .issues/132_...
Exception in thread Thread-1 (_readerthread):
  File "...\Python\pythoncore-3.14-64\Lib\threading.py", line 1082, in _bootstrap_inner
  ...
  File "...\Lib\subprocess.py", line 1614, in _readerthread
    buffer.append(fh.read())
sys.exit(main())
  File "...\scripts\claim_lease_check.py", line 786, in main
    fails = selftest()
  File "...\scripts\claim_lease_check.py", line 609, in selftest
    if refused.returncode != 1 or "REFUSED" not in refused.stdout:
TypeError: argument of type 'NoneType' is not a container or iterable
```

- `selftest()` → `sp.run([sys.executable, script, ...], capture_output=True, encoding="utf-8")` — the NESTED process's stdout comes back `None` (the reader thread dies: the `_readerthread` traceback above is the nested run's own spawn), and `selftest()` then reads `refused.stdout` unguarded → TypeError. Note the nested process ITSELF crashed with an internal `_readerthread` exception — the thread death class, not a plain exit.
- Identical under py3.10 (the system launcher) — the selftest's nested `sp.run` dies the same way, so it is NOT a py3.14-only regression; the selftest has never passed on this box.
- Amo/m3 are unaffected (their py builds run the selftest clean — every fleet lint call elsewhere works).

## Why it matters

- Every Windows-box `--lint` call dies → the lease scanner is UNREACHABLE from this box's lanes (the file verdicts parse fine when the call is reconstructed, but no Windows lane can run the tool's own lint gate at all).
- The blind-the-scanner hazard (the katgpt-rs Issue-1226 class) is the tool's own reason to exist — a tool whose lint gate cannot run on a fleet rig is the same failure one layer up.

## The fix sketch

- Guard `refused.stdout` for None in `selftest()` (loud `fails.append` naming the platform) — the minimal unblock; OR
- Skip the nested-process selftest when `sys.platform == "win32"` and the nested run's own spawn is the crasher, with a loud "selftest unavailable on this platform" verdict (never a silent pass); OR
- The real fix: find why the nested `sp.run` dies on this box (py3.14's `_readerthread` + Windows console encoding — plausibly the cp874/cp1252 codec the console hands back, the same class the fleet's other Windows scripts hit).

## Repro

Any `--lint <file>` call on shikuwa (Windows 11, py3.10.11 system + py3.14.6 launcher, Git Bash sh). The file verdicts themselves are irrelevant — the crash is in the selftest preamble.

## FIXED 2026-11-24 (zed-8fa4839b@amo) — the two-sided root cause, reproduced then fixed

**Root cause (both halves, the corpus's own `subprocess_encoding_gate` class):** (1) CHILD-ENCODER — the nested child printed the tool's ✓/✗/⚠ glyphs through the console-locale codec (cp874 on the Thai-locale box), so the pipe carried non-UTF-8 bytes; (2) DECODE — the parent's strict `encoding="utf-8"` reader thread died decoding them → `stdout=None` → the unguarded `refused.stdout` read crashed with TypeError. Reproduced the mechanism on amo (py3.9): a child emitting a truncated UTF-8 lead byte (`\xd1`) makes the strict parent RAISE `UnicodeDecodeError` in `communicate()` (on the py3.14-Windows build the same failure shape is the reader-thread death → stdout=None — the issue's traceback class); with `errors="replace"` the same bytes decode clean (`rc=0`, stdout never None).

**The fix (`scripts/claim_lease_check.py`, one commit):**
- selftest's nested spawn: `PYTHONIOENCODING=utf-8` into the child env (the child always writes UTF-8 regardless of console locale) + `errors="replace"` on the capture + the `run()` helper now returns a typed `(ok, rc, out)` triple — a None stdout (reader-thread death for ANY reason) lands as a LOUD named selftest failure ("nested run lost stdout — the console-locale decode class"), never a TypeError crash, never a silent pass;
- `fetch_head_age`'s git spawn: `errors="replace"` + a None-stdout typed skip (git's own localized output is the child-side byte source on the same box class).

**Verification:** selftest PASS + prove-fires PASS + a real lint clean through the fixed script; the four scripts-scanning gates green (subprocess-encoding 0 findings incl. 0 unpinned python children, python-syntax-floor 3.9 parse clean, console-encoding, locale-io); the end-to-end `--item` freshness run clean (holder-own-epoch posture correct). ⚠ The box-verified half (a real shikuwa lint lane) is the filer's re-run — the mechanism fix is proven here, the platform confirmation pending.
