# Vendored fork of `block` 0.1.6 (katgpt-rs issue 968)

Upstream: http://github.com/SSheldon/rust-block — ABANDONED (last release
2021-01; `block` 0.1.6 is the terminal registry version). MIT, preserved.

## Why this fork exists

`block` 0.1.6 declares `enum Class { }` (an uninhabited type) and uses it as
the type of `extern static _NSConcreteStackBlock`. rustc flags that as
`static of uninhabited type` (rust-lang/rust#74840) — a future-incompat that
becomes a HARD ERROR in a future release. Every `cargo build` in this
workspace that pulls the crate prints the warning today.

The static is only ever used for its ADDRESS (the Objective-C block ABI's
`_NSConcreteStackBlock` class pointer that stack blocks point into); the value
is never read. Upstream `objc2`-ecosystem crates fixed the same shape by using
an inhabited opaque type. This fork makes the identical minimal change:

```diff
-enum Class { }
+#[repr(C)]
+struct Class { _opaque: [u8; 0] }
```

That is the ONLY delta vs 0.1.6. All three `Class` use sites (the definition,
the extern static, the `isa: *const Class` ABI field) compile unchanged; the
ABI layout is untouched (pointer target is opaque either way).

## How it is wired

Root `Cargo.toml`:

```toml
[patch.crates-io]
block = { path = "vendor/block" }
```

## Removal condition (delete the whole patch when)

1. `katgpt-backend` migrates `metal` -> `objc2-metal` (the real exit — the
   legacy `block`+`objc` family leaves the graph), OR
2. upstream ever ships a fixed release (none as of 2026-10-27).

Note: a `[patch.crates-io]` entry applies per-workspace. Other repos that
consume `metal` (today: riir-train via the katgpt-rs dev-dep chain) need the
same patch in their own workspace to clear the warning from their builds.
