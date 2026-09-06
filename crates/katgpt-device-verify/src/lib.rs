//! # katgpt-device-verify — `no_std` device-side verification primitives
//!
//! The receiving side of `riir-chain` Issues 108 (fair-roll verify) and 109
//! (Merkle verify), filed against this repo as Issue 685. Design:
//! the ESP32 device-tier proposal (moved 2026-09-06 to the private POC repo
//! `riir-esp32`: `gist-rs/riir-esp32`, `.proposals/006_esp32_device_tier_ws_fallback.md`)
//! §9.3, §10 Q6/Q7.
//!
//! ## Why this crate exists
//!
//! A `Satellite` — an ESP32-class MCU or a browser mini-game at vote weight 0 —
//! must **independently verify** what a higher node hands it: the daily gacha
//! die it was dealt, and the Merkle root it is asked to attest. Both primitives
//! already existed, and both were reachable only through a std-only workspace
//! that fails to compile for a bare-metal target at its *first* dependency
//! (`error[E0463]: can't find crate for std` in `rustc-hash`, with zero
//! features enabled — measured 2026-08-24 on `riscv32imc-unknown-none-elf`).
//!
//! ## The invariant that matters more than the location
//!
//! **One implementation, consumed — never copied.** Two implementations of the
//! rejection-sampling threshold *will* drift, and when they do **every honest
//! claim looks like fraud**: device and node compute different items from the
//! same seed, indistinguishable from cheating.
//!
//! So the deliverable here is not the code — it is [`vectors`], the pinned
//! cross-target fixture. The code is the easy half. Every consumer asserts the
//! same table; a green test on one side proves nothing about the other.
//!
//! ## What is bit-identical to what
//!
//! | Here | Upstream-of-record | Consumed by |
//! |---|---|---|
//! | [`fair_roll::FairRollVerifier::roll_die`] | `riir-chain::split_key::FairRng::roll_die` | the daily-claim loop |
//! | [`merkle_verify::compute_root_from_proof`] | `riir_neuron_db::merkle::compute_root_from_proof` | `riir-chain::catchup::merkle` (a re-export) |
//!
//! Note the second row: `riir-chain` is **not** the home of the binary Merkle
//! tree — it re-exports `riir-neuron-db`'s. Issue 109 named `riir-chain` as the
//! consuming side; the actual owner is `riir-neuron-db`, and that is where the
//! vector assertion has to land to be worth anything.
//!
//! ## Discipline
//!
//! - `#![no_std]` unconditionally. `alloc` is a feature, and nothing on the
//!   verify path needs it.
//! - **No `getrandom`, ever.** Verification consumes entropy someone else
//!   produced; it never generates any. The `riir-wallet-signer` precedent
//!   deliberately has no entropy source either.
//! - Panic-free on the verify path — every fallible input is a `checked_`
//!   variant returning [`Option`]. On an MCU a panic is a reset.

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "fair_roll")]
pub mod fair_roll;

#[cfg(feature = "merkle_verify")]
pub mod merkle_verify;

pub mod vectors;

/// A BLAKE3 hash — fixed 32 bytes.
///
/// Same layout as the `[u8; 32]` used for `NeuronShard.commitment`,
/// `SyncBlock.commitment`, and the combined fair-roll seed, so no conversion
/// is needed at any seam.
pub type Hash = [u8; 32];

/// BLAKE3 hash output size in bytes.
pub const HASH_SIZE: usize = 32;
