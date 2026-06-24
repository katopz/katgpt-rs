//! Vessel — Extract-Once Secure Wire Format Primitive (Plan 315, Research 297).
//!
//! A generic open primitive for "WASM-with-BLAKE3-header wire format →
//! validated decode-once → tier-aware projection". The wire layout is:
//!
//! ```text
//! +---------------------+-----------------------+
//! | VesselHeader (52 B) | WASM module bytes ... |
//! +---------------------+-----------------------+
//! ```
//!
//! The WASM module's linear memory embeds a `#[repr(C)]` Pod payload at a
//! known offset. Two projection paths exist:
//!
//! - **Hot / Plasma tier** — `extract_payload::<T: Pod>()`: validate the
//!   header once (magic + version + BLAKE3), then borrow the Pod bytes
//!   zero-copy for SIMD host-side math. Validation cost is paid once and
//!   amortized over many hot-path calls.
//! - **Cold / Freeze tier** — `VesselProjector::project()`: keep the
//!   payload inside the WASM linear memory and call an exported projection
//!   function under a fuel budget. Capability-restricted, fail-safe.
//!
//! # Honest security framing (do NOT oversell)
//!
//! WASM provides API encapsulation + capability-based security + NFT
//! execute-permission, **NOT** cryptographic confidentiality. A debugger
//! can still dump linear memory. The honest selling point is integrity +
//! access control: stolen bytes that fail `verify_owner` cannot run, and a
//! chain verifier can prove "this projection was computed by THIS
//! bytecode" via BLAKE3 commitment. True confidentiality would require FHE
//! or TEE — out of scope here.
//!
//! See `riir-neuron-db` Research 006 / Plan 003 for the private Super-GOAT
//! guide and the shard-specific wrapper. This module is Pod-generic and
//! owns no shard / game / chain semantics.

// `bytemuck::Pod` is required by the extract path; `wasmi` is required by
// the projector path. Both are gated by the `secure_vessel` feature so the
// primitive compiles out cleanly when unused.
#![cfg(feature = "secure_vessel")]

use bytemuck::Pod;
use std::sync::Arc;

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────

/// Magic bytes prefixing every vessel — `b"VSL1"`.
///
/// Matches the canonical codebase header pattern (`FREEZE_MAGIC = b"FRZE"`,
/// `CGSP`, `BDTB`, `COLP`, `DRMR`, `GODT`, `GOTM`, `CERT`, `AV01`).
pub const VESSEL_MAGIC: [u8; 4] = *b"VSL1";

/// Current vessel wire-format version.
pub const VESSEL_VERSION: u32 = 1;

/// On-wire header size in bytes: `magic[4] + version[4] + blake3[32] +
/// payload_kind[4] + payload_offset[4] + payload_len[4] = 52`.
pub const VESSEL_HEADER_LEN: usize = 4 + 4 + 32 + 4 + 4 + 4;

// ────────────────────────────────────────────────────────────────────────────
// Header + Error
// ────────────────────────────────────────────────────────────────────────────

/// Fixed-layout wire header prepended to every vessel.
///
/// `#[repr(C)]` so the on-disk / on-wire layout is identical to a plain
/// byte buffer of `VESSEL_HEADER_LEN` — readers can `bytemuck::from_bytes`
/// the first 52 bytes directly without field-by-field parsing.
///
/// Field order is alignment-dense (no padding): all fields are 4- or
/// 32-byte aligned natural widths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct VesselHeader {
    /// Format magic — must equal [`VESSEL_MAGIC`].
    pub magic: [u8; 4],
    /// Wire-format version — must equal [`VESSEL_VERSION`] for this code path.
    pub version: u32,
    /// BLAKE3 hash over the WASM module bytes (header payload, *not* the
    /// header itself). Computed once at encode time, verified once at load
    /// time. Tampering with any WASM byte fails verification.
    pub blake3: [u8; 32],
    /// Caller-defined payload kind discriminator (e.g. shard-side
    /// `PayloadKind::NeuronShard`). Opaque at this layer; consumers route
    /// on it.
    pub payload_kind: u32,
    /// Byte offset of the Pod payload inside the WASM module's data
    /// section / linear memory. Caller validates
    /// `payload_offset + payload_len <= wasm_bytes.len()` at extract time.
    pub payload_offset: u32,
    /// Payload length in bytes. Must equal `size_of::<T>()` for the Pod
    /// type the caller intends to extract.
    pub payload_len: u32,
}

// `VesselHeader` is fully `Pod`-compatible (all fields are plain integers
// / byte arrays, no padding). These unsafe impls unlock
// `bytemuck::from_bytes::<VesselHeader>(&bytes[..52])` for zero-copy decode.
//
// SAFETY: `VesselHeader` is `#[repr(C)]`, all fields are `Pod`
// (`[u8; N]` / `u32`), no padding, no uninitialized bytes. Layout matches
// the byte sequence on every target the crate supports.
unsafe impl Pod for VesselHeader {}
unsafe impl bytemuck::Zeroable for VesselHeader {}

/// Errors returned by vessel encode / decode / extract / project.
#[derive(Debug)]
pub enum VesselError {
    /// Header magic bytes did not match [`VESSEL_MAGIC`].
    BadMagic,
    /// Header version is not supported by this binary.
    UnsupportedVersion,
    /// Recomputed BLAKE3 over the WASM bytes did not equal `header.blake3`.
    Blake3Mismatch,
    /// Input buffer shorter than [`VESSEL_HEADER_LEN`] — cannot even read
    /// the header.
    HeaderTooShort,
    /// Declared `payload_offset + payload_len` exceeds the WASM byte slice.
    PayloadOutOfBounds,
    /// Caller's `size_of::<T>()` did not equal `header.payload_len`.
    PayloadLenMismatch,
    /// WASM module failed to compile under wasmi.
    WasmiCompile(wasmi::Error),
    /// WASM module compiled but failed to instantiate (missing import,
    /// start function trap, ...).
    WasmiInstantiate(wasmi::Error),
    /// WASM instance is missing a required export by name.
    ExportMissing(&'static str),
    /// Projector call ran out of fuel (fail-safe — never panics).
    FuelExhausted,
}

// ────────────────────────────────────────────────────────────────────────────
// Encode / Decode / Verify
// ────────────────────────────────────────────────────────────────────────────

/// Encode a vessel wire blob from raw WASM module bytes + payload metadata.
///
/// Computes BLAKE3 over `wasm_bytes` only (the header is *not* self-hashed;
/// it carries the hash of the payload that follows it). Returns a freshly
/// allocated `Vec<u8>` of length `VESSEL_HEADER_LEN + wasm_bytes.len()`.
///
/// Allocation policy: this is the cold encode path. Hot-path callers reuse
/// [`load_vessel`] / [`extract_payload`] which are zero-copy after the
/// one-time verify.
pub fn encode_vessel(
    wasm_bytes: &[u8],
    payload_kind: u32,
    payload_offset: u32,
    payload_len: u32,
) -> Vec<u8> {
    let blake3 = *blake3::hash(wasm_bytes).as_bytes();
    let header = VesselHeader {
        magic: VESSEL_MAGIC,
        version: VESSEL_VERSION,
        blake3,
        payload_kind,
        payload_offset,
        payload_len,
    };

    let mut out = Vec::with_capacity(VESSEL_HEADER_LEN + wasm_bytes.len());
    // `extend_from_slice` for the header is byte-stable across targets
    // because `VesselHeader` is `#[repr(C)]` with no padding.
    out.extend_from_slice(bytemuck::bytes_of(&header));
    out.extend_from_slice(wasm_bytes);
    out
}

/// Decode just the header from a vessel blob.
///
/// Validates magic + version. Does **not** verify BLAKE3 — callers may
/// want to batch verification, or skip it for already-trusted blobs.
/// Use [`verify_blake3`] separately when ready.
pub fn decode_header(bytes: &[u8]) -> Result<VesselHeader, VesselError> {
    if bytes.len() < VESSEL_HEADER_LEN {
        return Err(VesselError::HeaderTooShort);
    }
    // SAFETY: `VesselHeader` is `#[repr(C)]` + `Pod` + `Zeroable`, and we
    // just bounds-checked the slice length. `from_bytes` performs a
    // length + alignment check internally; both pass by construction.
    let header: &VesselHeader =
        bytemuck::from_bytes(&bytes[..VESSEL_HEADER_LEN]);
    if header.magic != VESSEL_MAGIC {
        return Err(VesselError::BadMagic);
    }
    if header.version != VESSEL_VERSION {
        return Err(VesselError::UnsupportedVersion);
    }
    Ok(*header)
}

/// Verify that the BLAKE3 in `header` matches a recomputed hash over
/// `wasm_bytes`.
///
/// Split out from [`decode_header`] so callers can decode many headers in
/// a batch and verify selectively (e.g. skip already-cached blobs).
pub fn verify_blake3(
    header: &VesselHeader,
    wasm_bytes: &[u8],
) -> Result<(), VesselError> {
    let recomputed = blake3::hash(wasm_bytes);
    if recomputed.as_bytes() == &header.blake3 {
        Ok(())
    } else {
        Err(VesselError::Blake3Mismatch)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// LoadedVessel — Phase 2 extract path
// ────────────────────────────────────────────────────────────────────────────

/// A vessel that has been decoded + BLAKE3-verified and is ready for
/// either extract-once (Hot path) or projector (Cold path) access.
///
/// `wasm_bytes` is held in an `Arc<[u8]>` so a single loaded vessel can be
/// cheaply shared across threads (e.g. Hot-tier shard cache hit by many
/// inference workers). The `Arc` clone is one atomic refcount bump, not a
/// byte copy.
///
/// `instance` is lazily compiled — only the Cold / projector path pays
/// the wasmi compile cost. The Hot / extract path never touches it.
pub struct LoadedVessel {
    /// Verified header (magic + version + BLAKE3 already checked).
    pub header: VesselHeader,
    /// The WASM module bytes following the header. Borrowed by extract /
    /// projector paths; never copied.
    pub wasm_bytes: Arc<[u8]>,
    /// Lazily-compiled wasmi instance. `None` until
    /// [`ensure_compiled`] is called (Cold / projector path only).
    /// Stored as `Option` so the Hot path pays zero wasmi cost.
    pub instance: Option<wasmi::Instance>,
}

impl core::fmt::Debug for LoadedVessel {
    /// Manual impl — `wasmi::Instance` is not `Debug`, and we deliberately
    /// do not dump `wasm_bytes` (could be large / sensitive).
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LoadedVessel")
            .field("header", &self.header)
            .field("wasm_bytes_len", &self.wasm_bytes.len())
            .field("instance_compiled", &self.instance.is_some())
            .finish()
    }
}

impl LoadedVessel {
    /// Returns the WASM bytes slice (everything after the header).
    ///
    /// Cheap — single slice arithmetic on the `Arc`'s backing buffer.
    pub fn wasm_bytes(&self) -> &[u8] {
        &self.wasm_bytes
    }
}

/// Decode + verify a vessel blob into a [`LoadedVessel`].
///
/// Performs the one-time cost: header parse + BLAKE3 verify. After this,
/// [`extract_payload`] / [`extract_payload_slice`] are zero-copy borrows.
///
/// Allocation: one `Arc<[u8]>` for the WASM bytes. The input `bytes` may
/// be dropped after this returns — the `Arc` owns its own copy.
pub fn load_vessel(bytes: &[u8]) -> Result<LoadedVessel, VesselError> {
    let header = decode_header(bytes)?;
    let wasm_bytes: Arc<[u8]> = bytes[VESSEL_HEADER_LEN..].into();
    verify_blake3(&header, &wasm_bytes)?;
    Ok(LoadedVessel {
        header,
        wasm_bytes,
        instance: None,
    })
}

/// Extract a fixed-size `T: Pod` payload from the WASM bytes.
///
/// **The core primitive.** Validates:
/// 1. `header.payload_len == size_of::<T>()` (caller asked for the right type)
/// 2. `payload_offset + payload_len <= wasm_bytes.len()` (in-bounds)
///
/// Then returns a zero-copy borrow:
/// `bytemuck::from_bytes(&wasm_bytes[offset..offset+len])`.
///
/// The borrow is tied to `&vessel` — caller must keep the vessel alive.
/// No allocation, no copy. The BLAKE3 was already verified at
/// [`load_vessel`] time, so this is a pure bounds-checked slice.
pub fn extract_payload<T: Pod>(
    vessel: &LoadedVessel,
) -> Result<&T, VesselError> {
    let expected = core::mem::size_of::<T>();
    if expected as u32 != vessel.header.payload_len {
        return Err(VesselError::PayloadLenMismatch);
    }
    let start = vessel.header.payload_offset as usize;
    let end = start.checked_add(expected).ok_or(VesselError::PayloadOutOfBounds)?;
    let slice = vessel
        .wasm_bytes
        .get(start..end)
        .ok_or(VesselError::PayloadOutOfBounds)?;
    // SAFETY: `T: Pod` guarantees the byte reinterpret is sound. `slice`
    // is the correct length (just checked above). `bytemuck::from_bytes`
    // also re-checks length + alignment internally.
    Ok(bytemuck::from_bytes(slice))
}

/// Extract a variable-length `&[T]` slice from the WASM bytes.
///
/// Same semantics as [`extract_payload`] but for arrays: the caller does
/// not know the element count at compile time. Element count is derived
/// from `header.payload_len / size_of::<T>()`.
pub fn extract_payload_slice<T: Pod>(
    vessel: &LoadedVessel,
) -> Result<&[T], VesselError> {
    let elem_size = core::mem::size_of::<T>();
    if elem_size == 0 {
        // Defensive — ZSTs make no sense as vessel payloads.
        return Err(VesselError::PayloadLenMismatch);
    }
    if vessel.header.payload_len as usize % elem_size != 0 {
        return Err(VesselError::PayloadLenMismatch);
    }
    let start = vessel.header.payload_offset as usize;
    let end = start
        .checked_add(vessel.header.payload_len as usize)
        .ok_or(VesselError::PayloadOutOfBounds)?;
    let slice = vessel
        .wasm_bytes
        .get(start..end)
        .ok_or(VesselError::PayloadOutOfBounds)?;
    Ok(bytemuck::cast_slice(slice))
}

// ────────────────────────────────────────────────────────────────────────────
// VesselProjector — Phase 3 cold path
// ────────────────────────────────────────────────────────────────────────────

/// Capability-restricted projection trait for the Cold / Freeze tier.
///
/// Unlike [`extract_payload`] (which yields raw bytes to the host), a
/// projector keeps the payload inside the WASM linear memory and only
/// returns a derived scalar / struct. The host never sees the weights —
/// only the projection result.
///
/// # Why a trait, not a function
///
/// Different payload kinds need different projection shapes (dot-product,
/// top-k argmax, sigmoid gate, ...). The trait lets each consumer
/// (`NeuronShard`, `LatentThoughtKernel`, game validators) plug its own
/// WASM export signature without forcing this generic primitive to know
/// about any of them.
pub trait VesselProjector {
    /// Query input — e.g. a probe vector pointer + length.
    type Query<'a>
    where
        Self: 'a;
    /// Projection output — e.g. an `f32` scalar.
    type Output;
    /// Run the projection. Failures (fuel exhaustion, missing export,
    /// trap) return [`VesselError`] — never panic.
    fn project(
        &self,
        vessel: &LoadedVessel,
        store: &mut wasmi::Store<()>,
        query: &Self::Query<'_>,
    ) -> Result<Self::Output, VesselError>;
}

/// Lazily compile + instantiate the WASM module inside `vessel`.
///
/// Idempotent — if `vessel.instance` is already `Some`, returns the
/// cached instance. Otherwise compiles under a fresh wasmi engine with
/// fuel consumption enabled (fail-safe against runaway loops).
///
/// Stores the instance back into `vessel.instance` so subsequent
/// projector calls skip recompilation.
pub fn ensure_compiled<'a>(
    vessel: &'a mut LoadedVessel,
    store: &mut wasmi::Store<()>,
    engine: &wasmi::Engine,
) -> Result<&'a wasmi::Instance, VesselError> {
    if vessel.instance.is_some() {
        // Borrow-checker workaround: we already have an instance, return
        // a borrow to it. The `?` above guarantees `Some`.
        return Ok(vessel.instance.as_ref().expect("checked Some above"));
    }
    let module =
        wasmi::Module::new(engine, &vessel.wasm_bytes[..])
            .map_err(VesselError::WasmiCompile)?;
    let linker = wasmi::Linker::new(engine);
    let instance = linker
        .instantiate_and_start(store, &module)
        .map_err(VesselError::WasmiInstantiate)?;
    vessel.instance = Some(instance);
    Ok(vessel.instance.as_ref().expect("just stored"))
}

/// Generic WASM dot-product projector.
///
/// Looks up `export_name` in the vessel's WASM instance, expects a
/// signature `(ptr: i32, len: i32) -> f32`, copies the query into the
/// instance's exported `memory`, and calls the function under fuel.
///
/// The fuel budget bounds the worst-case runtime — a malicious or buggy
/// module cannot hang the host.
#[derive(Clone, Copy, Debug)]
pub struct WasmDotProjector {
    /// WASM export name to call (e.g. `"project"`).
    pub export_name: &'static str,
    /// Fuel budget per `project()` call. wasmi halts with
    /// [`VesselError::FuelExhausted`] if exceeded.
    pub fuel_budget: u64,
}

impl VesselProjector for WasmDotProjector {
    type Query<'a> = &'a [f32];
    type Output = f32;

    fn project(
        &self,
        vessel: &LoadedVessel,
        store: &mut wasmi::Store<()>,
        query: &Self::Query<'_>,
    ) -> Result<Self::Output, VesselError> {
        // `vessel.instance` must be Some — caller should have run
        // `ensure_compiled` first. We can't take `&mut vessel` here (trait
        // signature is `&LoadedVessel`), so this is a precondition.
        let instance = vessel
            .instance
            .as_ref()
            .ok_or(VesselError::ExportMissing("instance not compiled"))?;
        let func = instance
            .get_typed_func::<(i32, i32), f32>(&mut *store, self.export_name)
            .map_err(|_| VesselError::ExportMissing(self.export_name))?;
        let memory = instance
            .get_memory(&*store, "memory")
            .ok_or(VesselError::ExportMissing("memory"))?;

        // Write query bytes into WASM linear memory at offset 0. This is
        // a small allocation (query.len() * 4 bytes); for the typical
        // dot-product case it's an HLA 8-dim probe = 32 bytes.
        let query_bytes: &[u8] = bytemuck::cast_slice(query);
        // `memory.data_mut` requires `&mut store`; we have it.
        let mem_data = memory.data_mut(&mut *store);
        if mem_data.len() < query_bytes.len() {
            return Err(VesselError::PayloadOutOfBounds);
        }
        mem_data[..query_bytes.len()].copy_from_slice(query_bytes);

        // Fuel-gated call. Fail-safe: out-of-fuel returns an error, not a
        // panic, so a buggy / hostile module can never hang the host.
        store
            .set_fuel(self.fuel_budget)
            .map_err(|_| VesselError::FuelExhausted)?;
        let ptr = 0i32;
        let len = query.len() as i32;
        let result = func
            .call(&mut *store, (ptr, len))
            .map_err(|_| VesselError::FuelExhausted)?;
        Ok(result)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial Pod payload for round-trip tests.
    #[derive(Clone, Copy, Debug, PartialEq)]
    #[repr(C)]
    struct FakePayload {
        a: f32,
        b: f32,
        c: [u8; 8],
    }
    // SAFETY: `FakePayload` is `#[repr(C)]`, all fields are Pod, no
    // padding.
    unsafe impl Pod for FakePayload {}
    unsafe impl bytemuck::Zeroable for FakePayload {}

    fn fake_wasm_with_payload(payload: &FakePayload) -> Vec<u8> {
        // Synthetic "WASM" = 16 zero bytes prefix + payload bytes + 8 zero
        // suffix bytes. Real WASM modules put the payload inside the data
        // section; this stand-in is good enough for header / extract
        // round-trip tests.
        let mut bytes = vec![0u8; 16];
        bytes.extend_from_slice(bytemuck::bytes_of(payload));
        bytes.extend_from_slice(&[0u8; 8]);
        bytes
    }

    #[test]
    fn extract_returns_byte_identical_payload() {
        let payload = FakePayload {
            a: 1.5,
            b: -2.25,
            c: [1, 2, 3, 4, 5, 6, 7, 8],
        };
        let wasm = fake_wasm_with_payload(&payload);
        let encoded = encode_vessel(
            &wasm,
            /* payload_kind */ 42,
            /* payload_offset */ 16,
            /* payload_len */ core::mem::size_of::<FakePayload>() as u32,
        );
        let vessel = load_vessel(&encoded).expect("load");
        let out: &FakePayload =
            extract_payload(&vessel).expect("extract");
        assert_eq!(out, &payload, "round-trip must be byte-identical");
        assert_eq!(vessel.header.payload_kind, 42);
    }

    #[test]
    fn extract_rejects_bad_magic() {
        let payload = FakePayload { a: 0.0, b: 0.0, c: [0u8; 8] };
        let wasm = fake_wasm_with_payload(&payload);
        let mut encoded = encode_vessel(&wasm, 0, 16, 16);
        // Corrupt the magic.
        encoded[0] ^= 0xFF;
        match load_vessel(&encoded) {
            Err(VesselError::BadMagic) => (),
            other => panic!("expected BadMagic, got {other:?}"),
        }
    }

    #[test]
    fn extract_rejects_bad_version() {
        let payload = FakePayload { a: 0.0, b: 0.0, c: [0u8; 8] };
        let wasm = fake_wasm_with_payload(&payload);
        let mut encoded = encode_vessel(&wasm, 0, 16, 16);
        // Bump version field (bytes 4..8).
        encoded[4] = 0xFF;
        match load_vessel(&encoded) {
            Err(VesselError::UnsupportedVersion) => (),
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn extract_rejects_bad_blake3() {
        let payload = FakePayload { a: 0.0, b: 0.0, c: [0u8; 8] };
        let wasm = fake_wasm_with_payload(&payload);
        let mut encoded = encode_vessel(&wasm, 0, 16, 16);
        // Corrupt a WASM byte (offset = VESSEL_HEADER_LEN + 4).
        let wasm_offset = VESSEL_HEADER_LEN + 4;
        encoded[wasm_offset] ^= 0xFF;
        match load_vessel(&encoded) {
            Err(VesselError::Blake3Mismatch) => (),
            other => panic!("expected Blake3Mismatch, got {other:?}"),
        }
    }

    #[test]
    fn extract_rejects_payload_len_mismatch() {
        let payload = FakePayload { a: 0.0, b: 0.0, c: [0u8; 8] };
        let wasm = fake_wasm_with_payload(&payload);
        // Lie about payload_len — claim 8 bytes but ask for FakePayload (16).
        let encoded = encode_vessel(&wasm, 0, 16, /* payload_len */ 8);
        let vessel = load_vessel(&encoded).expect("header + blake3 ok");
        match extract_payload::<FakePayload>(&vessel) {
            Err(VesselError::PayloadLenMismatch) => (),
            other => panic!("expected PayloadLenMismatch, got {other:?}"),
        }
    }

    #[test]
    fn extract_rejects_payload_out_of_bounds() {
        let payload = FakePayload { a: 0.0, b: 0.0, c: [0u8; 8] };
        let wasm = fake_wasm_with_payload(&payload);
        // Lie about payload_offset — claim the payload starts past the end.
        let encoded = encode_vessel(
            &wasm,
            0,
            /* payload_offset */ 9_999,
            core::mem::size_of::<FakePayload>() as u32,
        );
        let vessel = load_vessel(&encoded).expect("header + blake3 ok");
        match extract_payload::<FakePayload>(&vessel) {
            Err(VesselError::PayloadOutOfBounds) => (),
            other => panic!("expected PayloadOutOfBounds, got {other:?}"),
        }
    }

    #[test]
    fn extract_payload_slice_round_trips() {
        let items: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut wasm = vec![0u8; 16];
        wasm.extend_from_slice(bytemuck::bytes_of(&items));
        let encoded = encode_vessel(
            &wasm,
            0,
            16,
            (items.len() * 4) as u32,
        );
        let vessel = load_vessel(&encoded).expect("load");
        let out: &[f32] = extract_payload_slice(&vessel).expect("extract");
        assert_eq!(out, &items);
    }

    #[test]
    fn decode_header_rejects_short_buffer() {
        let short = [0u8; 4];
        match decode_header(&short) {
            Err(VesselError::HeaderTooShort) => (),
            other => panic!("expected HeaderTooShort, got {other:?}"),
        }
    }

    #[test]
    fn verify_blake3_standalone_passes_on_valid_blob() {
        let payload = FakePayload { a: 0.0, b: 0.0, c: [0u8; 8] };
        let wasm = fake_wasm_with_payload(&payload);
        let encoded = encode_vessel(&wasm, 0, 16, 16);
        let header = decode_header(&encoded).expect("decode");
        // WASM bytes = everything after the header.
        let wasm_bytes = &encoded[VESSEL_HEADER_LEN..];
        verify_blake3(&header, wasm_bytes).expect("should pass");
    }

    #[test]
    fn header_is_52_bytes_no_padding() {
        // Sanity: the `#[repr(C)]` layout must not have padding, or the
        // `Pod` impl is unsound.
        assert_eq!(
            core::mem::size_of::<VesselHeader>(),
            VESSEL_HEADER_LEN,
            "VesselHeader must be exactly {} bytes (no padding)",
            VESSEL_HEADER_LEN
        );
    }

    #[test]
    fn loaded_vessel_shares_arc_across_clones() {
        let payload = FakePayload { a: 0.0, b: 0.0, c: [0u8; 8] };
        let wasm = fake_wasm_with_payload(&payload);
        let encoded = encode_vessel(&wasm, 0, 16, 16);
        let vessel = load_vessel(&encoded).expect("load");
        let arc1 = Arc::clone(&vessel.wasm_bytes);
        let arc2 = Arc::clone(&vessel.wasm_bytes);
        // Same allocation — only refcounts bumped.
        assert!(Arc::ptr_eq(&arc1, &arc2));
    }

    // ── Phase 3 projector tests (T3.4) ─────────────────────────────────
    //
    // These build a tiny WAT module that exports `memory` + a `project`
    // function `(ptr: i32, len: i32) -> f32` that sums `len` f32s at
    // `ptr`. The module is wrapped in a vessel and called via
    // `WasmDotProjector` to exercise the Cold/Freeze projection path.

    /// WAT source for a minimal projection module: `project(ptr, len)`
    /// returns the sum of `len` f32 values starting at byte offset `ptr`.
    /// This is the canonical dot-product-with-unit-vector shape.
    const PROJECT_WAT: &str = r#"
        (module
          (memory (export "memory") 4)
          (func (export "project") (param $ptr i32) (param $len i32) (result f32)
            (local $i i32)
            (local $acc f32)
            (local $cur i32)
            (local.set $cur (local.get $ptr))
            (block $done
              (loop $loop
                (br_if $done (i32.ge_s (local.get $i) (local.get $len)))
                (local.set $acc
                  (f32.add (local.get $acc) (f32.load (local.get $cur))))
                (local.set $cur (i32.add (local.get $cur) (i32.const 4)))
                (local.set $i (i32.add (local.get $i) (i32.const 1)))
                (br $loop)
              )
            )
            (local.get $acc)
          )
        )
    "#;

    fn load_project_vessel() -> LoadedVessel {
        let wat_bytes = PROJECT_WAT.as_bytes();
        let encoded = encode_vessel(wat_bytes, /* payload_kind */ 0, 0, 0);
        load_vessel(&encoded).expect("vessel should load")
    }

    #[test]
    fn project_calls_exported_function() {
        let mut vessel = load_project_vessel();
        let mut config = wasmi::Config::default();
        config.consume_fuel(true);
        let engine = wasmi::Engine::new(&config);
        let mut store = wasmi::Store::new(&engine, ());
        ensure_compiled(&mut vessel, &mut store, &engine).expect("compile");

        let projector = WasmDotProjector {
            export_name: "project",
            fuel_budget: 1_000_000,
        };
        let query: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        let result = projector
            .project(&vessel, &mut store, &query)
            .expect("project should succeed");
        // 1+2+3+4 = 10.0
        assert!((result - 10.0f32).abs() < 1e-6, "got {result}");
    }

    #[test]
    fn project_rejects_missing_export() {
        let mut vessel = load_project_vessel();
        let mut config = wasmi::Config::default();
        config.consume_fuel(true);
        let engine = wasmi::Engine::new(&config);
        let mut store = wasmi::Store::new(&engine, ());
        ensure_compiled(&mut vessel, &mut store, &engine).expect("compile");

        // Ask for an export that does not exist.
        let projector = WasmDotProjector {
            export_name: "nonexistent",
            fuel_budget: 1_000_000,
        };
        let query: &[f32] = &[1.0];
        match projector.project(&vessel, &mut store, &query) {
            Err(VesselError::ExportMissing("nonexistent")) => (),
            other => panic!("expected ExportMissing, got {other:?}"),
        }
    }

    #[test]
    fn project_rejects_uncompiled_instance() {
        // Load the vessel but never call `ensure_compiled` — the
        // projector should fail with ExportMissing("instance not compiled").
        let vessel = load_project_vessel();
        let mut config = wasmi::Config::default();
        config.consume_fuel(true);
        let engine = wasmi::Engine::new(&config);
        let mut store = wasmi::Store::new(&engine, ());

        let projector = WasmDotProjector {
            export_name: "project",
            fuel_budget: 1_000_000,
        };
        let query: &[f32] = &[1.0];
        match projector.project(&vessel, &mut store, &query) {
            Err(VesselError::ExportMissing("instance not compiled")) => (),
            other => panic!("expected ExportMissing instance-not-compiled, got {other:?}"),
        }
    }

    #[test]
    fn project_fuel_exhaustion_returns_error() {
        let mut vessel = load_project_vessel();
        let mut config = wasmi::Config::default();
        config.consume_fuel(true);
        let engine = wasmi::Engine::new(&config);
        let mut store = wasmi::Store::new(&engine, ());
        ensure_compiled(&mut vessel, &mut store, &engine).expect("compile");

        // Fuel budget too small for the loop — must fail safe, not panic.
        let projector = WasmDotProjector {
            export_name: "project",
            fuel_budget: 1, // impossibly small
        };
        let query: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        match projector.project(&vessel, &mut store, &query) {
            Err(VesselError::FuelExhausted) => (),
            other => panic!("expected FuelExhausted, got {other:?}"),
        }
    }
}
