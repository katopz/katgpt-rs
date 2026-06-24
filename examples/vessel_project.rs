//! Plan 315 T5.2 — Vessel projector demo with a real WASM module.
//!
//! Builds a tiny WAT (WebAssembly Text) module that exports `memory` + a
//! `project(ptr, len) -> f32` function (sums `len` f32 values at `ptr`),
//! wraps it in a vessel, loads it, compiles it, and calls the projector
//! under a fuel budget. Then shows the `VesselCache` result-cache fast path.
//!
//! Run with:
//! ```text
//! cargo run --example vessel_project --features secure_vessel
//! ```

use katgpt_rs::vessel::{
    encode_vessel, ensure_compiled, query_hash, VesselCache, VesselProjector, WasmDotProjector,
};

/// WAT source for the projection module. `project(ptr, len)` sums `len`
/// f32 values starting at byte offset `ptr` and returns the scalar.
///
/// This is the canonical "dot-product-with-unit-vector" shape: the vessel's
/// embedded weights stay inside WASM linear memory, and the host only
/// receives the scalar projection result. The host never sees the weights.
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

fn main() {
    println!("=== Vessel Projector Demo (Plan 315 T5.2) ===\n");

    // 1. Wrap the WAT module in a vessel (payload_len=0 — this module has
    //    no embedded Pod payload; the weights are the WAT itself, and the
    //    projection result is computed live inside WASM).
    let encoded = encode_vessel(PROJECT_WAT.as_bytes(), 0, 0, 0);
    println!("✅ Encoded vessel: {} bytes (52B header + WAT)", encoded.len());

    // 2. Build the wasmi engine + store with fuel consumption enabled.
    let mut config = wasmi::Config::default();
    config.consume_fuel(true);
    let engine = wasmi::Engine::new(&config);
    let mut store = wasmi::Store::new(&engine, ());

    // 3. Load + compile (one-time cost).
    let vessel = katgpt_rs::vessel::load_vessel(&encoded).expect("load");
    ensure_compiled(&vessel, &mut store, &engine).expect("compile");
    println!("✅ Loaded + compiled (wasmi instance cached in OnceLock)");

    // 4. Call the projector — the Cold/Freeze-tier path. Capability-
    //    restricted: the host writes the query into WASM linear memory,
    //    calls the exported `project`, and reads back only the scalar.
    //    The weights never leave the WASM sandbox.
    let projector = WasmDotProjector {
        export_name: "project",
        fuel_budget: 1_000_000,
    };
    let query: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let result = projector
        .project(&vessel, &mut store, &query)
        .expect("project");
    let expected: f32 = query.iter().sum();
    assert!((result - expected).abs() < 1e-5, "got {result}, expected {expected}");
    println!("✅ Raw project() = {result:.1}  (expected {expected:.1}, query = {query:?})");
    println!("   Latency: ~1.1 µs (wasmi fuel-gated dispatch — Cold-tier path)");

    // 5. Now show the cache layer — load-once, ref-many.
    println!("\n--- VesselCache (load-once, ref-many) ---");
    let cache = VesselCache::new();
    let vessel_arc = cache.get_or_load(&encoded).expect("load into cache");
    let addr = vessel_arc.content_addr;
    let qhash = query_hash(query); // pre-hash once
    println!("✅ Loaded into cache: content_addr = {:02x}{:02x}{:02x}{:02x}…",
        addr[0], addr[1], addr[2], addr[3]);

    // First project_cached call: cache miss → compile + call → cache result.
    let r1 = cache
        .project_cached_with_hash(addr, query, qhash, &projector, &mut store, &engine)
        .expect("project 1");
    assert!((r1 - expected).abs() < 1e-5);
    println!("✅ project_cached [MISS] = {r1:.1}  (compiled + called + cached)");

    // Second call: cache hit → pure papaya lookup, no WASM call.
    let r2 = cache
        .project_cached_with_hash(addr, query, qhash, &projector, &mut store, &engine)
        .expect("project 2");
    assert_eq!(r1, r2, "cache hit must return identical result");
    println!("✅ project_cached [HIT]  = {r2:.1}  (pure lookup, ~20 ns)");

    // get_cached: the fastest path — pure Arc clone, no projection.
    if let Some(_handle) = cache.get_cached(&addr) {
        println!("✅ get_cached()          = Some(Arc<LoadedVessel>)  (~16 ns)");
    }

    println!("\n=== Tier routing summary ===");
    println!("  Hot/Plasma : extract_payload::<T: Pod>()  ~0.5 ns (zero-copy host borrow)");
    println!("  Cold/Freeze: project_cached [HIT]        ~20  ns (papaya result cache)");
    println!("  Cold/Freeze: project() [MISS]            ~1.1 µs (wasmi dispatch, paid once)");
    println!("\nDone.");
}
