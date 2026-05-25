//! CachePrune GOAT Proof — Rolling hash segment matching demo (Plan 140).
//!
//! Validates G3 (collision resistance) and G4 (segment matching latency).
//!
//! Run: `cargo run --example cache_prune_02_segment_match --features cache_prune`

use katgpt_rs::cache_prune::{KvSegmentPool, RollingHash};

fn main() {
    println!("=== CachePrune Rolling Hash Segment Matching (Plan 140 GOAT) ===\n");

    collision_resistance();
    segment_matching_demo();
    segment_matching_latency();
}

fn collision_resistance() {
    println!("── G3: Rolling Hash Collision Resistance ──");

    let roller = RollingHash::new(512);
    let num_segments = 10_000;
    let mut hashes = std::collections::HashSet::new();
    let mut collisions = 0;

    for seed in 0..num_segments {
        let tokens: Vec<u32> = (0..64).map(|i| seed * 100 + i).collect();
        let prefixes = roller.prefix_hashes(&tokens);
        let h = roller.substring_hash(&prefixes, 0, tokens.len());
        if !hashes.insert(h) {
            collisions += 1;
        }
    }

    println!("  Segments: {num_segments}");
    println!("  Unique hashes: {}", hashes.len());
    println!("  Collisions: {collisions}");
    let status = if collisions == 0 {
        "✓ PASS"
    } else {
        "✗ FAIL"
    };
    println!("  Result: {status}");
    println!();
}

fn segment_matching_demo() {
    println!("── Segment Matching Demo ──");

    let roller = RollingHash::new(1024);
    let mut pool = KvSegmentPool::new();

    // Add some cached segments (simulating previous prompts).
    let seg1: Vec<u32> = (100..200).collect();
    let seg2: Vec<u32> = (300..450).collect();
    let seg3: Vec<u32> = (500..700).collect();

    pool.add_segment(&seg1, &roller, 0, 100);
    pool.add_segment(&seg2, &roller, 0, 150);
    pool.add_segment(&seg3, &roller, 0, 200);

    println!("  Pool: 3 segments (len=100, 150, 200)");

    // Request that contains seg1 embedded within a larger prompt.
    let mut request = vec![1_u32, 2, 3]; // prefix
    request.extend(&seg1); // embedded segment
    request.extend(&[999, 998, 997]); // suffix

    let matches = pool.find_matches(&request, &roller);
    let verified: Vec<_> = matches.iter().filter(|m| m.verified).collect();

    println!("  Request length: {} tokens", request.len());
    println!("  Candidates: {}", matches.len());
    println!("  Verified matches: {}", verified.len());

    for m in &verified {
        println!(
            "    Match: segment {} at [{}..{})",
            m.segment_idx, m.start, m.end
        );
    }

    println!();
}

fn segment_matching_latency() {
    println!("── G4: Segment Matching Latency ──");

    let roller = RollingHash::new(2048);
    let mut pool = KvSegmentPool::new();

    // Build pool with 1000 cached segments of varying lengths.
    for i in 0..1000u32 {
        let len = 64 + (i % 256) as usize;
        let tokens: Vec<u32> = (i * 1000..i * 1000 + len as u32).collect();
        pool.add_segment(&tokens, &roller, 0, len);
    }

    // 10K-token request containing a few embedded segments.
    let mut request = Vec::with_capacity(10_000);
    for i in 0..10_000u32 {
        request.push(i % 500);
    }
    // Embed a known segment.
    let known_seg: Vec<u32> = (500 * 1000..500 * 1000 + 100).collect();
    request[1000..1100].copy_from_slice(&known_seg);

    let start = std::time::Instant::now();
    let matches = pool.find_matches(&request, &roller);
    let elapsed = start.elapsed();

    let verified: Vec<_> = matches.iter().filter(|m| m.verified).collect();

    let status = if elapsed.as_millis() <= 10 {
        "✓"
    } else {
        "⚠ > 10ms"
    };

    println!("  Pool size: 1000 segments");
    println!("  Request length: 10,000 tokens");
    println!("  Candidates: {}", matches.len());
    println!("  Verified: {}", verified.len());
    println!("  Latency: {:?}", elapsed);
    println!("  Result: {status}");
    println!();
}
