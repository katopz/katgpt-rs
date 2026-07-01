//! Octahedral map: S² ↔ [-1,1]² equal-area parameterization.
//!
//! Based on Cigolle et al. "A Survey of Efficient Representations for
//! Independent Unit Vectors" (JCGT 2014), used by OCTOPUS (arXiv:2605.21226)
//! for joint triplet quantization of rotated KV cache coordinates.
//!
//! The octahedral map projects a unit vector on S² onto the octahedron
//! |x|+|y|+|z|=1, then unfolds the 4 bottom faces to yield a bijective
//! mapping to [-1,1]². Properties:
//!
//! - **Equal-area**: preserves spherical area (good for Lloyd-Max codebooks)
//! - **O(1) piecewise-linear**: encode/decode are branch-free aside from z-sign
//! - **Near-uniform Jacobian**: makes 1D scalar quantization a close
//!   approximation to true 2-sphere optimal quantization

/// Sign function that returns 1.0 for zero input.
///
/// Standard `f32::signum()` returns 0.0 for zero, which breaks bijectivity
/// of the octahedral map at coordinate-zero boundaries (equator edges).
/// Using `sign_positive(0) = 1` ensures the map is bijective everywhere.
///
/// Branch-free: uses `bool as u8 as f32` to avoid branch misprediction.
#[inline]
fn sign_positive(x: f32) -> f32 {
    1.0f32 - 2.0 * (x < 0.0) as u8 as f32
}

/// Encode a unit vector on S² to octahedral coordinates in [-1,1]².
///
/// # Algorithm
/// 1. Project onto octahedron: divide by L1 norm ℓ = |x|+|y|+|z|
/// 2. Top hemisphere (z ≥ 0): map directly → (x/ℓ, y/ℓ)
/// 3. Bottom hemisphere (z < 0): unfold → (sign(x)(1-|y|/ℓ), sign(y)(1-|x|/ℓ))
///
/// # Arguments
/// * `x`, `y`, `z` — components of a direction vector (need not be unit-normalized;
///   internally divided by L1 norm)
///
/// # Returns
/// `(ξ, η)` coordinates in [-1, 1]²
pub fn oct_encode(x: f32, y: f32, z: f32) -> (f32, f32) {
    let l1 = x.abs() + y.abs() + z.abs();
    if l1 < 1e-10 {
        return (0.0, 0.0);
    }
    let px = x / l1;
    let py = y / l1;
    let pz = z / l1;

    // Branch-free hemisphere selection: avoids branch misprediction.
    // top=1.0 when pz>=0, bottom=1.0 when pz<0
    let top = (pz >= 0.0) as u8 as f32;
    let bottom = 1.0 - top;
    let xi = px * top + sign_positive(px) * (1.0 - py.abs()) * bottom;
    let eta = py * top + sign_positive(py) * (1.0 - px.abs()) * bottom;
    (xi, eta)
}

/// Decode octahedral coordinates in [-1,1]² back to a unit vector on S².
///
/// # Algorithm
/// 1. Compute `r = 1 - |ξ| - |η|` (reconstructed z on octahedron)
/// 2. Top (r ≥ 0): direction is (ξ, η, r)
/// 3. Bottom (r < 0): fold back → (sign(ξ)(1-|η|), sign(η)(1-|ξ|), r)
/// 4. Normalize to unit length
///
/// # Arguments
/// * `xi` — first octahedral coordinate in [-1, 1]
/// * `eta` — second octahedral coordinate in [-1, 1]
///
/// # Returns
/// `(x, y, z)` unit vector on S²
pub fn oct_decode(xi: f32, eta: f32) -> (f32, f32, f32) {
    let r = 1.0 - xi.abs() - eta.abs();

    // Branch-free hemisphere reconstruction
    let top = (r >= 0.0) as u8 as f32;
    let bottom = 1.0 - top;
    let x = xi * top + sign_positive(xi) * (1.0 - eta.abs()) * bottom;
    let y = eta * top + sign_positive(eta) * (1.0 - xi.abs()) * bottom;
    let z = r;

    let norm = (x * x + y * y + z * z).sqrt();
    if norm < 1e-10 {
        return (0.0, 0.0, 1.0);
    }
    (x / norm, y / norm, z / norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Normalize a 3-vector to unit length.
    fn normalize(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let norm = (x * x + y * y + z * z).sqrt();
        if norm < 1e-10 {
            (0.0, 0.0, 1.0)
        } else {
            (x / norm, y / norm, z / norm)
        }
    }

    /// Assert two unit vectors are within cosine tolerance of 1.0.
    fn assert_unit_close(a: (f32, f32, f32), b: (f32, f32, f32), tol: f32, msg: &str) {
        let cos = a.0 * b.0 + a.1 * b.1 + a.2 * b.2;
        assert!(
            (cos - 1.0).abs() < tol,
            "{msg}: cosine = {cos}, expected ~1.0\n  got      ({}, {}, {})\n  expected ({}, {}, {})",
            a.0,
            a.1,
            a.2,
            b.0,
            b.1,
            b.2
        );
    }

    // ── Axis-aligned directions ──────────────────────────────

    #[test]
    fn test_encode_decode_positive_pole() {
        let (xi, eta) = oct_encode(0.0, 0.0, 1.0);
        assert!(xi.abs() < 1e-6, "xi = {xi}");
        assert!(eta.abs() < 1e-6, "eta = {eta}");
        let (x, y, z) = oct_decode(xi, eta);
        assert_unit_close((x, y, z), (0.0, 0.0, 1.0), 1e-5, "north pole");
    }

    #[test]
    fn test_encode_decode_negative_pole() {
        let (xi, eta) = oct_encode(0.0, 0.0, -1.0);
        let (x, y, z) = oct_decode(xi, eta);
        assert_unit_close((x, y, z), (0.0, 0.0, -1.0), 1e-5, "south pole");
    }

    #[test]
    fn test_encode_decode_all_axis_directions() {
        let axes = [
            (1.0f32, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ];
        for &dir in &axes {
            let (xi, eta) = oct_encode(dir.0, dir.1, dir.2);
            let (x, y, z) = oct_decode(xi, eta);
            assert_unit_close(
                (x, y, z),
                dir,
                1e-5,
                &format!("axis ({}, {}, {})", dir.0, dir.1, dir.2),
            );
        }
    }

    // ── Octant diagonals ─────────────────────────────────────

    #[test]
    fn test_encode_decode_all_octant_diagonals() {
        let inv_sqrt3 = 1.0f32 / 3.0f32.sqrt();
        for &sx in &[1.0f32, -1.0] {
            for &sy in &[1.0, -1.0] {
                for &sz in &[1.0, -1.0] {
                    let n = (sx * inv_sqrt3, sy * inv_sqrt3, sz * inv_sqrt3);
                    let (xi, eta) = oct_encode(n.0, n.1, n.2);
                    assert!(xi.abs() <= 1.0 + 1e-6, "xi = {xi} out of range");
                    assert!(eta.abs() <= 1.0 + 1e-6, "eta = {eta} out of range");
                    let (x, y, z) = oct_decode(xi, eta);
                    assert_unit_close((x, y, z), n, 1e-5, &format!("octant ({sx}, {sy}, {sz})"));
                }
            }
        }
    }

    // ── Equatorial vectors (z=0 plane) ──────────────────────

    #[test]
    fn test_encode_decode_equatorial() {
        let cases = [
            (1.0f32, 0.0, 0.0, "equatorial (1,0,0)"),
            (0.0, 1.0, 0.0, "equatorial (0,1,0)"),
            (1.0, 1.0, 0.0, "equatorial (1,1,0)"),
            (1.0, -1.0, 0.0, "equatorial (1,-1,0)"),
            (-1.0, 1.0, 0.0, "equatorial (-1,1,0)"),
        ];
        for &(cx, cy, cz, label) in &cases {
            let n = normalize(cx, cy, cz);
            let (xi, eta) = oct_encode(n.0, n.1, n.2);
            let (x, y, z) = oct_decode(xi, eta);
            assert_unit_close((x, y, z), n, 1e-5, label);
        }
    }

    // ── General roundtrip via Fibonacci sphere sampling ──────

    #[test]
    fn test_roundtrip_fibonacci_sphere() {
        let golden = (1.0 + 5.0f32.sqrt()) / 2.0;
        let n_samples = 256;
        for i in 0..n_samples {
            let theta = std::f32::consts::PI * (1.0 - 2.0 * (i as f32 + 0.5) / n_samples as f32);
            let phi = 2.0 * std::f32::consts::PI * i as f32 / golden;
            let x = theta.sin() * phi.cos();
            let y = theta.sin() * phi.sin();
            let z = theta.cos();

            let (xi, eta) = oct_encode(x, y, z);
            assert!(xi.abs() <= 1.0 + 1e-5, "sample {i}: xi={xi} out of [-1,1]");
            assert!(
                eta.abs() <= 1.0 + 1e-5,
                "sample {i}: eta={eta} out of [-1,1]"
            );
            let (xr, yr, zr) = oct_decode(xi, eta);
            assert_unit_close(
                (xr, yr, zr),
                (x, y, z),
                1e-5,
                &format!("fibonacci sample {i}"),
            );
        }
    }

    // ── Range verification ───────────────────────────────────

    #[test]
    fn test_encode_output_range() {
        let golden = (1.0 + 5.0f32.sqrt()) / 2.0;
        for i in 0..1000 {
            let theta = std::f32::consts::PI * (1.0 - 2.0 * (i as f32 + 0.5) / 1000.0);
            let phi = 2.0 * std::f32::consts::PI * i as f32 / golden;
            let x = theta.sin() * phi.cos();
            let y = theta.sin() * phi.sin();
            let z = theta.cos();
            let (xi, eta) = oct_encode(x, y, z);
            assert!(
                (-1.0 - 1e-6..=1.0 + 1e-6).contains(&xi),
                "xi out of range: {xi}"
            );
            assert!(
                (-1.0 - 1e-6..=1.0 + 1e-6).contains(&eta),
                "eta out of range: {eta}"
            );
        }
    }

    // ── Degenerate inputs ────────────────────────────────────

    #[test]
    fn test_zero_input() {
        let (xi, eta) = oct_encode(0.0, 0.0, 0.0);
        assert_eq!(xi, 0.0);
        assert_eq!(eta, 0.0);
    }

    #[test]
    fn test_non_unit_input_auto_normalizes() {
        let (xi1, eta1) = oct_encode(3.0, 4.0, 5.0);
        let n = normalize(3.0, 4.0, 5.0);
        let (xi2, eta2) = oct_encode(n.0, n.1, n.2);
        assert!((xi1 - xi2).abs() < 1e-5, "xi mismatch: {xi1} vs {xi2}");
        assert!((eta1 - eta2).abs() < 1e-5, "eta mismatch: {eta1} vs {eta2}");
    }

    // ── Specific coordinate values ───────────────────────────

    #[test]
    fn test_north_pole_maps_to_origin() {
        let (xi, eta) = oct_encode(0.0, 0.0, 1.0);
        assert!(xi.abs() < 1e-6);
        assert!(eta.abs() < 1e-6);
    }

    #[test]
    fn test_south_pole_maps_to_corner() {
        let (xi, eta) = oct_encode(0.0, 0.0, -1.0);
        assert!(
            (xi - 1.0).abs() < 1e-5,
            "south pole xi = {xi}, expected 1.0"
        );
        assert!(
            (eta - 1.0).abs() < 1e-5,
            "south pole eta = {eta}, expected 1.0"
        );
    }

    #[test]
    fn test_decode_origin_is_north_pole() {
        let (x, y, z) = oct_decode(0.0, 0.0);
        assert_unit_close((x, y, z), (0.0, 0.0, 1.0), 1e-6, "origin → north pole");
    }

    #[test]
    fn test_decode_corner_is_south_pole() {
        let (x, y, z) = oct_decode(1.0, 1.0);
        assert_unit_close(
            (x, y, z),
            (0.0, 0.0, -1.0),
            1e-6,
            "corner (1,1) → south pole",
        );
    }
}
