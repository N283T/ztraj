const std = @import("std");

// ============================================================================
// Lee-Richards SIMD helpers
// ============================================================================

/// SIMD-optimized batch xy-distance calculation for Lee-Richards.
/// Computes sqrt(dx^2 + dy^2) for 4 neighbors simultaneously.
///
/// # Parameters
/// - `xi`, `yi`: Coordinates of the reference atom
/// - `x_neighbors`, `y_neighbors`: Arrays of 4 neighbor x/y coordinates
///
/// # Returns
/// Array of 4 xy-distances
pub fn xyDistanceBatch4(
    xi: f64,
    yi: f64,
    x_neighbors: [4]f64,
    y_neighbors: [4]f64,
) [4]f64 {
    const px: @Vector(4, f64) = @splat(xi);
    const py: @Vector(4, f64) = @splat(yi);

    const nx: @Vector(4, f64) = x_neighbors;
    const ny: @Vector(4, f64) = y_neighbors;

    const dx = nx - px;
    const dy = ny - py;

    const dist_sq = dx * dx + dy * dy;
    const dist = @sqrt(dist_sq);

    return dist;
}

/// Compute slice radii (Rj' = sqrt(Rj^2 - dj^2)) for 4 neighbors.
/// Returns 0 for neighbors that don't intersect the slice (dj >= Rj).
///
/// # Parameters
/// - `slice_z`: Z-coordinate of the current slice
/// - `z_neighbors`: Array of 4 neighbor z-coordinates
/// - `radii`: Array of 4 neighbor radii
///
/// # Returns
/// Array of 4 slice radii (0 if no intersection)
pub fn sliceRadiiBatch4(
    slice_z: f64,
    z_neighbors: [4]f64,
    radii: [4]f64,
) [4]f64 {
    const sz: @Vector(4, f64) = @splat(slice_z);
    const zn: @Vector(4, f64) = z_neighbors;
    const rn: @Vector(4, f64) = radii;

    const dz = zn - sz;
    const dz_sq = dz * dz;
    const r_sq = rn * rn;

    // Rj_prime^2 = Rj^2 - dj^2
    const rp_sq = r_sq - dz_sq;

    // Clamp negative values to 0 (no intersection)
    const zero: @Vector(4, f64) = @splat(0.0);
    const rp_sq_clamped = @max(rp_sq, zero);

    return @sqrt(rp_sq_clamped);
}

/// Check if circles overlap (dij < Ri' + Rj') for 4 neighbors.
///
/// # Parameters
/// - `dij`: Array of 4 xy-distances
/// - `ri_prime`: Slice radius of reference atom
/// - `rj_primes`: Array of 4 neighbor slice radii
///
/// # Returns
/// Bitmask where bit i is set if circles overlap
pub fn circlesOverlapBatch4(
    dij: [4]f64,
    ri_prime: f64,
    rj_primes: [4]f64,
) u4 {
    const d: @Vector(4, f64) = dij;
    const ri: @Vector(4, f64) = @splat(ri_prime);
    const rj: @Vector(4, f64) = rj_primes;

    const sum_radii = ri + rj;
    const overlaps = d < sum_radii;

    return @bitCast(overlaps);
}

/// SIMD-optimized batch xy-distance calculation for Lee-Richards (8-wide).
/// Computes sqrt(dx^2 + dy^2) for 8 neighbors simultaneously.
///
/// # Parameters
/// - `xi`, `yi`: Coordinates of the reference atom
/// - `x_neighbors`, `y_neighbors`: Arrays of 8 neighbor x/y coordinates
///
/// # Returns
/// Array of 8 xy-distances
pub fn xyDistanceBatch8(
    xi: f64,
    yi: f64,
    x_neighbors: [8]f64,
    y_neighbors: [8]f64,
) [8]f64 {
    const px: @Vector(8, f64) = @splat(xi);
    const py: @Vector(8, f64) = @splat(yi);

    const nx: @Vector(8, f64) = x_neighbors;
    const ny: @Vector(8, f64) = y_neighbors;

    const dx = nx - px;
    const dy = ny - py;

    const dist_sq = dx * dx + dy * dy;
    const dist = @sqrt(dist_sq);

    return dist;
}

/// Compute slice radii (Rj' = sqrt(Rj^2 - dj^2)) for 8 neighbors.
/// Returns 0 for neighbors that don't intersect the slice (dj >= Rj).
///
/// # Parameters
/// - `slice_z`: Z-coordinate of the current slice
/// - `z_neighbors`: Array of 8 neighbor z-coordinates
/// - `radii`: Array of 8 neighbor radii
///
/// # Returns
/// Array of 8 slice radii (0 if no intersection)
pub fn sliceRadiiBatch8(
    slice_z: f64,
    z_neighbors: [8]f64,
    radii: [8]f64,
) [8]f64 {
    const sz: @Vector(8, f64) = @splat(slice_z);
    const zn: @Vector(8, f64) = z_neighbors;
    const rn: @Vector(8, f64) = radii;

    const dz = zn - sz;
    const dz_sq = dz * dz;
    const r_sq = rn * rn;

    // Rj_prime^2 = Rj^2 - dj^2
    const rp_sq = r_sq - dz_sq;

    // Clamp negative values to 0 (no intersection)
    const zero: @Vector(8, f64) = @splat(0.0);
    const rp_sq_clamped = @max(rp_sq, zero);

    return @sqrt(rp_sq_clamped);
}

/// Check if circles overlap (dij < Ri' + Rj') for 8 neighbors.
///
/// # Parameters
/// - `dij`: Array of 8 xy-distances
/// - `ri_prime`: Slice radius of reference atom
/// - `rj_primes`: Array of 8 neighbor slice radii
///
/// # Returns
/// Bitmask where bit i is set if circles overlap
pub fn circlesOverlapBatch8(
    dij: [8]f64,
    ri_prime: f64,
    rj_primes: [8]f64,
) u8 {
    const d: @Vector(8, f64) = dij;
    const ri: @Vector(8, f64) = @splat(ri_prime);
    const rj: @Vector(8, f64) = rj_primes;

    const sum_radii = ri + rj;
    const overlaps = d < sum_radii;

    return @bitCast(overlaps);
}

// ============================================================================
// Generic Lee-Richards SIMD helpers (f32/f64)
// ============================================================================

/// Generic SIMD-optimized batch xy-distance calculation for Lee-Richards (4-wide).
pub fn xyDistanceBatch4Gen(comptime T: type) type {
    return struct {
        pub fn compute(
            xi: T,
            yi: T,
            x_neighbors: [4]T,
            y_neighbors: [4]T,
        ) [4]T {
            const px: @Vector(4, T) = @splat(xi);
            const py: @Vector(4, T) = @splat(yi);

            const nx: @Vector(4, T) = x_neighbors;
            const ny: @Vector(4, T) = y_neighbors;

            const dx = nx - px;
            const dy = ny - py;

            const dist_sq = dx * dx + dy * dy;
            return @sqrt(dist_sq);
        }
    };
}

/// Generic compute slice radii (Rj' = sqrt(Rj^2 - dj^2)) for 4 neighbors.
pub fn sliceRadiiBatch4Gen(comptime T: type) type {
    return struct {
        pub fn compute(
            slice_z: T,
            z_neighbors: [4]T,
            radii: [4]T,
        ) [4]T {
            const sz: @Vector(4, T) = @splat(slice_z);
            const zn: @Vector(4, T) = z_neighbors;
            const rn: @Vector(4, T) = radii;

            const dz = zn - sz;
            const dz_sq = dz * dz;
            const r_sq = rn * rn;

            const rp_sq = r_sq - dz_sq;

            const zero: @Vector(4, T) = @splat(0.0);
            const rp_sq_clamped = @max(rp_sq, zero);

            return @sqrt(rp_sq_clamped);
        }
    };
}

/// Generic check if circles overlap (dij < Ri' + Rj') for 4 neighbors.
pub fn circlesOverlapBatch4Gen(comptime T: type) type {
    return struct {
        pub fn compute(
            dij: [4]T,
            ri_prime: T,
            rj_primes: [4]T,
        ) u4 {
            const d: @Vector(4, T) = dij;
            const ri: @Vector(4, T) = @splat(ri_prime);
            const rj: @Vector(4, T) = rj_primes;

            const sum_radii = ri + rj;
            const overlaps = d < sum_radii;

            return @bitCast(overlaps);
        }
    };
}

/// Generic SIMD-optimized batch xy-distance calculation for Lee-Richards (8-wide).
pub fn xyDistanceBatch8Gen(comptime T: type) type {
    return struct {
        pub fn compute(
            xi: T,
            yi: T,
            x_neighbors: [8]T,
            y_neighbors: [8]T,
        ) [8]T {
            const px: @Vector(8, T) = @splat(xi);
            const py: @Vector(8, T) = @splat(yi);

            const nx: @Vector(8, T) = x_neighbors;
            const ny: @Vector(8, T) = y_neighbors;

            const dx = nx - px;
            const dy = ny - py;

            const dist_sq = dx * dx + dy * dy;
            return @sqrt(dist_sq);
        }
    };
}

/// Generic compute slice radii (Rj' = sqrt(Rj^2 - dj^2)) for 8 neighbors.
pub fn sliceRadiiBatch8Gen(comptime T: type) type {
    return struct {
        pub fn compute(
            slice_z: T,
            z_neighbors: [8]T,
            radii: [8]T,
        ) [8]T {
            const sz: @Vector(8, T) = @splat(slice_z);
            const zn: @Vector(8, T) = z_neighbors;
            const rn: @Vector(8, T) = radii;

            const dz = zn - sz;
            const dz_sq = dz * dz;
            const r_sq = rn * rn;

            const rp_sq = r_sq - dz_sq;

            const zero: @Vector(8, T) = @splat(0.0);
            const rp_sq_clamped = @max(rp_sq, zero);

            return @sqrt(rp_sq_clamped);
        }
    };
}

/// Generic check if circles overlap (dij < Ri' + Rj') for 8 neighbors.
pub fn circlesOverlapBatch8Gen(comptime T: type) type {
    return struct {
        pub fn compute(
            dij: [8]T,
            ri_prime: T,
            rj_primes: [8]T,
        ) u8 {
            const d: @Vector(8, T) = dij;
            const ri: @Vector(8, T) = @splat(ri_prime);
            const rj: @Vector(8, T) = rj_primes;

            const sum_radii = ri + rj;
            const overlaps = d < sum_radii;

            return @bitCast(overlaps);
        }
    };
}

// Lee-Richards SIMD tests

test "xyDistanceBatch4 - correctness" {
    const result = xyDistanceBatch4(
        0.0,
        0.0,
        [4]f64{ 3.0, 0.0, 1.0, 3.0 },
        [4]f64{ 4.0, 5.0, 0.0, 4.0 },
    );

    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result[0], 1e-10); // 3-4-5 triangle
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result[3], 1e-10);
}

test "sliceRadiiBatch4 - correctness" {
    const result = sliceRadiiBatch4(
        0.0, // slice at z=0
        [4]f64{ 0.0, 0.6, 0.8, 2.0 }, // z-coords
        [4]f64{ 1.0, 1.0, 1.0, 1.0 }, // radii (R=1)
    );

    // R' = sqrt(R^2 - d^2)
    // Neighbor 0: sqrt(1 - 0) = 1.0
    // Neighbor 1: sqrt(1 - 0.36) = sqrt(0.64) = 0.8
    // Neighbor 2: sqrt(1 - 0.64) = sqrt(0.36) = 0.6
    // Neighbor 3: sqrt(1 - 4) -> clamped to 0
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), result[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), result[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result[3], 1e-10);
}

test "circlesOverlapBatch4 - mixed" {
    const result = circlesOverlapBatch4(
        [4]f64{ 1.0, 3.0, 0.5, 2.0 }, // xy-distances
        1.0, // Ri'
        [4]f64{ 1.0, 1.0, 1.0, 1.0 }, // Rj' values
    );

    // Ri' + Rj' = 2.0 for all
    // dij < 2.0?
    // Neighbor 0: 1.0 < 2.0 -> overlaps
    // Neighbor 1: 3.0 < 2.0 -> no
    // Neighbor 2: 0.5 < 2.0 -> overlaps
    // Neighbor 3: 2.0 < 2.0 -> no (equal, not less)
    try std.testing.expectEqual(@as(u4, 0b0101), result);
}

// Lee-Richards 8-wide SIMD tests

test "xyDistanceBatch8 - correctness" {
    const result = xyDistanceBatch8(
        0.0,
        0.0,
        [8]f64{ 3.0, 0.0, 1.0, 3.0, 4.0, 0.0, 5.0, 6.0 },
        [8]f64{ 4.0, 5.0, 0.0, 4.0, 3.0, 12.0, 12.0, 8.0 },
    );

    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result[0], 1e-10); // 3-4-5 triangle
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result[3], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result[4], 1e-10); // 4-3-5
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), result[5], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 13.0), result[6], 1e-10); // 5-12-13
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), result[7], 1e-10); // 6-8-10
}

test "sliceRadiiBatch8 - correctness" {
    const result = sliceRadiiBatch8(
        0.0, // slice at z=0
        [8]f64{ 0.0, 0.6, 0.8, 2.0, 0.0, 0.3, 0.4, 1.5 }, // z-coords
        [8]f64{ 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0 }, // radii
    );

    // R' = sqrt(R^2 - d^2)
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result[0], 1e-10); // sqrt(1 - 0)
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), result[1], 1e-10); // sqrt(1 - 0.36)
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), result[2], 1e-10); // sqrt(1 - 0.64)
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result[3], 1e-10); // clamped
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), result[4], 1e-10); // sqrt(0.25 - 0)
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), result[5], 1e-10); // sqrt(0.25 - 0.09)
    try std.testing.expectApproxEqAbs(@as(f64, 0.3), result[6], 1e-10); // sqrt(0.25 - 0.16)
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result[7], 1e-10); // clamped
}

test "circlesOverlapBatch8 - mixed" {
    const result = circlesOverlapBatch8(
        [8]f64{ 1.0, 3.0, 0.5, 2.0, 1.5, 2.5, 0.1, 1.9 }, // xy-distances
        1.0, // Ri'
        [8]f64{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }, // Rj' values
    );

    // Ri' + Rj' = 2.0 for all
    // dij < 2.0?
    // 0: 1.0 < 2.0 -> yes (bit 0)
    // 1: 3.0 < 2.0 -> no
    // 2: 0.5 < 2.0 -> yes (bit 2)
    // 3: 2.0 < 2.0 -> no
    // 4: 1.5 < 2.0 -> yes (bit 4)
    // 5: 2.5 < 2.0 -> no
    // 6: 0.1 < 2.0 -> yes (bit 6)
    // 7: 1.9 < 2.0 -> yes (bit 7)
    try std.testing.expectEqual(@as(u8, 0b11010101), result);
}

// Generic Lee-Richards SIMD tests

test "xyDistanceBatch4Gen f32 - correctness" {
    const result = xyDistanceBatch4Gen(f32).compute(
        0.0,
        0.0,
        [4]f32{ 3.0, 0.0, 1.0, 3.0 },
        [4]f32{ 4.0, 5.0, 0.0, 4.0 },
    );

    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result[3], 1e-5);
}

test "sliceRadiiBatch4Gen f32 - correctness" {
    const result = sliceRadiiBatch4Gen(f32).compute(
        0.0,
        [4]f32{ 0.0, 0.6, 0.8, 2.0 },
        [4]f32{ 1.0, 1.0, 1.0, 1.0 },
    );

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), result[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[3], 1e-5);
}

test "sliceRadiiBatch8Gen f32 - correctness" {
    const result = sliceRadiiBatch8Gen(f32).compute(
        0.0,
        [8]f32{ 0.0, 0.6, 0.8, 2.0, 0.0, 0.3, 0.4, 1.5 },
        [8]f32{ 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0 },
    );

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), result[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), result[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), result[5], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), result[6], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[7], 1e-5);
}
