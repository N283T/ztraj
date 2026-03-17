const std = @import("std");
const builtin = @import("builtin");
const vec = @import("vec.zig");

pub const Vec3Gen = vec.Vec3Gen;
pub const Vec3 = vec.Vec3;
pub const Vec3f32 = vec.Vec3f32;
pub const cpu_features = vec.cpu_features;
pub const optimal_vector_width = vec.optimal_vector_width;

// ============================================================================
// SIMD-optimized batch distance squared functions (f64)
// ============================================================================

/// SIMD-optimized batch distance squared calculation.
/// Process 4 positions simultaneously using @Vector(4, f64).
///
/// # Parameters
/// - `point`: The test point to measure distances from
/// - `positions`: Array of 4 Vec3 positions to measure to
///
/// # Returns
/// Array of 4 squared distances
pub fn distanceSquaredBatch4(
    point: Vec3,
    positions: [4]Vec3,
) [4]f64 {
    // Splat point coordinates into vectors
    const px: @Vector(4, f64) = @splat(point.x);
    const py: @Vector(4, f64) = @splat(point.y);
    const pz: @Vector(4, f64) = @splat(point.z);

    // Load other positions into vectors
    const ox = @Vector(4, f64){ positions[0].x, positions[1].x, positions[2].x, positions[3].x };
    const oy = @Vector(4, f64){ positions[0].y, positions[1].y, positions[2].y, positions[3].y };
    const oz = @Vector(4, f64){ positions[0].z, positions[1].z, positions[2].z, positions[3].z };

    // Calculate differences
    const dx = px - ox;
    const dy = py - oy;
    const dz = pz - oz;

    // Calculate squared distances: dx^2 + dy^2 + dz^2
    const dist_sq = dx * dx + dy * dy + dz * dz;

    return dist_sq;
}

/// Check if point is buried by any of 4 atoms.
///
/// # Parameters
/// - `point`: The test point to check
/// - `positions`: Array of 4 atom positions
/// - `radii_sq`: Array of 4 pre-computed (radius + probe)^2 values
///
/// # Returns
/// true if point is inside any of the 4 atoms, false otherwise
pub fn isPointBuriedBatch4(
    point: Vec3,
    positions: [4]Vec3,
    radii_sq: [4]f64,
) bool {
    const dist_sq = distanceSquaredBatch4(point, positions);
    const radii_v: @Vector(4, f64) = radii_sq;
    const dist_v: @Vector(4, f64) = dist_sq;

    // Check if any distance < radius (point inside atom)
    const inside = dist_v < radii_v;
    return @reduce(.Or, inside);
}

/// SIMD-optimized batch distance squared calculation for 8 atoms.
/// Process 8 positions simultaneously using @Vector(8, f64).
///
/// # Parameters
/// - `point`: The test point to measure distances from
/// - `positions`: Array of 8 Vec3 positions to measure to
///
/// # Returns
/// Array of 8 squared distances
pub fn distanceSquaredBatch8(
    point: Vec3,
    positions: [8]Vec3,
) [8]f64 {
    // Splat point coordinates into vectors
    const px: @Vector(8, f64) = @splat(point.x);
    const py: @Vector(8, f64) = @splat(point.y);
    const pz: @Vector(8, f64) = @splat(point.z);

    // Load other positions into vectors
    const ox = @Vector(8, f64){
        positions[0].x, positions[1].x, positions[2].x, positions[3].x,
        positions[4].x, positions[5].x, positions[6].x, positions[7].x,
    };
    const oy = @Vector(8, f64){
        positions[0].y, positions[1].y, positions[2].y, positions[3].y,
        positions[4].y, positions[5].y, positions[6].y, positions[7].y,
    };
    const oz = @Vector(8, f64){
        positions[0].z, positions[1].z, positions[2].z, positions[3].z,
        positions[4].z, positions[5].z, positions[6].z, positions[7].z,
    };

    // Calculate differences
    const dx = px - ox;
    const dy = py - oy;
    const dz = pz - oz;

    // Calculate squared distances: dx^2 + dy^2 + dz^2
    const dist_sq = dx * dx + dy * dy + dz * dz;

    return dist_sq;
}

/// Check if point is buried by any of 8 atoms.
///
/// # Parameters
/// - `point`: The test point to check
/// - `positions`: Array of 8 atom positions
/// - `radii_sq`: Array of 8 pre-computed (radius + probe)^2 values
///
/// # Returns
/// true if point is inside any of the 8 atoms, false otherwise
pub fn isPointBuriedBatch8(
    point: Vec3,
    positions: [8]Vec3,
    radii_sq: [8]f64,
) bool {
    const dist_sq = distanceSquaredBatch8(point, positions);
    const radii_v: @Vector(8, f64) = radii_sq;
    const dist_v: @Vector(8, f64) = dist_sq;

    // Check if any distance < radius (point inside atom)
    const inside = dist_v < radii_v;
    return @reduce(.Or, inside);
}

/// SIMD-optimized batch distance squared calculation for 16 atoms.
/// Process 16 positions simultaneously using @Vector(16, f64).
/// Best performance on AVX-512 enabled CPUs.
///
/// # Parameters
/// - `point`: The test point to measure distances from
/// - `positions`: Array of 16 Vec3 positions to measure to
///
/// # Returns
/// Array of 16 squared distances
pub fn distanceSquaredBatch16(
    point: Vec3,
    positions: [16]Vec3,
) [16]f64 {
    // Splat point coordinates into vectors
    const px: @Vector(16, f64) = @splat(point.x);
    const py: @Vector(16, f64) = @splat(point.y);
    const pz: @Vector(16, f64) = @splat(point.z);

    // Load other positions into vectors
    const ox = @Vector(16, f64){
        positions[0].x,  positions[1].x,  positions[2].x,  positions[3].x,
        positions[4].x,  positions[5].x,  positions[6].x,  positions[7].x,
        positions[8].x,  positions[9].x,  positions[10].x, positions[11].x,
        positions[12].x, positions[13].x, positions[14].x, positions[15].x,
    };
    const oy = @Vector(16, f64){
        positions[0].y,  positions[1].y,  positions[2].y,  positions[3].y,
        positions[4].y,  positions[5].y,  positions[6].y,  positions[7].y,
        positions[8].y,  positions[9].y,  positions[10].y, positions[11].y,
        positions[12].y, positions[13].y, positions[14].y, positions[15].y,
    };
    const oz = @Vector(16, f64){
        positions[0].z,  positions[1].z,  positions[2].z,  positions[3].z,
        positions[4].z,  positions[5].z,  positions[6].z,  positions[7].z,
        positions[8].z,  positions[9].z,  positions[10].z, positions[11].z,
        positions[12].z, positions[13].z, positions[14].z, positions[15].z,
    };

    // Calculate differences
    const dx = px - ox;
    const dy = py - oy;
    const dz = pz - oz;

    // Calculate squared distances: dx^2 + dy^2 + dz^2
    const dist_sq = dx * dx + dy * dy + dz * dz;

    return dist_sq;
}

/// Check if point is buried by any of 16 atoms.
///
/// # Parameters
/// - `point`: The test point to check
/// - `positions`: Array of 16 atom positions
/// - `radii_sq`: Array of 16 pre-computed (radius + probe)^2 values
///
/// # Returns
/// true if point is inside any of the 16 atoms, false otherwise
pub fn isPointBuriedBatch16(
    point: Vec3,
    positions: [16]Vec3,
    radii_sq: [16]f64,
) bool {
    const dist_sq = distanceSquaredBatch16(point, positions);
    const radii_v: @Vector(16, f64) = radii_sq;
    const dist_v: @Vector(16, f64) = dist_sq;

    // Check if any distance < radius (point inside atom)
    const inside = dist_v < radii_v;
    return @reduce(.Or, inside);
}

// ============================================================================
// Generic SIMD functions (f32/f64)
// ============================================================================

/// Generic SIMD-optimized batch distance squared calculation (4-wide).
/// Works with both f32 and f64.
pub fn distanceSquaredBatch4Gen(comptime T: type) type {
    const Vec = Vec3Gen(T);
    return struct {
        pub fn compute(point: Vec, positions: [4]Vec) [4]T {
            const px: @Vector(4, T) = @splat(point.x);
            const py: @Vector(4, T) = @splat(point.y);
            const pz: @Vector(4, T) = @splat(point.z);

            const ox = @Vector(4, T){ positions[0].x, positions[1].x, positions[2].x, positions[3].x };
            const oy = @Vector(4, T){ positions[0].y, positions[1].y, positions[2].y, positions[3].y };
            const oz = @Vector(4, T){ positions[0].z, positions[1].z, positions[2].z, positions[3].z };

            const dx = px - ox;
            const dy = py - oy;
            const dz = pz - oz;

            return dx * dx + dy * dy + dz * dz;
        }
    };
}

/// Generic check if point is buried by any of 4 atoms.
pub fn isPointBuriedBatch4Gen(comptime T: type) type {
    const Vec = Vec3Gen(T);
    return struct {
        pub fn compute(point: Vec, positions: [4]Vec, radii_sq: [4]T) bool {
            const dist_sq = distanceSquaredBatch4Gen(T).compute(point, positions);
            const radii_v: @Vector(4, T) = radii_sq;
            const dist_v: @Vector(4, T) = dist_sq;

            const inside = dist_v < radii_v;
            return @reduce(.Or, inside);
        }
    };
}

/// Generic SIMD-optimized batch distance squared calculation (8-wide).
/// Works with both f32 and f64.
pub fn distanceSquaredBatch8Gen(comptime T: type) type {
    const Vec = Vec3Gen(T);
    return struct {
        pub fn compute(point: Vec, positions: [8]Vec) [8]T {
            const px: @Vector(8, T) = @splat(point.x);
            const py: @Vector(8, T) = @splat(point.y);
            const pz: @Vector(8, T) = @splat(point.z);

            const ox = @Vector(8, T){
                positions[0].x, positions[1].x, positions[2].x, positions[3].x,
                positions[4].x, positions[5].x, positions[6].x, positions[7].x,
            };
            const oy = @Vector(8, T){
                positions[0].y, positions[1].y, positions[2].y, positions[3].y,
                positions[4].y, positions[5].y, positions[6].y, positions[7].y,
            };
            const oz = @Vector(8, T){
                positions[0].z, positions[1].z, positions[2].z, positions[3].z,
                positions[4].z, positions[5].z, positions[6].z, positions[7].z,
            };

            const dx = px - ox;
            const dy = py - oy;
            const dz = pz - oz;

            return dx * dx + dy * dy + dz * dz;
        }
    };
}

/// Generic check if point is buried by any of 8 atoms.
pub fn isPointBuriedBatch8Gen(comptime T: type) type {
    const Vec = Vec3Gen(T);
    return struct {
        pub fn compute(point: Vec, positions: [8]Vec, radii_sq: [8]T) bool {
            const dist_sq = distanceSquaredBatch8Gen(T).compute(point, positions);
            const radii_v: @Vector(8, T) = radii_sq;
            const dist_v: @Vector(8, T) = dist_sq;

            const inside = dist_v < radii_v;
            return @reduce(.Or, inside);
        }
    };
}

/// Generic SIMD-optimized batch distance squared calculation (16-wide).
/// Works with both f32 and f64. Best performance on AVX-512 enabled CPUs.
pub fn distanceSquaredBatch16Gen(comptime T: type) type {
    const Vec = Vec3Gen(T);
    return struct {
        pub fn compute(point: Vec, positions: [16]Vec) [16]T {
            const px: @Vector(16, T) = @splat(point.x);
            const py: @Vector(16, T) = @splat(point.y);
            const pz: @Vector(16, T) = @splat(point.z);

            const ox = @Vector(16, T){
                positions[0].x,  positions[1].x,  positions[2].x,  positions[3].x,
                positions[4].x,  positions[5].x,  positions[6].x,  positions[7].x,
                positions[8].x,  positions[9].x,  positions[10].x, positions[11].x,
                positions[12].x, positions[13].x, positions[14].x, positions[15].x,
            };
            const oy = @Vector(16, T){
                positions[0].y,  positions[1].y,  positions[2].y,  positions[3].y,
                positions[4].y,  positions[5].y,  positions[6].y,  positions[7].y,
                positions[8].y,  positions[9].y,  positions[10].y, positions[11].y,
                positions[12].y, positions[13].y, positions[14].y, positions[15].y,
            };
            const oz = @Vector(16, T){
                positions[0].z,  positions[1].z,  positions[2].z,  positions[3].z,
                positions[4].z,  positions[5].z,  positions[6].z,  positions[7].z,
                positions[8].z,  positions[9].z,  positions[10].z, positions[11].z,
                positions[12].z, positions[13].z, positions[14].z, positions[15].z,
            };

            const dx = px - ox;
            const dy = py - oy;
            const dz = pz - oz;

            return dx * dx + dy * dy + dz * dz;
        }
    };
}

/// Generic check if point is buried by any of 16 atoms.
pub fn isPointBuriedBatch16Gen(comptime T: type) type {
    const Vec = Vec3Gen(T);
    return struct {
        pub fn compute(point: Vec, positions: [16]Vec, radii_sq: [16]T) bool {
            const dist_sq = distanceSquaredBatch16Gen(T).compute(point, positions);
            const radii_v: @Vector(16, T) = radii_sq;
            const dist_v: @Vector(16, T) = dist_sq;

            const inside = dist_v < radii_v;
            return @reduce(.Or, inside);
        }
    };
}

// Tests

test "distanceSquaredBatch4 - correctness" {
    const point = Vec3{ .x = 0, .y = 0, .z = 0 };
    const positions = [4]Vec3{
        Vec3{ .x = 1, .y = 0, .z = 0 }, // dist^2 = 1
        Vec3{ .x = 0, .y = 2, .z = 0 }, // dist^2 = 4
        Vec3{ .x = 0, .y = 0, .z = 3 }, // dist^2 = 9
        Vec3{ .x = 1, .y = 1, .z = 1 }, // dist^2 = 3
    };

    const result = distanceSquaredBatch4(point, positions);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), result[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), result[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), result[3], 1e-10);
}

test "distanceSquaredBatch4 - non-origin point" {
    const point = Vec3{ .x = 1, .y = 2, .z = 3 };
    const positions = [4]Vec3{
        Vec3{ .x = 1, .y = 2, .z = 3 }, // dist^2 = 0
        Vec3{ .x = 2, .y = 2, .z = 3 }, // dist^2 = 1
        Vec3{ .x = 1, .y = 4, .z = 3 }, // dist^2 = 4
        Vec3{ .x = 4, .y = 6, .z = 3 }, // dist^2 = 9 + 16 = 25
    };

    const result = distanceSquaredBatch4(point, positions);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), result[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 25.0), result[3], 1e-10);
}

test "isPointBuriedBatch4 - none inside" {
    const point = Vec3{ .x = 0, .y = 0, .z = 0 };
    const positions = [4]Vec3{
        Vec3{ .x = 10, .y = 0, .z = 0 }, // dist^2 = 100
        Vec3{ .x = 0, .y = 10, .z = 0 }, // dist^2 = 100
        Vec3{ .x = 0, .y = 0, .z = 10 }, // dist^2 = 100
        Vec3{ .x = 10, .y = 10, .z = 10 }, // dist^2 = 300
    };
    const radii_sq = [4]f64{ 1.0, 1.0, 1.0, 1.0 }; // All radii^2 = 1

    try std.testing.expect(!isPointBuriedBatch4(point, positions, radii_sq));
}

test "isPointBuriedBatch4 - one inside (first)" {
    const point = Vec3{ .x = 0.5, .y = 0, .z = 0 };
    const positions = [4]Vec3{
        Vec3{ .x = 0, .y = 0, .z = 0 }, // dist^2 = 0.25 < 1.0
        Vec3{ .x = 10, .y = 0, .z = 0 }, // far
        Vec3{ .x = 10, .y = 0, .z = 0 }, // far
        Vec3{ .x = 10, .y = 0, .z = 0 }, // far
    };
    const radii_sq = [4]f64{ 1.0, 1.0, 1.0, 1.0 };

    try std.testing.expect(isPointBuriedBatch4(point, positions, radii_sq));
}

test "isPointBuriedBatch4 - one inside (last)" {
    const point = Vec3{ .x = 0.5, .y = 0, .z = 0 };
    const positions = [4]Vec3{
        Vec3{ .x = 10, .y = 0, .z = 0 }, // far
        Vec3{ .x = 10, .y = 0, .z = 0 }, // far
        Vec3{ .x = 10, .y = 0, .z = 0 }, // far
        Vec3{ .x = 0, .y = 0, .z = 0 }, // dist^2 = 0.25 < 1.0
    };
    const radii_sq = [4]f64{ 1.0, 1.0, 1.0, 1.0 };

    try std.testing.expect(isPointBuriedBatch4(point, positions, radii_sq));
}

test "isPointBuriedBatch4 - boundary case (exactly on radius)" {
    const point = Vec3{ .x = 1.0, .y = 0, .z = 0 };
    const positions = [4]Vec3{
        Vec3{ .x = 0, .y = 0, .z = 0 }, // dist^2 = 1.0 == radius^2
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 10, .y = 0, .z = 0 },
    };
    const radii_sq = [4]f64{ 1.0, 1.0, 1.0, 1.0 };

    // dist^2 == radius^2 means NOT inside (we use < not <=)
    try std.testing.expect(!isPointBuriedBatch4(point, positions, radii_sq));
}

test "isPointBuriedBatch4 - all inside" {
    const point = Vec3{ .x = 0, .y = 0, .z = 0 };
    const positions = [4]Vec3{
        Vec3{ .x = 0.1, .y = 0, .z = 0 },
        Vec3{ .x = 0, .y = 0.1, .z = 0 },
        Vec3{ .x = 0, .y = 0, .z = 0.1 },
        Vec3{ .x = 0.05, .y = 0.05, .z = 0 },
    };
    const radii_sq = [4]f64{ 1.0, 1.0, 1.0, 1.0 };

    try std.testing.expect(isPointBuriedBatch4(point, positions, radii_sq));
}

test "distanceSquaredBatch8 - correctness" {
    const point = Vec3{ .x = 0, .y = 0, .z = 0 };
    const positions = [8]Vec3{
        Vec3{ .x = 1, .y = 0, .z = 0 }, // dist^2 = 1
        Vec3{ .x = 0, .y = 2, .z = 0 }, // dist^2 = 4
        Vec3{ .x = 0, .y = 0, .z = 3 }, // dist^2 = 9
        Vec3{ .x = 1, .y = 1, .z = 1 }, // dist^2 = 3
        Vec3{ .x = 2, .y = 0, .z = 0 }, // dist^2 = 4
        Vec3{ .x = 0, .y = 3, .z = 0 }, // dist^2 = 9
        Vec3{ .x = 0, .y = 0, .z = 4 }, // dist^2 = 16
        Vec3{ .x = 1, .y = 1, .z = 0 }, // dist^2 = 2
    };

    const result = distanceSquaredBatch8(point, positions);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), result[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), result[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), result[3], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), result[4], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), result[5], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 16.0), result[6], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), result[7], 1e-10);
}

test "isPointBuriedBatch8 - none inside" {
    const point = Vec3{ .x = 0, .y = 0, .z = 0 };
    const positions = [8]Vec3{
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 0, .y = 10, .z = 0 },
        Vec3{ .x = 0, .y = 0, .z = 10 },
        Vec3{ .x = 10, .y = 10, .z = 0 },
        Vec3{ .x = 10, .y = 0, .z = 10 },
        Vec3{ .x = 0, .y = 10, .z = 10 },
        Vec3{ .x = 10, .y = 10, .z = 10 },
        Vec3{ .x = 5, .y = 5, .z = 5 },
    };
    const radii_sq = [8]f64{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

    try std.testing.expect(!isPointBuriedBatch8(point, positions, radii_sq));
}

test "isPointBuriedBatch8 - one inside (first)" {
    const point = Vec3{ .x = 0.5, .y = 0, .z = 0 };
    const positions = [8]Vec3{
        Vec3{ .x = 0, .y = 0, .z = 0 }, // dist^2 = 0.25 < 1.0
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 10, .y = 0, .z = 0 },
    };
    const radii_sq = [8]f64{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

    try std.testing.expect(isPointBuriedBatch8(point, positions, radii_sq));
}

test "isPointBuriedBatch8 - one inside (last)" {
    const point = Vec3{ .x = 0.5, .y = 0, .z = 0 };
    const positions = [8]Vec3{
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 0, .y = 0, .z = 0 }, // dist^2 = 0.25 < 1.0
    };
    const radii_sq = [8]f64{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

    try std.testing.expect(isPointBuriedBatch8(point, positions, radii_sq));
}

test "isPointBuriedBatch8 - boundary case (exactly on radius)" {
    const point = Vec3{ .x = 1.0, .y = 0, .z = 0 };
    const positions = [8]Vec3{
        Vec3{ .x = 0, .y = 0, .z = 0 }, // dist^2 = 1.0 == radius^2
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 10, .y = 0, .z = 0 },
        Vec3{ .x = 10, .y = 0, .z = 0 },
    };
    const radii_sq = [8]f64{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

    // dist^2 == radius^2 means NOT inside (we use < not <=)
    try std.testing.expect(!isPointBuriedBatch8(point, positions, radii_sq));
}

test "distanceSquaredBatch16 - correctness" {
    const point = Vec3{ .x = 0, .y = 0, .z = 0 };
    const positions = [16]Vec3{
        Vec3{ .x = 1, .y = 0, .z = 0 }, // dist^2 = 1
        Vec3{ .x = 0, .y = 2, .z = 0 }, // dist^2 = 4
        Vec3{ .x = 0, .y = 0, .z = 3 }, // dist^2 = 9
        Vec3{ .x = 1, .y = 1, .z = 1 }, // dist^2 = 3
        Vec3{ .x = 2, .y = 0, .z = 0 }, // dist^2 = 4
        Vec3{ .x = 0, .y = 3, .z = 0 }, // dist^2 = 9
        Vec3{ .x = 0, .y = 0, .z = 4 }, // dist^2 = 16
        Vec3{ .x = 1, .y = 1, .z = 0 }, // dist^2 = 2
        Vec3{ .x = 3, .y = 0, .z = 0 }, // dist^2 = 9
        Vec3{ .x = 0, .y = 4, .z = 0 }, // dist^2 = 16
        Vec3{ .x = 0, .y = 0, .z = 5 }, // dist^2 = 25
        Vec3{ .x = 2, .y = 2, .z = 0 }, // dist^2 = 8
        Vec3{ .x = 1, .y = 2, .z = 2 }, // dist^2 = 9
        Vec3{ .x = 2, .y = 1, .z = 2 }, // dist^2 = 9
        Vec3{ .x = 2, .y = 2, .z = 1 }, // dist^2 = 9
        Vec3{ .x = 1, .y = 0, .z = 1 }, // dist^2 = 2
    };

    const result = distanceSquaredBatch16(point, positions);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), result[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), result[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), result[3], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), result[4], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), result[5], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 16.0), result[6], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), result[7], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), result[8], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 16.0), result[9], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 25.0), result[10], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 8.0), result[11], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), result[12], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), result[13], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), result[14], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), result[15], 1e-10);
}

test "isPointBuriedBatch16 - none inside" {
    const point = Vec3{ .x = 0, .y = 0, .z = 0 };
    var positions: [16]Vec3 = undefined;
    for (0..16) |i| {
        positions[i] = Vec3{
            .x = 10.0 + @as(f64, @floatFromInt(i)),
            .y = 0,
            .z = 0,
        };
    }
    const radii_sq = [16]f64{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

    try std.testing.expect(!isPointBuriedBatch16(point, positions, radii_sq));
}

test "isPointBuriedBatch16 - one inside (first)" {
    const point = Vec3{ .x = 0.5, .y = 0, .z = 0 };
    var positions: [16]Vec3 = undefined;
    positions[0] = Vec3{ .x = 0, .y = 0, .z = 0 }; // dist^2 = 0.25 < 1.0
    for (1..16) |i| {
        positions[i] = Vec3{ .x = 10, .y = 0, .z = 0 }; // far
    }
    const radii_sq = [16]f64{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

    try std.testing.expect(isPointBuriedBatch16(point, positions, radii_sq));
}

test "isPointBuriedBatch16 - one inside (last)" {
    const point = Vec3{ .x = 0.5, .y = 0, .z = 0 };
    var positions: [16]Vec3 = undefined;
    for (0..15) |i| {
        positions[i] = Vec3{ .x = 10, .y = 0, .z = 0 }; // far
    }
    positions[15] = Vec3{ .x = 0, .y = 0, .z = 0 }; // dist^2 = 0.25 < 1.0
    const radii_sq = [16]f64{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

    try std.testing.expect(isPointBuriedBatch16(point, positions, radii_sq));
}

test "isPointBuriedBatch16 - one inside (middle)" {
    const point = Vec3{ .x = 0.5, .y = 0, .z = 0 };
    var positions: [16]Vec3 = undefined;
    for (0..16) |i| {
        if (i == 8) {
            positions[i] = Vec3{ .x = 0, .y = 0, .z = 0 }; // dist^2 = 0.25 < 1.0
        } else {
            positions[i] = Vec3{ .x = 10, .y = 0, .z = 0 }; // far
        }
    }
    const radii_sq = [16]f64{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

    try std.testing.expect(isPointBuriedBatch16(point, positions, radii_sq));
}

// ============================================================================
// CPU feature detection tests
// ============================================================================

test "cpu_features - compile-time detection works" {
    // These tests verify that the CPU feature detection compiles and
    // produces consistent values at compile time
    const has_avx512 = cpu_features.has_avx512f;
    const has_avx2 = cpu_features.has_avx2;
    const has_neon = cpu_features.has_neon;

    // On x86_64, if AVX-512 is available, AVX2 should also be available
    if (builtin.cpu.arch == .x86_64 and has_avx512) {
        try std.testing.expect(has_avx2);
    }

    // On aarch64, NEON should always be available
    if (builtin.cpu.arch == .aarch64) {
        try std.testing.expect(has_neon);
    }

    // Vector widths should be consistent with features
    if (has_avx512) {
        try std.testing.expectEqual(@as(comptime_int, 16), optimal_vector_width.f32_width);
        try std.testing.expectEqual(@as(comptime_int, 8), optimal_vector_width.f64_width);
    } else if (has_avx2 or has_neon) {
        try std.testing.expectEqual(@as(comptime_int, 8), optimal_vector_width.f32_width);
        if (has_avx2) {
            try std.testing.expectEqual(@as(comptime_int, 4), optimal_vector_width.f64_width);
        }
    }
}

// ============================================================================
// Generic SIMD tests (f32)
// ============================================================================

test "distanceSquaredBatch4Gen f32 - correctness" {
    const point = Vec3f32{ .x = 0, .y = 0, .z = 0 };
    const positions = [4]Vec3f32{
        Vec3f32{ .x = 1, .y = 0, .z = 0 }, // dist^2 = 1
        Vec3f32{ .x = 0, .y = 2, .z = 0 }, // dist^2 = 4
        Vec3f32{ .x = 0, .y = 0, .z = 3 }, // dist^2 = 9
        Vec3f32{ .x = 1, .y = 1, .z = 1 }, // dist^2 = 3
    };

    const result = distanceSquaredBatch4Gen(f32).compute(point, positions);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), result[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result[3], 1e-5);
}

test "isPointBuriedBatch4Gen f32 - one inside" {
    const point = Vec3f32{ .x = 0.5, .y = 0, .z = 0 };
    const positions = [4]Vec3f32{
        Vec3f32{ .x = 0, .y = 0, .z = 0 }, // dist^2 = 0.25 < 1.0
        Vec3f32{ .x = 10, .y = 0, .z = 0 }, // far
        Vec3f32{ .x = 10, .y = 0, .z = 0 }, // far
        Vec3f32{ .x = 10, .y = 0, .z = 0 }, // far
    };
    const radii_sq = [4]f32{ 1.0, 1.0, 1.0, 1.0 };

    try std.testing.expect(isPointBuriedBatch4Gen(f32).compute(point, positions, radii_sq));
}

test "isPointBuriedBatch8Gen f32 - one inside" {
    const point = Vec3f32{ .x = 0.5, .y = 0, .z = 0 };
    const positions = [8]Vec3f32{
        Vec3f32{ .x = 0, .y = 0, .z = 0 }, // dist^2 = 0.25 < 1.0
        Vec3f32{ .x = 10, .y = 0, .z = 0 },
        Vec3f32{ .x = 10, .y = 0, .z = 0 },
        Vec3f32{ .x = 10, .y = 0, .z = 0 },
        Vec3f32{ .x = 10, .y = 0, .z = 0 },
        Vec3f32{ .x = 10, .y = 0, .z = 0 },
        Vec3f32{ .x = 10, .y = 0, .z = 0 },
        Vec3f32{ .x = 10, .y = 0, .z = 0 },
    };
    const radii_sq = [8]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

    try std.testing.expect(isPointBuriedBatch8Gen(f32).compute(point, positions, radii_sq));
}

test "distanceSquaredBatch16Gen f32 - correctness" {
    const point = Vec3f32{ .x = 0, .y = 0, .z = 0 };
    var positions: [16]Vec3f32 = undefined;
    positions[0] = Vec3f32{ .x = 1, .y = 0, .z = 0 }; // dist^2 = 1
    positions[1] = Vec3f32{ .x = 0, .y = 2, .z = 0 }; // dist^2 = 4
    positions[2] = Vec3f32{ .x = 0, .y = 0, .z = 3 }; // dist^2 = 9
    positions[3] = Vec3f32{ .x = 1, .y = 1, .z = 1 }; // dist^2 = 3
    for (4..16) |i| {
        const fi: f32 = @floatFromInt(i);
        positions[i] = Vec3f32{ .x = fi, .y = 0, .z = 0 }; // dist^2 = i^2
    }

    const result = distanceSquaredBatch16Gen(f32).compute(point, positions);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), result[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result[3], 1e-5);
    for (4..16) |i| {
        const fi: f32 = @floatFromInt(i);
        try std.testing.expectApproxEqAbs(fi * fi, result[i], 1e-5);
    }
}

test "isPointBuriedBatch16Gen f32 - one inside" {
    const point = Vec3f32{ .x = 0.5, .y = 0, .z = 0 };
    var positions: [16]Vec3f32 = undefined;
    positions[0] = Vec3f32{ .x = 0, .y = 0, .z = 0 }; // dist^2 = 0.25 < 1.0
    for (1..16) |i| {
        positions[i] = Vec3f32{ .x = 10, .y = 0, .z = 0 }; // far
    }
    const radii_sq = [16]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

    try std.testing.expect(isPointBuriedBatch16Gen(f32).compute(point, positions, radii_sq));
}

test "isPointBuriedBatch16Gen f32 - none inside" {
    const point = Vec3f32{ .x = 0, .y = 0, .z = 0 };
    var positions: [16]Vec3f32 = undefined;
    for (0..16) |i| {
        positions[i] = Vec3f32{
            .x = 10.0 + @as(f32, @floatFromInt(i)),
            .y = 0,
            .z = 0,
        };
    }
    const radii_sq = [16]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

    try std.testing.expect(!isPointBuriedBatch16Gen(f32).compute(point, positions, radii_sq));
}
