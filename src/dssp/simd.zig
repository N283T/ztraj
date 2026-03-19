const std = @import("std");
const builtin = @import("builtin");
const math = std.math;

// ---------------------------------------------------------------------------
// SIMD utilities for f32 distance calculations
// ---------------------------------------------------------------------------

/// Optimal vector width for f32 on the current CPU target.
pub const vec_len: usize = blk: {
    if (builtin.cpu.arch == .x86_64 or builtin.cpu.arch == .x86) {
        if (std.Target.x86.featureSetHas(builtin.cpu.model.features, .avx512f)) {
            break :blk 16; // 512-bit = 16 × f32
        } else if (std.Target.x86.featureSetHas(builtin.cpu.model.features, .avx2)) {
            break :blk 8; // 256-bit = 8 × f32
        } else {
            break :blk 4; // 128-bit SSE = 4 × f32
        }
    } else if (builtin.cpu.arch == .aarch64) {
        break :blk 4; // NEON = 4 × f32
    } else {
        break :blk 4; // Fallback
    }
};

/// f32 vector type for the current CPU.
pub const F32xN = @Vector(vec_len, f32);

// ---------------------------------------------------------------------------
// Batch distance squared
// ---------------------------------------------------------------------------

/// Compute distance-squared from a reference point to 4 other points.
///
/// Returns [4]f32 where each element is the squared Euclidean distance
/// from (ref_x, ref_y, ref_z) to the corresponding point.
pub fn distanceSquaredBatch4(
    ref_x: f32,
    ref_y: f32,
    ref_z: f32,
    xs: [4]f32,
    ys: [4]f32,
    zs: [4]f32,
) [4]f32 {
    const rx: @Vector(4, f32) = @splat(ref_x);
    const ry: @Vector(4, f32) = @splat(ref_y);
    const rz: @Vector(4, f32) = @splat(ref_z);

    const vx: @Vector(4, f32) = xs;
    const vy: @Vector(4, f32) = ys;
    const vz: @Vector(4, f32) = zs;

    const dx = vx - rx;
    const dy = vy - ry;
    const dz = vz - rz;

    return dx * dx + dy * dy + dz * dz;
}

/// Generic batch point burial check using comptime-known batch size.
///
/// Returns true if the test point is within radius_sq of any neighbor.
pub fn isPointBuriedBatch(
    comptime N: usize,
    test_x: f32,
    test_y: f32,
    test_z: f32,
    nb_xs: [N]f32,
    nb_ys: [N]f32,
    nb_zs: [N]f32,
    radii_sq: [N]f32,
) bool {
    const rx: @Vector(N, f32) = @splat(test_x);
    const ry: @Vector(N, f32) = @splat(test_y);
    const rz: @Vector(N, f32) = @splat(test_z);

    const vx: @Vector(N, f32) = nb_xs;
    const vy: @Vector(N, f32) = nb_ys;
    const vz: @Vector(N, f32) = nb_zs;

    const dx = vx - rx;
    const dy = vy - ry;
    const dz = vz - rz;

    const dist_sq = dx * dx + dy * dy + dz * dz;
    const radii_v: @Vector(N, f32) = radii_sq;
    const inside = dist_sq < radii_v;
    return @reduce(.Or, inside);
}

/// Check if a surface test point is buried by any of 4 neighbor atoms.
/// Wrapper for backward compatibility.
pub fn isPointBuriedBatch4(
    test_x: f32,
    test_y: f32,
    test_z: f32,
    nb_xs: [4]f32,
    nb_ys: [4]f32,
    nb_zs: [4]f32,
    radii_sq: [4]f32,
) bool {
    return isPointBuriedBatch(4, test_x, test_y, test_z, nb_xs, nb_ys, nb_zs, radii_sq);
}

/// Check if a surface test point is buried by any of 8 neighbor atoms.
/// Wrapper for backward compatibility.
pub fn isPointBuriedBatch8(
    test_x: f32,
    test_y: f32,
    test_z: f32,
    nb_xs: [8]f32,
    nb_ys: [8]f32,
    nb_zs: [8]f32,
    radii_sq: [8]f32,
) bool {
    return isPointBuriedBatch(8, test_x, test_y, test_z, nb_xs, nb_ys, nb_zs, radii_sq);
}

// ---------------------------------------------------------------------------
// Batch 4 reciprocal distances for H-bond energy
// ---------------------------------------------------------------------------

/// Compute 4 distances simultaneously (HO, HC, NC, NO) and return
/// the reciprocals. This is used in the H-bond energy formula:
///   E = C * (1/d_HO - 1/d_HC + 1/d_NC - 1/d_NO)
///
/// Returns [4]f64: { 1/d_HO, 1/d_HC, 1/d_NC, 1/d_NO }
/// Returns null if any distance is below the minimal threshold.
pub fn hbondReciprocalDistances(
    h_x: f32,
    h_y: f32,
    h_z: f32,
    n_x: f32,
    n_y: f32,
    n_z: f32,
    o_x: f32,
    o_y: f32,
    o_z: f32,
    c_x: f32,
    c_y: f32,
    c_z: f32,
) ?[4]f64 {
    // Points: H, H, N, N  paired with  O, C, C, O
    const px: @Vector(4, f32) = .{ h_x, h_x, n_x, n_x };
    const py: @Vector(4, f32) = .{ h_y, h_y, n_y, n_y };
    const pz: @Vector(4, f32) = .{ h_z, h_z, n_z, n_z };
    const qx: @Vector(4, f32) = .{ o_x, c_x, c_x, o_x };
    const qy: @Vector(4, f32) = .{ o_y, c_y, c_y, o_y };
    const qz: @Vector(4, f32) = .{ o_z, c_z, c_z, o_z };

    const dx = px - qx;
    const dy = py - qy;
    const dz = pz - qz;

    const dist_sq = dx * dx + dy * dy + dz * dz;

    // Check minimum distance (0.5 Å)
    const min_sq: @Vector(4, f32) = @splat(0.5 * 0.5);
    const too_close = dist_sq < min_sq;
    if (@reduce(.Or, too_close)) return null;

    // Compute sqrt and reciprocal in f64 for energy precision
    const dist_sq_arr: [4]f32 = dist_sq;
    var result: [4]f64 = undefined;
    for (0..4) |i| {
        const d: f64 = @sqrt(@as(f64, @floatCast(dist_sq_arr[i])));
        result[i] = 1.0 / d;
    }

    return result;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "distanceSquaredBatch4 - basic" {
    const result = distanceSquaredBatch4(
        0.0,
        0.0,
        0.0,
        .{ 1.0, 2.0, 3.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 4.0 },
        .{ 0.0, 0.0, 0.0, 0.0 },
    );

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), result[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), result[3], 0.001);
}

test "isPointBuriedBatch4 - point inside one atom" {
    const buried = isPointBuriedBatch4(
        0.5,
        0.0,
        0.0,
        .{ 0.0, 10.0, 10.0, 10.0 },
        .{ 0.0, 10.0, 10.0, 10.0 },
        .{ 0.0, 10.0, 10.0, 10.0 },
        .{ 1.0, 1.0, 1.0, 1.0 }, // radius_sq = 1.0
    );
    try std.testing.expect(buried); // dist_sq = 0.25 < 1.0
}

test "isPointBuriedBatch4 - point outside all atoms" {
    const buried = isPointBuriedBatch4(
        5.0,
        5.0,
        5.0,
        .{ 0.0, 10.0, 0.0, 10.0 },
        .{ 0.0, 0.0, 10.0, 10.0 },
        .{ 0.0, 0.0, 0.0, 0.0 },
        .{ 1.0, 1.0, 1.0, 1.0 },
    );
    try std.testing.expect(!buried);
}

test "isPointBuriedBatch8 - point inside one atom" {
    const buried = isPointBuriedBatch8(
        0.5,
        0.0,
        0.0,
        .{ 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0 },
        .{ 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0 },
        .{ 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0 },
        .{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
    );
    try std.testing.expect(buried);
}

test "hbondReciprocalDistances - basic" {
    // H at (2, 0.5, 0), N at (3, 0, 0), O at (0, 1.2, 0), C at (0, 0, 0)
    const result = hbondReciprocalDistances(
        2.0,
        0.5,
        0.0, // H
        3.0,
        0.0,
        0.0, // N
        0.0,
        1.2,
        0.0, // O
        0.0,
        0.0,
        0.0, // C
    );

    try std.testing.expect(result != null);
    const r = result.?;
    // All reciprocals should be positive
    for (r) |v| {
        try std.testing.expect(v > 0.0);
    }
}

test "hbondReciprocalDistances - too close returns null" {
    // H and O at nearly the same position
    const result = hbondReciprocalDistances(
        0.0,
        0.0,
        0.0, // H
        3.0,
        0.0,
        0.0, // N
        0.1,
        0.0,
        0.0, // O (very close to H)
        0.0,
        0.0,
        5.0, // C
    );

    try std.testing.expectEqual(@as(?[4]f64, null), result);
}

test "vec_len - at least 4" {
    try std.testing.expect(vec_len >= 4);
}
