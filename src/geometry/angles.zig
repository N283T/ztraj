//! Bond angle calculation for atom triplets (i-j-k, angle at j).

const std = @import("std");

/// Compute bond angles (in radians) for a list of atom triplets.
///
/// For each triplet (i, j, k), computes the angle at atom j:
///   v1 = pos[i] - pos[j]
///   v2 = pos[k] - pos[j]
///   angle = acos(dot(v1,v2) / (|v1| * |v2|))
///
/// The dot product is clamped to [-1, 1] before acos to guard against
/// floating-point rounding outside the domain.
/// Uses f64 intermediate precision.
/// `result` must have the same length as `triplets`. Values are in radians.
pub fn compute(
    x: []const f32,
    y: []const f32,
    z: []const f32,
    triplets: []const [3]u32,
    result: []f32,
) void {
    std.debug.assert(result.len == triplets.len);

    for (triplets, 0..) |tri, idx| {
        const i = tri[0];
        const j = tri[1];
        const k = tri[2];

        // Vectors from j to i and from j to k
        const v1x: f64 = @as(f64, x[i]) - @as(f64, x[j]);
        const v1y: f64 = @as(f64, y[i]) - @as(f64, y[j]);
        const v1z: f64 = @as(f64, z[i]) - @as(f64, z[j]);

        const v2x: f64 = @as(f64, x[k]) - @as(f64, x[j]);
        const v2y: f64 = @as(f64, y[k]) - @as(f64, y[j]);
        const v2z: f64 = @as(f64, z[k]) - @as(f64, z[j]);

        const dot = v1x * v2x + v1y * v2y + v1z * v2z;
        const len1 = @sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
        const len2 = @sqrt(v2x * v2x + v2y * v2y + v2z * v2z);

        const cos_angle = std.math.clamp(dot / (len1 * len2), -1.0, 1.0);
        result[idx] = @floatCast(std.math.acos(cos_angle));
    }
}

// ============================================================================
// Tests
// ============================================================================

test "angles: 90 degree angle" {
    // atoms at (1,0,0), (0,0,0), (0,1,0) -> angle at origin = 90 deg
    const x = [_]f32{ 1.0, 0.0, 0.0 };
    const y = [_]f32{ 0.0, 0.0, 1.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0 };
    const triplets = [_][3]u32{.{ 0, 1, 2 }};
    var result = [_]f32{0.0};

    compute(&x, &y, &z, &triplets, &result);

    try std.testing.expectApproxEqAbs(
        @as(f32, std.math.pi / 2.0),
        result[0],
        1e-5,
    );
}

test "angles: 180 degree angle" {
    // atoms at (-1,0,0), (0,0,0), (1,0,0) -> straight line = pi radians
    const x = [_]f32{ -1.0, 0.0, 1.0 };
    const y = [_]f32{ 0.0, 0.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0 };
    const triplets = [_][3]u32{.{ 0, 1, 2 }};
    var result = [_]f32{0.0};

    compute(&x, &y, &z, &triplets, &result);

    try std.testing.expectApproxEqAbs(
        @as(f32, std.math.pi),
        result[0],
        1e-5,
    );
}

test "angles: 60 degree angle (equilateral triangle)" {
    // Equilateral triangle with side length 1
    // vertex 0: (1, 0, 0)
    // vertex 1: (0, 0, 0)  <- angle vertex
    // vertex 2: (0.5, sqrt(3)/2, 0)
    const sqrt3_over2: f32 = @floatCast(@sqrt(3.0) / 2.0);
    const x = [_]f32{ 1.0, 0.0, 0.5 };
    const y = [_]f32{ 0.0, 0.0, sqrt3_over2 };
    const z = [_]f32{ 0.0, 0.0, 0.0 };
    const triplets = [_][3]u32{.{ 0, 1, 2 }};
    var result = [_]f32{0.0};

    compute(&x, &y, &z, &triplets, &result);

    try std.testing.expectApproxEqAbs(
        @as(f32, std.math.pi / 3.0),
        result[0],
        1e-5,
    );
}

test "angles: multiple triplets" {
    // atoms: (1,0,0), (0,0,0), (0,1,0), (-1,0,0)
    // triplet 0: atoms 0-1-2 -> 90 deg
    // triplet 1: atoms 3-1-2 -> 90 deg
    // triplet 2: atoms 0-1-3 -> 180 deg
    const x = [_]f32{ 1.0, 0.0, 0.0, -1.0 };
    const y = [_]f32{ 0.0, 0.0, 1.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const triplets = [_][3]u32{
        .{ 0, 1, 2 },
        .{ 3, 1, 2 },
        .{ 0, 1, 3 },
    };
    var result = [_]f32{ 0.0, 0.0, 0.0 };

    compute(&x, &y, &z, &triplets, &result);

    try std.testing.expectApproxEqAbs(@as(f32, std.math.pi / 2.0), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, std.math.pi / 2.0), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, std.math.pi), result[2], 1e-5);
}

test "angles: zero triplets is no-op" {
    const x = [_]f32{1.0};
    const y = [_]f32{1.0};
    const z = [_]f32{1.0};
    const triplets = [_][3]u32{};
    var result = [_]f32{};

    compute(&x, &y, &z, &triplets, &result);
}
