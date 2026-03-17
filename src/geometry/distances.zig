//! Pairwise atom distance calculation.

const std = @import("std");

/// Compute Euclidean distances for given atom index pairs.
///
/// For each pair (i, j) in `pairs`, computes:
///   result[k] = sqrt((x[j]-x[i])^2 + (y[j]-y[i])^2 + (z[j]-z[i])^2)
///
/// Uses f64 intermediate precision to minimize floating-point error.
/// `result` must have the same length as `pairs`.
pub fn compute(
    x: []const f32,
    y: []const f32,
    z: []const f32,
    pairs: []const [2]u32,
    result: []f32,
) void {
    std.debug.assert(result.len == pairs.len);

    for (pairs, 0..) |pair, idx| {
        const i = pair[0];
        const j = pair[1];
        const dx: f64 = @as(f64, x[j]) - @as(f64, x[i]);
        const dy: f64 = @as(f64, y[j]) - @as(f64, y[i]);
        const dz: f64 = @as(f64, z[j]) - @as(f64, z[i]);
        result[idx] = @floatCast(@sqrt(dx * dx + dy * dy + dz * dz));
    }
}

// ============================================================================
// Tests
// ============================================================================

test "distances: 3-4-5 right triangle" {
    // atoms at (0,0,0) and (3,4,0) -> distance = 5.0
    const x = [_]f32{ 0.0, 3.0 };
    const y = [_]f32{ 0.0, 4.0 };
    const z = [_]f32{ 0.0, 0.0 };
    const pairs = [_][2]u32{.{ 0, 1 }};
    var result = [_]f32{0.0};

    compute(&x, &y, &z, &pairs, &result);

    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result[0], 1e-5);
}

test "distances: coincident atoms" {
    const x = [_]f32{ 1.5, 1.5 };
    const y = [_]f32{ 2.3, 2.3 };
    const z = [_]f32{ -0.7, -0.7 };
    const pairs = [_][2]u32{.{ 0, 1 }};
    var result = [_]f32{0.0};

    compute(&x, &y, &z, &pairs, &result);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 1e-6);
}

test "distances: multiple pairs" {
    // 3 atoms: origin, (1,0,0), (0,2,0)
    const x = [_]f32{ 0.0, 1.0, 0.0 };
    const y = [_]f32{ 0.0, 0.0, 2.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0 };

    const pairs = [_][2]u32{
        .{ 0, 1 }, // distance = 1.0
        .{ 0, 2 }, // distance = 2.0
        .{ 1, 2 }, // distance = sqrt(1+4) = sqrt(5)
    };
    var result = [_]f32{ 0.0, 0.0, 0.0 };

    compute(&x, &y, &z, &pairs, &result);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, @sqrt(5.0)), result[2], 1e-5);
}

test "distances: axis-aligned 3D" {
    // atoms at (0,0,0) and (0,0,7) -> distance = 7.0
    const x = [_]f32{ 0.0, 0.0 };
    const y = [_]f32{ 0.0, 0.0 };
    const z = [_]f32{ 0.0, 7.0 };
    const pairs = [_][2]u32{.{ 0, 1 }};
    var result = [_]f32{0.0};

    compute(&x, &y, &z, &pairs, &result);

    try std.testing.expectApproxEqAbs(@as(f32, 7.0), result[0], 1e-5);
}

test "distances: zero pairs is no-op" {
    const x = [_]f32{1.0};
    const y = [_]f32{1.0};
    const z = [_]f32{1.0};
    const pairs = [_][2]u32{};
    var result = [_]f32{};

    compute(&x, &y, &z, &pairs, &result); // must not crash
}
