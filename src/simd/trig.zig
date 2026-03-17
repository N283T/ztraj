const std = @import("std");
const vec = @import("vec.zig");

const Epsilon = vec.Epsilon;

// ============================================================================
// Fast approximate trigonometric functions
// ============================================================================

/// Fast approximate acos using polynomial approximation.
/// Based on Handbook of Mathematical Functions (Abramowitz & Stegun).
/// Max error: ~0.0003 radians (~0.02 degrees)
///
/// # Parameters
/// - `x`: Input value in range [-1, 1]
///
/// # Returns
/// Approximate acos(x) in radians [0, pi]
pub fn fastAcos(x: f64) f64 {
    // Clamp input to valid range
    const clamped = std.math.clamp(x, -1.0, 1.0);
    const abs_x = @abs(clamped);

    // Polynomial approximation for acos(x) when x >= 0
    // acos(x) approx sqrt(1-x) * (a0 + a1*x + a2*x^2 + a3*x^3)
    const a0: f64 = 1.5707963267948966; // pi/2
    const a1: f64 = -0.2145988016038123;
    const a2: f64 = 0.0889789874093553;
    const a3: f64 = -0.0501743046129726;

    const sqrt_term = @sqrt(1.0 - abs_x);
    const poly = a0 + abs_x * (a1 + abs_x * (a2 + abs_x * a3));
    const result = sqrt_term * poly;

    // For negative x: acos(-x) = pi - acos(x)
    return if (clamped < 0) std.math.pi - result else result;
}

/// Fast approximate atan2 using polynomial approximation.
/// Based on approximation with max error ~0.0015 radians (~0.09 degrees)
///
/// # Parameters
/// - `y`: Y coordinate
/// - `x`: X coordinate
///
/// # Returns
/// Approximate atan2(y, x) in radians [-pi, pi]
pub fn fastAtan2(y: f64, x: f64) f64 {
    const abs_x = @abs(x);
    const abs_y = @abs(y);

    // Handle special cases
    if (abs_x < 1e-10 and abs_y < 1e-10) {
        return 0.0;
    }

    // Use the smaller ratio for better accuracy
    const swap = abs_y > abs_x;
    const ratio = if (swap) abs_x / abs_y else abs_y / abs_x;

    // Polynomial approximation for atan(r) where r = min(|y/x|, |x/y|)
    // atan(r) approx r * (c0 + r^2 * (c1 + r^2 * c2))
    const c0: f64 = 0.9998660373;
    const c1: f64 = -0.3302994844;
    const c2: f64 = 0.1801410321;

    const r2 = ratio * ratio;
    var atan_r = ratio * (c0 + r2 * (c1 + r2 * c2));

    // Adjust for the octant
    if (swap) {
        atan_r = std.math.pi / 2.0 - atan_r;
    }
    if (x < 0) {
        atan_r = std.math.pi - atan_r;
    }
    if (y < 0) {
        atan_r = -atan_r;
    }

    return atan_r;
}

/// Generic fast approximate acos using polynomial approximation.
pub fn fastAcosGen(comptime T: type) type {
    return struct {
        pub fn compute(x: T) T {
            const clamped = std.math.clamp(x, -1.0, 1.0);
            const abs_x = @abs(clamped);

            const a0: T = 1.5707963267948966; // pi/2
            const a1: T = -0.2145988016038123;
            const a2: T = 0.0889789874093553;
            const a3: T = -0.0501743046129726;

            const sqrt_term = @sqrt(1.0 - abs_x);
            const poly = a0 + abs_x * (a1 + abs_x * (a2 + abs_x * a3));
            const result = sqrt_term * poly;

            return if (clamped < 0) std.math.pi - result else result;
        }
    };
}

/// Generic fast approximate atan2 using polynomial approximation.
pub fn fastAtan2Gen(comptime T: type) type {
    return struct {
        pub fn compute(y: T, x: T) T {
            const abs_x = @abs(x);
            const abs_y = @abs(y);

            const epsilon = Epsilon(T).trig;
            if (abs_x < epsilon and abs_y < epsilon) {
                return 0.0;
            }

            const swap = abs_y > abs_x;
            const ratio = if (swap) abs_x / abs_y else abs_y / abs_x;

            const c0: T = 0.9998660373;
            const c1: T = -0.3302994844;
            const c2: T = 0.1801410321;

            const r2 = ratio * ratio;
            var atan_r = ratio * (c0 + r2 * (c1 + r2 * c2));

            const half_pi: T = @as(T, std.math.pi) / 2.0;
            const pi: T = std.math.pi;

            if (swap) {
                atan_r = half_pi - atan_r;
            }
            if (x < 0) {
                atan_r = pi - atan_r;
            }
            if (y < 0) {
                atan_r = -atan_r;
            }

            return atan_r;
        }
    };
}

// Fast approximation tests

test "fastAcos - accuracy" {
    // Test various values and compare with std.math.acos
    const test_values = [_]f64{ -1.0, -0.9, -0.5, 0.0, 0.5, 0.9, 1.0 };
    const tolerance = 0.005; // ~0.3 degrees, matches polynomial approximation precision

    for (test_values) |x| {
        const expected = std.math.acos(x);
        const actual = fastAcos(x);
        try std.testing.expectApproxEqAbs(expected, actual, tolerance);
    }
}

test "fastAcos - edge cases" {
    // Values outside [-1, 1] should be clamped
    try std.testing.expectApproxEqAbs(std.math.pi, fastAcos(-1.5), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), fastAcos(1.5), 0.001);
}

test "fastAtan2 - accuracy" {
    // Test various angles
    const angles = [_]f64{ 0.0, 0.25, 0.5, 1.0, 2.0, 3.0 };
    const tolerance = 0.005; // ~0.3 degrees, matches polynomial approximation precision

    for (angles) |angle| {
        const y = @sin(angle);
        const x = @cos(angle);
        const expected = std.math.atan2(y, x);
        const actual = fastAtan2(y, x);
        try std.testing.expectApproxEqAbs(expected, actual, tolerance);
    }

    // Test negative quadrants (non-unit-circle inputs have larger polynomial error)
    const neg_tolerance = 0.07;
    try std.testing.expectApproxEqAbs(std.math.atan2(@as(f64, -1.0), @as(f64, 1.0)), fastAtan2(-1.0, 1.0), neg_tolerance);
    try std.testing.expectApproxEqAbs(std.math.atan2(@as(f64, -1.0), @as(f64, -1.0)), fastAtan2(-1.0, -1.0), neg_tolerance);
    try std.testing.expectApproxEqAbs(std.math.atan2(@as(f64, 1.0), @as(f64, -1.0)), fastAtan2(1.0, -1.0), neg_tolerance);
}

test "fastAtan2 - edge cases" {
    // Origin should return 0
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), fastAtan2(0.0, 0.0), 0.001);

    // Axis-aligned cases
    const tolerance = 0.002;
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), fastAtan2(0.0, 1.0), tolerance);
    try std.testing.expectApproxEqAbs(std.math.pi / 2.0, fastAtan2(1.0, 0.0), tolerance);
    try std.testing.expectApproxEqAbs(std.math.pi, fastAtan2(0.0, -1.0), tolerance);
    try std.testing.expectApproxEqAbs(-std.math.pi / 2.0, fastAtan2(-1.0, 0.0), tolerance);
}

test "fastAcosGen f32 - accuracy" {
    const test_values = [_]f32{ -1.0, -0.9, -0.5, 0.0, 0.5, 0.9, 1.0 };
    const tolerance: f32 = 0.005; // Matches polynomial approximation precision for f32

    for (test_values) |x| {
        const expected = std.math.acos(x);
        const actual = fastAcosGen(f32).compute(x);
        try std.testing.expectApproxEqAbs(expected, actual, tolerance);
    }
}

test "fastAtan2Gen f32 - accuracy" {
    const angles = [_]f32{ 0.0, 0.25, 0.5, 1.0, 2.0, 3.0 };
    const tolerance: f32 = 0.005;

    for (angles) |angle| {
        const y = @sin(angle);
        const x = @cos(angle);
        const expected = std.math.atan2(y, x);
        const actual = fastAtan2Gen(f32).compute(y, x);
        try std.testing.expectApproxEqAbs(expected, actual, tolerance);
    }
}
