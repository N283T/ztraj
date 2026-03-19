const std = @import("std");
const math = std.math;
const types = @import("types.zig");
const Vec3f32 = types.Vec3f32;

/// Dihedral angle (torsion angle) between four points, in degrees.
///
/// Computes the angle along the bond p2-p3.  Returns 360.0 when the
/// angle is undefined (degenerate geometry).
///
/// Port of dssp.cpp:98-122.
pub fn dihedralAngle(p1: Vec3f32, p2: Vec3f32, p3: Vec3f32, p4: Vec3f32) f32 {
    const v12 = p1.sub(p2);
    const v43 = p4.sub(p3);
    const z = p2.sub(p3);

    const p = z.cross(v12);
    const x = z.cross(v43);
    const y = z.cross(x);

    const u_sq = x.dot(x);
    const v_sq = y.dot(y);

    if (u_sq > 0 and v_sq > 0) {
        const u = p.dot(x) / @sqrt(u_sq);
        const v = p.dot(y) / @sqrt(v_sq);
        if (u != 0 or v != 0) {
            return math.atan2(v, u) * (180.0 / math.pi);
        }
    }
    return 360.0;
}

/// Cosine of the angle between vectors (p1-p2) and (p3-p4).
///
/// Returns the cosine in [-1, 1], or 0.0 when degenerate.
///
/// Port of dssp.cpp:124-136.
pub fn cosinusAngle(p1: Vec3f32, p2: Vec3f32, p3: Vec3f32, p4: Vec3f32) f32 {
    const v12 = p1.sub(p2);
    const v34 = p3.sub(p4);

    const denom = v12.dot(v12) * v34.dot(v34);
    if (denom > 0) {
        return v12.dot(v34) / @sqrt(denom);
    }
    return 0.0;
}

/// Kappa angle: virtual bond angle at Cα(i) defined by Cα(i-2), Cα(i), Cα(i+2).
///
/// Returns the angle in degrees, computed via acos of the cosine of the angle
/// between vectors (CA[i-2] - CA[i]) and (CA[i+2] - CA[i]).
///
/// Port of dssp.cpp:1504-1513.
pub fn kappaAngle(ca_prev2: Vec3f32, ca: Vec3f32, ca_next2: Vec3f32) f32 {
    // C++ DSSP: cosinus_angle(cur.mCAlpha, prevPrev.mCAlpha, nextNext.mCAlpha, cur.mCAlpha)
    // v12 = CA_i - CA_{i-2}, v34 = CA_{i+2} - CA_i
    const cosine = cosinusAngle(ca, ca_prev2, ca_next2, ca);
    const clamped = math.clamp(cosine, -1.0, 1.0);
    return math.acos(clamped) * (180.0 / math.pi);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "dihedral angle - known alpha helix geometry" {
    // Ideal alpha helix backbone geometry:
    // phi ≈ -57°, psi ≈ -47°
    // Using approximate coordinates for N, CA, C, N_next in an alpha helix

    // Simple test: dihedral of 4 points in XY plane should be ~0 or ~180
    const p1 = Vec3f32{ .x = 1.0, .y = 0.0, .z = 0.0 };
    const p2 = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const p3 = Vec3f32{ .x = 0.0, .y = 1.0, .z = 0.0 };
    const p4 = Vec3f32{ .x = 1.0, .y = 1.0, .z = 0.0 };

    const angle = dihedralAngle(p1, p2, p3, p4);
    // All points coplanar → dihedral = 0°
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), angle, 0.01);
}

test "dihedral angle - 90 degrees" {
    // p1-p2 along x, p2-p3 along y, p4 shifted along -z from p3
    const p1 = Vec3f32{ .x = 1.0, .y = 0.0, .z = 0.0 };
    const p2 = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const p3 = Vec3f32{ .x = 0.0, .y = 1.0, .z = 0.0 };
    const p4 = Vec3f32{ .x = 0.0, .y = 1.0, .z = -1.0 };

    const angle = dihedralAngle(p1, p2, p3, p4);
    try std.testing.expectApproxEqAbs(@as(f32, 90.0), angle, 0.01);
}

test "dihedral angle - negative 90 degrees" {
    const p1 = Vec3f32{ .x = 1.0, .y = 0.0, .z = 0.0 };
    const p2 = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const p3 = Vec3f32{ .x = 0.0, .y = 1.0, .z = 0.0 };
    const p4 = Vec3f32{ .x = 0.0, .y = 1.0, .z = 1.0 };

    const angle = dihedralAngle(p1, p2, p3, p4);
    try std.testing.expectApproxEqAbs(@as(f32, -90.0), angle, 0.01);
}

test "dihedral angle - degenerate returns 360" {
    const p = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const angle = dihedralAngle(p, p, p, p);
    try std.testing.expectApproxEqAbs(@as(f32, 360.0), angle, 0.01);
}

test "cosinus angle - parallel vectors" {
    const p1 = Vec3f32{ .x = 2.0, .y = 0.0, .z = 0.0 };
    const p2 = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const p3 = Vec3f32{ .x = 3.0, .y = 0.0, .z = 0.0 };
    const p4 = Vec3f32{ .x = 1.0, .y = 0.0, .z = 0.0 };

    const cos = cosinusAngle(p1, p2, p3, p4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cos, 1e-6);
}

test "cosinus angle - perpendicular vectors" {
    const p1 = Vec3f32{ .x = 1.0, .y = 0.0, .z = 0.0 };
    const p2 = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const p3 = Vec3f32{ .x = 0.0, .y = 1.0, .z = 0.0 };
    const p4 = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 };

    const cos = cosinusAngle(p1, p2, p3, p4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cos, 1e-6);
}

test "cosinus angle - antiparallel vectors" {
    const p1 = Vec3f32{ .x = -1.0, .y = 0.0, .z = 0.0 };
    const p2 = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const p3 = Vec3f32{ .x = 1.0, .y = 0.0, .z = 0.0 };
    const p4 = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 };

    const cos = cosinusAngle(p1, p2, p3, p4);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), cos, 1e-6);
}

test "kappa angle - straight chain" {
    // Three CA atoms in a straight line → kappa = 0° (no direction change)
    // C++ DSSP: cosinus_angle(CA_i, CA_{i-2}, CA_{i+2}, CA_i) measures
    // angle between direction vectors (CA_i - CA_{i-2}) and (CA_{i+2} - CA_i)
    const ca_prev2 = Vec3f32{ .x = -2.0, .y = 0.0, .z = 0.0 };
    const ca = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const ca_next2 = Vec3f32{ .x = 2.0, .y = 0.0, .z = 0.0 };

    const angle = kappaAngle(ca_prev2, ca, ca_next2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), angle, 0.01);
}

test "kappa angle - right angle" {
    // 90° direction change → kappa = 90°
    const ca_prev2 = Vec3f32{ .x = -1.0, .y = 0.0, .z = 0.0 };
    const ca = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const ca_next2 = Vec3f32{ .x = 0.0, .y = 1.0, .z = 0.0 };

    const angle = kappaAngle(ca_prev2, ca, ca_next2);
    try std.testing.expectApproxEqAbs(@as(f32, 90.0), angle, 0.01);
}

test "kappa angle - complete reversal" {
    // 180° direction change (U-turn) → kappa = 180°
    const ca_prev2 = Vec3f32{ .x = -2.0, .y = 0.0, .z = 0.0 };
    const ca = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const ca_next2 = Vec3f32{ .x = -1.0, .y = 0.0, .z = 0.0 };

    const angle = kappaAngle(ca_prev2, ca, ca_next2);
    try std.testing.expectApproxEqAbs(@as(f32, 180.0), angle, 0.01);
}
