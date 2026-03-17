//! Inertia tensor and principal moments of inertia.

const std = @import("std");
const center = @import("center.zig");

/// Compute the 3x3 inertia tensor around the center of mass.
///
/// If `atom_indices` is non-null, only the listed atom indices are used.
/// Otherwise all atoms (0..x.len) are used.
///
/// The tensor is:
///   I_xx = sum_i(m_i * (y_i^2 + z_i^2))
///   I_yy = sum_i(m_i * (x_i^2 + z_i^2))
///   I_zz = sum_i(m_i * (x_i^2 + y_i^2))
///   I_xy = I_yx = -sum_i(m_i * x_i * y_i)
///   I_xz = I_zx = -sum_i(m_i * x_i * z_i)
///   I_yz = I_zy = -sum_i(m_i * y_i * z_i)
///
/// Returns the tensor as [3][3]f64 where result[row][col].
pub fn compute(
    x: []const f32,
    y: []const f32,
    z: []const f32,
    masses: []const f64,
    atom_indices: ?[]const u32,
) [3][3]f64 {
    const com = center.ofMass(x, y, z, masses, atom_indices);

    var ixx: f64 = 0.0;
    var iyy: f64 = 0.0;
    var izz: f64 = 0.0;
    var ixy: f64 = 0.0;
    var ixz: f64 = 0.0;
    var iyz: f64 = 0.0;

    if (atom_indices) |indices| {
        for (indices) |idx| {
            const m = masses[idx];
            const rx = @as(f64, x[idx]) - com[0];
            const ry = @as(f64, y[idx]) - com[1];
            const rz = @as(f64, z[idx]) - com[2];
            ixx += m * (ry * ry + rz * rz);
            iyy += m * (rx * rx + rz * rz);
            izz += m * (rx * rx + ry * ry);
            ixy -= m * rx * ry;
            ixz -= m * rx * rz;
            iyz -= m * ry * rz;
        }
    } else {
        for (0..x.len) |idx| {
            const m = masses[idx];
            const rx = @as(f64, x[idx]) - com[0];
            const ry = @as(f64, y[idx]) - com[1];
            const rz = @as(f64, z[idx]) - com[2];
            ixx += m * (ry * ry + rz * rz);
            iyy += m * (rx * rx + rz * rz);
            izz += m * (rx * rx + ry * ry);
            ixy -= m * rx * ry;
            ixz -= m * rx * rz;
            iyz -= m * ry * rz;
        }
    }

    return .{
        .{ ixx, ixy, ixz },
        .{ ixy, iyy, iyz },
        .{ ixz, iyz, izz },
    };
}

/// Compute the three principal moments of inertia from the symmetric 3x3 tensor.
///
/// Uses an analytical closed-form eigenvalue solver for 3x3 symmetric matrices
/// (based on the characteristic polynomial). Returns eigenvalues sorted in
/// ascending order.
pub fn principalMoments(tensor: [3][3]f64) [3]f64 {
    // Coefficients of characteristic polynomial: lambda^3 - p1*lambda^2 + p2*lambda - p3 = 0
    // p1 = trace
    // p2 = sum of 2x2 principal minors
    // p3 = determinant
    const a = tensor[0][0];
    const b = tensor[1][1];
    const c = tensor[2][2];
    const d = tensor[0][1]; // = tensor[1][0]
    const e = tensor[0][2]; // = tensor[2][0]
    const f = tensor[1][2]; // = tensor[2][1]

    const p1 = a + b + c;
    const p2 = a * b + a * c + b * c - d * d - e * e - f * f;
    const p3 = a * (b * c - f * f) - d * (d * c - f * e) + e * (d * f - b * e);

    // Use Vieta's substitution: lambda = t + p1/3
    // t^3 + q*t + r = 0
    const p1_3 = p1 / 3.0;
    const q = p2 - p1 * p1 / 3.0;
    const r = -p3 + p1 * p2 / 3.0 - 2.0 * p1 * p1 * p1 / 27.0;

    // Discriminant for depressed cubic
    const disc = q * q * q / 27.0 + r * r / 4.0;

    var eigenvalues: [3]f64 = undefined;

    if (disc <= 0.0) {
        // Three real roots (symmetric matrix always has real eigenvalues)
        // Trigonometric method
        const m_val = 2.0 * @sqrt(-q / 3.0);
        const theta = std.math.acos(std.math.clamp(
            3.0 * r / (q * m_val),
            -1.0,
            1.0,
        ));

        eigenvalues[0] = m_val * @cos(theta / 3.0) + p1_3;
        eigenvalues[1] = m_val * @cos((theta - 2.0 * std.math.pi) / 3.0) + p1_3;
        eigenvalues[2] = m_val * @cos((theta - 4.0 * std.math.pi) / 3.0) + p1_3;
    } else {
        // One real root (degenerate case — shouldn't happen for valid inertia tensor)
        const sqrt_disc = @sqrt(disc);
        const u = std.math.cbrt(-r / 2.0 + sqrt_disc);
        const v = std.math.cbrt(-r / 2.0 - sqrt_disc);
        eigenvalues[0] = (u + v) + p1_3;
        eigenvalues[1] = eigenvalues[0];
        eigenvalues[2] = eigenvalues[0];
    }

    // Sort ascending
    if (eigenvalues[0] > eigenvalues[1]) {
        const tmp = eigenvalues[0];
        eigenvalues[0] = eigenvalues[1];
        eigenvalues[1] = tmp;
    }
    if (eigenvalues[1] > eigenvalues[2]) {
        const tmp = eigenvalues[1];
        eigenvalues[1] = eigenvalues[2];
        eigenvalues[2] = tmp;
    }
    if (eigenvalues[0] > eigenvalues[1]) {
        const tmp = eigenvalues[0];
        eigenvalues[0] = eigenvalues[1];
        eigenvalues[1] = tmp;
    }

    return eigenvalues;
}

// ============================================================================
// Tests
// ============================================================================

test "inertia: single point mass" {
    // Single mass at (r, 0, 0): Ixx=0, Iyy=Izz=m*r^2, off-diagonals=0
    const x = [_]f32{3.0};
    const y = [_]f32{0.0};
    const z = [_]f32{0.0};
    const masses = [_]f64{2.0};

    const tensor = compute(&x, &y, &z, &masses, null);

    // COM is at the atom itself, so all displacements are zero -> tensor is zero
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), tensor[0][0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), tensor[1][1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), tensor[2][2], 1e-12);
}

test "inertia: tensor is symmetric" {
    const x = [_]f32{ 1.0, -1.0, 0.0, 0.5 };
    const y = [_]f32{ 0.5, 0.5, 2.0, -1.0 };
    const z = [_]f32{ 0.0, 0.0, 0.5, 0.5 };
    const masses = [_]f64{ 1.0, 2.0, 1.5, 0.5 };

    const tensor = compute(&x, &y, &z, &masses, null);

    // Check symmetry: I_xy = I_yx, etc.
    try std.testing.expectApproxEqAbs(tensor[0][1], tensor[1][0], 1e-12);
    try std.testing.expectApproxEqAbs(tensor[0][2], tensor[2][0], 1e-12);
    try std.testing.expectApproxEqAbs(tensor[1][2], tensor[2][1], 1e-12);
}

test "inertia: two atoms on x-axis" {
    // Two equal masses at (-d,0,0) and (d,0,0)
    // COM at origin
    // Ixx = 0 (no y,z displacement)
    // Iyy = 2*m*d^2
    // Izz = 2*m*d^2
    const d: f32 = 3.0;
    const m: f64 = 5.0;
    const x = [_]f32{ -d, d };
    const y = [_]f32{ 0.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0 };
    const masses = [_]f64{ m, m };

    const tensor = compute(&x, &y, &z, &masses, null);

    const expected = 2.0 * m * @as(f64, d) * @as(f64, d);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), tensor[0][0], 1e-10);
    try std.testing.expectApproxEqAbs(expected, tensor[1][1], 1e-10);
    try std.testing.expectApproxEqAbs(expected, tensor[2][2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), tensor[0][1], 1e-10);
}

test "inertia: principal moments sum equals trace" {
    const x = [_]f32{ 1.0, -1.0, 0.0 };
    const y = [_]f32{ 0.0, 0.0, 2.0 };
    const z = [_]f32{ 0.5, -0.5, 0.0 };
    const masses = [_]f64{ 1.0, 1.0, 1.0 };

    const tensor = compute(&x, &y, &z, &masses, null);
    const trace = tensor[0][0] + tensor[1][1] + tensor[2][2];
    const moments = principalMoments(tensor);
    const moments_sum = moments[0] + moments[1] + moments[2];

    try std.testing.expectApproxEqAbs(trace, moments_sum, 1e-8);
}

test "inertia: diagonal tensor has sorted principal moments" {
    // Build a diagonal tensor directly by placing atoms on coordinate axes
    // (1,0,0) and (-1,0,0) with mass 1 each
    // (0,2,0) and (0,-2,0) with mass 1 each
    // (0,0,3) and (0,0,-3) with mass 1 each
    // COM = origin
    // Ixx = 2*(4+9) = 26, Iyy = 2*(1+9) = 20, Izz = 2*(1+4) = 10
    const x = [_]f32{ 1.0, -1.0, 0.0, 0.0, 0.0, 0.0 };
    const y = [_]f32{ 0.0, 0.0, 2.0, -2.0, 0.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0, 0.0, 3.0, -3.0 };
    const masses = [_]f64{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

    const tensor = compute(&x, &y, &z, &masses, null);
    const moments = principalMoments(tensor);

    // Should be sorted ascending: 10, 20, 26
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), moments[0], 1e-8);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), moments[1], 1e-8);
    try std.testing.expectApproxEqAbs(@as(f64, 26.0), moments[2], 1e-8);
}
