//! Radius of gyration calculation.

const std = @import("std");
const center = @import("center.zig");
const simd = @import("../simd.zig");

const vec_len = simd.optimal_vector_width.f64_width;

/// Compute the mass-weighted radius of gyration.
///
/// Rg = sqrt(sum_i(m_i * |r_i - r_com|^2) / sum_i(m_i))
///
/// If `atom_indices` is non-null, only the listed atom indices are included.
/// Otherwise all atoms (0..x.len) are used.
/// Uses f64 accumulation throughout for precision.
pub fn compute(
    x: []const f32,
    y: []const f32,
    z: []const f32,
    masses: []const f64,
    atom_indices: ?[]const u32,
) f64 {
    std.debug.assert(x.len == y.len);
    std.debug.assert(x.len == z.len);
    std.debug.assert(x.len == masses.len);

    const com = center.ofMass(x, y, z, masses, atom_indices);

    if (atom_indices) |indices| {
        return computeIndexed(x, y, z, masses, indices, com);
    } else {
        return computeAll(x, y, z, masses, com);
    }
}

/// SIMD-optimized MSD accumulation for all atoms.
fn computeAll(
    x: []const f32,
    y: []const f32,
    z: []const f32,
    masses: []const f64,
    com: [3]f64,
) f64 {
    const n = x.len;
    if (n == 0) return 0.0;

    const V = vec_len;
    const com_x: @Vector(V, f64) = @splat(com[0]);
    const com_y: @Vector(V, f64) = @splat(com[1]);
    const com_z: @Vector(V, f64) = @splat(com[2]);

    var sum_msd: @Vector(V, f64) = @splat(0.0);
    var sum_mass: @Vector(V, f64) = @splat(0.0);

    var i: usize = 0;
    while (i + V <= n) : (i += V) {
        const vx: @Vector(V, f32) = x[i..][0..V].*;
        const vy: @Vector(V, f32) = y[i..][0..V].*;
        const vz: @Vector(V, f32) = z[i..][0..V].*;
        const vm: @Vector(V, f64) = masses[i..][0..V].*;

        const dx = @as(@Vector(V, f64), @floatCast(vx)) - com_x;
        const dy = @as(@Vector(V, f64), @floatCast(vy)) - com_y;
        const dz = @as(@Vector(V, f64), @floatCast(vz)) - com_z;

        sum_msd += vm * (dx * dx + dy * dy + dz * dz);
        sum_mass += vm;
    }

    var total_msd: f64 = @reduce(.Add, sum_msd);
    var total_mass: f64 = @reduce(.Add, sum_mass);

    // Scalar tail
    while (i < n) : (i += 1) {
        const m = masses[i];
        const dx = @as(f64, x[i]) - com[0];
        const dy = @as(f64, y[i]) - com[1];
        const dz = @as(f64, z[i]) - com[2];
        total_msd += m * (dx * dx + dy * dy + dz * dz);
        total_mass += m;
    }

    if (total_mass == 0.0) return 0.0;

    return @sqrt(total_msd / total_mass);
}

/// Scalar MSD accumulation for indexed subset (indirect indexing defeats SIMD).
fn computeIndexed(
    x: []const f32,
    y: []const f32,
    z: []const f32,
    masses: []const f64,
    indices: []const u32,
    com: [3]f64,
) f64 {
    var sum_msd: f64 = 0.0;
    var total_mass: f64 = 0.0;

    for (indices) |idx| {
        const m = masses[idx];
        const dx = @as(f64, x[idx]) - com[0];
        const dy = @as(f64, y[idx]) - com[1];
        const dz = @as(f64, z[idx]) - com[2];
        sum_msd += m * (dx * dx + dy * dy + dz * dz);
        total_mass += m;
    }

    if (total_mass == 0.0) return 0.0;

    return @sqrt(sum_msd / total_mass);
}

// ============================================================================
// Tests
// ============================================================================

test "rg: single atom returns zero" {
    const x = [_]f32{5.0};
    const y = [_]f32{3.0};
    const z = [_]f32{1.0};
    const masses = [_]f64{12.0};

    const rg = compute(&x, &y, &z, &masses, null);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rg, 1e-12);
}

test "rg: two equal-mass atoms equidistant from center" {
    // Two atoms at distance d from origin along x-axis
    // COM = (0,0,0), Rg = d
    const d: f32 = 5.0;
    const x = [_]f32{ -d, d };
    const y = [_]f32{ 0.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0 };
    const masses = [_]f64{ 1.0, 1.0 };

    const rg = compute(&x, &y, &z, &masses, null);
    try std.testing.expectApproxEqAbs(@as(f64, d), rg, 1e-10);
}

test "rg: hand-calculated value" {
    // Three atoms: (1,0,0), (-1,0,0), (0,2,0) with equal mass 1.0
    // COM = (0, 2/3, 0)
    // Rg^2 = [1*(1^2 + (2/3)^2) + 1*(1^2 + (2/3)^2) + 1*(0 + (4/3)^2)] / 3
    //      = [1 + 4/9 + 1 + 4/9 + 16/9] / 3
    //      = [2 + 24/9] / 3
    //      = [2 + 8/3] / 3
    //      = [6/3 + 8/3] / 3
    //      = (14/3) / 3 = 14/9
    // Rg = sqrt(14/9) = sqrt(14)/3
    const x = [_]f32{ 1.0, -1.0, 0.0 };
    const y = [_]f32{ 0.0, 0.0, 2.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0 };
    const masses = [_]f64{ 1.0, 1.0, 1.0 };

    const expected: f64 = @sqrt(14.0) / 3.0;
    const rg = compute(&x, &y, &z, &masses, null);

    try std.testing.expectApproxEqAbs(expected, rg, 1e-10);
}

test "rg: with atom_indices subset" {
    // 4 atoms; only use indices 0 and 1 which are at ±5 along x
    const x = [_]f32{ -5.0, 5.0, 100.0, 200.0 };
    const y = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const masses = [_]f64{ 1.0, 1.0, 999.0, 999.0 };

    const indices = [_]u32{ 0, 1 };
    const rg = compute(&x, &y, &z, &masses, &indices);

    // COM of subset = (0,0,0), Rg = 5.0
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), rg, 1e-10);
}

test "rg: zero mass returns zero" {
    const x = [_]f32{ 1.0, 2.0 };
    const y = [_]f32{ 0.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0 };
    const masses = [_]f64{ 0.0, 0.0 };

    const rg = compute(&x, &y, &z, &masses, null);
    try std.testing.expectEqual(@as(f64, 0.0), rg);
}

test "rg: large array exercises SIMD path" {
    // N=103 is not divisible by common SIMD vector widths (2, 4, 8, 16),
    // ensuring the scalar tail path is exercised.
    const N = 103;
    var x: [N]f32 = undefined;
    var y: [N]f32 = undefined;
    var z: [N]f32 = undefined;
    var masses: [N]f64 = undefined;

    // Place atoms uniformly along x-axis: x_i = i, y=0, z=0, mass=1
    for (0..N) |i| {
        x[i] = @floatFromInt(i);
        y[i] = 0.0;
        z[i] = 0.0;
        masses[i] = 1.0;
    }

    const rg = compute(&x, &y, &z, &masses, null);

    // For N equal-mass atoms at positions 0, 1, ..., N-1 along x-axis:
    // COM_x = (N-1)/2
    // Rg^2 = (1/N) * sum_i (i - (N-1)/2)^2 = variance of {0, 1, ..., N-1}
    // Variance = (N^2 - 1) / 12
    const n_f64: f64 = @floatFromInt(N);
    const expected = @sqrt((n_f64 * n_f64 - 1.0) / 12.0);

    try std.testing.expectApproxEqAbs(expected, rg, 1e-4);
}
