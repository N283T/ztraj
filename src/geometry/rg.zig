//! Radius of gyration calculation.

const std = @import("std");
const center = @import("center.zig");

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
    const com = center.ofMass(x, y, z, masses, atom_indices);

    var sum_msd: f64 = 0.0;
    var total_mass: f64 = 0.0;

    if (atom_indices) |indices| {
        for (indices) |idx| {
            const m = masses[idx];
            const dx = @as(f64, x[idx]) - com[0];
            const dy = @as(f64, y[idx]) - com[1];
            const dz = @as(f64, z[idx]) - com[2];
            sum_msd += m * (dx * dx + dy * dy + dz * dz);
            total_mass += m;
        }
    } else {
        for (0..x.len) |idx| {
            const m = masses[idx];
            const dx = @as(f64, x[idx]) - com[0];
            const dy = @as(f64, y[idx]) - com[1];
            const dz = @as(f64, z[idx]) - com[2];
            sum_msd += m * (dx * dx + dy * dy + dz * dz);
            total_mass += m;
        }
    }

    if (total_mass == 0.0) {
        return 0.0;
    }

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
