//! Center of mass and center of geometry calculations.

const std = @import("std");

/// Compute the center of mass for a set of atoms.
///
/// If `atom_indices` is non-null, only the listed atom indices are used.
/// Otherwise all atoms (0..x.len) are used.
/// Uses f64 accumulation throughout for precision.
/// Returns [3]f64 = {x_com, y_com, z_com}.
pub fn ofMass(
    x: []const f32,
    y: []const f32,
    z: []const f32,
    masses: []const f64,
    atom_indices: ?[]const u32,
) [3]f64 {
    var sum_x: f64 = 0.0;
    var sum_y: f64 = 0.0;
    var sum_z: f64 = 0.0;
    var total_mass: f64 = 0.0;

    if (atom_indices) |indices| {
        for (indices) |idx| {
            const m = masses[idx];
            sum_x += m * @as(f64, x[idx]);
            sum_y += m * @as(f64, y[idx]);
            sum_z += m * @as(f64, z[idx]);
            total_mass += m;
        }
    } else {
        for (0..x.len) |idx| {
            const m = masses[idx];
            sum_x += m * @as(f64, x[idx]);
            sum_y += m * @as(f64, y[idx]);
            sum_z += m * @as(f64, z[idx]);
            total_mass += m;
        }
    }

    if (total_mass == 0.0) {
        return .{ 0.0, 0.0, 0.0 };
    }

    return .{
        sum_x / total_mass,
        sum_y / total_mass,
        sum_z / total_mass,
    };
}

/// Compute the center of geometry (unweighted mean position).
///
/// If `atom_indices` is non-null, only the listed atom indices are used.
/// Otherwise all atoms (0..x.len) are used.
/// Uses f64 accumulation throughout for precision.
/// Returns [3]f64 = {x_cog, y_cog, z_cog}.
pub fn ofGeometry(
    x: []const f32,
    y: []const f32,
    z: []const f32,
    atom_indices: ?[]const u32,
) [3]f64 {
    var sum_x: f64 = 0.0;
    var sum_y: f64 = 0.0;
    var sum_z: f64 = 0.0;
    var n: f64 = 0.0;

    if (atom_indices) |indices| {
        for (indices) |idx| {
            sum_x += @as(f64, x[idx]);
            sum_y += @as(f64, y[idx]);
            sum_z += @as(f64, z[idx]);
            n += 1.0;
        }
    } else {
        for (0..x.len) |idx| {
            sum_x += @as(f64, x[idx]);
            sum_y += @as(f64, y[idx]);
            sum_z += @as(f64, z[idx]);
            n += 1.0;
        }
    }

    if (n == 0.0) {
        return .{ 0.0, 0.0, 0.0 };
    }

    return .{
        sum_x / n,
        sum_y / n,
        sum_z / n,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "center ofGeometry: 3 equal-mass atoms" {
    // COM should equal geometric center
    const x = [_]f32{ 0.0, 6.0, 3.0 };
    const y = [_]f32{ 0.0, 0.0, 6.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0 };

    const cog = ofGeometry(&x, &y, &z, null);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), cog[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), cog[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), cog[2], 1e-10);
}

test "center ofMass: equal masses equal geometry" {
    const x = [_]f32{ 0.0, 6.0, 3.0 };
    const y = [_]f32{ 0.0, 0.0, 6.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0 };
    const masses = [_]f64{ 1.0, 1.0, 1.0 };

    const com = ofMass(&x, &y, &z, &masses, null);
    const cog = ofGeometry(&x, &y, &z, null);

    try std.testing.expectApproxEqAbs(cog[0], com[0], 1e-10);
    try std.testing.expectApproxEqAbs(cog[1], com[1], 1e-10);
    try std.testing.expectApproxEqAbs(cog[2], com[2], 1e-10);
}

test "center ofMass: unequal masses" {
    // Two atoms: (0,0,0) mass=1, (10,0,0) mass=9
    // COM_x = (0*1 + 10*9) / 10 = 9.0
    const x = [_]f32{ 0.0, 10.0 };
    const y = [_]f32{ 0.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0 };
    const masses = [_]f64{ 1.0, 9.0 };

    const com = ofMass(&x, &y, &z, &masses, null);

    try std.testing.expectApproxEqAbs(@as(f64, 9.0), com[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), com[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), com[2], 1e-10);
}

test "center ofMass: with atom_indices subset" {
    // 4 atoms; use only indices 1,2
    const x = [_]f32{ 100.0, 0.0, 4.0, 200.0 };
    const y = [_]f32{ 100.0, 0.0, 0.0, 200.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const masses = [_]f64{ 99.0, 1.0, 1.0, 99.0 };

    const indices = [_]u32{ 1, 2 };
    const com = ofMass(&x, &y, &z, &masses, &indices);

    // Both selected masses = 1.0, so COM = geometric center of atoms 1 and 2
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), com[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), com[1], 1e-10);
}

test "center ofGeometry: with atom_indices subset" {
    // 4 atoms; use only indices 0,3
    const x = [_]f32{ 2.0, 100.0, 100.0, 8.0 };
    const y = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    const indices = [_]u32{ 0, 3 };
    const cog = ofGeometry(&x, &y, &z, &indices);

    try std.testing.expectApproxEqAbs(@as(f64, 5.0), cog[0], 1e-10);
}

test "center: empty atoms returns zero" {
    const x = [_]f32{};
    const y = [_]f32{};
    const z = [_]f32{};
    const masses = [_]f64{};

    const com = ofMass(&x, &y, &z, &masses, null);
    const cog = ofGeometry(&x, &y, &z, null);

    try std.testing.expectEqual(@as(f64, 0.0), com[0]);
    try std.testing.expectEqual(@as(f64, 0.0), cog[0]);
}
