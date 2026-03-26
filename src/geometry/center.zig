//! Center of mass and center of geometry calculations.

const std = @import("std");
const simd = @import("../simd.zig");

const vec_len_f32 = simd.optimal_vector_width.f32_width;
const vec_len_f64 = simd.optimal_vector_width.f64_width;

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
    if (atom_indices) |indices| {
        return ofMassIndexed(x, y, z, masses, indices);
    } else {
        return ofMassAll(x, y, z, masses);
    }
}

/// SIMD-optimized center of mass for all atoms.
/// Accumulates in f64 vectors, loads f32 coords and widens to f64.
fn ofMassAll(
    x: []const f32,
    y: []const f32,
    z: []const f32,
    masses: []const f64,
) [3]f64 {
    const n = x.len;
    if (n == 0) return .{ 0.0, 0.0, 0.0 };

    const V = vec_len_f64;
    var sum_x: @Vector(V, f64) = @splat(0.0);
    var sum_y: @Vector(V, f64) = @splat(0.0);
    var sum_z: @Vector(V, f64) = @splat(0.0);
    var sum_m: @Vector(V, f64) = @splat(0.0);

    var i: usize = 0;
    while (i + V <= n) : (i += V) {
        const vx: @Vector(V, f32) = x[i..][0..V].*;
        const vy: @Vector(V, f32) = y[i..][0..V].*;
        const vz: @Vector(V, f32) = z[i..][0..V].*;
        const vm: @Vector(V, f64) = masses[i..][0..V].*;

        sum_x += vm * @as(@Vector(V, f64), @floatCast(vx));
        sum_y += vm * @as(@Vector(V, f64), @floatCast(vy));
        sum_z += vm * @as(@Vector(V, f64), @floatCast(vz));
        sum_m += vm;
    }

    var rx: f64 = @reduce(.Add, sum_x);
    var ry: f64 = @reduce(.Add, sum_y);
    var rz: f64 = @reduce(.Add, sum_z);
    var total_mass: f64 = @reduce(.Add, sum_m);

    // Scalar tail
    while (i < n) : (i += 1) {
        const m = masses[i];
        rx += m * @as(f64, x[i]);
        ry += m * @as(f64, y[i]);
        rz += m * @as(f64, z[i]);
        total_mass += m;
    }

    if (total_mass == 0.0) return .{ 0.0, 0.0, 0.0 };

    return .{
        rx / total_mass,
        ry / total_mass,
        rz / total_mass,
    };
}

/// Scalar center of mass for indexed subset (indirect indexing defeats SIMD).
fn ofMassIndexed(
    x: []const f32,
    y: []const f32,
    z: []const f32,
    masses: []const f64,
    indices: []const u32,
) [3]f64 {
    var sum_x: f64 = 0.0;
    var sum_y: f64 = 0.0;
    var sum_z: f64 = 0.0;
    var total_mass: f64 = 0.0;

    for (indices) |idx| {
        const m = masses[idx];
        sum_x += m * @as(f64, x[idx]);
        sum_y += m * @as(f64, y[idx]);
        sum_z += m * @as(f64, z[idx]);
        total_mass += m;
    }

    if (total_mass == 0.0) return .{ 0.0, 0.0, 0.0 };

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
/// Returns [3]f64 = {x_cog, y_cog, z_cog}.
pub fn ofGeometry(
    x: []const f32,
    y: []const f32,
    z: []const f32,
    atom_indices: ?[]const u32,
) [3]f64 {
    if (atom_indices) |indices| {
        return ofGeometryIndexed(x, y, z, indices);
    } else {
        return ofGeometryAll(x, y, z);
    }
}

/// SIMD-optimized center of geometry for all atoms.
/// Accumulates in f32 vectors for speed, reduces to f64 at end.
fn ofGeometryAll(
    x: []const f32,
    y: []const f32,
    z: []const f32,
) [3]f64 {
    const n = x.len;
    if (n == 0) return .{ 0.0, 0.0, 0.0 };

    const V = vec_len_f32;
    var sum_x: @Vector(V, f32) = @splat(0.0);
    var sum_y: @Vector(V, f32) = @splat(0.0);
    var sum_z: @Vector(V, f32) = @splat(0.0);

    var i: usize = 0;
    while (i + V <= n) : (i += V) {
        sum_x += x[i..][0..V].*;
        sum_y += y[i..][0..V].*;
        sum_z += z[i..][0..V].*;
    }

    var rx: f64 = @as(f64, @reduce(.Add, sum_x));
    var ry: f64 = @as(f64, @reduce(.Add, sum_y));
    var rz: f64 = @as(f64, @reduce(.Add, sum_z));

    // Scalar tail
    while (i < n) : (i += 1) {
        rx += @as(f64, x[i]);
        ry += @as(f64, y[i]);
        rz += @as(f64, z[i]);
    }

    const count: f64 = @floatFromInt(n);

    return .{
        rx / count,
        ry / count,
        rz / count,
    };
}

/// Scalar center of geometry for indexed subset (indirect indexing defeats SIMD).
fn ofGeometryIndexed(
    x: []const f32,
    y: []const f32,
    z: []const f32,
    indices: []const u32,
) [3]f64 {
    var sum_x: f64 = 0.0;
    var sum_y: f64 = 0.0;
    var sum_z: f64 = 0.0;

    for (indices) |idx| {
        sum_x += @as(f64, x[idx]);
        sum_y += @as(f64, y[idx]);
        sum_z += @as(f64, z[idx]);
    }

    if (indices.len == 0) return .{ 0.0, 0.0, 0.0 };

    const n: f64 = @floatFromInt(indices.len);

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

test "large array exercises SIMD path: ofGeometry" {
    const N = 100;
    var x: [N]f32 = undefined;
    var y: [N]f32 = undefined;
    var z: [N]f32 = undefined;

    for (0..N) |i| {
        const fi: f32 = @floatFromInt(i);
        x[i] = fi;
        y[i] = fi * 2.0;
        z[i] = fi * 3.0;
    }

    const cog = ofGeometry(&x, &y, &z, null);

    // Mean of 0..99 = 49.5
    try std.testing.expectApproxEqAbs(@as(f64, 49.5), cog[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f64, 99.0), cog[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f64, 148.5), cog[2], 1e-4);
}

test "large array exercises SIMD path: ofMass" {
    const N = 100;
    var x: [N]f32 = undefined;
    var y: [N]f32 = undefined;
    var z: [N]f32 = undefined;
    var masses: [N]f64 = undefined;

    for (0..N) |i| {
        const fi: f32 = @floatFromInt(i);
        x[i] = fi;
        y[i] = fi * 2.0;
        z[i] = fi * 3.0;
        masses[i] = 1.0; // equal masses => COM = geometric center
    }

    const com = ofMass(&x, &y, &z, &masses, null);

    // Mean of 0..99 = 49.5
    try std.testing.expectApproxEqAbs(@as(f64, 49.5), com[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f64, 99.0), com[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f64, 148.5), com[2], 1e-4);
}
