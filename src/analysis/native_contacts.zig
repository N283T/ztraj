//! Native contacts (Q value): fraction of contacts preserved relative to a reference.
//!
//! A "native contact" is a pair of atoms within a distance cutoff in the
//! reference structure. Q is the fraction of those contacts preserved in
//! each trajectory frame.

const std = @import("std");

/// Compute native contacts Q value (hard-cut method).
///
/// 1. Identify native contacts: pairs where ref distance ≤ cutoff.
/// 2. For each frame, count how many native contacts have distance ≤ cutoff.
/// 3. Q = count / n_native.
///
/// `ref_x/y/z` and `x/y/z` are SOA coordinate arrays.
/// `atom_indices_a` and `atom_indices_b` define the two groups of atoms.
/// Returns Q in [0, 1]. Returns 0 if no native contacts exist.
pub fn computeQ(
    ref_x: []const f32,
    ref_y: []const f32,
    ref_z: []const f32,
    x: []const f32,
    y: []const f32,
    z: []const f32,
    atom_indices_a: []const u32,
    atom_indices_b: []const u32,
    cutoff: f32,
) f64 {
    const cutoff_sq: f64 = @as(f64, cutoff) * @as(f64, cutoff);

    // Count native contacts in reference and preserved contacts in current frame
    var n_native: u32 = 0;
    var n_preserved: u32 = 0;

    for (atom_indices_a) |ia| {
        for (atom_indices_b) |ib| {
            // Reference distance
            const rdx: f64 = @as(f64, ref_x[ia]) - @as(f64, ref_x[ib]);
            const rdy: f64 = @as(f64, ref_y[ia]) - @as(f64, ref_y[ib]);
            const rdz: f64 = @as(f64, ref_z[ia]) - @as(f64, ref_z[ib]);
            const ref_dist_sq = rdx * rdx + rdy * rdy + rdz * rdz;

            if (ref_dist_sq <= cutoff_sq) {
                n_native += 1;

                // Current frame distance
                const dx: f64 = @as(f64, x[ia]) - @as(f64, x[ib]);
                const dy: f64 = @as(f64, y[ia]) - @as(f64, y[ib]);
                const dz: f64 = @as(f64, z[ia]) - @as(f64, z[ib]);
                const dist_sq = dx * dx + dy * dy + dz * dz;

                if (dist_sq <= cutoff_sq) {
                    n_preserved += 1;
                }
            }
        }
    }

    if (n_native == 0) return 0.0;
    return @as(f64, @floatFromInt(n_preserved)) / @as(f64, @floatFromInt(n_native));
}

/// Soft-cut Q value: uses sigmoid to smoothly transition contacts.
///
/// Q_soft = sum(1 / (1 + exp(beta * (r - lambda * r0)))) / n_native
pub fn computeQSoft(
    ref_x: []const f32,
    ref_y: []const f32,
    ref_z: []const f32,
    x: []const f32,
    y: []const f32,
    z: []const f32,
    atom_indices_a: []const u32,
    atom_indices_b: []const u32,
    cutoff: f32,
    beta: f64,
    lambda: f64,
) f64 {
    const cutoff_sq: f64 = @as(f64, cutoff) * @as(f64, cutoff);

    var n_native: u32 = 0;
    var q_sum: f64 = 0.0;

    for (atom_indices_a) |ia| {
        for (atom_indices_b) |ib| {
            const rdx: f64 = @as(f64, ref_x[ia]) - @as(f64, ref_x[ib]);
            const rdy: f64 = @as(f64, ref_y[ia]) - @as(f64, ref_y[ib]);
            const rdz: f64 = @as(f64, ref_z[ia]) - @as(f64, ref_z[ib]);
            const ref_dist_sq = rdx * rdx + rdy * rdy + rdz * rdz;

            if (ref_dist_sq <= cutoff_sq) {
                n_native += 1;
                const r0 = @sqrt(ref_dist_sq);

                const dx: f64 = @as(f64, x[ia]) - @as(f64, x[ib]);
                const dy: f64 = @as(f64, y[ia]) - @as(f64, y[ib]);
                const dz: f64 = @as(f64, z[ia]) - @as(f64, z[ib]);
                const r = @sqrt(dx * dx + dy * dy + dz * dz);

                q_sum += 1.0 / (1.0 + @exp(beta * (r - lambda * r0)));
            }
        }
    }

    if (n_native == 0) return 0.0;
    return q_sum / @as(f64, @floatFromInt(n_native));
}

// ============================================================================
// Tests
// ============================================================================

test "computeQ: identical structures → Q = 1.0" {
    const x = [_]f32{ 0.0, 3.0, 10.0 };
    const y = [_]f32{ 0.0, 0.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0 };
    const a = [_]u32{0};
    const b = [_]u32{ 1, 2 };

    // Only pair (0,1) is within cutoff 5.0 (distance 3.0)
    const q = computeQ(&x, &y, &z, &x, &y, &z, &a, &b, 5.0);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), q, 1e-10);
}

test "computeQ: all contacts broken → Q = 0.0" {
    const ref_x = [_]f32{ 0.0, 1.0 };
    const ref_y = [_]f32{ 0.0, 0.0 };
    const ref_z = [_]f32{ 0.0, 0.0 };

    // Move atoms far apart
    const x = [_]f32{ 0.0, 100.0 };
    const y = [_]f32{ 0.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0 };

    const a = [_]u32{0};
    const b = [_]u32{1};
    const q = computeQ(&ref_x, &ref_y, &ref_z, &x, &y, &z, &a, &b, 5.0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), q, 1e-10);
}

test "computeQ: no native contacts → Q = 0.0" {
    const x = [_]f32{ 0.0, 100.0 };
    const y = [_]f32{ 0.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0 };
    const a = [_]u32{0};
    const b = [_]u32{1};

    const q = computeQ(&x, &y, &z, &x, &y, &z, &a, &b, 5.0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), q, 1e-10);
}

test "computeQSoft: identical structures → Q ≈ 1.0" {
    const x = [_]f32{ 0.0, 3.0 };
    const y = [_]f32{ 0.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0 };
    const a = [_]u32{0};
    const b = [_]u32{1};

    const q = computeQSoft(&x, &y, &z, &x, &y, &z, &a, &b, 5.0, 5.0, 1.8);
    // When r == r0, sigmoid = 1/(1+exp(beta*r0*(1-lambda))) ≈ 1 for lambda>1
    try std.testing.expect(q > 0.9);
}
