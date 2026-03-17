//! Per-atom Root Mean Square Fluctuation (RMSF) over a trajectory.

const std = @import("std");
const types = @import("../types.zig");

/// Compute per-atom RMSF over a set of trajectory frames.
///
/// RMSF_i = sqrt( mean_t( |r_i(t) - <r_i>|^2 ) )
///
/// Algorithm:
///   1. Compute mean position for each atom across all frames.
///   2. Compute mean squared displacement from that mean.
///   3. Take square root.
///
/// If `atom_indices` is non-null, only those atom indices are processed and
/// the returned slice has `atom_indices.len` elements in the same order.
/// Otherwise all atoms are processed and the slice has `frames[0].nAtoms()`
/// elements.
///
/// Returns a heap-allocated []f64 owned by the caller (free with `allocator`).
/// Returns error.NoFrames if `frames` is empty.
pub fn compute(
    allocator: std.mem.Allocator,
    frames: []const types.Frame,
    atom_indices: ?[]const u32,
) ![]f64 {
    if (frames.len == 0) return error.NoFrames;

    const n_frames: f64 = @floatFromInt(frames.len);
    const first = frames[0];

    // Determine which atoms to process
    const n_atoms: usize = if (atom_indices) |idx| idx.len else first.nAtoms();

    // Allocate mean position arrays (f64)
    const mean_x = try allocator.alloc(f64, n_atoms);
    defer allocator.free(mean_x);
    const mean_y = try allocator.alloc(f64, n_atoms);
    defer allocator.free(mean_y);
    const mean_z = try allocator.alloc(f64, n_atoms);
    defer allocator.free(mean_z);

    @memset(mean_x, 0.0);
    @memset(mean_y, 0.0);
    @memset(mean_z, 0.0);

    // Step 1: Accumulate mean positions
    for (frames) |frame| {
        if (atom_indices) |indices| {
            for (indices, 0..) |atom_idx, i| {
                mean_x[i] += @as(f64, frame.x[atom_idx]);
                mean_y[i] += @as(f64, frame.y[atom_idx]);
                mean_z[i] += @as(f64, frame.z[atom_idx]);
            }
        } else {
            for (0..n_atoms) |i| {
                mean_x[i] += @as(f64, frame.x[i]);
                mean_y[i] += @as(f64, frame.y[i]);
                mean_z[i] += @as(f64, frame.z[i]);
            }
        }
    }

    for (0..n_atoms) |i| {
        mean_x[i] /= n_frames;
        mean_y[i] /= n_frames;
        mean_z[i] /= n_frames;
    }

    // Step 2: Accumulate mean squared displacement
    const result = try allocator.alloc(f64, n_atoms);
    @memset(result, 0.0);

    for (frames) |frame| {
        if (atom_indices) |indices| {
            for (indices, 0..) |atom_idx, i| {
                const dx = @as(f64, frame.x[atom_idx]) - mean_x[i];
                const dy = @as(f64, frame.y[atom_idx]) - mean_y[i];
                const dz = @as(f64, frame.z[atom_idx]) - mean_z[i];
                result[i] += dx * dx + dy * dy + dz * dz;
            }
        } else {
            for (0..n_atoms) |i| {
                const dx = @as(f64, frame.x[i]) - mean_x[i];
                const dy = @as(f64, frame.y[i]) - mean_y[i];
                const dz = @as(f64, frame.z[i]) - mean_z[i];
                result[i] += dx * dx + dy * dy + dz * dz;
            }
        }
    }

    // Step 3: Divide by n_frames and take sqrt
    for (result) |*v| {
        v.* = @sqrt(v.* / n_frames);
    }

    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "rmsf: all frames identical -> zero fluctuation" {
    const allocator = std.testing.allocator;

    // 3 atoms, 4 identical frames
    const n_frames = 4;
    const n_atoms = 3;

    var frames: [n_frames]types.Frame = undefined;
    for (&frames) |*f| {
        f.* = try types.Frame.init(allocator, n_atoms);
        f.x[0] = 1.0;
        f.y[0] = 2.0;
        f.z[0] = 3.0;
        f.x[1] = -1.0;
        f.y[1] = 0.0;
        f.z[1] = 5.0;
        f.x[2] = 4.0;
        f.y[2] = 4.0;
        f.z[2] = 4.0;
    }
    defer for (&frames) |*f| f.deinit();

    const rmsf = try compute(allocator, &frames, null);
    defer allocator.free(rmsf);

    try std.testing.expectEqual(n_atoms, rmsf.len);
    for (rmsf) |v| {
        try std.testing.expectApproxEqAbs(@as(f64, 0.0), v, 1e-10);
    }
}

test "rmsf: single atom oscillating" {
    const allocator = std.testing.allocator;

    // 1 atom oscillating between x=-1 and x=+1 (y=z=0)
    // mean = 0, MSD = ((1)^2 + (1)^2) / 2 = 1, RMSF = 1
    var frame1 = try types.Frame.init(allocator, 1);
    defer frame1.deinit();
    frame1.x[0] = -1.0;
    frame1.y[0] = 0.0;
    frame1.z[0] = 0.0;

    var frame2 = try types.Frame.init(allocator, 1);
    defer frame2.deinit();
    frame2.x[0] = 1.0;
    frame2.y[0] = 0.0;
    frame2.z[0] = 0.0;

    const frames = [_]types.Frame{ frame1, frame2 };
    const rmsf = try compute(allocator, &frames, null);
    defer allocator.free(rmsf);

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), rmsf[0], 1e-10);
}

test "rmsf: with atom_indices subset" {
    const allocator = std.testing.allocator;

    // 3 atoms; only track atom 1 (which oscillates) and atom 2 (static)
    // Atom 0 oscillates wildly but is excluded
    var frame1 = try types.Frame.init(allocator, 3);
    defer frame1.deinit();
    frame1.x[0] = 999.0;
    frame1.y[0] = 0.0;
    frame1.z[0] = 0.0;
    frame1.x[1] = -2.0;
    frame1.y[1] = 0.0;
    frame1.z[1] = 0.0;
    frame1.x[2] = 5.0;
    frame1.y[2] = 5.0;
    frame1.z[2] = 5.0;

    var frame2 = try types.Frame.init(allocator, 3);
    defer frame2.deinit();
    frame2.x[0] = -999.0;
    frame2.y[0] = 0.0;
    frame2.z[0] = 0.0;
    frame2.x[1] = 2.0;
    frame2.y[1] = 0.0;
    frame2.z[1] = 0.0;
    frame2.x[2] = 5.0;
    frame2.y[2] = 5.0;
    frame2.z[2] = 5.0;

    const frames = [_]types.Frame{ frame1, frame2 };
    const indices = [_]u32{ 1, 2 };
    const rmsf = try compute(allocator, &frames, &indices);
    defer allocator.free(rmsf);

    try std.testing.expectEqual(@as(usize, 2), rmsf.len);
    // Atom 1: oscillates between -2 and +2 along x, RMSF = 2
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), rmsf[0], 1e-10);
    // Atom 2: static at (5,5,5), RMSF = 0
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rmsf[1], 1e-10);
}

test "rmsf: error on empty frames" {
    const allocator = std.testing.allocator;
    const frames = [_]types.Frame{};
    const result = compute(allocator, &frames, null);
    try std.testing.expectError(error.NoFrames, result);
}
