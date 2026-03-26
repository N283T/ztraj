//! Per-atom Root Mean Square Fluctuation (RMSF) over a trajectory.

const std = @import("std");
const types = @import("../types.zig");
const simd = @import("../simd.zig");

const vec_len = simd.optimal_vector_width.f64_width;

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

/// Pass 1 worker: accumulate coordinates into thread-local sum arrays.
fn pass1Worker(
    frames: []const types.Frame,
    atom_indices: ?[]const u32,
    sum_x: []f64,
    sum_y: []f64,
    sum_z: []f64,
) void {
    const n = sum_x.len;
    for (frames) |frame| {
        if (atom_indices) |indices| {
            // Indexed path: scalar only
            for (indices, 0..) |atom_idx, i| {
                sum_x[i] += @as(f64, frame.x[atom_idx]);
                sum_y[i] += @as(f64, frame.y[atom_idx]);
                sum_z[i] += @as(f64, frame.z[atom_idx]);
            }
        } else {
            // Non-indexed path: SIMD vectorized
            const V = vec_len;
            var i: usize = 0;
            while (i + V <= n) : (i += V) {
                const vx: @Vector(V, f64) = @floatCast(@as(@Vector(V, f32), frame.x[i..][0..V].*));
                const vy: @Vector(V, f64) = @floatCast(@as(@Vector(V, f32), frame.y[i..][0..V].*));
                const vz: @Vector(V, f64) = @floatCast(@as(@Vector(V, f32), frame.z[i..][0..V].*));
                const sx: @Vector(V, f64) = sum_x[i..][0..V].*;
                const sy: @Vector(V, f64) = sum_y[i..][0..V].*;
                const sz: @Vector(V, f64) = sum_z[i..][0..V].*;
                sum_x[i..][0..V].* = sx + vx;
                sum_y[i..][0..V].* = sy + vy;
                sum_z[i..][0..V].* = sz + vz;
            }
            // Scalar tail
            while (i < n) : (i += 1) {
                sum_x[i] += @as(f64, frame.x[i]);
                sum_y[i] += @as(f64, frame.y[i]);
                sum_z[i] += @as(f64, frame.z[i]);
            }
        }
    }
}

/// Pass 2 worker: accumulate squared deviations from mean into thread-local msd array.
fn pass2Worker(
    frames: []const types.Frame,
    atom_indices: ?[]const u32,
    mean_x: []const f64,
    mean_y: []const f64,
    mean_z: []const f64,
    msd: []f64,
) void {
    const n = msd.len;
    for (frames) |frame| {
        if (atom_indices) |indices| {
            // Indexed path: scalar only
            for (indices, 0..) |atom_idx, i| {
                const dx = @as(f64, frame.x[atom_idx]) - mean_x[i];
                const dy = @as(f64, frame.y[atom_idx]) - mean_y[i];
                const dz = @as(f64, frame.z[atom_idx]) - mean_z[i];
                msd[i] += dx * dx + dy * dy + dz * dz;
            }
        } else {
            // Non-indexed path: SIMD vectorized
            const V = vec_len;
            var i: usize = 0;
            while (i + V <= n) : (i += V) {
                const vx: @Vector(V, f64) = @floatCast(@as(@Vector(V, f32), frame.x[i..][0..V].*));
                const vy: @Vector(V, f64) = @floatCast(@as(@Vector(V, f32), frame.y[i..][0..V].*));
                const vz: @Vector(V, f64) = @floatCast(@as(@Vector(V, f32), frame.z[i..][0..V].*));
                const mx: @Vector(V, f64) = mean_x[i..][0..V].*;
                const my: @Vector(V, f64) = mean_y[i..][0..V].*;
                const mz: @Vector(V, f64) = mean_z[i..][0..V].*;
                const dx = vx - mx;
                const dy = vy - my;
                const dz = vz - mz;
                const sq = dx * dx + dy * dy + dz * dz;
                const prev: @Vector(V, f64) = msd[i..][0..V].*;
                msd[i..][0..V].* = prev + sq;
            }
            // Scalar tail
            while (i < n) : (i += 1) {
                const dx = @as(f64, frame.x[i]) - mean_x[i];
                const dy = @as(f64, frame.y[i]) - mean_y[i];
                const dz = @as(f64, frame.z[i]) - mean_z[i];
                msd[i] += dx * dx + dy * dy + dz * dz;
            }
        }
    }
}

/// Compute per-atom RMSF in parallel using multiple threads with SIMD inner loops.
///
/// Falls back to single-threaded `compute` when `n_threads <= 1` or
/// `frames.len < 4`.
///
/// Returns a heap-allocated []f64 owned by the caller (free with `allocator`).
pub fn computeParallel(
    allocator: std.mem.Allocator,
    frames: []const types.Frame,
    atom_indices: ?[]const u32,
    n_threads: usize,
) ![]f64 {
    // Fallback to single-threaded for small workloads
    if (n_threads <= 1 or frames.len < 4) {
        return compute(allocator, frames, atom_indices);
    }

    const n_frames: f64 = @floatFromInt(frames.len);
    const first = frames[0];
    const n_atoms: usize = if (atom_indices) |idx| idx.len else first.nAtoms();

    const cpu_count = std.Thread.getCpuCount() catch {
        return compute(allocator, frames, atom_indices);
    };
    const actual_threads = @min(n_threads, cpu_count);
    // Don't use more threads than frames
    const thread_count = @min(actual_threads, frames.len);

    // Allocate thread-local sum arrays for pass 1
    const tl_sum_x = try allocator.alloc([]f64, thread_count);
    defer allocator.free(tl_sum_x);
    const tl_sum_y = try allocator.alloc([]f64, thread_count);
    defer allocator.free(tl_sum_y);
    const tl_sum_z = try allocator.alloc([]f64, thread_count);
    defer allocator.free(tl_sum_z);

    // Initialize to empty slices so defer-free is safe on partial allocation failure
    for (0..thread_count) |t| {
        tl_sum_x[t] = &.{};
        tl_sum_y[t] = &.{};
        tl_sum_z[t] = &.{};
    }
    defer for (0..thread_count) |t| {
        if (tl_sum_x[t].len > 0) allocator.free(tl_sum_x[t]);
        if (tl_sum_y[t].len > 0) allocator.free(tl_sum_y[t]);
        if (tl_sum_z[t].len > 0) allocator.free(tl_sum_z[t]);
    };

    for (0..thread_count) |t| {
        tl_sum_x[t] = try allocator.alloc(f64, n_atoms);
        @memset(tl_sum_x[t], 0.0);
        tl_sum_y[t] = try allocator.alloc(f64, n_atoms);
        @memset(tl_sum_y[t], 0.0);
        tl_sum_z[t] = try allocator.alloc(f64, n_atoms);
        @memset(tl_sum_z[t], 0.0);
    }

    // Spawn threads for pass 1 (mean positions)
    const threads = try allocator.alloc(std.Thread, thread_count);
    defer allocator.free(threads);

    var pass1_spawned: usize = 0;
    errdefer for (threads[0..pass1_spawned]) |thread| {
        thread.join();
    };
    for (0..thread_count) |t| {
        const frame_start = t * frames.len / thread_count;
        const frame_end = (t + 1) * frames.len / thread_count;
        threads[t] = try std.Thread.spawn(.{}, pass1Worker, .{
            frames[frame_start..frame_end],
            atom_indices,
            tl_sum_x[t],
            tl_sum_y[t],
            tl_sum_z[t],
        });
        pass1_spawned += 1;
    }
    for (threads) |thread| {
        thread.join();
    }
    pass1_spawned = 0;

    // Reduce thread-local sums into mean_x/y/z (reuse tl_sum_x/y/z[0])
    const mean_x = tl_sum_x[0];
    const mean_y = tl_sum_y[0];
    const mean_z = tl_sum_z[0];
    for (1..thread_count) |t| {
        for (0..n_atoms) |i| {
            mean_x[i] += tl_sum_x[t][i];
            mean_y[i] += tl_sum_y[t][i];
            mean_z[i] += tl_sum_z[t][i];
        }
    }
    for (0..n_atoms) |i| {
        mean_x[i] /= n_frames;
        mean_y[i] /= n_frames;
        mean_z[i] /= n_frames;
    }

    // Allocate thread-local msd arrays for pass 2
    const tl_msd = try allocator.alloc([]f64, thread_count);
    defer allocator.free(tl_msd);

    // Initialize to empty slices so defer-free is safe on partial allocation failure
    for (0..thread_count) |t| {
        tl_msd[t] = &.{};
    }
    defer for (1..thread_count) |t| {
        if (tl_msd[t].len > 0) allocator.free(tl_msd[t]);
    };
    errdefer if (tl_msd[0].len > 0) allocator.free(tl_msd[0]);

    for (0..thread_count) |t| {
        tl_msd[t] = try allocator.alloc(f64, n_atoms);
        @memset(tl_msd[t], 0.0);
    }

    // Spawn threads for pass 2 (MSD)
    var pass2_spawned: usize = 0;
    errdefer for (threads[0..pass2_spawned]) |thread| {
        thread.join();
    };
    for (0..thread_count) |t| {
        const frame_start = t * frames.len / thread_count;
        const frame_end = (t + 1) * frames.len / thread_count;
        threads[t] = try std.Thread.spawn(.{}, pass2Worker, .{
            frames[frame_start..frame_end],
            atom_indices,
            mean_x,
            mean_y,
            mean_z,
            tl_msd[t],
        });
        pass2_spawned += 1;
    }
    for (threads) |thread| {
        thread.join();
    }
    pass2_spawned = 0;

    // Reduce MSD and compute final result; reuse tl_msd[0] as output
    const result = tl_msd[0];
    for (1..thread_count) |t| {
        for (0..n_atoms) |i| {
            result[i] += tl_msd[t][i];
        }
    }
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

test "rmsf: computeParallel matches single-threaded compute" {
    const allocator = std.testing.allocator;

    // 5 atoms, 8 frames with varying positions
    const n_frames = 8;
    const n_atoms = 5;

    var frames: [n_frames]types.Frame = undefined;
    for (&frames, 0..) |*f, fi| {
        f.* = try types.Frame.init(allocator, n_atoms);
        const t: f32 = @floatFromInt(fi);
        for (0..n_atoms) |a| {
            const ai: f32 = @floatFromInt(a);
            // Varying positions: each atom oscillates differently per frame
            f.x[a] = ai * 1.5 + @sin(t * 0.7 + ai);
            f.y[a] = ai * 0.8 - @cos(t * 1.1 + ai * 0.5);
            f.z[a] = (t - 4.0) * (ai + 1.0) * 0.3;
        }
    }
    defer for (&frames) |*f| f.deinit();

    // Single-threaded reference
    const ref = try compute(allocator, &frames, null);
    defer allocator.free(ref);

    // Multi-threaded (4 threads)
    const par = try computeParallel(allocator, &frames, null, 4);
    defer allocator.free(par);

    try std.testing.expectEqual(ref.len, par.len);
    for (ref, par) |r, p| {
        try std.testing.expectApproxEqAbs(r, p, 1e-10);
    }

    // Also test with atom_indices
    const indices = [_]u32{ 0, 2, 4 };
    const ref_idx = try compute(allocator, &frames, &indices);
    defer allocator.free(ref_idx);

    const par_idx = try computeParallel(allocator, &frames, &indices, 4);
    defer allocator.free(par_idx);

    try std.testing.expectEqual(ref_idx.len, par_idx.len);
    for (ref_idx, par_idx) |r, p| {
        try std.testing.expectApproxEqAbs(r, p, 1e-10);
    }
}

test "rmsf: computeParallel with 32 atoms exercises SIMD path" {
    const allocator = std.testing.allocator;

    const n_frames = 8;
    const n_atoms = 32;

    var frames: [n_frames]types.Frame = undefined;
    for (&frames, 0..) |*f, fi| {
        f.* = try types.Frame.init(allocator, n_atoms);
        const t: f32 = @floatFromInt(fi);
        for (0..n_atoms) |a| {
            const ai: f32 = @floatFromInt(a);
            f.x[a] = ai * 0.5 + @sin(t * 0.3 + ai * 0.1);
            f.y[a] = ai * 0.3 - @cos(t * 0.5 + ai * 0.2);
            f.z[a] = (t - 3.0) * (ai + 1.0) * 0.1;
        }
    }
    defer for (&frames) |*f| f.deinit();

    const ref = try compute(allocator, &frames, null);
    defer allocator.free(ref);

    const par = try computeParallel(allocator, &frames, null, 4);
    defer allocator.free(par);

    try std.testing.expectEqual(ref.len, par.len);
    for (ref, par) |r, p| {
        try std.testing.expectApproxEqAbs(r, p, 1e-10);
    }
}

test "rmsf: computeParallel caps threads when more than frames" {
    const allocator = std.testing.allocator;

    // Only 2 frames but request 16 threads -- should cap to 2 threads
    const n_atoms = 4;

    var frame1 = try types.Frame.init(allocator, n_atoms);
    defer frame1.deinit();
    var frame2 = try types.Frame.init(allocator, n_atoms);
    defer frame2.deinit();

    for (0..n_atoms) |a| {
        const ai: f32 = @floatFromInt(a);
        frame1.x[a] = ai;
        frame1.y[a] = 0.0;
        frame1.z[a] = 0.0;
        frame2.x[a] = ai + 1.0;
        frame2.y[a] = 0.0;
        frame2.z[a] = 0.0;
    }

    const frames = [_]types.Frame{ frame1, frame2 };

    const ref = try compute(allocator, &frames, null);
    defer allocator.free(ref);

    // Request 16 threads with only 2 frames; falls back to single-threaded
    // because frames.len < 4, exercising the fallback path
    const par = try computeParallel(allocator, &frames, null, 16);
    defer allocator.free(par);

    try std.testing.expectEqual(ref.len, par.len);
    for (ref, par) |r, p| {
        try std.testing.expectApproxEqAbs(r, p, 1e-10);
    }
}
