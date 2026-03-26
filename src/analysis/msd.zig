//! Mean Square Displacement (MSD) for computing diffusion coefficients.
//!
//! MSD(τ) = <|r(t+τ) - r(t)|²>
//!
//! Averaged over all time origins t and (optionally) selected atoms.

const std = @import("std");
const types = @import("../types.zig");
const simd = @import("../simd/vec.zig");

/// Compute MSD as a function of lag time τ.
///
/// Output: MSD values for τ = 0, 1, ..., n_frames-1 (in Å²).
/// Caller owns the returned slice.
pub fn compute(
    allocator: std.mem.Allocator,
    frames: []const types.Frame,
    atom_indices: ?[]const u32,
) ![]f64 {
    const n_frames = frames.len;
    if (n_frames == 0) return error.NoFrames;

    const frame_n = frames[0].nAtoms();
    // Validate atom indices
    if (atom_indices) |indices| {
        for (indices) |idx| {
            if (idx >= frame_n) return error.IndexOutOfBounds;
        }
    }

    const msd = try allocator.alloc(f64, n_frames);
    @memset(msd, 0.0);

    const n_atoms: usize = if (atom_indices) |idx| idx.len else frame_n;
    if (n_atoms == 0) return msd;

    const n_atoms_f: f64 = @floatFromInt(n_atoms);

    for (0..n_frames) |tau| {
        var sum: f64 = 0.0;
        var count: usize = 0;

        for (0..n_frames - tau) |t| {
            const frame_t = frames[t];
            const frame_tau = frames[t + tau];

            if (atom_indices) |indices| {
                for (indices) |idx| {
                    const dx: f64 = @as(f64, frame_tau.x[idx]) - @as(f64, frame_t.x[idx]);
                    const dy: f64 = @as(f64, frame_tau.y[idx]) - @as(f64, frame_t.y[idx]);
                    const dz: f64 = @as(f64, frame_tau.z[idx]) - @as(f64, frame_t.z[idx]);
                    sum += dx * dx + dy * dy + dz * dz;
                }
            } else {
                for (0..frame_n) |idx| {
                    const dx: f64 = @as(f64, frame_tau.x[idx]) - @as(f64, frame_t.x[idx]);
                    const dy: f64 = @as(f64, frame_tau.y[idx]) - @as(f64, frame_t.y[idx]);
                    const dz: f64 = @as(f64, frame_tau.z[idx]) - @as(f64, frame_t.z[idx]);
                    sum += dx * dx + dy * dy + dz * dz;
                }
            }
            count += 1;
        }

        if (count > 0) {
            msd[tau] = sum / (@as(f64, @floatFromInt(count)) * n_atoms_f);
        }
    }

    return msd;
}

/// Worker function for parallel MSD computation.
/// Each worker computes MSD for a range of tau values [tau_start, tau_end).
fn msdWorker(
    tau_start: usize,
    tau_end: usize,
    frames: []const types.Frame,
    atom_indices: ?[]const u32,
    n_atoms: usize,
    n_atoms_f: f64,
    msd: []f64,
) void {
    const n_frames = frames.len;
    const vec_len = simd.optimal_vector_width.f32_width;

    for (tau_start..tau_end) |tau| {
        var sum: f64 = 0.0;
        var count: usize = 0;

        for (0..n_frames - tau) |t| {
            const frame_t = frames[t];
            const frame_tau = frames[t + tau];

            if (atom_indices) |indices| {
                for (indices) |idx| {
                    const dx: f64 = @as(f64, frame_tau.x[idx]) - @as(f64, frame_t.x[idx]);
                    const dy: f64 = @as(f64, frame_tau.y[idx]) - @as(f64, frame_t.y[idx]);
                    const dz: f64 = @as(f64, frame_tau.z[idx]) - @as(f64, frame_t.z[idx]);
                    sum += dx * dx + dy * dy + dz * dz;
                }
            } else {
                // SIMD path for non-indexed atoms.
                const V = vec_len;
                var sum_vec: @Vector(V, f64) = @splat(0.0);
                var a: usize = 0;
                while (a + V <= n_atoms) : (a += V) {
                    const vx_t: @Vector(V, f64) = @floatCast(@as(@Vector(V, f32), frame_t.x[a..][0..V].*));
                    const vx_tau: @Vector(V, f64) = @floatCast(@as(@Vector(V, f32), frame_tau.x[a..][0..V].*));
                    const dx = vx_tau - vx_t;

                    const vy_t: @Vector(V, f64) = @floatCast(@as(@Vector(V, f32), frame_t.y[a..][0..V].*));
                    const vy_tau: @Vector(V, f64) = @floatCast(@as(@Vector(V, f32), frame_tau.y[a..][0..V].*));
                    const dy = vy_tau - vy_t;

                    const vz_t: @Vector(V, f64) = @floatCast(@as(@Vector(V, f32), frame_t.z[a..][0..V].*));
                    const vz_tau: @Vector(V, f64) = @floatCast(@as(@Vector(V, f32), frame_tau.z[a..][0..V].*));
                    const dz = vz_tau - vz_t;

                    sum_vec += dx * dx + dy * dy + dz * dz;
                }
                sum += @reduce(.Add, sum_vec);

                // Scalar tail for remaining atoms.
                while (a < n_atoms) : (a += 1) {
                    const dx: f64 = @as(f64, frame_tau.x[a]) - @as(f64, frame_t.x[a]);
                    const dy: f64 = @as(f64, frame_tau.y[a]) - @as(f64, frame_t.y[a]);
                    const dz: f64 = @as(f64, frame_tau.z[a]) - @as(f64, frame_t.z[a]);
                    sum += dx * dx + dy * dy + dz * dz;
                }
            }
            count += 1;
        }

        if (count > 0) {
            msd[tau] = sum / (@as(f64, @floatFromInt(count)) * n_atoms_f);
        }
    }
}

/// Compute MSD with SIMD + multi-threading.
///
/// Partitions tau values across threads. Falls back to single-threaded
/// `compute` when n_threads <= 1 or frames.len < 4.
pub fn computeParallel(
    allocator: std.mem.Allocator,
    frames: []const types.Frame,
    atom_indices: ?[]const u32,
    n_threads: usize,
) ![]f64 {
    const n_frames = frames.len;
    if (n_frames == 0) return error.NoFrames;

    // Validate atom indices.
    const frame_n = frames[0].nAtoms();
    if (atom_indices) |indices| {
        for (indices) |idx| {
            if (idx >= frame_n) return error.IndexOutOfBounds;
        }
    }

    // Fallback to single-threaded for small workloads.
    if (n_threads <= 1 or n_frames < 4) {
        return compute(allocator, frames, atom_indices);
    }

    const cpu_count = std.Thread.getCpuCount() catch {
        return compute(allocator, frames, atom_indices);
    };
    const actual_threads = @min(n_threads, cpu_count);
    const thread_count = @min(actual_threads, n_frames);

    const n_atoms: usize = if (atom_indices) |idx| idx.len else frame_n;
    const n_atoms_f: f64 = @floatFromInt(n_atoms);

    const msd = try allocator.alloc(f64, n_frames);
    @memset(msd, 0.0);
    errdefer allocator.free(msd);

    if (n_atoms == 0) return msd;

    // Spawn threads, partitioning tau values.
    const threads = try allocator.alloc(std.Thread, thread_count);
    defer allocator.free(threads);

    var spawned: usize = 0;
    errdefer for (threads[0..spawned]) |thread| {
        thread.join();
    };

    const chunk = n_frames / thread_count;
    const remainder = n_frames % thread_count;

    for (0..thread_count) |t| {
        const start = t * chunk + @min(t, remainder);
        const end = start + chunk + @as(usize, if (t < remainder) 1 else 0);

        threads[t] = try std.Thread.spawn(.{}, msdWorker, .{
            start,
            end,
            frames,
            atom_indices,
            n_atoms,
            n_atoms_f,
            msd,
        });
        spawned += 1;
    }

    // Join all threads.
    for (threads[0..spawned]) |thread| {
        thread.join();
    }
    spawned = 0;

    return msd;
}

test "msd: stationary atoms → MSD = 0" {
    const allocator = std.testing.allocator;
    var f1 = try types.Frame.init(allocator, 2);
    defer f1.deinit();
    f1.x[0] = 0.0;
    f1.x[1] = 1.0;

    const frames = [_]types.Frame{ f1, f1, f1 };
    const msd = try compute(allocator, &frames, null);
    defer allocator.free(msd);

    for (msd) |v| {
        try std.testing.expectApproxEqAbs(@as(f64, 0.0), v, 1e-10);
    }
}

test "msd: linear motion → MSD = τ²" {
    const allocator = std.testing.allocator;

    var f0 = try types.Frame.init(allocator, 1);
    defer f0.deinit();
    f0.x[0] = 0.0;
    f0.y[0] = 0.0;
    f0.z[0] = 0.0;

    var f1 = try types.Frame.init(allocator, 1);
    defer f1.deinit();
    f1.x[0] = 1.0;
    f1.y[0] = 0.0;
    f1.z[0] = 0.0;

    var f2 = try types.Frame.init(allocator, 1);
    defer f2.deinit();
    f2.x[0] = 2.0;
    f2.y[0] = 0.0;
    f2.z[0] = 0.0;

    var f3 = try types.Frame.init(allocator, 1);
    defer f3.deinit();
    f3.x[0] = 3.0;
    f3.y[0] = 0.0;
    f3.z[0] = 0.0;

    const frames = [_]types.Frame{ f0, f1, f2, f3 };
    const msd = try compute(allocator, &frames, null);
    defer allocator.free(msd);

    try std.testing.expectApproxEqAbs(@as(f64, 0.0), msd[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), msd[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), msd[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), msd[3], 1e-10);
}

test "msd: computeParallel matches single-threaded" {
    const allocator = std.testing.allocator;

    // Linear motion: 4 frames, 1 atom, x = 0,1,2,3
    var f0 = try types.Frame.init(allocator, 1);
    defer f0.deinit();
    f0.x[0] = 0.0;
    f0.y[0] = 0.0;
    f0.z[0] = 0.0;

    var f1 = try types.Frame.init(allocator, 1);
    defer f1.deinit();
    f1.x[0] = 1.0;
    f1.y[0] = 0.0;
    f1.z[0] = 0.0;

    var f2 = try types.Frame.init(allocator, 1);
    defer f2.deinit();
    f2.x[0] = 2.0;
    f2.y[0] = 0.0;
    f2.z[0] = 0.0;

    var f3 = try types.Frame.init(allocator, 1);
    defer f3.deinit();
    f3.x[0] = 3.0;
    f3.y[0] = 0.0;
    f3.z[0] = 0.0;

    const frames = [_]types.Frame{ f0, f1, f2, f3 };

    const msd_single = try compute(allocator, &frames, null);
    defer allocator.free(msd_single);

    const msd_parallel = try computeParallel(allocator, &frames, null, 4);
    defer allocator.free(msd_parallel);

    try std.testing.expectEqual(msd_single.len, msd_parallel.len);
    for (0..msd_single.len) |i| {
        try std.testing.expectApproxEqAbs(msd_single[i], msd_parallel[i], 1e-10);
    }

    // Verify MSD[tau] = tau^2 for linear motion.
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), msd_parallel[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), msd_parallel[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), msd_parallel[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), msd_parallel[3], 1e-10);
}

test "msd: computeParallel with more frames exercises threading" {
    const allocator = std.testing.allocator;

    const n_frames = 10;
    const n_atoms = 5;

    var frame_list: [n_frames]types.Frame = undefined;
    var initialized: usize = 0;
    errdefer for (frame_list[0..initialized]) |*f| f.deinit();

    for (0..n_frames) |i| {
        frame_list[i] = try types.Frame.init(allocator, n_atoms);
        initialized += 1;
        for (0..n_atoms) |a| {
            const fi: f32 = @floatFromInt(i);
            const fa: f32 = @floatFromInt(a);
            frame_list[i].x[a] = fi * 0.5 + fa * 0.1;
            frame_list[i].y[a] = fi * 0.3 - fa * 0.2;
            frame_list[i].z[a] = fi * 0.1 + fa * 0.4;
        }
    }
    defer for (&frame_list) |*f| f.deinit();

    const frames: []const types.Frame = &frame_list;

    const msd_single = try compute(allocator, frames, null);
    defer allocator.free(msd_single);

    const msd_parallel = try computeParallel(allocator, frames, null, 4);
    defer allocator.free(msd_parallel);

    try std.testing.expectEqual(msd_single.len, msd_parallel.len);
    for (0..msd_single.len) |i| {
        try std.testing.expectApproxEqAbs(msd_single[i], msd_parallel[i], 1e-10);
    }
}
