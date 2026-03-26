//! Principal Component Analysis (PCA) of atomic coordinate fluctuations.
//!
//! Computes the 3N × 3N covariance matrix. Eigendecomposition is left
//! to the caller (e.g., numpy.linalg.eigh on the Python side).

const std = @import("std");
const types = @import("../types.zig");
const simd = @import("../simd/vec.zig");

/// Compute the covariance matrix of atomic coordinate fluctuations.
/// Returns flat (3N × 3N) in row-major order (f64). Caller owns the slice.
pub fn computeCovarianceMatrix(
    allocator: std.mem.Allocator,
    frames: []const types.Frame,
    atom_indices: ?[]const u32,
) ![]f64 {
    const n_frames = frames.len;
    if (n_frames < 2) return error.TooFewFrames;

    const frame_n = frames[0].nAtoms();
    // Validate atom indices
    if (atom_indices) |indices| {
        for (indices) |idx| {
            if (idx >= frame_n) return error.IndexOutOfBounds;
        }
    }

    const n_sel: usize = if (atom_indices) |idx| idx.len else frame_n;
    const dim = n_sel * 3;

    // Guard against unreasonable dimension (3N × 3N matrix)
    // 10,000 dimensions = ~800 MB (10000^2 * 8 bytes)
    if (dim > 30_000) return error.DimensionTooLarge;
    const n_f: f64 = @floatFromInt(n_frames);

    const mean = try allocator.alloc(f64, dim);
    defer allocator.free(mean);
    @memset(mean, 0.0);

    for (frames) |frame| {
        if (atom_indices) |indices| {
            for (indices, 0..) |idx, i| {
                mean[i * 3 + 0] += @as(f64, frame.x[idx]);
                mean[i * 3 + 1] += @as(f64, frame.y[idx]);
                mean[i * 3 + 2] += @as(f64, frame.z[idx]);
            }
        } else {
            for (0..n_sel) |i| {
                mean[i * 3 + 0] += @as(f64, frame.x[i]);
                mean[i * 3 + 1] += @as(f64, frame.y[i]);
                mean[i * 3 + 2] += @as(f64, frame.z[i]);
            }
        }
    }
    for (mean) |*m| m.* /= n_f;

    const cov = try allocator.alloc(f64, dim * dim);
    @memset(cov, 0.0);

    const dev = try allocator.alloc(f64, dim);
    defer allocator.free(dev);

    for (frames) |frame| {
        if (atom_indices) |indices| {
            for (indices, 0..) |idx, i| {
                dev[i * 3 + 0] = @as(f64, frame.x[idx]) - mean[i * 3 + 0];
                dev[i * 3 + 1] = @as(f64, frame.y[idx]) - mean[i * 3 + 1];
                dev[i * 3 + 2] = @as(f64, frame.z[idx]) - mean[i * 3 + 2];
            }
        } else {
            for (0..n_sel) |i| {
                dev[i * 3 + 0] = @as(f64, frame.x[i]) - mean[i * 3 + 0];
                dev[i * 3 + 1] = @as(f64, frame.y[i]) - mean[i * 3 + 1];
                dev[i * 3 + 2] = @as(f64, frame.z[i]) - mean[i * 3 + 2];
            }
        }

        for (0..dim) |i| {
            for (i..dim) |j| {
                cov[i * dim + j] += dev[i] * dev[j];
            }
        }
    }

    const norm = 1.0 / (@as(f64, @floatFromInt(n_frames)) - 1.0);
    for (0..dim) |i| {
        for (i..dim) |j| {
            cov[i * dim + j] *= norm;
            cov[j * dim + i] = cov[i * dim + j];
        }
    }

    return cov;
}

/// Worker: accumulate mean positions for a chunk of frames.
fn pcaMeanWorker(
    frames: []const types.Frame,
    atom_indices: ?[]const u32,
    n_sel: usize,
    local_mean: []f64,
) void {
    for (frames) |frame| {
        if (atom_indices) |indices| {
            for (indices, 0..) |idx, i| {
                local_mean[i * 3 + 0] += @as(f64, frame.x[idx]);
                local_mean[i * 3 + 1] += @as(f64, frame.y[idx]);
                local_mean[i * 3 + 2] += @as(f64, frame.z[idx]);
            }
        } else {
            for (0..n_sel) |i| {
                local_mean[i * 3 + 0] += @as(f64, frame.x[i]);
                local_mean[i * 3 + 1] += @as(f64, frame.y[i]);
                local_mean[i * 3 + 2] += @as(f64, frame.z[i]);
            }
        }
    }
}

/// Worker: accumulate covariance outer products for a chunk of frames (SIMD inner loop).
fn pcaCovWorker(
    frames: []const types.Frame,
    atom_indices: ?[]const u32,
    n_sel: usize,
    dim: usize,
    mean: []const f64,
    local_cov: []f64,
    local_dev: []f64,
) void {
    const V = simd.optimal_vector_width.f64_width;

    for (frames) |frame| {
        // Compute deviation vector.
        if (atom_indices) |indices| {
            for (indices, 0..) |idx, i| {
                local_dev[i * 3 + 0] = @as(f64, frame.x[idx]) - mean[i * 3 + 0];
                local_dev[i * 3 + 1] = @as(f64, frame.y[idx]) - mean[i * 3 + 1];
                local_dev[i * 3 + 2] = @as(f64, frame.z[idx]) - mean[i * 3 + 2];
            }
        } else {
            for (0..n_sel) |i| {
                local_dev[i * 3 + 0] = @as(f64, frame.x[i]) - mean[i * 3 + 0];
                local_dev[i * 3 + 1] = @as(f64, frame.y[i]) - mean[i * 3 + 1];
                local_dev[i * 3 + 2] = @as(f64, frame.z[i]) - mean[i * 3 + 2];
            }
        }

        // Accumulate upper triangle of outer product (SIMD inner loop).
        for (0..dim) |i| {
            const dev_i_splat: @Vector(V, f64) = @splat(local_dev[i]);
            var j: usize = i;
            while (j + V <= dim) : (j += V) {
                const dev_j: @Vector(V, f64) = local_dev[j..][0..V].*;
                const old: @Vector(V, f64) = local_cov[(i * dim + j)..][0..V].*;
                local_cov[(i * dim + j)..][0..V].* = old + dev_i_splat * dev_j;
            }
            // Scalar tail.
            while (j < dim) : (j += 1) {
                local_cov[i * dim + j] += local_dev[i] * local_dev[j];
            }
        }
    }
}

/// Threaded + SIMD covariance matrix computation.
/// Falls back to single-threaded `computeCovarianceMatrix` for small workloads.
/// Returns flat (3N × 3N) in row-major order (f64). Caller owns the slice.
pub fn computeCovarianceMatrixParallel(
    allocator: std.mem.Allocator,
    frames: []const types.Frame,
    atom_indices: ?[]const u32,
    n_threads: usize,
) ![]f64 {
    const n_frames = frames.len;
    if (n_frames < 2) return error.TooFewFrames;

    const frame_n = frames[0].nAtoms();
    if (atom_indices) |indices| {
        for (indices) |idx| {
            if (idx >= frame_n) return error.IndexOutOfBounds;
        }
    }

    const n_sel: usize = if (atom_indices) |idx| idx.len else frame_n;
    const dim = n_sel * 3;
    if (dim > 30_000) return error.DimensionTooLarge;

    // Fallback for small workloads.
    if (n_threads <= 1 or n_frames < 4) {
        return computeCovarianceMatrix(allocator, frames, atom_indices);
    }

    const cpu_count = std.Thread.getCpuCount() catch {
        return computeCovarianceMatrix(allocator, frames, atom_indices);
    };
    const actual_threads = @min(n_threads, cpu_count);
    const thread_count = @min(actual_threads, n_frames);

    // --- Phase 1: Mean calculation (threaded) ---

    // Allocate thread-local mean arrays.
    const tl_means = try allocator.alloc([]f64, thread_count);
    defer allocator.free(tl_means);
    for (0..thread_count) |t| {
        tl_means[t] = &.{};
    }
    defer for (0..thread_count) |t| {
        if (tl_means[t].len > 0) allocator.free(tl_means[t]);
    };
    for (0..thread_count) |t| {
        tl_means[t] = try allocator.alloc(f64, dim);
        @memset(tl_means[t], 0.0);
    }

    // Spawn mean-computation threads.
    const threads_mean = try allocator.alloc(std.Thread, thread_count);
    defer allocator.free(threads_mean);

    var spawned: usize = 0;
    errdefer for (threads_mean[0..spawned]) |thread| {
        thread.join();
    };

    const chunk_m = n_frames / thread_count;
    const rem_m = n_frames % thread_count;

    for (0..thread_count) |t| {
        const start = t * chunk_m + @min(t, rem_m);
        const end = start + chunk_m + @as(usize, if (t < rem_m) 1 else 0);

        threads_mean[t] = try std.Thread.spawn(.{}, pcaMeanWorker, .{
            frames[start..end],
            atom_indices,
            n_sel,
            tl_means[t],
        });
        spawned += 1;
    }

    for (threads_mean[0..spawned]) |thread| {
        thread.join();
    }
    spawned = 0;

    // Reduce means.
    const mean = tl_means[0];
    for (1..thread_count) |t| {
        for (0..dim) |k| {
            mean[k] += tl_means[t][k];
        }
    }
    const n_f: f64 = @floatFromInt(n_frames);
    for (mean) |*m| m.* /= n_f;

    // --- Phase 2: Covariance accumulation (threaded + SIMD) ---

    // Allocate thread-local covariance matrices and dev buffers.
    const tl_covs = try allocator.alloc([]f64, thread_count);
    defer allocator.free(tl_covs);
    for (0..thread_count) |t| {
        tl_covs[t] = &.{};
    }
    defer for (1..thread_count) |t| {
        if (tl_covs[t].len > 0) allocator.free(tl_covs[t]);
    };

    const tl_devs = try allocator.alloc([]f64, thread_count);
    defer allocator.free(tl_devs);
    for (0..thread_count) |t| {
        tl_devs[t] = &.{};
    }
    defer for (0..thread_count) |t| {
        if (tl_devs[t].len > 0) allocator.free(tl_devs[t]);
    };

    // tl_covs[0] is the output buffer (caller-owned), rest are temporary.
    const cov = try allocator.alloc(f64, dim * dim);
    errdefer allocator.free(cov);
    @memset(cov, 0.0);
    tl_covs[0] = cov;

    for (1..thread_count) |t| {
        tl_covs[t] = try allocator.alloc(f64, dim * dim);
        @memset(tl_covs[t], 0.0);
    }

    for (0..thread_count) |t| {
        tl_devs[t] = try allocator.alloc(f64, dim);
        @memset(tl_devs[t], 0.0);
    }

    // Spawn covariance-computation threads.
    const threads_cov = try allocator.alloc(std.Thread, thread_count);
    defer allocator.free(threads_cov);

    errdefer for (threads_cov[0..spawned]) |thread| {
        thread.join();
    };

    for (0..thread_count) |t| {
        const start = t * chunk_m + @min(t, rem_m);
        const end = start + chunk_m + @as(usize, if (t < rem_m) 1 else 0);

        threads_cov[t] = try std.Thread.spawn(.{}, pcaCovWorker, .{
            frames[start..end],
            atom_indices,
            n_sel,
            dim,
            mean,
            tl_covs[t],
            tl_devs[t],
        });
        spawned += 1;
    }

    for (threads_cov[0..spawned]) |thread| {
        thread.join();
    }
    spawned = 0;

    // --- Phase 3: Reduce and normalize ---

    // Element-wise sum all thread-local cov matrices into cov (tl_covs[0]).
    for (1..thread_count) |t| {
        for (0..dim * dim) |k| {
            cov[k] += tl_covs[t][k];
        }
    }

    // Normalize by (n_frames - 1) and symmetrize.
    const norm = 1.0 / (n_f - 1.0);
    for (0..dim) |i| {
        for (i..dim) |j| {
            cov[i * dim + j] *= norm;
            cov[j * dim + i] = cov[i * dim + j];
        }
    }

    return cov;
}

test "covariance: stationary → zeros" {
    const allocator = std.testing.allocator;
    var f1 = try types.Frame.init(allocator, 1);
    defer f1.deinit();
    f1.x[0] = 5.0;
    f1.y[0] = 3.0;
    f1.z[0] = 1.0;

    const frames = [_]types.Frame{ f1, f1 };
    const cov = try computeCovarianceMatrix(allocator, &frames, null);
    defer allocator.free(cov);

    for (cov) |v| {
        try std.testing.expectApproxEqAbs(@as(f64, 0.0), v, 1e-10);
    }
}

test "covariance: x-oscillation → variance in x only" {
    const allocator = std.testing.allocator;
    var f1 = try types.Frame.init(allocator, 1);
    defer f1.deinit();
    f1.x[0] = -1.0;
    f1.y[0] = 0.0;
    f1.z[0] = 0.0;

    var f2 = try types.Frame.init(allocator, 1);
    defer f2.deinit();
    f2.x[0] = 1.0;
    f2.y[0] = 0.0;
    f2.z[0] = 0.0;

    const frames = [_]types.Frame{ f1, f2 };
    const cov = try computeCovarianceMatrix(allocator, &frames, null);
    defer allocator.free(cov);

    try std.testing.expectApproxEqAbs(@as(f64, 2.0), cov[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), cov[4], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), cov[8], 1e-10);
}

test "covariance: too few frames → error" {
    const allocator = std.testing.allocator;
    var f1 = try types.Frame.init(allocator, 1);
    defer f1.deinit();
    const frames = [_]types.Frame{f1};
    try std.testing.expectError(error.TooFewFrames, computeCovarianceMatrix(allocator, &frames, null));
}

test "pca: computeCovarianceMatrixParallel matches single-threaded" {
    const allocator = std.testing.allocator;

    // x-oscillation pattern: 2 frames, 1 atom.
    var f1 = try types.Frame.init(allocator, 1);
    defer f1.deinit();
    f1.x[0] = -1.0;
    f1.y[0] = 0.0;
    f1.z[0] = 0.0;

    var f2 = try types.Frame.init(allocator, 1);
    defer f2.deinit();
    f2.x[0] = 1.0;
    f2.y[0] = 0.0;
    f2.z[0] = 0.0;

    const frames = [_]types.Frame{ f1, f2 };

    // Single-threaded reference.
    const cov_st = try computeCovarianceMatrix(allocator, &frames, null);
    defer allocator.free(cov_st);

    // Parallel (falls back to single-threaded for < 4 frames).
    const cov_par = try computeCovarianceMatrixParallel(allocator, &frames, null, 4);
    defer allocator.free(cov_par);

    for (cov_st, cov_par) |a, b| {
        try std.testing.expectApproxEqAbs(a, b, 1e-10);
    }
}

test "pca: computeCovarianceMatrixParallel multi-frame multi-atom" {
    const allocator = std.testing.allocator;

    // 8 frames, 3 atoms (dim = 9) to exercise threading + SIMD.
    const n_atoms: usize = 3;
    const n_frames: usize = 8;
    var frame_list: [n_frames]types.Frame = undefined;

    for (0..n_frames) |f| {
        frame_list[f] = try types.Frame.init(allocator, n_atoms);
        const ff: f32 = @floatFromInt(f);
        // Varied positions per frame to create non-trivial covariance.
        frame_list[f].x[0] = 1.0 + ff * 0.5;
        frame_list[f].y[0] = 2.0 - ff * 0.3;
        frame_list[f].z[0] = 0.1 * ff;
        frame_list[f].x[1] = -1.0 + ff * 0.2;
        frame_list[f].y[1] = 0.5 + ff * 0.1;
        frame_list[f].z[1] = 3.0 - ff * 0.4;
        frame_list[f].x[2] = ff * 0.7;
        frame_list[f].y[2] = -ff * 0.6;
        frame_list[f].z[2] = 2.0 + ff * 0.15;
    }
    defer for (&frame_list) |*frame| {
        frame.deinit();
    };

    const frames: []const types.Frame = &frame_list;

    // Single-threaded reference.
    const cov_st = try computeCovarianceMatrix(allocator, frames, null);
    defer allocator.free(cov_st);

    // Parallel (8 frames >= 4, should use threads).
    const cov_par = try computeCovarianceMatrixParallel(allocator, frames, null, 4);
    defer allocator.free(cov_par);

    for (cov_st, cov_par) |a, b| {
        try std.testing.expectApproxEqAbs(a, b, 1e-10);
    }
}
