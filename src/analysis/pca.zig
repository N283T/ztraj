//! Principal Component Analysis (PCA) of atomic coordinate fluctuations.
//!
//! Computes the 3N × 3N covariance matrix. Eigendecomposition is left
//! to the caller (e.g., numpy.linalg.eigh on the Python side).

const std = @import("std");
const types = @import("../types.zig");

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
