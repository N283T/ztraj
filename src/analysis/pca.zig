//! Principal Component Analysis (PCA) of atomic coordinate fluctuations.
//!
//! Computes the 3N × 3N covariance matrix of atomic positions across
//! trajectory frames. Eigendecomposition is left to the caller (e.g.,
//! NumPy's eigh on the Python side) since Zig lacks a general LAPACK.
//!
//! Typical workflow:
//! 1. Select atoms (e.g., CA only) to reduce dimensionality
//! 2. Compute covariance matrix (this module)
//! 3. Eigendecompose (Python/NumPy)
//! 4. Project trajectory onto top principal components

const std = @import("std");
const types = @import("../types.zig");

/// Compute the covariance matrix of atomic coordinate fluctuations.
///
/// Input: trajectory frames + optional atom indices.
/// Output: flat (3N × 3N) covariance matrix in row-major order (f64).
///
/// The covariance is computed as:
///   C_ij = <(x_i - <x_i>) * (x_j - <x_j>)>
///
/// where x_i are the 3N coordinate components (x0,y0,z0,x1,y1,z1,...).
///
/// Caller owns the returned slice.
pub fn computeCovarianceMatrix(
    allocator: std.mem.Allocator,
    frames: []const types.Frame,
    atom_indices: ?[]const u32,
) ![]f64 {
    const n_frames = frames.len;
    if (n_frames < 2) return error.TooFewFrames;

    const n_sel: usize = if (atom_indices) |idx| idx.len else frames[0].nAtoms();
    const dim = n_sel * 3; // 3N dimensions

    const n_f: f64 = @floatFromInt(n_frames);

    // Step 1: Compute mean positions
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

    // Step 2: Compute covariance matrix C = (1/(N-1)) * sum((x - mean)(x - mean)^T)
    const cov = try allocator.alloc(f64, dim * dim);
    @memset(cov, 0.0);

    // Allocate deviation buffer
    const dev = try allocator.alloc(f64, dim);
    defer allocator.free(dev);

    for (frames) |frame| {
        // Compute deviation from mean
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

        // Outer product: cov += dev * dev^T
        for (0..dim) |i| {
            for (i..dim) |j| {
                cov[i * dim + j] += dev[i] * dev[j];
            }
        }
    }

    // Normalize and symmetrize
    const norm = 1.0 / (@as(f64, @floatFromInt(n_frames)) - 1.0);
    for (0..dim) |i| {
        for (i..dim) |j| {
            cov[i * dim + j] *= norm;
            cov[j * dim + i] = cov[i * dim + j]; // symmetric
        }
    }

    return cov;
}

// ============================================================================
// Tests
// ============================================================================

test "covariance: single-atom stationary → all zeros" {
    const allocator = std.testing.allocator;

    var f1 = try types.Frame.init(allocator, 1);
    defer f1.deinit();
    f1.x[0] = 5.0;
    f1.y[0] = 3.0;
    f1.z[0] = 1.0;

    // Two identical frames
    const frames = [_]types.Frame{ f1, f1 };
    const cov = try computeCovarianceMatrix(allocator, &frames, null);
    defer allocator.free(cov);

    // 3x3 matrix, all zeros (no variance)
    for (cov) |v| {
        try std.testing.expectApproxEqAbs(@as(f64, 0.0), v, 1e-10);
    }
}

test "covariance: atom oscillating in x → variance only in x" {
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

    // 3x3 matrix: variance in x (cov[0,0] = 2.0), zero elsewhere
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), cov[0], 1e-10); // Cxx
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), cov[1], 1e-10); // Cxy
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), cov[4], 1e-10); // Cyy
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), cov[8], 1e-10); // Czz
}

test "covariance: too few frames → error" {
    const allocator = std.testing.allocator;
    var f1 = try types.Frame.init(allocator, 1);
    defer f1.deinit();
    const frames = [_]types.Frame{f1};
    try std.testing.expectError(error.TooFewFrames, computeCovarianceMatrix(allocator, &frames, null));
}
