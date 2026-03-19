//! Mean Square Displacement (MSD) for computing diffusion coefficients.
//!
//! MSD(τ) = <|r(t+τ) - r(t)|²>
//!
//! Averaged over all time origins t and (optionally) selected atoms.
//! Uses direct averaging over time origins (not FFT).

const std = @import("std");
const types = @import("../types.zig");

/// Compute MSD as a function of lag time τ.
///
/// Input: array of Frames (trajectory), optional atom indices.
/// Output: MSD values for τ = 0, 1, 2, ..., n_frames-1 (in Å²).
///
/// For each lag τ, averages |r(t+τ) - r(t)|² over all valid t and all
/// selected atoms. Uses f64 accumulation for precision.
///
/// Caller owns the returned slice.
pub fn compute(
    allocator: std.mem.Allocator,
    frames: []const types.Frame,
    atom_indices: ?[]const u32,
) ![]f64 {
    const n_frames = frames.len;
    if (n_frames == 0) return error.NoFrames;

    const msd = try allocator.alloc(f64, n_frames);
    @memset(msd, 0.0);

    const n_atoms: usize = if (atom_indices) |idx| idx.len else frames[0].nAtoms();
    if (n_atoms == 0) {
        return msd; // all zeros
    }

    const n_atoms_f: f64 = @floatFromInt(n_atoms);

    // For each lag τ
    for (0..n_frames) |tau| {
        var sum: f64 = 0.0;
        var count: u32 = 0;

        // Average over all time origins t
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
                for (0..frames[0].nAtoms()) |idx| {
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

// ============================================================================
// Tests
// ============================================================================

test "msd: stationary atoms → MSD = 0 for all lags" {
    const allocator = std.testing.allocator;

    var f1 = try types.Frame.init(allocator, 2);
    defer f1.deinit();
    f1.x[0] = 0.0;
    f1.x[1] = 1.0;
    f1.y[0] = 0.0;
    f1.y[1] = 0.0;
    f1.z[0] = 0.0;
    f1.z[1] = 0.0;

    const frames = [_]types.Frame{ f1, f1, f1 };
    const msd = try compute(allocator, &frames, null);
    defer allocator.free(msd);

    for (msd) |v| {
        try std.testing.expectApproxEqAbs(@as(f64, 0.0), v, 1e-10);
    }
}

test "msd: uniform linear motion → MSD ∝ τ²" {
    const allocator = std.testing.allocator;

    // Atom moves 1 Å per frame along x
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

    // MSD(0) = 0, MSD(1) = 1, MSD(2) = 4, MSD(3) = 9
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), msd[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), msd[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), msd[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), msd[3], 1e-10);
}
