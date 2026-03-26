//! Radial distribution function g(r) computation.
//!
//! g(r) describes how atomic density varies as a function of distance from a
//! reference atom. For an ideal gas, g(r) = 1.0 everywhere.
//!
//! Normalization:
//!   g(r) = N(r, r+dr) / (4π r² dr ρ N_sel1)
//! where ρ = N_sel2 / V_box is the bulk number density.

const std = @import("std");

// ============================================================================
// Public types
// ============================================================================

/// Result of an RDF calculation. Owns its memory; call deinit() when done.
pub const RdfResult = struct {
    /// Bin center positions in Angstroms (length = n_bins).
    r: []f64,
    /// g(r) values at each bin center (length = n_bins).
    g_r: []f64,
    allocator: std.mem.Allocator,

    /// Free all owned slices.
    pub fn deinit(self: *RdfResult) void {
        self.allocator.free(self.r);
        self.allocator.free(self.g_r);
    }
};

/// Configuration for RDF calculation.
pub const Config = struct {
    /// Minimum distance to consider (Å). Default: 0.0.
    r_min: f32 = 0.0,
    /// Maximum distance to consider (Å). Default: 10.0.
    r_max: f32 = 10.0,
    /// Number of histogram bins. Default: 100.
    n_bins: u32 = 100,
};

// ============================================================================
// Public API
// ============================================================================

/// Compute the radial distribution function between two atom selections.
///
/// `sel1_*` and `sel2_*` are SOA coordinate arrays for the two selections.
/// `box_volume` is the simulation box volume in Å³, used to normalize by the
/// ideal-gas density ρ = N_sel2 / V.
///
/// All pairs (i from sel1, j from sel2) are considered, including i == j when
/// the same atom appears in both selections. Caller is responsible for passing
/// the correct selections.
///
/// Returns an RdfResult owned by the caller; call `.deinit()` when done.
pub const ComputeError = error{
    /// Coordinate slices for the same selection have mismatched lengths.
    MismatchedSelectionLengths,
    /// n_bins must be greater than zero.
    ZeroBins,
    /// r_max must be greater than r_min.
    InvalidRange,
    /// box_volume must be positive.
    InvalidBoxVolume,
};

pub fn compute(
    allocator: std.mem.Allocator,
    sel1_x: []const f32,
    sel1_y: []const f32,
    sel1_z: []const f32,
    sel2_x: []const f32,
    sel2_y: []const f32,
    sel2_z: []const f32,
    box_volume: f64,
    config: Config,
) !RdfResult {
    if (sel1_x.len != sel1_y.len or sel1_x.len != sel1_z.len) return ComputeError.MismatchedSelectionLengths;
    if (sel2_x.len != sel2_y.len or sel2_x.len != sel2_z.len) return ComputeError.MismatchedSelectionLengths;
    if (config.n_bins == 0) return ComputeError.ZeroBins;
    if (config.r_max <= config.r_min) return ComputeError.InvalidRange;
    if (box_volume <= 0.0) return ComputeError.InvalidBoxVolume;

    const n_bins = config.n_bins;
    const r_min: f64 = config.r_min;
    const r_max: f64 = config.r_max;
    const bin_width: f64 = (r_max - r_min) / @as(f64, @floatFromInt(n_bins));

    // Histogram counts (using f64 for accumulation).
    const hist = try allocator.alloc(f64, n_bins);
    defer allocator.free(hist);
    @memset(hist, 0.0);

    // Accumulate pairwise distances into histogram bins.
    for (0..sel1_x.len) |i| {
        const ix: f64 = sel1_x[i];
        const iy: f64 = sel1_y[i];
        const iz: f64 = sel1_z[i];

        for (0..sel2_x.len) |j| {
            const dx: f64 = @as(f64, sel2_x[j]) - ix;
            const dy: f64 = @as(f64, sel2_y[j]) - iy;
            const dz: f64 = @as(f64, sel2_z[j]) - iz;
            const r: f64 = @sqrt(dx * dx + dy * dy + dz * dz);

            if (r < r_min or r >= r_max) continue;

            const bin_f: f64 = (r - r_min) / bin_width;
            const bin: u32 = @intFromFloat(bin_f);
            // Guard against floating-point edge at r_max.
            if (bin < n_bins) {
                hist[bin] += 1.0;
            }
        }
    }

    // Build output arrays.
    const r_out = try allocator.alloc(f64, n_bins);
    errdefer allocator.free(r_out);

    const g_r = try allocator.alloc(f64, n_bins);
    errdefer allocator.free(g_r);

    // ρ = N_sel2 / V_box  (number density of sel2 atoms)
    const rho: f64 = @as(f64, @floatFromInt(sel2_x.len)) / box_volume;
    const n_sel1: f64 = @floatFromInt(sel1_x.len);

    for (0..n_bins) |k| {
        // Bin center.
        const r_center: f64 = r_min + ((@as(f64, @floatFromInt(k)) + 0.5) * bin_width);
        r_out[k] = r_center;

        // Volume of the spherical shell [r, r+dr).
        const shell_vol: f64 = (4.0 / 3.0) * std.math.pi *
            ((r_center + 0.5 * bin_width) * (r_center + 0.5 * bin_width) * (r_center + 0.5 * bin_width) -
                (r_center - 0.5 * bin_width) * (r_center - 0.5 * bin_width) * (r_center - 0.5 * bin_width));

        // Expected count in this shell for an ideal gas.
        const expected: f64 = rho * shell_vol * n_sel1;

        if (expected > 0.0) {
            g_r[k] = hist[k] / expected;
        } else {
            g_r[k] = 0.0;
        }
    }

    return RdfResult{
        .r = r_out,
        .g_r = g_r,
        .allocator = allocator,
    };
}

// ============================================================================
// Parallel implementation
// ============================================================================

const simd = @import("../simd/vec.zig");

/// SIMD vector length for f32 operations.
const vec_len = simd.optimal_vector_width.f32_width;

/// Worker function for parallel RDF computation.
/// Each worker processes a contiguous range of sel1 atoms and accumulates
/// pair distances into a thread-local histogram.
fn rdfWorker(
    sel1_x: []const f32,
    sel1_y: []const f32,
    sel1_z: []const f32,
    sel2_x: []const f32,
    sel2_y: []const f32,
    sel2_z: []const f32,
    start: usize,
    end: usize,
    r_min: f64,
    r_max: f64,
    bin_width: f64,
    n_bins: u32,
    local_hist: []f64,
) void {
    const n2 = sel2_x.len;
    const simd_end = n2 - (n2 % vec_len);

    for (start..end) |i| {
        const ix: f64 = sel1_x[i];
        const iy: f64 = sel1_y[i];
        const iz: f64 = sel1_z[i];

        // SIMD inner loop: process sel2 in batches of vec_len.
        const ix_f32: @Vector(vec_len, f32) = @splat(sel1_x[i]);
        const iy_f32: @Vector(vec_len, f32) = @splat(sel1_y[i]);
        const iz_f32: @Vector(vec_len, f32) = @splat(sel1_z[i]);

        var j: usize = 0;
        while (j < simd_end) : (j += vec_len) {
            // Load sel2 coords as f32 vectors.
            const sx: @Vector(vec_len, f32) = sel2_x[j..][0..vec_len].*;
            const sy: @Vector(vec_len, f32) = sel2_y[j..][0..vec_len].*;
            const sz: @Vector(vec_len, f32) = sel2_z[j..][0..vec_len].*;

            // Compute differences in f32.
            const dx_f32 = sx - ix_f32;
            const dy_f32 = sy - iy_f32;
            const dz_f32 = sz - iz_f32;

            // Widen to f64 for precision and compute distance.
            const dx: @Vector(vec_len, f64) = @floatCast(dx_f32);
            const dy: @Vector(vec_len, f64) = @floatCast(dy_f32);
            const dz: @Vector(vec_len, f64) = @floatCast(dz_f32);
            const r_vec: @Vector(vec_len, f64) = @sqrt(dx * dx + dy * dy + dz * dz);

            // Extract to scalar for histogram binning (bins may collide).
            const r_arr: [vec_len]f64 = r_vec;
            for (r_arr) |r| {
                if (r < r_min or r >= r_max) continue;
                const bin_f: f64 = (r - r_min) / bin_width;
                const bin: u32 = @intFromFloat(bin_f);
                if (bin < n_bins) {
                    local_hist[bin] += 1.0;
                }
            }
        }

        // Scalar tail for remaining sel2 atoms.
        while (j < n2) : (j += 1) {
            const dx: f64 = @as(f64, sel2_x[j]) - ix;
            const dy: f64 = @as(f64, sel2_y[j]) - iy;
            const dz: f64 = @as(f64, sel2_z[j]) - iz;
            const r: f64 = @sqrt(dx * dx + dy * dy + dz * dz);

            if (r < r_min or r >= r_max) continue;
            const bin_f: f64 = (r - r_min) / bin_width;
            const bin: u32 = @intFromFloat(bin_f);
            if (bin < n_bins) {
                local_hist[bin] += 1.0;
            }
        }
    }
}

/// Multi-threaded + SIMD version of `compute`.
///
/// Partitions sel1 atoms across threads, each with a thread-local histogram.
/// After joining, histograms are reduced (element-wise sum) and normalized
/// identically to the single-threaded version.
///
/// Falls back to single-threaded `compute` when `n_threads <= 1` or the
/// selection is too small to benefit from parallelism.
pub fn computeParallel(
    allocator: std.mem.Allocator,
    sel1_x: []const f32,
    sel1_y: []const f32,
    sel1_z: []const f32,
    sel2_x: []const f32,
    sel2_y: []const f32,
    sel2_z: []const f32,
    box_volume: f64,
    config: Config,
    n_threads: usize,
) !RdfResult {
    // Validate inputs (same checks as single-threaded compute).
    if (sel1_x.len != sel1_y.len or sel1_x.len != sel1_z.len) return ComputeError.MismatchedSelectionLengths;
    if (sel2_x.len != sel2_y.len or sel2_x.len != sel2_z.len) return ComputeError.MismatchedSelectionLengths;
    if (config.n_bins == 0) return ComputeError.ZeroBins;
    if (config.r_max <= config.r_min) return ComputeError.InvalidRange;
    if (box_volume <= 0.0) return ComputeError.InvalidBoxVolume;

    // Fallback to single-threaded for small workloads.
    if (n_threads <= 1 or sel1_x.len < 16) {
        return compute(allocator, sel1_x, sel1_y, sel1_z, sel2_x, sel2_y, sel2_z, box_volume, config);
    }

    const cpu_count = std.Thread.getCpuCount() catch {
        return compute(allocator, sel1_x, sel1_y, sel1_z, sel2_x, sel2_y, sel2_z, box_volume, config);
    };
    const actual_threads = @min(n_threads, cpu_count);
    const thread_count = @min(actual_threads, sel1_x.len);

    const n_bins = config.n_bins;
    const r_min: f64 = config.r_min;
    const r_max: f64 = config.r_max;
    const bin_width: f64 = (r_max - r_min) / @as(f64, @floatFromInt(n_bins));

    // Allocate thread-local histograms.
    const tl_hists = try allocator.alloc([]f64, thread_count);
    defer allocator.free(tl_hists);

    // Initialize to empty slices so errdefer cleanup is safe.
    for (0..thread_count) |t| {
        tl_hists[t] = &.{};
    }
    defer for (0..thread_count) |t| {
        if (tl_hists[t].len > 0) allocator.free(tl_hists[t]);
    };

    for (0..thread_count) |t| {
        tl_hists[t] = try allocator.alloc(f64, n_bins);
        @memset(tl_hists[t], 0.0);
    }

    // Spawn threads.
    const threads = try allocator.alloc(std.Thread, thread_count);
    defer allocator.free(threads);

    var spawned: usize = 0;
    errdefer for (threads[0..spawned]) |thread| {
        thread.join();
    };

    const n1 = sel1_x.len;
    const chunk = n1 / thread_count;
    const remainder = n1 % thread_count;

    for (0..thread_count) |t| {
        const start = t * chunk + @min(t, remainder);
        const end = start + chunk + @as(usize, if (t < remainder) 1 else 0);

        threads[t] = try std.Thread.spawn(.{}, rdfWorker, .{
            sel1_x,
            sel1_y,
            sel1_z,
            sel2_x,
            sel2_y,
            sel2_z,
            start,
            end,
            r_min,
            r_max,
            bin_width,
            n_bins,
            tl_hists[t],
        });
        spawned += 1;
    }

    // Join all threads.
    for (threads[0..spawned]) |thread| {
        thread.join();
    }
    spawned = 0;

    // Reduce: element-wise sum of all thread-local histograms into tl_hists[0].
    for (1..thread_count) |t| {
        for (0..n_bins) |k| {
            tl_hists[0][k] += tl_hists[t][k];
        }
    }
    const hist = tl_hists[0];

    // Build output arrays (same normalization as single-threaded).
    const r_out = try allocator.alloc(f64, n_bins);
    errdefer allocator.free(r_out);

    const g_r = try allocator.alloc(f64, n_bins);
    errdefer allocator.free(g_r);

    const rho: f64 = @as(f64, @floatFromInt(sel2_x.len)) / box_volume;
    const n_sel1: f64 = @floatFromInt(sel1_x.len);

    for (0..n_bins) |k| {
        const r_center: f64 = r_min + ((@as(f64, @floatFromInt(k)) + 0.5) * bin_width);
        r_out[k] = r_center;

        const shell_vol: f64 = (4.0 / 3.0) * std.math.pi *
            ((r_center + 0.5 * bin_width) * (r_center + 0.5 * bin_width) * (r_center + 0.5 * bin_width) -
                (r_center - 0.5 * bin_width) * (r_center - 0.5 * bin_width) * (r_center - 0.5 * bin_width));

        const expected: f64 = rho * shell_vol * n_sel1;

        if (expected > 0.0) {
            g_r[k] = hist[k] / expected;
        } else {
            g_r[k] = 0.0;
        }
    }

    return RdfResult{
        .r = r_out,
        .g_r = g_r,
        .allocator = allocator,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "rdf: bin centers are correct" {
    // r_min=0, r_max=10, n_bins=10 -> bin width 1.0 -> centers at 0.5, 1.5, ..., 9.5.
    const allocator = std.testing.allocator;

    // Two atoms at the same position so all pair counts are trivially 0
    // except at r=0. We only care about bin center positions here.
    const x = [_]f32{0.0};
    const y = [_]f32{0.0};
    const z = [_]f32{0.0};

    var result = try compute(
        allocator,
        &x,
        &y,
        &z,
        &x,
        &y,
        &z,
        1000.0,
        .{ .r_min = 0.0, .r_max = 10.0, .n_bins = 10 },
    );
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 10), result.r.len);
    try std.testing.expectApproxEqAbs(0.5, result.r[0], 1e-10);
    try std.testing.expectApproxEqAbs(1.5, result.r[1], 1e-10);
    try std.testing.expectApproxEqAbs(9.5, result.r[9], 1e-10);
}

test "rdf: two atoms at known distance produce peak in correct bin" {
    // sel1: one atom at origin. sel2: one atom at (3.5, 0, 0).
    // With r_min=0, r_max=10, n_bins=10 (bin_width=1), distance 3.5 falls in bin 3
    // (center at 3.5). g(r) should be non-zero exactly there.
    const allocator = std.testing.allocator;

    const sel1_x = [_]f32{0.0};
    const sel1_y = [_]f32{0.0};
    const sel1_z = [_]f32{0.0};

    const sel2_x = [_]f32{3.5};
    const sel2_y = [_]f32{0.0};
    const sel2_z = [_]f32{0.0};

    // Use a large box volume so rho is tiny and normalization doesn't obscure peak.
    var result = try compute(
        allocator,
        &sel1_x,
        &sel1_y,
        &sel1_z,
        &sel2_x,
        &sel2_y,
        &sel2_z,
        1e9,
        .{ .r_min = 0.0, .r_max = 10.0, .n_bins = 10 },
    );
    defer result.deinit();

    // Bin 3 (center 3.5) should be the only non-zero bin.
    for (result.g_r, 0..) |g, k| {
        if (k == 3) {
            try std.testing.expect(g > 0.0);
        } else {
            try std.testing.expectApproxEqAbs(0.0, g, 1e-30);
        }
    }
}

test "rdf: uniform distribution approaches g(r)=1 at large r" {
    // Place N atoms randomly but deterministically in a cubic box of side L.
    // For large enough N the average g(r) over the non-trivial bins should
    // be close to 1.0. We use a simple pseudo-random lattice here to avoid
    // importing a PRNG.
    const allocator = std.testing.allocator;

    const N = 500;
    const L: f32 = 50.0; // Å
    const box_vol: f64 = @as(f64, L) * L * L;

    var xs: [N]f32 = undefined;
    var ys: [N]f32 = undefined;
    var zs: [N]f32 = undefined;

    // Simple LCG to spread atoms around the box.
    var state: u64 = 12345;
    for (0..N) |k| {
        state = state *% 6364136223846793005 +% 1442695040888963407;
        xs[k] = @as(f32, @floatFromInt(state >> 32 & 0xFFFF)) / 65535.0 * L;
        state = state *% 6364136223846793005 +% 1442695040888963407;
        ys[k] = @as(f32, @floatFromInt(state >> 32 & 0xFFFF)) / 65535.0 * L;
        state = state *% 6364136223846793005 +% 1442695040888963407;
        zs[k] = @as(f32, @floatFromInt(state >> 32 & 0xFFFF)) / 65535.0 * L;
    }

    var result = try compute(
        allocator,
        &xs,
        &ys,
        &zs,
        &xs,
        &ys,
        &zs,
        box_vol,
        .{ .r_min = 0.0, .r_max = 20.0, .n_bins = 50 },
    );
    defer result.deinit();

    // Bins at r > 5 Å (far from the self-pair peak at r=0) should have g ≈ 1.
    // We allow a generous tolerance because N=500 has statistical noise.
    var sum: f64 = 0.0;
    var count: usize = 0;
    for (result.r, result.g_r) |r, g| {
        if (r > 5.0 and r < 18.0) {
            sum += g;
            count += 1;
        }
    }
    const mean = sum / @as(f64, @floatFromInt(count));
    // Mean g(r) in the mid-range should be within 20% of 1.0.
    try std.testing.expect(mean > 0.8 and mean < 1.2);
}

test "rdf: g_r length matches n_bins" {
    const allocator = std.testing.allocator;

    const x = [_]f32{0.0};
    const y = [_]f32{0.0};
    const z = [_]f32{0.0};

    var result = try compute(
        allocator,
        &x,
        &y,
        &z,
        &x,
        &y,
        &z,
        1000.0,
        .{ .r_min = 0.0, .r_max = 5.0, .n_bins = 25 },
    );
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 25), result.r.len);
    try std.testing.expectEqual(@as(usize, 25), result.g_r.len);
}

test "rdf: atoms outside [r_min, r_max) are excluded" {
    // sel2 atom at distance 15, but r_max=10. Should produce all-zero g_r.
    const allocator = std.testing.allocator;

    const sel1_x = [_]f32{0.0};
    const sel1_y = [_]f32{0.0};
    const sel1_z = [_]f32{0.0};

    const sel2_x = [_]f32{15.0};
    const sel2_y = [_]f32{0.0};
    const sel2_z = [_]f32{0.0};

    var result = try compute(
        allocator,
        &sel1_x,
        &sel1_y,
        &sel1_z,
        &sel2_x,
        &sel2_y,
        &sel2_z,
        1000.0,
        .{ .r_min = 0.0, .r_max = 10.0, .n_bins = 10 },
    );
    defer result.deinit();

    for (result.g_r) |g| {
        try std.testing.expectApproxEqAbs(0.0, g, 1e-30);
    }
}

test "rdf: computeParallel matches single-threaded" {
    // Two atoms at known distance — verify parallel gives same result as serial.
    const allocator = std.testing.allocator;

    // Need at least 16 sel1 atoms to avoid fallback. Place 20 atoms at origin
    // and one sel2 atom at (3.5, 0, 0).
    const N1 = 20;
    var s1x: [N1]f32 = undefined;
    var s1y: [N1]f32 = undefined;
    var s1z: [N1]f32 = undefined;
    for (0..N1) |k| {
        s1x[k] = 0.0;
        s1y[k] = 0.0;
        s1z[k] = 0.0;
    }

    const s2x = [_]f32{3.5};
    const s2y = [_]f32{0.0};
    const s2z = [_]f32{0.0};

    const cfg: Config = .{ .r_min = 0.0, .r_max = 10.0, .n_bins = 10 };
    const box_vol: f64 = 1e9;

    var st = try compute(allocator, &s1x, &s1y, &s1z, &s2x, &s2y, &s2z, box_vol, cfg);
    defer st.deinit();

    var mt = try computeParallel(allocator, &s1x, &s1y, &s1z, &s2x, &s2y, &s2z, box_vol, cfg, 4);
    defer mt.deinit();

    try std.testing.expectEqual(st.g_r.len, mt.g_r.len);
    for (st.g_r, mt.g_r) |s, m| {
        try std.testing.expectApproxEqAbs(s, m, 1e-10);
    }
    for (st.r, mt.r) |s, m| {
        try std.testing.expectApproxEqAbs(s, m, 1e-10);
    }
}

test "rdf: computeParallel uniform distribution approaches g(r)=1" {
    // N=500 atoms in a box — parallel computation should yield mean g(r) near 1.0.
    const allocator = std.testing.allocator;

    const N = 500;
    const L: f32 = 50.0;
    const box_vol: f64 = @as(f64, L) * L * L;

    var xs: [N]f32 = undefined;
    var ys: [N]f32 = undefined;
    var zs: [N]f32 = undefined;

    var state: u64 = 12345;
    for (0..N) |k| {
        state = state *% 6364136223846793005 +% 1442695040888963407;
        xs[k] = @as(f32, @floatFromInt(state >> 32 & 0xFFFF)) / 65535.0 * L;
        state = state *% 6364136223846793005 +% 1442695040888963407;
        ys[k] = @as(f32, @floatFromInt(state >> 32 & 0xFFFF)) / 65535.0 * L;
        state = state *% 6364136223846793005 +% 1442695040888963407;
        zs[k] = @as(f32, @floatFromInt(state >> 32 & 0xFFFF)) / 65535.0 * L;
    }

    var result = try computeParallel(
        allocator,
        &xs,
        &ys,
        &zs,
        &xs,
        &ys,
        &zs,
        box_vol,
        .{ .r_min = 0.0, .r_max = 20.0, .n_bins = 50 },
        4,
    );
    defer result.deinit();

    var sum: f64 = 0.0;
    var count: usize = 0;
    for (result.r, result.g_r) |r, g| {
        if (r > 5.0 and r < 18.0) {
            sum += g;
            count += 1;
        }
    }
    const mean = sum / @as(f64, @floatFromInt(count));
    try std.testing.expect(mean > 0.8 and mean < 1.2);
}
