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
        &x, &y, &z,
        &x, &y, &z,
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
        &sel1_x, &sel1_y, &sel1_z,
        &sel2_x, &sel2_y, &sel2_z,
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
        &xs, &ys, &zs,
        &xs, &ys, &zs,
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
        &x, &y, &z,
        &x, &y, &z,
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
        &sel1_x, &sel1_y, &sel1_z,
        &sel2_x, &sel2_y, &sel2_z,
        1000.0,
        .{ .r_min = 0.0, .r_max = 10.0, .n_bins = 10 },
    );
    defer result.deinit();

    for (result.g_r) |g| {
        try std.testing.expectApproxEqAbs(0.0, g, 1e-30);
    }
}
