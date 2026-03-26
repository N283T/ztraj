// H-bond energy calculation for native DSSP.
//
// Implements the Kabsch-Sander electrostatic approximation. Atom coordinates
// are read from the Frame on demand using atom indices stored in DsspResidue.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ztraj_types = @import("../../types.zig");
const types = @import("types.zig");
const backbone = @import("backbone.zig");

const Frame = ztraj_types.Frame;
const DsspResidue = types.DsspResidue;
const HBond = types.HBond;
const Vec3f32 = types.Vec3f32;

// ============================================================================
// Distance helper (f32 — matches C++ mkdssp behavior)
// ============================================================================

fn distanceF32(a: Vec3f32, b: Vec3f32) f32 {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    const dz = a.z - b.z;
    return @sqrt(dx * dx + dy * dy + dz * dz);
}

// ============================================================================
// H-bond energy (Kabsch-Sander)
// ============================================================================

/// Calculate the Kabsch-Sander H-bond energy between a donor and an acceptor.
///
/// Formula: E = C/d_HO - C/d_HC + C/d_NC - C/d_NO
///   C = -27.888 kcal/mol·Å
///
/// Mixed precision to match C++ mkdssp:
///   - distances computed as f32
///   - energy accumulation in f64
///   - result rounded to 3 decimals, clamped, stored as f32
///
/// Also updates the best-2 tracking arrays in donor and acceptor.
pub fn calculateHBondEnergy(
    residues: []DsspResidue,
    frame: Frame,
    donor_idx: u32,
    acceptor_idx: u32,
) f32 {
    const donor = &residues[donor_idx];
    const acceptor = &residues[acceptor_idx];

    if (donor.isProline()) return 0.0;
    if (donor_idx == acceptor_idx) return 0.0;

    // Donor positions
    const h = donor.getH();
    const n = backbone.getPos(frame, donor.n_idx);

    // Acceptor positions
    const o = backbone.getPos(frame, acceptor.o_idx);
    const c = backbone.getPos(frame, acceptor.c_idx);

    const d_ho: f64 = @floatCast(distanceF32(h, o));
    const d_hc: f64 = @floatCast(distanceF32(h, c));
    const d_nc: f64 = @floatCast(distanceF32(n, c));
    const d_no: f64 = @floatCast(distanceF32(n, o));

    var energy: f64 = 0.0;
    const kC: f32 = types.kCouplingConstantF32;

    if (d_ho < types.kMinimalDistance or d_hc < types.kMinimalDistance or
        d_nc < types.kMinimalDistance or d_no < types.kMinimalDistance)
    {
        energy = types.kMinHBondEnergyF32;
    } else {
        energy = @as(f64, kC) / d_ho - @as(f64, kC) / d_hc +
            @as(f64, kC) / d_nc - @as(f64, kC) / d_no;
    }

    energy = @round(energy * 1000.0) / 1000.0;
    if (energy < types.kMinHBondEnergyF32) {
        energy = types.kMinHBondEnergyF32;
    }

    const energy_f32: f32 = @floatCast(energy);

    updateHBondPair(&residues[donor_idx].hbond_acceptor, acceptor_idx, energy_f32);
    updateHBondPair(&residues[acceptor_idx].hbond_donor, donor_idx, energy_f32);

    return energy_f32;
}

fn updateHBondPair(pair: *[2]HBond, partner_idx: u32, energy: f32) void {
    if (energy < pair[0].energy) {
        pair[1] = pair[0];
        pair[0] = .{ .residue_index = partner_idx, .energy = energy };
    } else if (energy < pair[1].energy) {
        pair[1] = .{ .residue_index = partner_idx, .energy = energy };
    }
}

// ============================================================================
// Test if an H-bond exists
// ============================================================================

/// Returns true if there is an H-bond from donor_idx to acceptor_idx.
///
/// Checks the donor's two best acceptors; the bond exists when the energy is
/// below the threshold (-0.5 kcal/mol).
pub fn testBond(residues: []const DsspResidue, donor_idx: u32, acceptor_idx: u32) bool {
    const donor = &residues[donor_idx];
    for (donor.hbond_acceptor) |hb| {
        if (hb.residue_index) |idx| {
            if (idx == acceptor_idx and hb.energy < types.kMaxHBondEnergyF32) {
                return true;
            }
        }
    }
    return false;
}

// ============================================================================
// Near-pair finding (CA–CA < 9 Å)
// ============================================================================

/// Find all pairs (i, j) of complete residues whose CA atoms are within
/// kMinimalCADistance. Pairs where |i - j| <= 1 are excluded.
///
/// Caller owns the returned slice and must free it with the same allocator.
pub fn findNearPairs(
    residues: []const DsspResidue,
    frame: Frame,
    allocator: Allocator,
) ![][2]u32 {
    var pairs: std.ArrayListAligned([2]u32, null) = .empty;
    errdefer pairs.deinit(allocator);

    const n: u32 = @intCast(residues.len);
    const max_dist_sq: f32 = types.kMinimalCADistance * types.kMinimalCADistance;

    for (0..n) |i_usize| {
        const i: u32 = @intCast(i_usize);
        if (!residues[i].complete) continue;
        if (i + 2 >= n) continue;
        const ca_i = backbone.getPos(frame, residues[i].ca_idx);

        for (i + 2..n) |j_usize| {
            const j: u32 = @intCast(j_usize);
            if (!residues[j].complete) continue;
            const ca_j = backbone.getPos(frame, residues[j].ca_idx);

            const dist_sq = ca_i.distanceSq(ca_j);
            if (dist_sq < max_dist_sq) {
                try pairs.append(allocator, .{ i, j });
            }
        }
    }

    return pairs.toOwnedSlice(allocator);
}

/// Find near pairs using cell-list spatial indexing (O(n) expected).
///
/// Same semantics as `findNearPairs` but avoids the O(n²) all-pairs scan by
/// partitioning CA atoms into a uniform grid with cell size equal to the
/// distance cutoff. Only atoms in neighboring cells (3×3×3 neighbourhood)
/// are tested, giving ~O(n) runtime for uniformly distributed coordinates.
pub fn findNearPairsFast(
    residues: []const DsspResidue,
    frame: Frame,
    allocator: Allocator,
) ![][2]u32 {
    const max_dist: f32 = types.kMinimalCADistance;
    const max_dist_sq: f32 = max_dist * max_dist;

    // -- 1. Extract CA positions for complete residues -----------------------
    const n_res: u32 = @intCast(residues.len);

    // Upper bound: all residues could be complete
    var ca_x = try allocator.alloc(f32, n_res);
    defer allocator.free(ca_x);
    var ca_y = try allocator.alloc(f32, n_res);
    defer allocator.free(ca_y);
    var ca_z = try allocator.alloc(f32, n_res);
    defer allocator.free(ca_z);
    var idx_map = try allocator.alloc(u32, n_res); // pos-idx -> residue-idx
    defer allocator.free(idx_map);

    var n_ca: u32 = 0;
    for (0..n_res) |r| {
        if (!residues[r].complete) continue;
        const pos = backbone.getPos(frame, residues[r].ca_idx);
        ca_x[n_ca] = pos.x;
        ca_y[n_ca] = pos.y;
        ca_z[n_ca] = pos.z;
        idx_map[n_ca] = @intCast(r);
        n_ca += 1;
    }

    if (n_ca == 0) {
        const empty: [][2]u32 = &.{};
        return allocator.dupe([2]u32, empty);
    }

    // -- 2. Bounding box & grid dimensions -----------------------------------
    var min_x: f32 = ca_x[0];
    var min_y: f32 = ca_y[0];
    var min_z: f32 = ca_z[0];
    var max_x: f32 = ca_x[0];
    var max_y: f32 = ca_y[0];
    var max_z: f32 = ca_z[0];
    for (1..n_ca) |i| {
        min_x = @min(min_x, ca_x[i]);
        min_y = @min(min_y, ca_y[i]);
        min_z = @min(min_z, ca_z[i]);
        max_x = @max(max_x, ca_x[i]);
        max_y = @max(max_y, ca_y[i]);
        max_z = @max(max_z, ca_z[i]);
    }

    const inv_cell: f32 = 1.0 / max_dist;
    const nx: u32 = @intCast(@as(i64, @intFromFloat(@floor((max_x - min_x) * inv_cell))) + 1);
    const ny: u32 = @intCast(@as(i64, @intFromFloat(@floor((max_y - min_y) * inv_cell))) + 1);
    const nz: u32 = @intCast(@as(i64, @intFromFloat(@floor((max_z - min_z) * inv_cell))) + 1);
    const n_cells: u32 = nx * ny * nz;

    // -- 3. Counting sort into cells -----------------------------------------
    // cell_counts[c] = number of atoms in cell c
    var cell_counts = try allocator.alloc(u32, n_cells + 1);
    defer allocator.free(cell_counts);
    @memset(cell_counts, 0);

    // Temporary array holding the cell index for each CA
    var atom_cell = try allocator.alloc(u32, n_ca);
    defer allocator.free(atom_cell);

    for (0..n_ca) |i| {
        const ix: u32 = @intFromFloat(@floor((ca_x[i] - min_x) * inv_cell));
        const iy: u32 = @intFromFloat(@floor((ca_y[i] - min_y) * inv_cell));
        const iz: u32 = @intFromFloat(@floor((ca_z[i] - min_z) * inv_cell));
        const cix = @min(ix, nx - 1);
        const ciy = @min(iy, ny - 1);
        const ciz = @min(iz, nz - 1);
        const cell_idx = ciz * nx * ny + ciy * nx + cix;
        atom_cell[i] = cell_idx;
        cell_counts[cell_idx] += 1;
    }

    // Prefix sum -> offsets
    var cell_offsets = try allocator.alloc(u32, n_cells + 1);
    defer allocator.free(cell_offsets);
    cell_offsets[0] = 0;
    for (1..n_cells + 1) |c| {
        cell_offsets[c] = cell_offsets[c - 1] + cell_counts[c - 1];
    }

    // Scatter atoms into sorted order
    var sorted_atoms = try allocator.alloc(u32, n_ca);
    defer allocator.free(sorted_atoms);
    // Reuse cell_counts as write cursors (reset to offsets)
    @memcpy(cell_counts[0..n_cells], cell_offsets[0..n_cells]);

    for (0..n_ca) |i| {
        const c = atom_cell[i];
        sorted_atoms[cell_counts[c]] = @intCast(i);
        cell_counts[c] += 1;
    }

    // -- 4. Search neighboring cells -----------------------------------------
    var pairs: std.ArrayListAligned([2]u32, null) = .empty;
    errdefer pairs.deinit(allocator);

    const offsets = [_]i64{ -1, 0, 1 };

    for (0..nz) |cz_usize| {
        for (0..ny) |cy_usize| {
            for (0..nx) |cx_usize| {
                const cell_a: u32 = @intCast(cz_usize * nx * ny + cy_usize * nx + cx_usize);
                const a_start = cell_offsets[cell_a];
                const a_end = cell_offsets[cell_a + 1];
                if (a_start == a_end) continue;

                for (offsets) |dz| {
                    const nz_i: i64 = @as(i64, @intCast(cz_usize)) + dz;
                    if (nz_i < 0 or nz_i >= @as(i64, @intCast(nz))) continue;

                    for (offsets) |dy| {
                        const ny_i: i64 = @as(i64, @intCast(cy_usize)) + dy;
                        if (ny_i < 0 or ny_i >= @as(i64, @intCast(ny))) continue;

                        for (offsets) |dx| {
                            const nx_i: i64 = @as(i64, @intCast(cx_usize)) + dx;
                            if (nx_i < 0 or nx_i >= @as(i64, @intCast(nx))) continue;

                            const cell_b: u32 = @intCast(@as(i64, @intCast(nz_i)) * @as(i64, @intCast(nx * ny)) +
                                @as(i64, @intCast(ny_i)) * @as(i64, @intCast(nx)) +
                                nx_i);

                            // Half-pair: only process cell_b >= cell_a
                            if (cell_b < cell_a) continue;

                            const b_start = cell_offsets[cell_b];
                            const b_end = cell_offsets[cell_b + 1];
                            if (b_start == b_end) continue;

                            if (cell_a == cell_b) {
                                // Same cell: only pairs where a_atom < b_atom
                                for (a_start..a_end) |ai_usize| {
                                    const ai = sorted_atoms[ai_usize];
                                    for (ai_usize + 1..a_end) |bi_usize| {
                                        const bi = sorted_atoms[bi_usize];
                                        const ri = idx_map[ai];
                                        const rj = idx_map[bi];
                                        const r_lo = @min(ri, rj);
                                        const r_hi = @max(ri, rj);
                                        if (r_hi - r_lo <= 1) continue;
                                        const dx2 = ca_x[ai] - ca_x[bi];
                                        const dy2 = ca_y[ai] - ca_y[bi];
                                        const dz2 = ca_z[ai] - ca_z[bi];
                                        const dist_sq = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
                                        if (dist_sq < max_dist_sq) {
                                            try pairs.append(allocator, .{ r_lo, r_hi });
                                        }
                                    }
                                }
                            } else {
                                // Different cells
                                for (a_start..a_end) |ai_usize| {
                                    const ai = sorted_atoms[ai_usize];
                                    for (b_start..b_end) |bi_usize| {
                                        const bi = sorted_atoms[bi_usize];
                                        const ri = idx_map[ai];
                                        const rj = idx_map[bi];
                                        const r_lo = @min(ri, rj);
                                        const r_hi = @max(ri, rj);
                                        if (r_hi - r_lo <= 1) continue;
                                        const dx2 = ca_x[ai] - ca_x[bi];
                                        const dy2 = ca_y[ai] - ca_y[bi];
                                        const dz2 = ca_z[ai] - ca_z[bi];
                                        const dist_sq = dx2 * dx2 + dy2 * dy2 + dz2 * dz2;
                                        if (dist_sq < max_dist_sq) {
                                            try pairs.append(allocator, .{ r_lo, r_hi });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // -- 5. Sort pairs by (i, j) to match findNearPairs ordering -------------
    const items = pairs.items;
    std.mem.sortUnstable([2]u32, items, {}, struct {
        fn lessThan(_: void, a: [2]u32, b: [2]u32) bool {
            if (a[0] != b[0]) return a[0] < b[0];
            return a[1] < b[1];
        }
    }.lessThan);

    return pairs.toOwnedSlice(allocator);
}

// ============================================================================
// Calculate all H-bond energies for near pairs
// ============================================================================

/// Calculate H-bond energies for all near pairs.
///
/// For each pair (i, j): test i as donor and j as donor. The reverse
/// direction j->i is skipped when j == i+1 (adjacent peptide bond).
pub fn calculateHBondEnergies(
    residues: []DsspResidue,
    frame: Frame,
    pairs: [][2]u32,
) void {
    for (pairs) |pair| {
        const i = pair[0];
        const j = pair[1];
        _ = calculateHBondEnergy(residues, frame, i, j);
        if (j != i + 1) {
            _ = calculateHBondEnergy(residues, frame, j, i);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "calculateHBondEnergy - proline returns zero" {
    const allocator = std.testing.allocator;

    var frame = try Frame.init(allocator, 8);
    defer frame.deinit();

    var residues = [_]DsspResidue{
        DsspResidue{ .c_idx = 0, .o_idx = 1, .n_idx = 2, .ca_idx = 3, .complete = true },
        DsspResidue{
            .n_idx = 4,
            .h_x = 3.0,
            .h_y = 0.0,
            .h_z = 0.0,
            .c_idx = 5,
            .o_idx = 6,
            .ca_idx = 7,
            .residue_type = .pro,
            .complete = true,
        },
    };
    // acceptor O at index 1
    frame.x[1] = 0.0;
    frame.y[1] = 1.2;
    frame.z[1] = 0.0;

    const energy = calculateHBondEnergy(&residues, frame, 1, 0);
    try std.testing.expectEqual(@as(f32, 0.0), energy);
}

test "calculateHBondEnergy - same residue returns zero" {
    const allocator = std.testing.allocator;
    var frame = try Frame.init(allocator, 8);
    defer frame.deinit();

    var residues = [_]DsspResidue{
        DsspResidue{ .complete = true },
    };
    const energy = calculateHBondEnergy(&residues, frame, 0, 0);
    try std.testing.expectEqual(@as(f32, 0.0), energy);
}

test "calculateHBondEnergy - normal energy is finite" {
    const allocator = std.testing.allocator;
    var frame = try Frame.init(allocator, 8);
    defer frame.deinit();

    // Acceptor: C at 0, O at 1; Donor: N at 4, H at stored pos
    frame.x[0] = 0.0;
    frame.y[0] = 0.0;
    frame.z[0] = 0.0;
    frame.x[1] = 0.0;
    frame.y[1] = 1.2;
    frame.z[1] = 0.0;
    frame.x[4] = 3.0;
    frame.y[4] = 0.0;
    frame.z[4] = 0.0;

    var residues = [_]DsspResidue{
        DsspResidue{ .c_idx = 0, .o_idx = 1, .n_idx = 2, .ca_idx = 3, .complete = true },
        DsspResidue{
            .n_idx = 4,
            .h_x = 2.0,
            .h_y = 0.5,
            .h_z = 0.0,
            .c_idx = 5,
            .o_idx = 6,
            .ca_idx = 7,
            .residue_type = .ala,
            .complete = true,
        },
    };

    const energy = calculateHBondEnergy(&residues, frame, 1, 0);
    try std.testing.expect(!std.math.isNan(energy));
}

test "testBond - no bond by default" {
    const residues = [_]DsspResidue{ DsspResidue{}, DsspResidue{} };
    try std.testing.expect(!testBond(&residues, 0, 1));
}

test "testBond - bond present" {
    var residues = [_]DsspResidue{ DsspResidue{}, DsspResidue{} };
    residues[0].hbond_acceptor[0] = .{ .residue_index = 1, .energy = -2.0 };
    try std.testing.expect(testBond(&residues, 0, 1));
}

test "testBond - bond present but energy too weak" {
    var residues = [_]DsspResidue{ DsspResidue{}, DsspResidue{} };
    residues[0].hbond_acceptor[0] = .{ .residue_index = 1, .energy = -0.3 }; // > -0.5
    try std.testing.expect(!testBond(&residues, 0, 1));
}

test "findNearPairs - close residues included" {
    const allocator = std.testing.allocator;
    var frame = try Frame.init(allocator, 8);
    defer frame.deinit();

    // CA indices: res0=1, res1=3, res2=5
    // Place CA of res0 at origin, res2 at 5 Å
    frame.x[1] = 0.0;
    frame.x[5] = 5.0; // 5 < 9

    var residues = [_]DsspResidue{
        DsspResidue{ .ca_idx = 1, .complete = true },
        DsspResidue{ .ca_idx = 3, .complete = true },
        DsspResidue{ .ca_idx = 5, .complete = true },
    };

    const pairs = try findNearPairs(&residues, frame, allocator);
    defer allocator.free(pairs);

    // Only (0,2) qualifies: |0-2|=2>1, dist=5<9. (0,1) and (1,2) are adjacent.
    try std.testing.expectEqual(@as(usize, 1), pairs.len);
    try std.testing.expectEqual(@as(u32, 0), pairs[0][0]);
    try std.testing.expectEqual(@as(u32, 2), pairs[0][1]);
}

test "findNearPairs - far residues excluded" {
    const allocator = std.testing.allocator;
    var frame = try Frame.init(allocator, 6);
    defer frame.deinit();

    // CA indices: res0=1, res2=3 — place them 20 Å apart
    frame.x[1] = 0.0;
    frame.x[3] = 20.0;

    var residues = [_]DsspResidue{
        DsspResidue{ .ca_idx = 1, .complete = true },
        DsspResidue{ .ca_idx = 2, .complete = true },
        DsspResidue{ .ca_idx = 3, .complete = true },
    };

    const pairs = try findNearPairs(&residues, frame, allocator);
    defer allocator.free(pairs);

    // (0,2) is 20 Å > 9 Å, so no pairs
    try std.testing.expectEqual(@as(usize, 0), pairs.len);
}

test "findNearPairsFast - matches findNearPairs on 4 residues" {
    const allocator = std.testing.allocator;

    // 4 residues with CA at indices 0..3
    // Positions: res0=(0,0,0), res1=(1,0,0), res2=(5,0,0), res3=(20,0,0)
    // Expected pairs (dist < 9, |i-j| > 1):
    //   (0,2) dist=5 ✓   (0,3) dist=20 ✗
    //   (1,3) dist=19 ✗
    var frame = try Frame.init(allocator, 4);
    defer frame.deinit();
    frame.x[0] = 0.0;
    frame.x[1] = 1.0;
    frame.x[2] = 5.0;
    frame.x[3] = 20.0;

    var residues = [_]DsspResidue{
        DsspResidue{ .ca_idx = 0, .complete = true },
        DsspResidue{ .ca_idx = 1, .complete = true },
        DsspResidue{ .ca_idx = 2, .complete = true },
        DsspResidue{ .ca_idx = 3, .complete = true },
    };

    const naive = try findNearPairs(&residues, frame, allocator);
    defer allocator.free(naive);
    const fast = try findNearPairsFast(&residues, frame, allocator);
    defer allocator.free(fast);

    // Both should return exactly (0,2) and (1,3) — wait, let's check:
    //   (0,2) dist=5 < 9, |0-2|=2 > 1 ✓
    //   (0,3) dist=20 >= 9 ✗
    //   (1,3) dist=19 >= 9 ✗
    // So only (0,2).
    try std.testing.expectEqual(naive.len, fast.len);
    for (naive, fast) |n, f| {
        try std.testing.expectEqual(n[0], f[0]);
        try std.testing.expectEqual(n[1], f[1]);
    }

    // Also verify correct result
    try std.testing.expectEqual(@as(usize, 1), fast.len);
    try std.testing.expectEqual(@as(u32, 0), fast[0][0]);
    try std.testing.expectEqual(@as(u32, 2), fast[0][1]);
}

test "findNearPairsFast - incomplete residues skipped" {
    const allocator = std.testing.allocator;

    var frame = try Frame.init(allocator, 4);
    defer frame.deinit();
    frame.x[0] = 0.0;
    frame.x[1] = 1.0;
    frame.x[2] = 5.0;
    frame.x[3] = 6.0;

    var residues = [_]DsspResidue{
        DsspResidue{ .ca_idx = 0, .complete = true },
        DsspResidue{ .ca_idx = 1, .complete = false }, // incomplete
        DsspResidue{ .ca_idx = 2, .complete = true },
        DsspResidue{ .ca_idx = 3, .complete = true },
    };

    const naive = try findNearPairs(&residues, frame, allocator);
    defer allocator.free(naive);
    const fast = try findNearPairsFast(&residues, frame, allocator);
    defer allocator.free(fast);

    try std.testing.expectEqual(naive.len, fast.len);
    for (naive, fast) |n, f| {
        try std.testing.expectEqual(n[0], f[0]);
        try std.testing.expectEqual(n[1], f[1]);
    }
}

test "findNearPairsFast - multiple pairs in 3D" {
    const allocator = std.testing.allocator;

    // 5 residues in 3D space
    var frame = try Frame.init(allocator, 5);
    defer frame.deinit();
    // res0=(0,0,0) res1=(3,0,0) res2=(0,4,0) res3=(0,0,7) res4=(0,0,15)
    frame.x[0] = 0.0;
    frame.y[0] = 0.0;
    frame.z[0] = 0.0;
    frame.x[1] = 3.0;
    frame.y[1] = 0.0;
    frame.z[1] = 0.0;
    frame.x[2] = 0.0;
    frame.y[2] = 4.0;
    frame.z[2] = 0.0;
    frame.x[3] = 0.0;
    frame.y[3] = 0.0;
    frame.z[3] = 7.0;
    frame.x[4] = 0.0;
    frame.y[4] = 0.0;
    frame.z[4] = 15.0;

    var residues = [_]DsspResidue{
        DsspResidue{ .ca_idx = 0, .complete = true },
        DsspResidue{ .ca_idx = 1, .complete = true },
        DsspResidue{ .ca_idx = 2, .complete = true },
        DsspResidue{ .ca_idx = 3, .complete = true },
        DsspResidue{ .ca_idx = 4, .complete = true },
    };

    const naive = try findNearPairs(&residues, frame, allocator);
    defer allocator.free(naive);
    const fast = try findNearPairsFast(&residues, frame, allocator);
    defer allocator.free(fast);

    try std.testing.expectEqual(naive.len, fast.len);
    for (naive, fast) |n, f| {
        try std.testing.expectEqual(n[0], f[0]);
        try std.testing.expectEqual(n[1], f[1]);
    }
}
