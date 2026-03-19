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
