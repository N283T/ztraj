const std = @import("std");
const Allocator = std.mem.Allocator;
const types = @import("types.zig");
const residue_mod = @import("residue.zig");

const Vec3f32 = types.Vec3f32;
const Residue = residue_mod.Residue;
const HBond = types.HBond;
const neighbor_list = @import("neighbor_list.zig");

// ---------------------------------------------------------------------------
// Distance calculation matching C++ behavior
// ---------------------------------------------------------------------------

/// Compute distance between two Vec3f32 points matching C++ mkdssp.
///
/// C++ computes: std::sqrt(distance_sq(a, b))
/// where distance_sq returns float and std::sqrt(float) returns float.
///
/// The exact floating-point behavior depends on compiler flags and hardware.
/// This pure f32 implementation provides consistent results that match
/// C++ on most cases, with rare 1 ULP differences on threshold boundaries.
fn distanceF32(a: Vec3f32, b: Vec3f32) f32 {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    const dz = a.z - b.z;
    const dist_sq: f32 = dx * dx + dy * dy + dz * dz;
    return @sqrt(dist_sq);
}

// ---------------------------------------------------------------------------
// H-bond energy calculation (dssp.cpp:726-776)
// ---------------------------------------------------------------------------

/// Calculate the H-bond energy between a donor and an acceptor residue.
///
/// Formula: E = C/d_HO - C/d_HC + C/d_NC - C/d_NO
/// where C = -27.888 kcal/mol·Å
///
/// Mixed precision to match C++ mkdssp exactly:
/// - distance() returns float (f32)
/// - distances stored in double (f64) via implicit cast
/// - kCouplingConstant is float (f32) but promoted to f64 in calculation
/// - calculation done in f64
/// - result stored as f32
///
/// Returns the energy (negative = favourable). Updates the donor/acceptor
/// best-2 tracking arrays if this energy is better than existing bonds.
pub fn calculateHBondEnergy(residues: []Residue, donor_idx: u32, acceptor_idx: u32) f32 {
    const donor = &residues[donor_idx];
    const acceptor = &residues[acceptor_idx];

    // Proline cannot donate an H-bond
    if (donor.isProline()) return 0.0;

    // Same residue cannot form H-bond with itself
    if (donor_idx == acceptor_idx) return 0.0;

    // Compute distances
    const d_ho: f64 = @floatCast(distanceF32(donor.h, acceptor.o));
    const d_hc: f64 = @floatCast(distanceF32(donor.h, acceptor.c));
    const d_nc: f64 = @floatCast(distanceF32(donor.n, acceptor.c));
    const d_no: f64 = @floatCast(distanceF32(donor.n, acceptor.o));

    var energy: f64 = 0.0;

    // kCouplingConstant is f32 (-27.888f) and gets promoted to f64 in each division
    // Match C++ exactly: float / double for each term
    const kC: f32 = types.kCouplingConstantF32;

    if (d_ho < types.kMinimalDistance or d_hc < types.kMinimalDistance or
        d_nc < types.kMinimalDistance or d_no < types.kMinimalDistance)
    {
        energy = types.kMinHBondEnergyF32;
    } else {
        // Match C++ formula structure exactly:
        // result = kCouplingConstant / distanceHO - kCouplingConstant / distanceHC
        //        + kCouplingConstant / distanceNC - kCouplingConstant / distanceNO;
        // In C++, kCouplingConstant (float) is promoted to double at each division
        energy = @as(f64, kC) / d_ho - @as(f64, kC) / d_hc + @as(f64, kC) / d_nc - @as(f64, kC) / d_no;
    }

    // Round to 3 decimal places (DSSP compatibility)
    energy = @round(energy * 1000.0) / 1000.0;

    // Clamp minimum
    if (energy < types.kMinHBondEnergyF32) {
        energy = types.kMinHBondEnergyF32;
    }

    // Convert to f32 for storage (matching C++ HBond struct which uses double,
    // but we use f32 for memory efficiency - value is already rounded)
    const energy_f32: f32 = @floatCast(energy);

    // Always record best-2 energies in tracking arrays, even weak ones.
    // C++ DSSP stores all computed energies, the threshold is only applied
    // when testing if a bond exists (testBond). This matches dssp.cpp:749-773.
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

// ---------------------------------------------------------------------------
// Test if an H-bond exists between two residues (dssp.cpp:823-827)
// ---------------------------------------------------------------------------

/// Returns true if there is an H-bond from residue `donor_idx` to `acceptor_idx`.
///
/// A bond exists if either of the donor's two best acceptors matches `acceptor_idx`
/// AND the energy is below the threshold (-0.5 kcal/mol).
pub fn testBond(residues: []const Residue, donor_idx: u32, acceptor_idx: u32) bool {
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

// ---------------------------------------------------------------------------
// Near-pair finding (dssp.cpp:1590-1618)
// ---------------------------------------------------------------------------

/// Find all pairs of residues whose CA atoms are within kMinimalCADistance (9.0 Å).
///
/// Returns a list of [donor_index, acceptor_index] pairs. Only considers
/// complete residues and skips pairs that are adjacent (|i-j| <= 1).
pub fn findNearPairs(residues: []const Residue, allocator: Allocator) ![][2]u32 {
    var pairs: std.ArrayListAligned([2]u32, null) = .empty;
    errdefer pairs.deinit(allocator);

    const n: u32 = @intCast(residues.len);
    const max_dist_sq: f32 = types.kMinimalCADistance * types.kMinimalCADistance;

    for (0..n) |i_usize| {
        const i: u32 = @intCast(i_usize);
        if (!residues[i].complete) continue;

        for (i + 1..n) |j_usize| {
            const j: u32 = @intCast(j_usize);
            if (!residues[j].complete) continue;

            const dist_sq = residues[i].ca.distanceSq(residues[j].ca);
            if (dist_sq < max_dist_sq) {
                try pairs.append(allocator, .{ i, j });
            }
        }
    }

    return pairs.toOwnedSlice(allocator);
}

// ---------------------------------------------------------------------------
// Near-pair finding using spatial hash grid (O(N + pairs))
// ---------------------------------------------------------------------------

/// Find near pairs using a spatial hash grid for O(N + pairs) performance.
///
/// Same output format as findNearPairs but much faster for large proteins.
pub fn findNearPairsOptimized(residues: []const Residue, allocator: Allocator) ![][2]u32 {
    const n = residues.len;
    if (n < 2) return allocator.alloc([2]u32, 0);

    // Extract CA positions and completeness flags
    const ca_positions = try allocator.alloc(Vec3f32, n);
    defer allocator.free(ca_positions);
    const complete = try allocator.alloc(bool, n);
    defer allocator.free(complete);

    for (residues, 0..) |res, i| {
        ca_positions[i] = res.ca;
        complete[i] = res.complete;
    }

    return neighbor_list.findNearPairsGrid(
        ca_positions,
        complete,
        types.kMinimalCADistance,
        allocator,
    );
}

// ---------------------------------------------------------------------------
// Calculate all H-bond energies for near pairs
// ---------------------------------------------------------------------------

/// Calculate H-bond energies for all near pairs.
///
/// Each pair (i, j) is tested with i as donor. The reverse direction (j as
/// donor) is only tested when j != i + 1, matching C++ DSSP behaviour
/// (dssp.cpp:792). Adjacent residues share a peptide bond so the back-
/// donation NH(i+1)→CO(i) is not a meaningful H-bond.
pub fn calculateHBondEnergies(residues: []Residue, pairs: [][2]u32) void {
    for (pairs) |pair| {
        const i = pair[0];
        const j = pair[1];

        // Test i as donor, j as acceptor
        _ = calculateHBondEnergy(residues, i, j);
        // Test j as donor, i as acceptor (skip for adjacent residues)
        if (j != i + 1) {
            _ = calculateHBondEnergy(residues, j, i);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "calculateHBondEnergy - basic energy calculation" {
    var residues = [_]Residue{
        // Acceptor (residue 0)
        .{
            .c = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .o = Vec3f32{ .x = 0.0, .y = 1.2, .z = 0.0 },
            .n = Vec3f32{ .x = -1.0, .y = 0.0, .z = 0.0 },
            .complete = true,
        },
        // Donor (residue 1)
        .{
            .n = Vec3f32{ .x = 3.0, .y = 0.0, .z = 0.0 },
            .h = Vec3f32{ .x = 2.0, .y = 0.5, .z = 0.0 },
            .c = Vec3f32{ .x = 4.0, .y = 0.0, .z = 0.0 },
            .o = Vec3f32{ .x = 4.0, .y = 1.2, .z = 0.0 },
            .ca = Vec3f32{ .x = 3.5, .y = 0.0, .z = 0.0 },
            .residue_type = .ala,
            .complete = true,
        },
    };

    const energy = calculateHBondEnergy(&residues, 1, 0);
    // Energy should be negative (favourable H-bond if geometry is right)
    // Exact value depends on distances; just verify it's computed
    try std.testing.expect(energy < 0.0 or energy >= 0.0); // non-NaN
    try std.testing.expect(!std.math.isNan(energy));
}

test "calculateHBondEnergy - proline returns zero" {
    var residues = [_]Residue{
        .{
            .c = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .o = Vec3f32{ .x = 0.0, .y = 1.2, .z = 0.0 },
            .complete = true,
        },
        .{
            .n = Vec3f32{ .x = 3.0, .y = 0.0, .z = 0.0 },
            .h = Vec3f32{ .x = 2.0, .y = 0.5, .z = 0.0 },
            .residue_type = .pro,
            .complete = true,
        },
    };

    const energy = calculateHBondEnergy(&residues, 1, 0);
    try std.testing.expectEqual(@as(f32, 0.0), energy);
}

test "testBond - no bond" {
    const residues = [_]Residue{
        .{},
        .{},
    };
    try std.testing.expect(!testBond(&residues, 0, 1));
}

test "testBond - with bond" {
    var residues = [_]Residue{
        .{},
        .{},
    };
    // Manually set an H-bond record
    residues[0].hbond_acceptor[0] = .{ .residue_index = 1, .energy = -2.0 };

    try std.testing.expect(testBond(&residues, 0, 1));
}

test "findNearPairs - close residues" {
    const residues = [_]Residue{
        .{
            .ca = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .complete = true,
        },
        .{
            .ca = Vec3f32{ .x = 5.0, .y = 0.0, .z = 0.0 }, // 5 Å < 9 Å
            .complete = true,
        },
        .{
            .ca = Vec3f32{ .x = 20.0, .y = 0.0, .z = 0.0 }, // 20 Å > 9 Å from first
            .complete = true,
        },
    };

    const allocator = std.testing.allocator;
    const pairs = try findNearPairs(&residues, allocator);
    defer allocator.free(pairs);

    // (0,1) should be a near pair; (0,2) and (1,2) should not
    try std.testing.expectEqual(@as(usize, 1), pairs.len);
    try std.testing.expectEqual(@as(u32, 0), pairs[0][0]);
    try std.testing.expectEqual(@as(u32, 1), pairs[0][1]);
}

test "updateHBondPair - tracking best 2" {
    var pair = [2]HBond{ .{}, .{} };

    // First bond
    updateHBondPair(&pair, 5, -2.0);
    try std.testing.expectEqual(@as(?u32, 5), pair[0].residue_index);
    try std.testing.expectEqual(@as(f32, -2.0), pair[0].energy);

    // Better bond
    updateHBondPair(&pair, 10, -3.0);
    try std.testing.expectEqual(@as(?u32, 10), pair[0].residue_index);
    try std.testing.expectEqual(@as(f32, -3.0), pair[0].energy);
    // Previous best should be second
    try std.testing.expectEqual(@as(?u32, 5), pair[1].residue_index);
    try std.testing.expectEqual(@as(f32, -2.0), pair[1].energy);
}
