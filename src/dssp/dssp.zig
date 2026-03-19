const std = @import("std");
const Allocator = std.mem.Allocator;
const types = @import("types.zig");
const residue_mod = @import("residue.zig");
const json_parser = @import("json_parser.zig");
const hbond_mod = @import("hbond.zig");
const beta_sheet = @import("beta_sheet.zig");
const helix = @import("helix.zig");
const accessibility_mod = @import("accessibility.zig");

const Residue = residue_mod.Residue;
const StructureType = types.StructureType;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

pub const DsspConfig = struct {
    /// PP-II helix minimum stretch length (2 or 3, default 3 matching C++ mkdssp)
    pp_stretch: u32 = 3,
    /// Prefer pi helices over alpha helices (matches C++ default)
    prefer_pi_helices: bool = true,
    /// Calculate surface accessibility
    calculate_accessibility: bool = true,
    /// Number of threads for accessibility calculation (0 = auto-detect)
    n_threads: usize = 0,
    /// Collect timing information for each step
    collect_timing: bool = false,
};

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

pub const Statistics = struct {
    total_residues: u32 = 0,
    complete_residues: u32 = 0,
    chain_breaks: u32 = 0,

    // Per-structure-type counts
    structure_counts: [9]u32 = .{0} ** 9,

    // H-bond statistics
    hbond_count: u32 = 0,
    ss_bond_count: u32 = 0,

    pub fn getCount(self: *const Statistics, st: StructureType) u32 {
        return self.structure_counts[structureIndex(st)];
    }

    fn structureIndex(st: StructureType) usize {
        return switch (st) {
            .loop => 0,
            .alpha_helix => 1,
            .beta_bridge => 2,
            .strand => 3,
            .helix_3 => 4,
            .helix_5 => 5,
            .helix_pp2 => 6,
            .turn => 7,
            .bend => 8,
        };
    }
};

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Timing breakdown for each calculation step (nanoseconds)
pub const StepTimings = struct {
    chain_breaks_ns: u64 = 0,
    hydrogen_ns: u64 = 0,
    geometry_ns: u64 = 0,
    near_pairs_ns: u64 = 0,
    hbond_ns: u64 = 0,
    beta_sheet_ns: u64 = 0,
    helix_ns: u64 = 0,
    pp_helix_ns: u64 = 0,
    accessibility_ns: u64 = 0,
    total_ns: u64 = 0,

    pub fn totalMs(self: StepTimings) f64 {
        return @as(f64, @floatFromInt(self.total_ns)) / 1_000_000.0;
    }
};

pub const DsspResult = struct {
    residues: []Residue,
    ss_bonds: []json_parser.SSBond,
    statistics: Statistics,
    side_chain_storage: []residue_mod.SideChainAtom,
    near_pairs: [][2]u32,
    timings: ?StepTimings = null,
    allocator: Allocator,

    pub fn deinit(self: *DsspResult) void {
        self.allocator.free(self.residues);
        self.allocator.free(self.ss_bonds);
        self.allocator.free(self.side_chain_storage);
        self.allocator.free(self.near_pairs);
    }
};

// ---------------------------------------------------------------------------
// Main calculation (dssp.cpp:1416-1692)
// ---------------------------------------------------------------------------

pub const CalculateError = json_parser.ParseError || error{OutOfMemory};

/// Run the full DSSP algorithm on JSON input.
///
/// Orchestration order:
///  1. Parse JSON input
///  2. Detect chain breaks
///  3. Assign hydrogens
///  4. Calculate backbone geometry (phi, psi, omega, kappa, alpha, TCO)
///  5. Find near pairs (CA-CA < 9 Å)
///  6. Calculate H-bond energies
///  7. Calculate beta sheets (bridges → ladders → sheets)
///  8. Calculate alpha/3-10/pi helices
///  9. Calculate PP-II helices
/// 10. (Optional) Calculate surface accessibility
/// 11. Collect statistics
pub fn calculate(allocator: Allocator, json_data: []const u8, config: DsspConfig) CalculateError!DsspResult {
    // Step 1: Parse JSON input
    const parse_result = try json_parser.parseJsonInput(allocator, json_data);
    return calculateFromParseResult(allocator, parse_result, config);
}

/// Run the full DSSP algorithm on a pre-parsed structure.
///
/// This allows using any parser (JSON, mmCIF, PDB) that produces ParseResult.
pub fn calculateFromParseResult(allocator: Allocator, parse_result: json_parser.ParseResult, config: DsspConfig) CalculateError!DsspResult {
    const ss_bonds = parse_result.ss_bonds;
    const side_chain_storage = parse_result.side_chain_storage;

    // Filter to only complete residues (matching mkdssp behavior: dssp.cpp:1456-1460)
    // mkdssp uses std::erase_if() to remove incomplete residues before processing
    const residues = try filterCompleteResidues(allocator, parse_result.residues);
    // Free the original residues array since we made a filtered copy
    allocator.free(parse_result.residues);

    var timings = StepTimings{};
    const collect_timing = config.collect_timing;
    var total_start: i128 = 0;
    var step_start: i128 = 0;

    if (collect_timing) {
        total_start = std.time.nanoTimestamp();
        step_start = total_start;
    }

    // Step 2: Detect chain breaks
    residue_mod.detectChainBreaks(residues);
    if (collect_timing) {
        const now = std.time.nanoTimestamp();
        timings.chain_breaks_ns = @intCast(now - step_start);
        step_start = now;
    }

    // Step 3: Number residues sequentially
    for (residues, 0..) |*res, i| {
        res.number = @intCast(i);
    }

    // Step 4: Assign hydrogens
    residue_mod.assignHydrogen(residues);
    if (collect_timing) {
        const now = std.time.nanoTimestamp();
        timings.hydrogen_ns = @intCast(now - step_start);
        step_start = now;
    }

    // Step 5: Calculate backbone geometry
    residue_mod.calculateGeometry(residues);
    if (collect_timing) {
        const now = std.time.nanoTimestamp();
        timings.geometry_ns = @intCast(now - step_start);
        step_start = now;
    }

    // Step 6: Mark SS bonds - each bond gets a unique number shared by both cysteines
    // Use u16 to support proteins with more than 255 disulfide bonds
    var ss_bond_number: u16 = 0;
    for (ss_bonds) |bond| {
        const idx1 = findResidue(residues, bond.chain1[0..bond.chain1_len], bond.seq1);
        const idx2 = findResidue(residues, bond.chain2[0..bond.chain2_len], bond.seq2);
        if (idx1 != null and idx2 != null) {
            ss_bond_number += 1;
            residues[idx1.?].ss_bridge_nr = ss_bond_number;
            residues[idx2.?].ss_bridge_nr = ss_bond_number;
        }
    }

    // Step 7: Find near pairs (spatial hash grid for O(N + pairs) instead of O(N²))
    const near_pairs = hbond_mod.findNearPairsOptimized(residues, allocator) catch return CalculateError.OutOfMemory;

    // Sort pairs by (i, j) to match mkdssp's sequential order. This ensures
    // consistent tie-breaking when multiple H-bonds have the same energy.
    const PairComparator = struct {
        fn lessThan(_: @This(), a: [2]u32, b: [2]u32) bool {
            if (a[0] != b[0]) return a[0] < b[0];
            return a[1] < b[1];
        }
    };
    std.mem.sort([2]u32, near_pairs, PairComparator{}, PairComparator.lessThan);

    if (collect_timing) {
        const now = std.time.nanoTimestamp();
        timings.near_pairs_ns = @intCast(now - step_start);
        step_start = now;
    }

    // Step 8: Calculate H-bond energies
    hbond_mod.calculateHBondEnergies(residues, near_pairs);
    if (collect_timing) {
        const now = std.time.nanoTimestamp();
        timings.hbond_ns = @intCast(now - step_start);
        step_start = now;
    }

    // Step 9: Calculate beta sheets
    beta_sheet.calculateBetaSheets(residues, near_pairs, allocator) catch return CalculateError.OutOfMemory;
    if (collect_timing) {
        const now = std.time.nanoTimestamp();
        timings.beta_sheet_ns = @intCast(now - step_start);
        step_start = now;
    }

    // Step 10: Calculate helices
    helix.calculateAlphaHelices(residues, config.prefer_pi_helices);
    if (collect_timing) {
        const now = std.time.nanoTimestamp();
        timings.helix_ns = @intCast(now - step_start);
        step_start = now;
    }

    // Step 11: Calculate PP-II helices
    helix.calculatePPHelices(residues, config.pp_stretch);
    if (collect_timing) {
        const now = std.time.nanoTimestamp();
        timings.pp_helix_ns = @intCast(now - step_start);
        step_start = now;
    }

    // Step 12: Calculate surface accessibility (spatial hash grid optimized)
    if (config.calculate_accessibility) {
        accessibility_mod.calculateAccessibilitiesParallel(residues, allocator, config.n_threads) catch return CalculateError.OutOfMemory;
    }
    if (collect_timing) {
        const now = std.time.nanoTimestamp();
        timings.accessibility_ns = @intCast(now - step_start);
        timings.total_ns = @intCast(now - total_start);
    }

    // Step 13: Collect statistics
    const stats = collectStatistics(residues, ss_bonds);

    return DsspResult{
        .residues = residues,
        .ss_bonds = ss_bonds,
        .statistics = stats,
        .side_chain_storage = side_chain_storage,
        .near_pairs = near_pairs,
        .timings = if (collect_timing) timings else null,
        .allocator = allocator,
    };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Filter residues to match mkdssp behavior:
/// 1. Keep only complete residues (have all backbone atoms N, CA, C, O)
/// 2. Deduplicate by (chain_id, seq_id) - keep only first occurrence
///
/// mkdssp removes incomplete residues with std::erase_if() before processing,
/// and implicitly deduplicates through its CIF parsing which keeps first altloc.
fn filterCompleteResidues(allocator: Allocator, residues: []Residue) ![]Residue {
    // Use a hash set to track seen (chain_id, seq_id) pairs
    const SeenKey = struct {
        chain_id: [4]u8,
        chain_id_len: u8,
        seq_id: i32,
    };
    var seen = std.AutoHashMap(SeenKey, void).init(allocator);
    defer seen.deinit();

    // First pass: count residues that pass the filter
    var count: usize = 0;
    for (residues) |res| {
        if (!res.complete) continue;

        const key = SeenKey{
            .chain_id = res.chain_id,
            .chain_id_len = res.chain_id_len,
            .seq_id = res.seq_id,
        };
        const result = try seen.getOrPut(key);
        if (!result.found_existing) {
            count += 1;
        }
    }

    // Allocate filtered array
    const filtered = try allocator.alloc(Residue, count);

    // Reset seen set for second pass
    seen.clearRetainingCapacity();

    // Second pass: copy residues that pass the filter
    var idx: usize = 0;
    for (residues) |res| {
        if (!res.complete) continue;

        const key = SeenKey{
            .chain_id = res.chain_id,
            .chain_id_len = res.chain_id_len,
            .seq_id = res.seq_id,
        };
        const result = try seen.getOrPut(key);
        if (!result.found_existing) {
            filtered[idx] = res;
            idx += 1;
        }
    }

    return filtered;
}

fn findResidue(residues: []const Residue, chain_id: []const u8, seq_id: i32) ?usize {
    for (residues, 0..) |res, i| {
        if (std.mem.eql(u8, res.getChainId(), chain_id) and res.seq_id == seq_id) {
            return i;
        }
    }
    return null;
}

fn collectStatistics(residues: []const Residue, ss_bonds: []const json_parser.SSBond) Statistics {
    var stats = Statistics{};
    stats.total_residues = @intCast(residues.len);
    stats.ss_bond_count = @intCast(ss_bonds.len);

    for (residues) |res| {
        if (res.complete) stats.complete_residues += 1;
        if (res.chain_break != .none) stats.chain_breaks += 1;
        stats.structure_counts[Statistics.structureIndex(res.secondary_structure)] += 1;
    }

    // Count H-bonds
    for (residues) |res| {
        for (res.hbond_acceptor) |hb| {
            if (hb.residue_index != null and hb.energy < types.kMaxHBondEnergyF32) {
                stats.hbond_count += 1;
            }
        }
    }

    return stats;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "calculate - minimal input" {
    const input =
        \\{
        \\  "residues": [
        \\    {
        \\      "chain_id": "A",
        \\      "seq_id": 1,
        \\      "compound_id": "ALA",
        \\      "atoms": {
        \\        "N": [0.0, 0.0, 0.0],
        \\        "CA": [1.5, 0.0, 0.0],
        \\        "C": [2.5, 0.0, 0.0],
        \\        "O": [2.5, 1.2, 0.0]
        \\      }
        \\    },
        \\    {
        \\      "chain_id": "A",
        \\      "seq_id": 2,
        \\      "compound_id": "GLY",
        \\      "atoms": {
        \\        "N": [3.5, 0.0, 0.0],
        \\        "CA": [5.0, 0.0, 0.0],
        \\        "C": [6.0, 0.0, 0.0],
        \\        "O": [6.0, 1.2, 0.0]
        \\      }
        \\    },
        \\    {
        \\      "chain_id": "A",
        \\      "seq_id": 3,
        \\      "compound_id": "ALA",
        \\      "atoms": {
        \\        "N": [7.0, 0.0, 0.0],
        \\        "CA": [8.5, 0.0, 0.0],
        \\        "C": [9.5, 0.0, 0.0],
        \\        "O": [9.5, 1.2, 0.0]
        \\      }
        \\    }
        \\  ]
        \\}
    ;

    const allocator = std.testing.allocator;
    var result = try calculate(allocator, input, .{});
    defer result.deinit();

    try std.testing.expectEqual(@as(u32, 3), result.statistics.total_residues);
    try std.testing.expectEqual(@as(u32, 3), result.statistics.complete_residues);
}

test "calculate - chain break detection" {
    const input =
        \\{
        \\  "residues": [
        \\    {
        \\      "chain_id": "A",
        \\      "seq_id": 1,
        \\      "compound_id": "ALA",
        \\      "atoms": {
        \\        "N": [0.0, 0.0, 0.0],
        \\        "CA": [1.5, 0.0, 0.0],
        \\        "C": [2.5, 0.0, 0.0],
        \\        "O": [2.5, 1.2, 0.0]
        \\      }
        \\    },
        \\    {
        \\      "chain_id": "B",
        \\      "seq_id": 1,
        \\      "compound_id": "GLY",
        \\      "atoms": {
        \\        "N": [30.0, 0.0, 0.0],
        \\        "CA": [31.5, 0.0, 0.0],
        \\        "C": [32.5, 0.0, 0.0],
        \\        "O": [32.5, 1.2, 0.0]
        \\      }
        \\    }
        \\  ]
        \\}
    ;

    const allocator = std.testing.allocator;
    var result = try calculate(allocator, input, .{});
    defer result.deinit();

    try std.testing.expectEqual(@as(u32, 1), result.statistics.chain_breaks);
}

test "Statistics - structure counting" {
    var stats = Statistics{};
    try std.testing.expectEqual(@as(u32, 0), stats.getCount(.alpha_helix));
    stats.structure_counts[Statistics.structureIndex(.alpha_helix)] = 5;
    try std.testing.expectEqual(@as(u32, 5), stats.getCount(.alpha_helix));
}

test "calculate - filters incomplete residues" {
    const input =
        \\{
        \\  "residues": [
        \\    {
        \\      "chain_id": "A",
        \\      "seq_id": 1,
        \\      "compound_id": "ALA",
        \\      "atoms": {
        \\        "N": [0.0, 0.0, 0.0],
        \\        "CA": [1.5, 0.0, 0.0],
        \\        "C": [2.5, 0.0, 0.0],
        \\        "O": [2.5, 1.2, 0.0]
        \\      }
        \\    },
        \\    {
        \\      "chain_id": "A",
        \\      "seq_id": 2,
        \\      "compound_id": "GLY",
        \\      "atoms": {
        \\        "N": [3.5, 0.0, 0.0],
        \\        "CA": [5.0, 0.0, 0.0]
        \\      }
        \\    },
        \\    {
        \\      "chain_id": "A",
        \\      "seq_id": 3,
        \\      "compound_id": "ALA",
        \\      "atoms": {
        \\        "N": [7.0, 0.0, 0.0],
        \\        "CA": [8.5, 0.0, 0.0],
        \\        "C": [9.5, 0.0, 0.0],
        \\        "O": [9.5, 1.2, 0.0]
        \\      }
        \\    }
        \\  ]
        \\}
    ;

    const allocator = std.testing.allocator;
    var result = try calculate(allocator, input, .{});
    defer result.deinit();

    // Second residue is incomplete (missing C, O), should be filtered out
    try std.testing.expectEqual(@as(u32, 2), result.statistics.total_residues);
    try std.testing.expectEqual(@as(u32, 2), result.statistics.complete_residues);
}

test "calculate - deduplicates by chain_id and seq_id" {
    // Simulates alternate conformations with same seq_id
    const input =
        \\{
        \\  "residues": [
        \\    {
        \\      "chain_id": "A",
        \\      "seq_id": 1,
        \\      "compound_id": "ALA",
        \\      "atoms": {
        \\        "N": [0.0, 0.0, 0.0],
        \\        "CA": [1.5, 0.0, 0.0],
        \\        "C": [2.5, 0.0, 0.0],
        \\        "O": [2.5, 1.2, 0.0]
        \\      }
        \\    },
        \\    {
        \\      "chain_id": "A",
        \\      "seq_id": 1,
        \\      "compound_id": "SER",
        \\      "atoms": {
        \\        "N": [0.1, 0.0, 0.0],
        \\        "CA": [1.6, 0.0, 0.0],
        \\        "C": [2.6, 0.0, 0.0],
        \\        "O": [2.6, 1.2, 0.0]
        \\      }
        \\    },
        \\    {
        \\      "chain_id": "A",
        \\      "seq_id": 2,
        \\      "compound_id": "GLY",
        \\      "atoms": {
        \\        "N": [3.5, 0.0, 0.0],
        \\        "CA": [5.0, 0.0, 0.0],
        \\        "C": [6.0, 0.0, 0.0],
        \\        "O": [6.0, 1.2, 0.0]
        \\      }
        \\    }
        \\  ]
        \\}
    ;

    const allocator = std.testing.allocator;
    var result = try calculate(allocator, input, .{});
    defer result.deinit();

    // Two residues have seq_id=1, only first should be kept
    try std.testing.expectEqual(@as(u32, 2), result.statistics.total_residues);
    // First kept residue should be ALA, not SER
    try std.testing.expect(std.mem.eql(u8, result.residues[0].getCompoundId(), "ALA"));
}
