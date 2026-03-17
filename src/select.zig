//! Atom selection utilities.
//!
//! Provides functions that return a slice of 0-based atom indices satisfying
//! a given criterion. All returned slices are owned by the caller and must be
//! freed with the provided allocator.
//!
//! Phase 1 supports:
//!   - Index specification strings ("0,1,5-10")
//!   - Keyword selections (backbone, protein, water)
//!   - Name-based selection ("CA")
//!   - Element-based selection (.N, .C, ...)

const std = @import("std");
const types = @import("types.zig");
const element_mod = @import("element.zig");
pub const Element = element_mod.Element;

// ============================================================================
// Keyword definitions
// ============================================================================

/// Pre-defined atom selection keywords.
pub const Keyword = enum {
    /// Protein backbone atoms: N, CA, C, O.
    backbone,
    /// All atoms belonging to standard amino acid residues.
    protein,
    /// Water molecules (HOH, WAT, SOL, TIP3, TIP4, SPC).
    water,
};

/// Backbone atom names (exact match).
const backbone_names = [_][]const u8{ "N", "CA", "C", "O" };

/// Standard amino acid residue names (3-letter codes).
const protein_residues = [_][]const u8{
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
};

/// Water residue names.
const water_residues = [_][]const u8{
    "HOH", "WAT", "SOL", "TIP3", "TIP4", "SPC",
};

// ============================================================================
// byIndex — parse "0,1,5-10" style specifications
// ============================================================================

/// Parse an index specification string and return a sorted, deduplicated slice
/// of atom indices.
///
/// Supported syntax:
///   - Single index: "3"
///   - Comma-separated: "0,1,2"
///   - Ranges (inclusive): "5-10"
///   - Mixed: "0,1,5-10,20"
///
/// Returns error.InvalidSpec if the string is malformed.
/// Caller owns the returned slice.
pub fn byIndex(allocator: std.mem.Allocator, spec: []const u8) ![]u32 {
    var result = std.ArrayList(u32){};
    defer result.deinit(allocator);

    var token_iter = std.mem.tokenizeScalar(u8, spec, ',');
    while (token_iter.next()) |token| {
        const trimmed = std.mem.trim(u8, token, " \t\r\n");
        if (trimmed.len == 0) continue;

        if (std.mem.indexOfScalar(u8, trimmed, '-')) |dash_pos| {
            // Range: "start-end"
            const start_str = std.mem.trim(u8, trimmed[0..dash_pos], " ");
            const end_str = std.mem.trim(u8, trimmed[dash_pos + 1 ..], " ");

            const start = std.fmt.parseInt(u32, start_str, 10) catch return error.InvalidSpec;
            const end = std.fmt.parseInt(u32, end_str, 10) catch return error.InvalidSpec;
            if (end < start) return error.InvalidSpec;

            var idx: u32 = start;
            while (idx <= end) : (idx += 1) {
                try result.append(allocator, idx);
            }
        } else {
            // Single index.
            const idx = std.fmt.parseInt(u32, trimmed, 10) catch return error.InvalidSpec;
            try result.append(allocator, idx);
        }
    }

    // Sort and deduplicate in-place before transferring ownership.
    std.mem.sort(u32, result.items, {}, std.sort.asc(u32));
    const unique_len = dedupSorted(result.items);
    result.items.len = unique_len;

    return result.toOwnedSlice(allocator);
}

/// Compact a sorted slice in-place and return the number of unique elements.
fn dedupSorted(items: []u32) usize {
    if (items.len == 0) return 0;
    var write: usize = 1;
    for (1..items.len) |read| {
        if (items[read] != items[write - 1]) {
            items[write] = items[read];
            write += 1;
        }
    }
    return write;
}

// ============================================================================
// byKeyword
// ============================================================================

/// Return indices of atoms matching the given keyword.
/// Caller owns the returned slice.
pub fn byKeyword(
    allocator: std.mem.Allocator,
    topology: types.Topology,
    kw: Keyword,
) ![]u32 {
    var result = std.ArrayList(u32){};
    defer result.deinit(allocator);

    switch (kw) {
        .backbone => {
            for (topology.atoms, 0..) |atom, i| {
                const name = atom.name.slice();
                for (backbone_names) |bn| {
                    if (std.mem.eql(u8, name, bn)) {
                        try result.append(allocator, @intCast(i));
                        break;
                    }
                }
            }
        },

        .protein => {
            for (topology.atoms, 0..) |atom, i| {
                const res_idx = atom.residue_index;
                if (res_idx >= topology.residues.len) continue;
                const res_name = topology.residues[res_idx].name.slice();
                for (protein_residues) |pr| {
                    if (std.mem.eql(u8, res_name, pr)) {
                        try result.append(allocator, @intCast(i));
                        break;
                    }
                }
            }
        },

        .water => {
            for (topology.atoms, 0..) |atom, i| {
                const res_idx = atom.residue_index;
                if (res_idx >= topology.residues.len) continue;
                const res_name = topology.residues[res_idx].name.slice();
                for (water_residues) |wr| {
                    if (std.mem.eql(u8, res_name, wr)) {
                        try result.append(allocator, @intCast(i));
                        break;
                    }
                }
            }
        },
    }

    return result.toOwnedSlice(allocator);
}

// ============================================================================
// byName
// ============================================================================

/// Return indices of all atoms whose name exactly matches `name`.
/// Caller owns the returned slice.
pub fn byName(
    allocator: std.mem.Allocator,
    topology: types.Topology,
    name: []const u8,
) ![]u32 {
    var result = std.ArrayList(u32){};
    defer result.deinit(allocator);

    for (topology.atoms, 0..) |atom, i| {
        if (atom.name.eqlSlice(name)) {
            try result.append(allocator, @intCast(i));
        }
    }

    return result.toOwnedSlice(allocator);
}

// ============================================================================
// byElement
// ============================================================================

/// Return indices of all atoms with the given element.
/// Caller owns the returned slice.
pub fn byElement(
    allocator: std.mem.Allocator,
    topology: types.Topology,
    elem: Element,
) ![]u32 {
    var result = std.ArrayList(u32){};
    defer result.deinit(allocator);

    for (topology.atoms, 0..) |atom, i| {
        if (atom.element == elem) {
            try result.append(allocator, @intCast(i));
        }
    }

    return result.toOwnedSlice(allocator);
}

// ============================================================================
// Tests
// ============================================================================

// --- byIndex tests ---

test "byIndex single index" {
    const allocator = std.testing.allocator;
    const indices = try byIndex(allocator, "3");
    defer allocator.free(indices);

    try std.testing.expectEqual(@as(usize, 1), indices.len);
    try std.testing.expectEqual(@as(u32, 3), indices[0]);
}

test "byIndex comma-separated" {
    const allocator = std.testing.allocator;
    const indices = try byIndex(allocator, "0,1,2");
    defer allocator.free(indices);

    try std.testing.expectEqual(@as(usize, 3), indices.len);
    try std.testing.expectEqual(@as(u32, 0), indices[0]);
    try std.testing.expectEqual(@as(u32, 1), indices[1]);
    try std.testing.expectEqual(@as(u32, 2), indices[2]);
}

test "byIndex range" {
    const allocator = std.testing.allocator;
    const indices = try byIndex(allocator, "5-10");
    defer allocator.free(indices);

    try std.testing.expectEqual(@as(usize, 6), indices.len);
    try std.testing.expectEqual(@as(u32, 5), indices[0]);
    try std.testing.expectEqual(@as(u32, 10), indices[5]);
}

test "byIndex mixed spec" {
    const allocator = std.testing.allocator;
    const indices = try byIndex(allocator, "0,1,5-10");
    defer allocator.free(indices);

    // 0, 1, 5, 6, 7, 8, 9, 10 = 8 elements
    try std.testing.expectEqual(@as(usize, 8), indices.len);
    try std.testing.expectEqual(@as(u32, 0), indices[0]);
    try std.testing.expectEqual(@as(u32, 1), indices[1]);
    try std.testing.expectEqual(@as(u32, 5), indices[2]);
    try std.testing.expectEqual(@as(u32, 10), indices[7]);
}

test "byIndex deduplication" {
    const allocator = std.testing.allocator;
    const indices = try byIndex(allocator, "1,1,2");
    defer allocator.free(indices);

    try std.testing.expectEqual(@as(usize, 2), indices.len);
    try std.testing.expectEqual(@as(u32, 1), indices[0]);
    try std.testing.expectEqual(@as(u32, 2), indices[1]);
}

test "byIndex sorted output" {
    const allocator = std.testing.allocator;
    const indices = try byIndex(allocator, "10,2,5");
    defer allocator.free(indices);

    try std.testing.expectEqual(@as(usize, 3), indices.len);
    try std.testing.expectEqual(@as(u32, 2), indices[0]);
    try std.testing.expectEqual(@as(u32, 5), indices[1]);
    try std.testing.expectEqual(@as(u32, 10), indices[2]);
}

test "byIndex invalid spec returns error" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.InvalidSpec, byIndex(allocator, "abc"));
    try std.testing.expectError(error.InvalidSpec, byIndex(allocator, "1-"));
}

// --- Test topology helper ---

/// Build a minimal topology for testing.
/// Chain A: residues GLY, ALA, HOH
/// GLY: atoms N, CA, C, O (indices 0-3)
/// ALA: atoms N, CA, C, O, CB (indices 4-8)
/// HOH: atom O (index 9)
fn buildTestTopology(allocator: std.mem.Allocator) !types.Topology {
    var topo = try types.Topology.init(allocator, .{
        .n_atoms = 10,
        .n_residues = 3,
        .n_chains = 1,
        .n_bonds = 0,
    });

    topo.chains[0] = .{
        .name = types.FixedString(4).fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 3 },
    };

    // Residue 0: GLY (atoms 0-3)
    topo.residues[0] = .{
        .name = types.FixedString(5).fromSlice("GLY"),
        .chain_index = 0,
        .atom_range = .{ .start = 0, .len = 4 },
        .resid = 1,
    };
    topo.atoms[0] = .{ .name = types.FixedString(4).fromSlice("N"), .element = .N, .residue_index = 0 };
    topo.atoms[1] = .{ .name = types.FixedString(4).fromSlice("CA"), .element = .C, .residue_index = 0 };
    topo.atoms[2] = .{ .name = types.FixedString(4).fromSlice("C"), .element = .C, .residue_index = 0 };
    topo.atoms[3] = .{ .name = types.FixedString(4).fromSlice("O"), .element = .O, .residue_index = 0 };

    // Residue 1: ALA (atoms 4-8)
    topo.residues[1] = .{
        .name = types.FixedString(5).fromSlice("ALA"),
        .chain_index = 0,
        .atom_range = .{ .start = 4, .len = 5 },
        .resid = 2,
    };
    topo.atoms[4] = .{ .name = types.FixedString(4).fromSlice("N"), .element = .N, .residue_index = 1 };
    topo.atoms[5] = .{ .name = types.FixedString(4).fromSlice("CA"), .element = .C, .residue_index = 1 };
    topo.atoms[6] = .{ .name = types.FixedString(4).fromSlice("C"), .element = .C, .residue_index = 1 };
    topo.atoms[7] = .{ .name = types.FixedString(4).fromSlice("O"), .element = .O, .residue_index = 1 };
    topo.atoms[8] = .{ .name = types.FixedString(4).fromSlice("CB"), .element = .C, .residue_index = 1 };

    // Residue 2: HOH (atom 9)
    topo.residues[2] = .{
        .name = types.FixedString(5).fromSlice("HOH"),
        .chain_index = 0,
        .atom_range = .{ .start = 9, .len = 1 },
        .resid = 100,
    };
    topo.atoms[9] = .{ .name = types.FixedString(4).fromSlice("O"), .element = .O, .residue_index = 2 };

    return topo;
}

// --- byKeyword tests ---

test "byKeyword backbone selects N CA C O atoms" {
    const allocator = std.testing.allocator;
    var topo = try buildTestTopology(allocator);
    defer topo.deinit();

    const indices = try byKeyword(allocator, topo, .backbone);
    defer allocator.free(indices);

    // backbone_names = {N, CA, C, O}
    // GLY: 0(N),1(CA),2(C),3(O)  ALA: 4(N),5(CA),6(C),7(O)  HOH: 9(O)
    try std.testing.expectEqual(@as(usize, 9), indices.len);
    try std.testing.expectEqual(@as(u32, 0), indices[0]); // GLY-N
    try std.testing.expectEqual(@as(u32, 1), indices[1]); // GLY-CA
    try std.testing.expectEqual(@as(u32, 2), indices[2]); // GLY-C
    try std.testing.expectEqual(@as(u32, 3), indices[3]); // GLY-O
    try std.testing.expectEqual(@as(u32, 4), indices[4]); // ALA-N
    try std.testing.expectEqual(@as(u32, 5), indices[5]); // ALA-CA
    try std.testing.expectEqual(@as(u32, 6), indices[6]); // ALA-C
    try std.testing.expectEqual(@as(u32, 7), indices[7]); // ALA-O
    try std.testing.expectEqual(@as(u32, 9), indices[8]); // HOH-O
}

test "byKeyword protein excludes water" {
    const allocator = std.testing.allocator;
    var topo = try buildTestTopology(allocator);
    defer topo.deinit();

    const indices = try byKeyword(allocator, topo, .protein);
    defer allocator.free(indices);

    // GLY (4 atoms) + ALA (5 atoms) = 9; HOH excluded
    try std.testing.expectEqual(@as(usize, 9), indices.len);
    try std.testing.expectEqual(@as(u32, 8), indices[8]); // ALA-CB
}

test "byKeyword water selects HOH only" {
    const allocator = std.testing.allocator;
    var topo = try buildTestTopology(allocator);
    defer topo.deinit();

    const indices = try byKeyword(allocator, topo, .water);
    defer allocator.free(indices);

    try std.testing.expectEqual(@as(usize, 1), indices.len);
    try std.testing.expectEqual(@as(u32, 9), indices[0]);
}

// --- byName tests ---

test "byName CA selects alpha carbons" {
    const allocator = std.testing.allocator;
    var topo = try buildTestTopology(allocator);
    defer topo.deinit();

    const indices = try byName(allocator, topo, "CA");
    defer allocator.free(indices);

    try std.testing.expectEqual(@as(usize, 2), indices.len);
    try std.testing.expectEqual(@as(u32, 1), indices[0]); // GLY-CA
    try std.testing.expectEqual(@as(u32, 5), indices[1]); // ALA-CA
}

test "byName returns empty for nonexistent name" {
    const allocator = std.testing.allocator;
    var topo = try buildTestTopology(allocator);
    defer topo.deinit();

    const indices = try byName(allocator, topo, "ZZ");
    defer allocator.free(indices);

    try std.testing.expectEqual(@as(usize, 0), indices.len);
}

// --- byElement tests ---

test "byElement nitrogen selects N atoms" {
    const allocator = std.testing.allocator;
    var topo = try buildTestTopology(allocator);
    defer topo.deinit();

    const indices = try byElement(allocator, topo, .N);
    defer allocator.free(indices);

    // GLY-N (0) and ALA-N (4)
    try std.testing.expectEqual(@as(usize, 2), indices.len);
    try std.testing.expectEqual(@as(u32, 0), indices[0]);
    try std.testing.expectEqual(@as(u32, 4), indices[1]);
}

test "byElement oxygen selects O atoms" {
    const allocator = std.testing.allocator;
    var topo = try buildTestTopology(allocator);
    defer topo.deinit();

    const indices = try byElement(allocator, topo, .O);
    defer allocator.free(indices);

    // GLY-O (3), ALA-O (7), HOH-O (9)
    try std.testing.expectEqual(@as(usize, 3), indices.len);
    try std.testing.expectEqual(@as(u32, 3), indices[0]);
    try std.testing.expectEqual(@as(u32, 7), indices[1]);
    try std.testing.expectEqual(@as(u32, 9), indices[2]);
}

test "byElement returns empty for absent element" {
    const allocator = std.testing.allocator;
    var topo = try buildTestTopology(allocator);
    defer topo.deinit();

    const indices = try byElement(allocator, topo, .Fe);
    defer allocator.free(indices);

    try std.testing.expectEqual(@as(usize, 0), indices.len);
}
