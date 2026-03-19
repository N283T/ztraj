// Backbone extraction and preprocessing for native DSSP.
//
// Reads Topology + Frame to build a []DsspResidue with atom indices into
// the Frame arrays. No coordinates are copied except the computed hydrogen.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ztraj_types = @import("../../types.zig");
const types = @import("types.zig");

const Topology = ztraj_types.Topology;
const Frame = ztraj_types.Frame;
const DsspResidue = types.DsspResidue;
const Vec3f32 = types.Vec3f32;
const ChainBreakType = types.ChainBreakType;
const ResidueType = types.ResidueType;

// ============================================================================
// Coordinate helper
// ============================================================================

/// Read a coordinate from Frame at atom index idx.
pub fn getPos(frame: Frame, idx: u32) Vec3f32 {
    return .{ .x = frame.x[idx], .y = frame.y[idx], .z = frame.z[idx] };
}

// ============================================================================
// Backbone extraction
// ============================================================================

/// Build a DsspResidue slice from topology + frame.
///
/// For each residue in topology, scans its atom range for the four backbone
/// atoms N, CA, C, O. Residues with all four atoms have complete=true.
/// Residue type and chain assignment are taken from topology.
///
/// Caller owns the returned slice and must free it with the same allocator.
pub fn extractBackbone(
    allocator: Allocator,
    topology: Topology,
    frame: Frame,
) ![]DsspResidue {
    const n_residues = topology.residues.len;
    const residues = try allocator.alloc(DsspResidue, n_residues);

    for (topology.residues, 0..) |topo_res, r| {
        var dssp_res = DsspResidue{};
        dssp_res.residue_index = @intCast(r);
        dssp_res.chain_index = topo_res.chain_index;
        dssp_res.residue_type = ResidueType.fromCompoundId(topo_res.name.slice());

        var found_n = false;
        var found_ca = false;
        var found_c = false;
        var found_o = false;

        const atom_start = topo_res.atom_range.start;
        const atom_end = topo_res.atom_range.end();

        for (atom_start..atom_end) |a| {
            const atom = topology.atoms[a];
            const name = atom.name.slice();
            if (!found_n and std.mem.eql(u8, name, "N")) {
                dssp_res.n_idx = @intCast(a);
                found_n = true;
            } else if (!found_ca and std.mem.eql(u8, name, "CA")) {
                dssp_res.ca_idx = @intCast(a);
                found_ca = true;
            } else if (!found_c and std.mem.eql(u8, name, "C")) {
                dssp_res.c_idx = @intCast(a);
                found_c = true;
            } else if (!found_o and std.mem.eql(u8, name, "O")) {
                dssp_res.o_idx = @intCast(a);
                found_o = true;
            }
        }

        dssp_res.complete = found_n and found_ca and found_c and found_o;
        _ = frame; // atom indices are stored, coords accessed on demand
        residues[r] = dssp_res;
    }

    return residues;
}

// ============================================================================
// Chain break detection
// ============================================================================

/// Detect chain breaks using topology chain boundaries and peptide bond length.
///
/// A break is:
///  - .new_chain when the chain_index differs from the previous residue
///  - .gap when the C(i-1)–N(i) distance exceeds kMaxPeptideBondLength
pub fn detectChainBreaks(
    residues: []DsspResidue,
    topology: Topology,
    frame: Frame,
) void {
    _ = topology; // chain_index is already stored in each DsspResidue
    if (residues.len == 0) return;

    for (residues[1..], 1..) |*res, i| {
        const prev = &residues[i - 1];

        if (res.chain_index != prev.chain_index) {
            res.chain_break = .new_chain;
            continue;
        }

        // Check C(i-1) to N(i) peptide bond length
        const c_pos = getPos(frame, prev.c_idx);
        const n_pos = getPos(frame, res.n_idx);
        const dist = c_pos.distance(n_pos);
        if (dist > types.kMaxPeptideBondLength) {
            res.chain_break = .gap;
        }
    }
}

// ============================================================================
// Hydrogen placement
// ============================================================================

/// Place the backbone amide hydrogen for each non-proline, non-chain-break
/// residue.
///
/// H = N(i) + normalize(C(i-1) - O(i-1))
///   (unit vector opposite to the previous C=O, length 1.0 Å)
///
/// Residues at chain starts or prolines receive H = (0,0,0).
pub fn assignHydrogen(residues: []DsspResidue, frame: Frame) void {
    for (residues, 0..) |*res, i| {
        if (i == 0 or res.chain_break != .none or res.isProline()) {
            res.h_x = 0;
            res.h_y = 0;
            res.h_z = 0;
            continue;
        }
        const prev = &residues[i - 1];
        const prev_c = getPos(frame, prev.c_idx);
        const prev_o = getPos(frame, prev.o_idx);
        const n = getPos(frame, res.n_idx);

        const co = prev_o.sub(prev_c);
        const co_len = co.length();
        if (co_len < 1e-6) {
            res.h_x = 0;
            res.h_y = 0;
            res.h_z = 0;
            continue;
        }
        const unit_co = co.scale(1.0 / co_len);
        const h = n.sub(unit_co);
        res.h_x = h.x;
        res.h_y = h.y;
        res.h_z = h.z;
    }
}

// ============================================================================
// Chain-break range helper
// ============================================================================

/// Return true when there is no chain break in residues[from+1..to] (inclusive).
pub fn noChainBreak(residues: []const DsspResidue, from: u32, to: u32) bool {
    if (from >= to) return true;
    const start = from + 1;
    const end_idx = @min(to + 1, @as(u32, @intCast(residues.len)));
    for (residues[start..end_idx]) |res| {
        if (res.chain_break != .none) return false;
    }
    return true;
}

// ============================================================================
// Tests
// ============================================================================

test "extractBackbone - finds all four backbone atoms" {
    const allocator = std.testing.allocator;

    // Build a minimal topology: 1 chain, 2 residues, 8 atoms (N CA C O each)
    var topo = try Topology.init(allocator, .{
        .n_atoms = 8,
        .n_residues = 2,
        .n_chains = 1,
        .n_bonds = 0,
    });
    defer topo.deinit();

    topo.chains[0] = .{
        .name = ztraj_types.FixedString(4).fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 2 },
    };

    const atom_names = [_][]const u8{ "N", "CA", "C", "O", "N", "CA", "C", "O" };
    for (topo.atoms, 0..) |*atom, i| {
        atom.* = .{
            .name = ztraj_types.FixedString(4).fromSlice(atom_names[i]),
            .element = .C,
            .residue_index = @intCast(i / 4),
        };
    }
    topo.residues[0] = .{
        .name = ztraj_types.FixedString(5).fromSlice("ALA"),
        .chain_index = 0,
        .atom_range = .{ .start = 0, .len = 4 },
        .resid = 1,
    };
    topo.residues[1] = .{
        .name = ztraj_types.FixedString(5).fromSlice("GLY"),
        .chain_index = 0,
        .atom_range = .{ .start = 4, .len = 4 },
        .resid = 2,
    };

    var frame = try Frame.init(allocator, 8);
    defer frame.deinit();

    const residues = try extractBackbone(allocator, topo, frame);
    defer allocator.free(residues);

    try std.testing.expectEqual(@as(usize, 2), residues.len);
    try std.testing.expect(residues[0].complete);
    try std.testing.expect(residues[1].complete);
    try std.testing.expectEqual(@as(u32, 0), residues[0].n_idx);
    try std.testing.expectEqual(@as(u32, 1), residues[0].ca_idx);
    try std.testing.expectEqual(@as(u32, 2), residues[0].c_idx);
    try std.testing.expectEqual(@as(u32, 3), residues[0].o_idx);
    try std.testing.expectEqual(@as(u32, 4), residues[1].n_idx);
}

test "extractBackbone - incomplete residue" {
    const allocator = std.testing.allocator;

    var topo = try Topology.init(allocator, .{
        .n_atoms = 3,
        .n_residues = 1,
        .n_chains = 1,
        .n_bonds = 0,
    });
    defer topo.deinit();

    topo.chains[0] = .{
        .name = ztraj_types.FixedString(4).fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 1 },
    };
    // Only N CA C — missing O
    const names = [_][]const u8{ "N", "CA", "C" };
    for (topo.atoms, 0..) |*atom, i| {
        atom.* = .{
            .name = ztraj_types.FixedString(4).fromSlice(names[i]),
            .element = .C,
            .residue_index = 0,
        };
    }
    topo.residues[0] = .{
        .name = ztraj_types.FixedString(5).fromSlice("ALA"),
        .chain_index = 0,
        .atom_range = .{ .start = 0, .len = 3 },
        .resid = 1,
    };

    var frame = try Frame.init(allocator, 3);
    defer frame.deinit();

    const residues = try extractBackbone(allocator, topo, frame);
    defer allocator.free(residues);

    try std.testing.expect(!residues[0].complete);
}

test "detectChainBreaks - same chain, normal bond" {
    const allocator = std.testing.allocator;

    var topo = try Topology.init(allocator, .{
        .n_atoms = 8,
        .n_residues = 2,
        .n_chains = 1,
        .n_bonds = 0,
    });
    defer topo.deinit();

    topo.chains[0] = .{
        .name = ztraj_types.FixedString(4).fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 2 },
    };
    const names = [_][]const u8{ "N", "CA", "C", "O", "N", "CA", "C", "O" };
    for (topo.atoms, 0..) |*atom, i| {
        atom.* = .{
            .name = ztraj_types.FixedString(4).fromSlice(names[i]),
            .element = .C,
            .residue_index = @intCast(i / 4),
        };
    }
    topo.residues[0] = .{
        .name = ztraj_types.FixedString(5).fromSlice("ALA"),
        .chain_index = 0,
        .atom_range = .{ .start = 0, .len = 4 },
        .resid = 1,
    };
    topo.residues[1] = .{
        .name = ztraj_types.FixedString(5).fromSlice("GLY"),
        .chain_index = 0,
        .atom_range = .{ .start = 4, .len = 4 },
        .resid = 2,
    };

    var frame = try Frame.init(allocator, 8);
    defer frame.deinit();

    // C at index 2, N at index 4 — place them 1.3 Å apart (< 2.5 Å)
    frame.x[2] = 0.0;
    frame.y[2] = 0.0;
    frame.z[2] = 0.0;
    frame.x[4] = 1.3;
    frame.y[4] = 0.0;
    frame.z[4] = 0.0;

    const residues = try extractBackbone(allocator, topo, frame);
    defer allocator.free(residues);

    detectChainBreaks(residues, topo, frame);

    try std.testing.expectEqual(ChainBreakType.none, residues[1].chain_break);
}

test "detectChainBreaks - gap (bond too long)" {
    const allocator = std.testing.allocator;

    var topo = try Topology.init(allocator, .{
        .n_atoms = 8,
        .n_residues = 2,
        .n_chains = 1,
        .n_bonds = 0,
    });
    defer topo.deinit();

    topo.chains[0] = .{
        .name = ztraj_types.FixedString(4).fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 2 },
    };
    const names = [_][]const u8{ "N", "CA", "C", "O", "N", "CA", "C", "O" };
    for (topo.atoms, 0..) |*atom, i| {
        atom.* = .{
            .name = ztraj_types.FixedString(4).fromSlice(names[i]),
            .element = .C,
            .residue_index = @intCast(i / 4),
        };
    }
    topo.residues[0] = .{
        .name = ztraj_types.FixedString(5).fromSlice("ALA"),
        .chain_index = 0,
        .atom_range = .{ .start = 0, .len = 4 },
        .resid = 1,
    };
    topo.residues[1] = .{
        .name = ztraj_types.FixedString(5).fromSlice("GLY"),
        .chain_index = 0,
        .atom_range = .{ .start = 4, .len = 4 },
        .resid = 2,
    };

    var frame = try Frame.init(allocator, 8);
    defer frame.deinit();

    // C at index 2 and N at index 4 are 10 Å apart (> 2.5 Å)
    frame.x[2] = 0.0;
    frame.x[4] = 10.0;

    const residues = try extractBackbone(allocator, topo, frame);
    defer allocator.free(residues);

    detectChainBreaks(residues, topo, frame);

    try std.testing.expectEqual(ChainBreakType.gap, residues[1].chain_break);
}

test "detectChainBreaks - new chain" {
    const allocator = std.testing.allocator;

    var topo = try Topology.init(allocator, .{
        .n_atoms = 8,
        .n_residues = 2,
        .n_chains = 2,
        .n_bonds = 0,
    });
    defer topo.deinit();

    topo.chains[0] = .{
        .name = ztraj_types.FixedString(4).fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 1 },
    };
    topo.chains[1] = .{
        .name = ztraj_types.FixedString(4).fromSlice("B"),
        .residue_range = .{ .start = 1, .len = 1 },
    };
    const names = [_][]const u8{ "N", "CA", "C", "O", "N", "CA", "C", "O" };
    for (topo.atoms, 0..) |*atom, i| {
        atom.* = .{
            .name = ztraj_types.FixedString(4).fromSlice(names[i]),
            .element = .C,
            .residue_index = @intCast(i / 4),
        };
    }
    topo.residues[0] = .{
        .name = ztraj_types.FixedString(5).fromSlice("ALA"),
        .chain_index = 0,
        .atom_range = .{ .start = 0, .len = 4 },
        .resid = 1,
    };
    topo.residues[1] = .{
        .name = ztraj_types.FixedString(5).fromSlice("GLY"),
        .chain_index = 1,
        .atom_range = .{ .start = 4, .len = 4 },
        .resid = 1,
    };

    var frame = try Frame.init(allocator, 8);
    defer frame.deinit();

    const residues = try extractBackbone(allocator, topo, frame);
    defer allocator.free(residues);

    detectChainBreaks(residues, topo, frame);

    try std.testing.expectEqual(ChainBreakType.new_chain, residues[1].chain_break);
}

test "assignHydrogen - basic placement" {
    var residues = [_]DsspResidue{
        DsspResidue{ .c_idx = 2, .o_idx = 3, .complete = true },
        DsspResidue{ .n_idx = 4, .complete = true, .residue_type = .ala },
    };

    var frame_buf: [5][3]f32 = undefined;
    // C at atom 2 = (0,0,0), O at atom 3 = (0,0,1.2), N at atom 4 = (1.5,0,0)
    frame_buf[2] = .{ 0.0, 0.0, 0.0 };
    frame_buf[3] = .{ 0.0, 0.0, 1.2 };
    frame_buf[4] = .{ 1.5, 0.0, 0.0 };

    // Build a dummy Frame by directly setting coordinates
    const allocator = std.testing.allocator;
    var frame = try Frame.init(allocator, 8);
    defer frame.deinit();
    frame.x[2] = 0.0;
    frame.y[2] = 0.0;
    frame.z[2] = 0.0;
    frame.x[3] = 0.0;
    frame.y[3] = 0.0;
    frame.z[3] = 1.2;
    frame.x[4] = 1.5;
    frame.y[4] = 0.0;
    frame.z[4] = 0.0;

    assignHydrogen(&residues, frame);

    // H should be ~1 Å from N, in the direction opposite to C=O
    const n = Vec3f32{ .x = 1.5, .y = 0.0, .z = 0.0 };
    const h = residues[1].getH();
    const dist = n.distance(h);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), dist, 0.01);
    // H.z should be negative (opposite to O direction)
    try std.testing.expect(h.z < 0.0);
}

test "assignHydrogen - proline gets zero" {
    var residues = [_]DsspResidue{
        DsspResidue{ .c_idx = 2, .o_idx = 3, .complete = true },
        DsspResidue{ .n_idx = 4, .complete = true, .residue_type = .pro },
    };

    const allocator = std.testing.allocator;
    var frame = try Frame.init(allocator, 8);
    defer frame.deinit();
    frame.x[2] = 0.0;
    frame.z[3] = 1.2;
    frame.x[4] = 1.5;

    assignHydrogen(&residues, frame);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), residues[1].h_x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), residues[1].h_y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), residues[1].h_z, 1e-6);
}

test "noChainBreak" {
    var residues = [_]DsspResidue{
        DsspResidue{},
        DsspResidue{},
        DsspResidue{ .chain_break = .new_chain },
        DsspResidue{},
    };
    try std.testing.expect(noChainBreak(&residues, 0, 1));
    try std.testing.expect(!noChainBreak(&residues, 0, 2));
    try std.testing.expect(!noChainBreak(&residues, 1, 3));
    try std.testing.expect(noChainBreak(&residues, 3, 3));
}
