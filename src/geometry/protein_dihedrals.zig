//! Protein-specific dihedral angles: phi, psi, omega, and chi (1-4).
//!
//! Automatically detects the 4 atoms forming each dihedral from the topology.
//! Backbone dihedrals (phi, psi, omega) use standard N, CA, C atoms.
//! Side-chain chi angles use residue-specific atom name patterns.

const std = @import("std");
const types = @import("../types.zig");
const dihedrals = @import("dihedrals.zig");

// ============================================================================
// Chi angle atom name patterns (residue-specific)
// ============================================================================

const AtomPattern = [4][]const u8;

const chi1_patterns = [_]AtomPattern{
    .{ "N", "CA", "CB", "CG" },
    .{ "N", "CA", "CB", "CG1" },
    .{ "N", "CA", "CB", "SG" },
    .{ "N", "CA", "CB", "OG" },
    .{ "N", "CA", "CB", "OG1" },
};

const chi2_patterns = [_]AtomPattern{
    .{ "CA", "CB", "CG", "CD" },
    .{ "CA", "CB", "CG", "CD1" },
    .{ "CA", "CB", "CG1", "CD1" },
    .{ "CA", "CB", "CG", "OD1" },
    .{ "CA", "CB", "CG", "ND1" },
    .{ "CA", "CB", "CG", "SD" },
};

const chi3_patterns = [_]AtomPattern{
    .{ "CB", "CG", "CD", "NE" },
    .{ "CB", "CG", "CD", "CE" },
    .{ "CB", "CG", "CD", "OE1" },
    .{ "CB", "CG", "SD", "CE" },
};

const chi4_patterns = [_]AtomPattern{
    .{ "CG", "CD", "NE", "CZ" },
    .{ "CG", "CD", "CE", "NZ" },
};

// ============================================================================
// Backbone dihedral extraction
// ============================================================================

/// Find atom index by name within a residue's atom range.
fn findAtom(topology: types.Topology, res_idx: u32, name: []const u8) ?u32 {
    const res = topology.residues[res_idx];
    const end = res.atom_range.end();
    var a: u32 = res.atom_range.start;
    while (a < end) : (a += 1) {
        if (topology.atoms[a].name.eqlSlice(name)) return a;
    }
    return null;
}

/// Check if two residues are in the same chain.
fn sameChain(topology: types.Topology, r1: u32, r2: u32) bool {
    // Find chain for each residue
    for (topology.chains) |chain| {
        const start = chain.residue_range.start;
        const end = chain.residue_range.end();
        const in1 = r1 >= start and r1 < end;
        const in2 = r2 >= start and r2 < end;
        if (in1 and in2) return true;
        if (in1 or in2) return false;
    }
    return false;
}

/// Compute phi angles (C(i-1) - N(i) - CA(i) - C(i)) for all residues.
///
/// Returns a slice of ?f32 (null if the angle is undefined, e.g., first
/// residue or chain break). Values are in radians [-pi, pi].
/// Caller owns the returned slice.
pub fn computePhi(
    allocator: std.mem.Allocator,
    topology: types.Topology,
    frame: types.Frame,
) ![]?f32 {
    const n_res: u32 = @intCast(topology.residues.len);
    const result = try allocator.alloc(?f32, n_res);

    for (0..n_res) |ri| {
        const r: u32 = @intCast(ri);
        if (r == 0 or !sameChain(topology, r - 1, r)) {
            result[ri] = null;
            continue;
        }
        const c_prev = findAtom(topology, r - 1, "C");
        const n = findAtom(topology, r, "N");
        const ca = findAtom(topology, r, "CA");
        const c = findAtom(topology, r, "C");

        if (c_prev != null and n != null and ca != null and c != null) {
            var out: [1]f32 = undefined;
            dihedrals.compute(frame.x, frame.y, frame.z, &.{.{ c_prev.?, n.?, ca.?, c.? }}, &out);
            result[ri] = out[0];
        } else {
            result[ri] = null;
        }
    }

    return result;
}

/// Compute psi angles (N(i) - CA(i) - C(i) - N(i+1)) for all residues.
pub fn computePsi(
    allocator: std.mem.Allocator,
    topology: types.Topology,
    frame: types.Frame,
) ![]?f32 {
    const n_res: u32 = @intCast(topology.residues.len);
    const result = try allocator.alloc(?f32, n_res);

    for (0..n_res) |ri| {
        const r: u32 = @intCast(ri);
        if (r + 1 >= n_res or !sameChain(topology, r, r + 1)) {
            result[ri] = null;
            continue;
        }
        const n = findAtom(topology, r, "N");
        const ca = findAtom(topology, r, "CA");
        const c = findAtom(topology, r, "C");
        const n_next = findAtom(topology, r + 1, "N");

        if (n != null and ca != null and c != null and n_next != null) {
            var out: [1]f32 = undefined;
            dihedrals.compute(frame.x, frame.y, frame.z, &.{.{ n.?, ca.?, c.?, n_next.? }}, &out);
            result[ri] = out[0];
        } else {
            result[ri] = null;
        }
    }

    return result;
}

/// Compute omega angles (CA(i) - C(i) - N(i+1) - CA(i+1)) for all residues.
pub fn computeOmega(
    allocator: std.mem.Allocator,
    topology: types.Topology,
    frame: types.Frame,
) ![]?f32 {
    const n_res: u32 = @intCast(topology.residues.len);
    const result = try allocator.alloc(?f32, n_res);

    for (0..n_res) |ri| {
        const r: u32 = @intCast(ri);
        if (r + 1 >= n_res or !sameChain(topology, r, r + 1)) {
            result[ri] = null;
            continue;
        }
        const ca = findAtom(topology, r, "CA");
        const c = findAtom(topology, r, "C");
        const n_next = findAtom(topology, r + 1, "N");
        const ca_next = findAtom(topology, r + 1, "CA");

        if (ca != null and c != null and n_next != null and ca_next != null) {
            var out: [1]f32 = undefined;
            dihedrals.compute(frame.x, frame.y, frame.z, &.{.{ ca.?, c.?, n_next.?, ca_next.? }}, &out);
            result[ri] = out[0];
        } else {
            result[ri] = null;
        }
    }

    return result;
}

// ============================================================================
// Chi angles
// ============================================================================

/// Compute chi angle for a given level (1-4) for all residues.
///
/// Tries multiple atom name patterns per residue. Returns null for residues
/// that don't have the required side-chain atoms (e.g., Gly for chi1).
pub fn computeChi(
    allocator: std.mem.Allocator,
    topology: types.Topology,
    frame: types.Frame,
    level: u8,
) ![]?f32 {
    const patterns: []const AtomPattern = switch (level) {
        1 => &chi1_patterns,
        2 => &chi2_patterns,
        3 => &chi3_patterns,
        4 => &chi4_patterns,
        else => return error.InvalidChiLevel,
    };

    const n_res: u32 = @intCast(topology.residues.len);
    const result = try allocator.alloc(?f32, n_res);

    for (0..n_res) |ri| {
        const r: u32 = @intCast(ri);
        result[ri] = null;

        for (patterns) |pattern| {
            const a0 = findAtom(topology, r, pattern[0]);
            const a1 = findAtom(topology, r, pattern[1]);
            const a2 = findAtom(topology, r, pattern[2]);
            const a3 = findAtom(topology, r, pattern[3]);

            if (a0 != null and a1 != null and a2 != null and a3 != null) {
                var out: [1]f32 = undefined;
                dihedrals.compute(frame.x, frame.y, frame.z, &.{.{ a0.?, a1.?, a2.?, a3.? }}, &out);
                result[ri] = out[0];
                break;
            }
        }
    }

    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "computePhi: first residue is null" {
    const allocator = std.testing.allocator;

    const topo_sizes = types.TopologySizes{ .n_atoms = 8, .n_residues = 2, .n_chains = 1, .n_bonds = 0 };
    var topo = try types.Topology.init(allocator, topo_sizes);
    defer topo.deinit();

    // Two residues with N, CA, C, O each
    topo.atoms[0] = .{ .name = types.FixedString(4).fromSlice("N"), .element = .N, .residue_index = 0 };
    topo.atoms[1] = .{ .name = types.FixedString(4).fromSlice("CA"), .element = .C, .residue_index = 0 };
    topo.atoms[2] = .{ .name = types.FixedString(4).fromSlice("C"), .element = .C, .residue_index = 0 };
    topo.atoms[3] = .{ .name = types.FixedString(4).fromSlice("O"), .element = .O, .residue_index = 0 };
    topo.atoms[4] = .{ .name = types.FixedString(4).fromSlice("N"), .element = .N, .residue_index = 1 };
    topo.atoms[5] = .{ .name = types.FixedString(4).fromSlice("CA"), .element = .C, .residue_index = 1 };
    topo.atoms[6] = .{ .name = types.FixedString(4).fromSlice("C"), .element = .C, .residue_index = 1 };
    topo.atoms[7] = .{ .name = types.FixedString(4).fromSlice("O"), .element = .O, .residue_index = 1 };

    topo.residues[0] = .{ .name = types.FixedString(5).fromSlice("ALA"), .chain_index = 0, .atom_range = .{ .start = 0, .len = 4 }, .resid = 1 };
    topo.residues[1] = .{ .name = types.FixedString(5).fromSlice("ALA"), .chain_index = 0, .atom_range = .{ .start = 4, .len = 4 }, .resid = 2 };
    topo.chains[0] = .{ .name = types.FixedString(4).fromSlice("A"), .residue_range = .{ .start = 0, .len = 2 } };

    var frame = try types.Frame.init(allocator, 8);
    defer frame.deinit();
    // Place atoms along x-axis with slight bends
    frame.x[0] = 0.0;
    frame.y[0] = 0.0;
    frame.z[0] = 0.0;
    frame.x[1] = 1.5;
    frame.y[1] = 0.0;
    frame.z[1] = 0.0;
    frame.x[2] = 2.5;
    frame.y[2] = 0.5;
    frame.z[2] = 0.0;
    frame.x[3] = 2.5;
    frame.y[3] = 1.5;
    frame.z[3] = 0.0;
    frame.x[4] = 3.5;
    frame.y[4] = 0.0;
    frame.z[4] = 0.0;
    frame.x[5] = 5.0;
    frame.y[5] = 0.0;
    frame.z[5] = 0.0;
    frame.x[6] = 6.0;
    frame.y[6] = 0.5;
    frame.z[6] = 0.0;
    frame.x[7] = 6.0;
    frame.y[7] = 1.5;
    frame.z[7] = 0.0;

    const phi = try computePhi(allocator, topo, frame);
    defer allocator.free(phi);

    try std.testing.expect(phi[0] == null); // First residue has no phi
    try std.testing.expect(phi[1] != null); // Second residue has phi

    const psi = try computePsi(allocator, topo, frame);
    defer allocator.free(psi);

    try std.testing.expect(psi[0] != null); // First residue has psi
    try std.testing.expect(psi[1] == null); // Last residue has no psi
}

test "computeChi: glycine has no chi1" {
    const allocator = std.testing.allocator;

    const topo_sizes = types.TopologySizes{ .n_atoms = 4, .n_residues = 1, .n_chains = 1, .n_bonds = 0 };
    var topo = try types.Topology.init(allocator, topo_sizes);
    defer topo.deinit();

    topo.atoms[0] = .{ .name = types.FixedString(4).fromSlice("N"), .element = .N, .residue_index = 0 };
    topo.atoms[1] = .{ .name = types.FixedString(4).fromSlice("CA"), .element = .C, .residue_index = 0 };
    topo.atoms[2] = .{ .name = types.FixedString(4).fromSlice("C"), .element = .C, .residue_index = 0 };
    topo.atoms[3] = .{ .name = types.FixedString(4).fromSlice("O"), .element = .O, .residue_index = 0 };

    topo.residues[0] = .{ .name = types.FixedString(5).fromSlice("GLY"), .chain_index = 0, .atom_range = .{ .start = 0, .len = 4 }, .resid = 1 };
    topo.chains[0] = .{ .name = types.FixedString(4).fromSlice("A"), .residue_range = .{ .start = 0, .len = 1 } };

    var frame = try types.Frame.init(allocator, 4);
    defer frame.deinit();
    frame.x[0] = 0.0;
    frame.x[1] = 1.5;
    frame.x[2] = 3.0;
    frame.x[3] = 3.0;
    frame.y[3] = 1.2;

    const chi1 = try computeChi(allocator, topo, frame, 1);
    defer allocator.free(chi1);

    try std.testing.expect(chi1[0] == null); // Gly has no CB → no chi1
}
