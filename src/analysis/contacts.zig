//! Residue-residue contact analysis.
//!
//! A contact between residues i and j exists when their representative
//! distance is below a given cutoff. Three measurement schemes are supported:
//!
//! - `closest`:       minimum distance between any atom pair (including H)
//! - `ca`:            distance between Cα atoms only
//! - `closest_heavy`: minimum distance excluding hydrogen atoms

const std = @import("std");
const types = @import("../types.zig");

// ============================================================================
// Public types
// ============================================================================

/// How to measure the inter-residue distance.
pub const Scheme = enum {
    /// Minimum distance between any atom pair across the two residues.
    closest,
    /// Distance between the Cα atoms of each residue.
    ca,
    /// Minimum distance between non-hydrogen atom pairs.
    closest_heavy,
};

/// A detected residue-residue contact.
pub const Contact = struct {
    /// 0-based index of the first residue.
    residue_i: u32,
    /// 0-based index of the second residue.
    residue_j: u32,
    /// Representative distance in Angstroms (scheme-dependent).
    distance: f32,
};

// ============================================================================
// Implementation helpers
// ============================================================================

/// Squared Euclidean distance between atoms `a` and `b` using f64 arithmetic.
inline fn distSq(frame: types.Frame, a: u32, b: u32) f64 {
    const dx: f64 = @as(f64, frame.x[b]) - @as(f64, frame.x[a]);
    const dy: f64 = @as(f64, frame.y[b]) - @as(f64, frame.y[a]);
    const dz: f64 = @as(f64, frame.z[b]) - @as(f64, frame.z[a]);
    return dx * dx + dy * dy + dz * dz;
}

/// Find the Cα atom index within an atom range, or null if not present.
fn findCa(topology: types.Topology, range: types.Range) ?u32 {
    const end = range.end();
    var i: u32 = range.start;
    while (i < end) : (i += 1) {
        if (topology.atoms[i].name.eqlSlice("CA")) return i;
    }
    return null;
}

/// Minimum distance between two residues under the given scheme.
/// Returns null when the scheme cannot be applied (e.g. missing Cα).
fn residueDistance(
    topology: types.Topology,
    frame: types.Frame,
    ri: types.Residue,
    rj: types.Residue,
    scheme: Scheme,
) ?f32 {
    switch (scheme) {
        .ca => {
            const ca_i = findCa(topology, ri.atom_range) orelse return null;
            const ca_j = findCa(topology, rj.atom_range) orelse return null;
            return @floatCast(@sqrt(distSq(frame, ca_i, ca_j)));
        },
        .closest, .closest_heavy => {
            var min_sq: f64 = std.math.floatMax(f64);
            const end_i = ri.atom_range.end();
            const end_j = rj.atom_range.end();
            var ai: u32 = ri.atom_range.start;
            while (ai < end_i) : (ai += 1) {
                if (scheme == .closest_heavy and topology.atoms[ai].element == .H) continue;
                var aj: u32 = rj.atom_range.start;
                while (aj < end_j) : (aj += 1) {
                    if (scheme == .closest_heavy and topology.atoms[aj].element == .H) continue;
                    const d2 = distSq(frame, ai, aj);
                    if (d2 < min_sq) min_sq = d2;
                }
            }
            if (min_sq == std.math.floatMax(f64)) return null;
            return @floatCast(@sqrt(min_sq));
        },
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Compute residue-residue contacts for all unique pairs (i < j).
///
/// Returns every pair whose representative distance (determined by `scheme`)
/// is strictly less than `cutoff` Angstroms.
///
/// The returned slice is owned by the caller; free with `allocator.free()`.
pub fn compute(
    allocator: std.mem.Allocator,
    topology: types.Topology,
    frame: types.Frame,
    scheme: Scheme,
    cutoff: f32,
) ![]Contact {
    var result = std.ArrayList(Contact){};
    errdefer result.deinit(allocator);

    const n_res = topology.residues.len;

    for (0..n_res) |ri_raw| {
        for (ri_raw + 1..n_res) |rj_raw| {
            const ri: u32 = @intCast(ri_raw);
            const rj: u32 = @intCast(rj_raw);

            const dist = residueDistance(
                topology,
                frame,
                topology.residues[ri],
                topology.residues[rj],
                scheme,
            ) orelse continue;

            if (dist < cutoff) {
                try result.append(allocator, .{
                    .residue_i = ri,
                    .residue_j = rj,
                    .distance = dist,
                });
            }
        }
    }

    return result.toOwnedSlice(allocator);
}

// ============================================================================
// Tests
// ============================================================================

/// Build a two-residue topology where residue 0 has atoms at `pos0` and
/// residue 1 has atoms at `pos1`. Each residue has a single Cα.
fn makeTwoResidueSystem(
    allocator: std.mem.Allocator,
    pos0: [3]f32,
    pos1: [3]f32,
) !struct { topo: types.Topology, frame: types.Frame } {
    // 2 atoms, 2 residues, 1 chain, 0 bonds.
    var topo = try types.Topology.init(allocator, .{
        .n_atoms = 2,
        .n_residues = 2,
        .n_chains = 1,
        .n_bonds = 0,
    });
    errdefer topo.deinit();

    const FS4 = types.FixedString(4);
    const FS5 = types.FixedString(5);

    topo.atoms[0] = .{ .name = FS4.fromSlice("CA"), .element = .C, .residue_index = 0 };
    topo.atoms[1] = .{ .name = FS4.fromSlice("CA"), .element = .C, .residue_index = 1 };

    topo.residues[0] = .{
        .name = FS5.fromSlice("ALA"),
        .chain_index = 0,
        .atom_range = .{ .start = 0, .len = 1 },
        .resid = 1,
    };
    topo.residues[1] = .{
        .name = FS5.fromSlice("GLY"),
        .chain_index = 0,
        .atom_range = .{ .start = 1, .len = 1 },
        .resid = 2,
    };
    topo.chains[0] = .{
        .name = FS4.fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 2 },
    };

    var frame = try types.Frame.init(allocator, 2);
    errdefer frame.deinit();

    frame.x[0] = pos0[0]; frame.y[0] = pos0[1]; frame.z[0] = pos0[2];
    frame.x[1] = pos1[0]; frame.y[1] = pos1[1]; frame.z[1] = pos1[2];

    return .{ .topo = topo, .frame = frame };
}

test "contacts: closest — two residues in contact" {
    const allocator = std.testing.allocator;

    var sys = try makeTwoResidueSystem(allocator, .{ 0.0, 0.0, 0.0 }, .{ 4.0, 0.0, 0.0 });
    defer sys.topo.deinit();
    defer sys.frame.deinit();

    const contacts = try compute(allocator, sys.topo, sys.frame, .closest, 5.0);
    defer allocator.free(contacts);

    try std.testing.expectEqual(@as(usize, 1), contacts.len);
    try std.testing.expectEqual(@as(u32, 0), contacts[0].residue_i);
    try std.testing.expectEqual(@as(u32, 1), contacts[0].residue_j);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), contacts[0].distance, 1e-4);
}

test "contacts: closest — two residues too far apart" {
    const allocator = std.testing.allocator;

    var sys = try makeTwoResidueSystem(allocator, .{ 0.0, 0.0, 0.0 }, .{ 10.0, 0.0, 0.0 });
    defer sys.topo.deinit();
    defer sys.frame.deinit();

    const contacts = try compute(allocator, sys.topo, sys.frame, .closest, 5.0);
    defer allocator.free(contacts);

    try std.testing.expectEqual(@as(usize, 0), contacts.len);
}

test "contacts: ca scheme picks CA atoms" {
    const allocator = std.testing.allocator;

    // Residue 0: CA at (0,0,0) + sidechain C at (10,10,10) (far from res 1).
    // Residue 1: CA at (3,0,0).
    // CA distance = 3.0 Å; closest distance would be ~3.0 as well here,
    // but we verify the CA scheme returns the correct distance.
    var topo = try types.Topology.init(allocator, .{
        .n_atoms = 3,
        .n_residues = 2,
        .n_chains = 1,
        .n_bonds = 0,
    });
    defer topo.deinit();

    const FS4 = types.FixedString(4);
    const FS5 = types.FixedString(5);

    topo.atoms[0] = .{ .name = FS4.fromSlice("CA"), .element = .C, .residue_index = 0 };
    topo.atoms[1] = .{ .name = FS4.fromSlice("CB"), .element = .C, .residue_index = 0 };
    topo.atoms[2] = .{ .name = FS4.fromSlice("CA"), .element = .C, .residue_index = 1 };

    topo.residues[0] = .{
        .name = FS5.fromSlice("ALA"),
        .chain_index = 0,
        .atom_range = .{ .start = 0, .len = 2 },
        .resid = 1,
    };
    topo.residues[1] = .{
        .name = FS5.fromSlice("GLY"),
        .chain_index = 0,
        .atom_range = .{ .start = 2, .len = 1 },
        .resid = 2,
    };
    topo.chains[0] = .{
        .name = FS4.fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 2 },
    };

    var frame = try types.Frame.init(allocator, 3);
    defer frame.deinit();
    frame.x[0] = 0.0; frame.y[0] = 0.0; frame.z[0] = 0.0; // CA res 0
    frame.x[1] = 0.0; frame.y[1] = 0.0; frame.z[1] = 0.0; // CB res 0 (same pos)
    frame.x[2] = 3.0; frame.y[2] = 0.0; frame.z[2] = 0.0; // CA res 1

    const contacts = try compute(allocator, topo, frame, .ca, 5.0);
    defer allocator.free(contacts);

    try std.testing.expectEqual(@as(usize, 1), contacts.len);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), contacts[0].distance, 1e-4);
}

test "contacts: closest_heavy excludes hydrogen" {
    // Res 0: heavy C at (0,0,0), H at (0.5,0,0).
    // Res 1: heavy N at (3,0,0).
    // closest would be H...N = 2.5, closest_heavy would be C...N = 3.0.
    const allocator = std.testing.allocator;

    var topo = try types.Topology.init(allocator, .{
        .n_atoms = 3,
        .n_residues = 2,
        .n_chains = 1,
        .n_bonds = 0,
    });
    defer topo.deinit();

    const FS4 = types.FixedString(4);
    const FS5 = types.FixedString(5);

    topo.atoms[0] = .{ .name = FS4.fromSlice("CA"), .element = .C, .residue_index = 0 };
    topo.atoms[1] = .{ .name = FS4.fromSlice("H"), .element = .H, .residue_index = 0 };
    topo.atoms[2] = .{ .name = FS4.fromSlice("N"), .element = .N, .residue_index = 1 };

    topo.residues[0] = .{
        .name = FS5.fromSlice("ALA"),
        .chain_index = 0,
        .atom_range = .{ .start = 0, .len = 2 },
        .resid = 1,
    };
    topo.residues[1] = .{
        .name = FS5.fromSlice("GLY"),
        .chain_index = 0,
        .atom_range = .{ .start = 2, .len = 1 },
        .resid = 2,
    };
    topo.chains[0] = .{
        .name = FS4.fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 2 },
    };

    var frame = try types.Frame.init(allocator, 3);
    defer frame.deinit();
    frame.x[0] = 0.0; frame.y[0] = 0.0; frame.z[0] = 0.0; // CA res 0
    frame.x[1] = 0.5; frame.y[1] = 0.0; frame.z[1] = 0.0; // H  res 0
    frame.x[2] = 3.0; frame.y[2] = 0.0; frame.z[2] = 0.0; // N  res 1

    // closest_heavy: CA...N = 3.0, within cutoff 4.0.
    const contacts_h = try compute(allocator, topo, frame, .closest_heavy, 4.0);
    defer allocator.free(contacts_h);

    try std.testing.expectEqual(@as(usize, 1), contacts_h.len);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), contacts_h[0].distance, 1e-4);

    // closest: H...N = 2.5, within cutoff 2.6.
    const contacts_c = try compute(allocator, topo, frame, .closest, 2.6);
    defer allocator.free(contacts_c);

    try std.testing.expectEqual(@as(usize, 1), contacts_c.len);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), contacts_c[0].distance, 1e-4);

    // closest_heavy: CA...N = 3.0 > 2.6, so no contact.
    const contacts_nh = try compute(allocator, topo, frame, .closest_heavy, 2.6);
    defer allocator.free(contacts_nh);

    try std.testing.expectEqual(@as(usize, 0), contacts_nh.len);
}

test "contacts: ca scheme — missing CA returns no contact" {
    // Neither residue has a CA atom.
    const allocator = std.testing.allocator;

    var topo = try types.Topology.init(allocator, .{
        .n_atoms = 2,
        .n_residues = 2,
        .n_chains = 1,
        .n_bonds = 0,
    });
    defer topo.deinit();

    const FS4 = types.FixedString(4);
    const FS5 = types.FixedString(5);

    topo.atoms[0] = .{ .name = FS4.fromSlice("N"), .element = .N, .residue_index = 0 };
    topo.atoms[1] = .{ .name = FS4.fromSlice("N"), .element = .N, .residue_index = 1 };

    topo.residues[0] = .{
        .name = FS5.fromSlice("ALA"),
        .chain_index = 0,
        .atom_range = .{ .start = 0, .len = 1 },
        .resid = 1,
    };
    topo.residues[1] = .{
        .name = FS5.fromSlice("GLY"),
        .chain_index = 0,
        .atom_range = .{ .start = 1, .len = 1 },
        .resid = 2,
    };
    topo.chains[0] = .{
        .name = FS4.fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 2 },
    };

    var frame = try types.Frame.init(allocator, 2);
    defer frame.deinit();
    frame.x[0] = 0.0; frame.y[0] = 0.0; frame.z[0] = 0.0;
    frame.x[1] = 1.0; frame.y[1] = 0.0; frame.z[1] = 0.0;

    const contacts = try compute(allocator, topo, frame, .ca, 5.0);
    defer allocator.free(contacts);

    try std.testing.expectEqual(@as(usize, 0), contacts.len);
}

test "contacts: empty topology returns no contacts" {
    const allocator = std.testing.allocator;

    var topo = try types.Topology.init(allocator, .{
        .n_atoms = 0,
        .n_residues = 0,
        .n_chains = 0,
        .n_bonds = 0,
    });
    defer topo.deinit();

    var frame = try types.Frame.init(allocator, 0);
    defer frame.deinit();

    const contacts = try compute(allocator, topo, frame, .closest, 5.0);
    defer allocator.free(contacts);

    try std.testing.expectEqual(@as(usize, 0), contacts.len);
}
