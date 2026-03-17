//! Hydrogen bond detection using Baker-Hubbard criteria.
//!
//! A hydrogen bond D-H...A is detected when:
//! - D (donor) is N or O covalently bonded to H
//! - A (acceptor) is N, O, or S
//! - H...A distance < dist_cutoff (default 2.5 Å)
//! - D-H...A angle > angle_cutoff (default 120°)
//!
//! Reference: Baker & Hubbard (1984) Prog. Biophys. Mol. Biol. 44, 97–179.

const std = @import("std");
const types = @import("../types.zig");

// ============================================================================
// Public types
// ============================================================================

/// A detected hydrogen bond.
pub const HBond = struct {
    /// Index of the donor heavy atom (N or O) in the topology.
    donor: u32,
    /// Index of the hydrogen atom in the topology.
    hydrogen: u32,
    /// Index of the acceptor atom (N, O, or S) in the topology.
    acceptor: u32,
    /// H...A distance in Angstroms.
    distance: f32,
    /// D-H...A angle in degrees.
    angle: f32,
};

/// Detection parameters.
pub const Config = struct {
    /// Maximum H...A distance to consider (Å). Default: 2.5.
    dist_cutoff: f32 = 2.5,
    /// Minimum D-H...A angle to consider (degrees). Default: 120.0.
    angle_cutoff: f32 = 120.0,
};

// ============================================================================
// Detection
// ============================================================================

/// Detect hydrogen bonds in a single frame using the Baker-Hubbard criteria.
///
/// Iterates over all covalent bonds in `topology` looking for D-H pairs where
/// D is N or O, then tests every potential acceptor (N, O, S) against the
/// distance and angle thresholds.
///
/// The returned slice is owned by the caller; free with `allocator.free()`.
pub fn detect(
    allocator: std.mem.Allocator,
    topology: types.Topology,
    frame: types.Frame,
    config: Config,
) ![]HBond {
    var result = std.ArrayList(HBond){};
    errdefer result.deinit(allocator);

    const n_atoms = topology.atoms.len;

    for (topology.bonds) |bond| {
        const a1 = topology.atoms[bond.atom_i];
        const a2 = topology.atoms[bond.atom_j];

        // Identify which atom is H and which is the donor heavy atom.
        var donor_idx: u32 = undefined;
        var h_idx: u32 = undefined;
        const is_dh: bool = blk: {
            if (a1.element == .H and (a2.element == .N or a2.element == .O)) {
                h_idx = bond.atom_i;
                donor_idx = bond.atom_j;
                break :blk true;
            }
            if (a2.element == .H and (a1.element == .N or a1.element == .O)) {
                h_idx = bond.atom_j;
                donor_idx = bond.atom_i;
                break :blk true;
            }
            break :blk false;
        };

        if (!is_dh) continue;

        // Pre-fetch H position in f64 to avoid repeated conversions.
        const hx: f64 = @floatCast(frame.x[h_idx]);
        const hy: f64 = @floatCast(frame.y[h_idx]);
        const hz: f64 = @floatCast(frame.z[h_idx]);

        // D-H vector (donor minus hydrogen).
        const v1x: f64 = @as(f64, frame.x[donor_idx]) - hx;
        const v1y: f64 = @as(f64, frame.y[donor_idx]) - hy;
        const v1z: f64 = @as(f64, frame.z[donor_idx]) - hz;
        const mag1 = @sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
        if (mag1 < 1e-10) continue;

        for (0..n_atoms) |acc_raw| {
            const acc_idx: u32 = @intCast(acc_raw);
            if (acc_idx == donor_idx or acc_idx == h_idx) continue;

            const acc_elem = topology.atoms[acc_idx].element;
            if (acc_elem != .N and acc_elem != .O and acc_elem != .S) continue;

            // H...A vector.
            const v2x: f64 = @as(f64, frame.x[acc_idx]) - hx;
            const v2y: f64 = @as(f64, frame.y[acc_idx]) - hy;
            const v2z: f64 = @as(f64, frame.z[acc_idx]) - hz;

            // H...A distance.
            const dist_sq = v2x * v2x + v2y * v2y + v2z * v2z;
            const dist: f32 = @floatCast(@sqrt(dist_sq));
            if (dist > config.dist_cutoff) continue;

            const mag2 = @sqrt(dist_sq);
            if (mag2 < 1e-10) continue;

            // D-H...A angle via dot product.
            const dot_val = v1x * v2x + v1y * v2y + v1z * v2z;
            const cos_angle = std.math.clamp(dot_val / (mag1 * mag2), -1.0, 1.0);
            const angle_rad = std.math.acos(cos_angle);
            const angle_deg: f32 = @floatCast(angle_rad * (180.0 / std.math.pi));

            if (angle_deg >= config.angle_cutoff) {
                try result.append(allocator, .{
                    .donor = donor_idx,
                    .hydrogen = h_idx,
                    .acceptor = acc_idx,
                    .distance = dist,
                    .angle = angle_deg,
                });
            }
        }
    }

    return result.toOwnedSlice(allocator);
}

// ============================================================================
// Tests
// ============================================================================

/// Build a minimal Topology + Frame for testing.
/// Layout:
///   atom 0: donor N at origin
///   atom 1: H bonded to donor
///   atom 2: acceptor O
/// Bond: 0-1 (N-H).
fn makeTestSystem(
    allocator: std.mem.Allocator,
    hx: f32,
    hy: f32,
    hz: f32,
    ax: f32,
    ay: f32,
    az: f32,
) !struct { topo: types.Topology, frame: types.Frame } {
    var topo = try types.Topology.init(allocator, .{
        .n_atoms = 3,
        .n_residues = 1,
        .n_chains = 1,
        .n_bonds = 1,
    });
    errdefer topo.deinit();

    const FS4 = types.FixedString(4);
    const FS5 = types.FixedString(5);

    topo.atoms[0] = .{ .name = FS4.fromSlice("N"), .element = .N, .residue_index = 0 };
    topo.atoms[1] = .{ .name = FS4.fromSlice("H"), .element = .H, .residue_index = 0 };
    topo.atoms[2] = .{ .name = FS4.fromSlice("O"), .element = .O, .residue_index = 0 };

    topo.residues[0] = .{
        .name = FS5.fromSlice("ALA"),
        .chain_index = 0,
        .atom_range = .{ .start = 0, .len = 3 },
        .resid = 1,
    };
    topo.chains[0] = .{
        .name = FS4.fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 1 },
    };
    topo.bonds[0] = .{ .atom_i = 0, .atom_j = 1 }; // N-H bond

    var frame = try types.Frame.init(allocator, 3);
    errdefer frame.deinit();

    // Donor N at origin.
    frame.x[0] = 0.0;
    frame.y[0] = 0.0;
    frame.z[0] = 0.0;
    // Hydrogen.
    frame.x[1] = hx;
    frame.y[1] = hy;
    frame.z[1] = hz;
    // Acceptor O.
    frame.x[2] = ax;
    frame.y[2] = ay;
    frame.z[2] = az;

    return .{ .topo = topo, .frame = frame };
}

test "hbonds: detects valid N-H...O bond" {
    // Linear geometry: N at (0,0,0), H at (1,0,0), O at (2.5,0,0).
    // H...O distance = 1.5 Å, D-H...A angle = 180°.
    const allocator = std.testing.allocator;

    var sys = try makeTestSystem(allocator, 1.0, 0.0, 0.0, 2.5, 0.0, 0.0);
    defer sys.topo.deinit();
    defer sys.frame.deinit();

    const bonds = try detect(allocator, sys.topo, sys.frame, .{});
    defer allocator.free(bonds);

    try std.testing.expectEqual(@as(usize, 1), bonds.len);
    try std.testing.expectEqual(@as(u32, 0), bonds[0].donor);
    try std.testing.expectEqual(@as(u32, 1), bonds[0].hydrogen);
    try std.testing.expectEqual(@as(u32, 2), bonds[0].acceptor);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), bonds[0].distance, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 180.0), bonds[0].angle, 1e-2);
}

test "hbonds: H...A distance too large — no bond" {
    // H...O = 3.5 Å (> default cutoff of 2.5).
    const allocator = std.testing.allocator;

    var sys = try makeTestSystem(allocator, 1.0, 0.0, 0.0, 4.5, 0.0, 0.0);
    defer sys.topo.deinit();
    defer sys.frame.deinit();

    const bonds = try detect(allocator, sys.topo, sys.frame, .{});
    defer allocator.free(bonds);

    try std.testing.expectEqual(@as(usize, 0), bonds.len);
}

test "hbonds: angle too small — no bond" {
    // N at origin, H at (1,0,0), O at (1,2,0).
    // D-H...A angle ~ 63.4° (< 120° threshold).
    const allocator = std.testing.allocator;

    var sys = try makeTestSystem(allocator, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0);
    defer sys.topo.deinit();
    defer sys.frame.deinit();

    const bonds = try detect(allocator, sys.topo, sys.frame, .{});
    defer allocator.free(bonds);

    try std.testing.expectEqual(@as(usize, 0), bonds.len);
}

test "hbonds: non-donor atom bonded to H — no bond detected" {
    // Carbon bonded to H should not be treated as a donor.
    const allocator = std.testing.allocator;

    var topo = try types.Topology.init(allocator, .{
        .n_atoms = 3,
        .n_residues = 1,
        .n_chains = 1,
        .n_bonds = 1,
    });
    defer topo.deinit();

    const FS4 = types.FixedString(4);
    const FS5 = types.FixedString(5);

    topo.atoms[0] = .{ .name = FS4.fromSlice("C"), .element = .C, .residue_index = 0 };
    topo.atoms[1] = .{ .name = FS4.fromSlice("H"), .element = .H, .residue_index = 0 };
    topo.atoms[2] = .{ .name = FS4.fromSlice("O"), .element = .O, .residue_index = 0 };
    topo.residues[0] = .{
        .name = FS5.fromSlice("ALA"),
        .chain_index = 0,
        .atom_range = .{ .start = 0, .len = 3 },
        .resid = 1,
    };
    topo.chains[0] = .{
        .name = FS4.fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 1 },
    };
    topo.bonds[0] = .{ .atom_i = 0, .atom_j = 1 }; // C-H bond

    var frame = try types.Frame.init(allocator, 3);
    defer frame.deinit();
    frame.x[0] = 0.0; frame.y[0] = 0.0; frame.z[0] = 0.0; // C
    frame.x[1] = 1.0; frame.y[1] = 0.0; frame.z[1] = 0.0; // H
    frame.x[2] = 2.3; frame.y[2] = 0.0; frame.z[2] = 0.0; // O

    const bonds = try detect(allocator, topo, frame, .{});
    defer allocator.free(bonds);

    try std.testing.expectEqual(@as(usize, 0), bonds.len);
}

test "hbonds: sulfur acceptor is detected" {
    // N-H...S should trigger on a sulfur acceptor.
    const allocator = std.testing.allocator;

    var topo = try types.Topology.init(allocator, .{
        .n_atoms = 3,
        .n_residues = 1,
        .n_chains = 1,
        .n_bonds = 1,
    });
    defer topo.deinit();

    const FS4 = types.FixedString(4);
    const FS5 = types.FixedString(5);

    topo.atoms[0] = .{ .name = FS4.fromSlice("N"), .element = .N, .residue_index = 0 };
    topo.atoms[1] = .{ .name = FS4.fromSlice("H"), .element = .H, .residue_index = 0 };
    topo.atoms[2] = .{ .name = FS4.fromSlice("S"), .element = .S, .residue_index = 0 };
    topo.residues[0] = .{
        .name = FS5.fromSlice("MET"),
        .chain_index = 0,
        .atom_range = .{ .start = 0, .len = 3 },
        .resid = 1,
    };
    topo.chains[0] = .{
        .name = FS4.fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 1 },
    };
    topo.bonds[0] = .{ .atom_i = 0, .atom_j = 1 };

    var frame = try types.Frame.init(allocator, 3);
    defer frame.deinit();
    frame.x[0] = 0.0; frame.y[0] = 0.0; frame.z[0] = 0.0;
    frame.x[1] = 1.0; frame.y[1] = 0.0; frame.z[1] = 0.0;
    frame.x[2] = 2.3; frame.y[2] = 0.0; frame.z[2] = 0.0;

    const bonds = try detect(allocator, topo, frame, .{});
    defer allocator.free(bonds);

    try std.testing.expectEqual(@as(usize, 1), bonds.len);
    try std.testing.expectEqual(@as(u32, 2), bonds[0].acceptor);
}

test "hbonds: no bonds in empty topology" {
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

    const bonds = try detect(allocator, topo, frame, .{});
    defer allocator.free(bonds);

    try std.testing.expectEqual(@as(usize, 0), bonds.len);
}
