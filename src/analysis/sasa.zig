//! Solvent Accessible Surface Area (SASA) via Shrake-Rupley algorithm.
//!
//! Wraps zsasa's Shrake-Rupley implementation with ztraj's coordinate
//! and topology conventions. Uses element-based van der Waals radii
//! from ztraj's element table.

const std = @import("std");
const types = @import("../types.zig");
const element_mod = @import("../element.zig");
const zsasa = @import("zsasa");

const AtomInput = zsasa.types.AtomInput;
const Config = zsasa.types.Config;
const SasaResult = zsasa.types.SasaResult;
const shrake_rupley = zsasa.shrake_rupley;

/// SASA calculation parameters.
pub const SasaConfig = struct {
    /// Number of test points per atom sphere. Default: 100.
    n_points: u32 = 100,
    /// Probe radius in Angstroms. Default: 1.4 (water).
    probe_radius: f64 = 1.4,
    /// Number of threads (0 = auto-detect).
    n_threads: usize = 0,
};

/// SASA result owning its memory.
pub const SasaOutput = struct {
    /// Total SASA in square Angstroms.
    total_area: f64,
    /// Per-atom SASA in square Angstroms (length = n_atoms).
    atom_areas: []f64,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *SasaOutput) void {
        self.allocator.free(self.atom_areas);
    }
};

/// Compute SASA for a set of atoms using the Shrake-Rupley algorithm.
///
/// Coordinates are in Angstroms (SOA layout). Radii are derived from
/// element-based van der Waals radii via `topology.atoms[i].element.vdwRadius()`.
///
/// If `atom_indices` is non-null, only those atoms are included in the
/// calculation. Otherwise all atoms are used.
pub fn compute(
    allocator: std.mem.Allocator,
    x: []const f32,
    y: []const f32,
    z: []const f32,
    topology: types.Topology,
    atom_indices: ?[]const u32,
    config: SasaConfig,
) !SasaOutput {
    const n_selected = if (atom_indices) |idx| idx.len else x.len;
    if (n_selected == 0) return error.NoAtoms;

    // Convert f32 SOA coords to f64 for zsasa
    const x64 = try allocator.alloc(f64, n_selected);
    defer allocator.free(x64);
    const y64 = try allocator.alloc(f64, n_selected);
    defer allocator.free(y64);
    const z64 = try allocator.alloc(f64, n_selected);
    defer allocator.free(z64);
    const radii = try allocator.alloc(f64, n_selected);
    defer allocator.free(radii);

    if (atom_indices) |indices| {
        for (indices, 0..) |idx, i| {
            x64[i] = @floatCast(x[idx]);
            y64[i] = @floatCast(y[idx]);
            z64[i] = @floatCast(z[idx]);
            radii[i] = topology.atoms[idx].element.vdwRadius();
        }
    } else {
        for (0..n_selected) |i| {
            x64[i] = @floatCast(x[i]);
            y64[i] = @floatCast(y[i]);
            z64[i] = @floatCast(z[i]);
            radii[i] = topology.atoms[i].element.vdwRadius();
        }
    }

    const input = AtomInput{
        .x = x64,
        .y = y64,
        .z = z64,
        .r = radii,
        .residue = null,
        .atom_name = null,
        .element = null,
        .chain_id = null,
        .residue_num = null,
        .insertion_code = null,
        .allocator = allocator,
    };

    const zsasa_config = Config{
        .n_points = config.n_points,
        .probe_radius = config.probe_radius,
    };

    var result = if (config.n_threads == 1)
        try shrake_rupley.calculateSasa(allocator, input, zsasa_config)
    else
        try shrake_rupley.calculateSasaParallel(allocator, input, zsasa_config, config.n_threads);

    // Copy atom_areas out (zsasa result owns its own copy)
    const atom_areas = try allocator.alloc(f64, n_selected);
    @memcpy(atom_areas, result.atom_areas);
    const total = result.total_area;
    result.deinit();

    return SasaOutput{
        .total_area = total,
        .atom_areas = atom_areas,
        .allocator = allocator,
    };
}

// =============================================================================
// Tests
// =============================================================================

test "sasa: single atom has expected surface area" {
    const allocator = std.testing.allocator;

    // Single carbon atom: VdW radius 1.70 Å, probe 1.4 Å
    // Expected area ≈ 4π(1.70+1.4)² ≈ 4π(3.1)² ≈ 120.76 Å²
    const x = [_]f32{0.0};
    const y = [_]f32{0.0};
    const z = [_]f32{0.0};

    const topo_sizes = types.TopologySizes{
        .n_atoms = 1,
        .n_residues = 1,
        .n_chains = 1,
        .n_bonds = 0,
    };
    var topo = try types.Topology.init(allocator, topo_sizes);
    defer topo.deinit();

    topo.atoms[0] = .{
        .name = types.FixedString(4).fromSlice("C"),
        .element = .C,
        .residue_index = 0,
    };
    topo.residues[0] = .{
        .name = types.FixedString(5).fromSlice("ALA"),
        .chain_index = 0,
        .atom_range = .{ .start = 0, .len = 1 },
        .resid = 1,
    };
    topo.chains[0] = .{
        .name = types.FixedString(4).fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 1 },
    };

    var result = try compute(allocator, &x, &y, &z, topo, null, .{
        .n_points = 960,
        .probe_radius = 1.4,
        .n_threads = 1,
    });
    defer result.deinit();

    // 4π(1.70+1.4)² ≈ 120.76
    const expected = 4.0 * std.math.pi * (1.70 + 1.4) * (1.70 + 1.4);
    try std.testing.expectApproxEqRel(expected, result.total_area, 0.02);
    try std.testing.expectEqual(@as(usize, 1), result.atom_areas.len);
}

test "sasa: two overlapping atoms have less than 2x isolated area" {
    const allocator = std.testing.allocator;

    // Two carbon atoms close together — overlapping spheres
    const x = [_]f32{ 0.0, 2.0 };
    const y = [_]f32{ 0.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0 };

    const topo_sizes = types.TopologySizes{
        .n_atoms = 2,
        .n_residues = 1,
        .n_chains = 1,
        .n_bonds = 0,
    };
    var topo = try types.Topology.init(allocator, topo_sizes);
    defer topo.deinit();

    topo.atoms[0] = .{ .name = types.FixedString(4).fromSlice("C"), .element = .C, .residue_index = 0 };
    topo.atoms[1] = .{ .name = types.FixedString(4).fromSlice("C"), .element = .C, .residue_index = 0 };
    topo.residues[0] = .{
        .name = types.FixedString(5).fromSlice("ALA"),
        .chain_index = 0,
        .atom_range = .{ .start = 0, .len = 2 },
        .resid = 1,
    };
    topo.chains[0] = .{
        .name = types.FixedString(4).fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 1 },
    };

    var result = try compute(allocator, &x, &y, &z, topo, null, .{
        .n_points = 960,
        .probe_radius = 1.4,
        .n_threads = 1,
    });
    defer result.deinit();

    // Single isolated C: ~120.76 Å². Two overlapping should be < 2 * 120.76
    const single_area = 4.0 * std.math.pi * (1.70 + 1.4) * (1.70 + 1.4);
    try std.testing.expect(result.total_area < 2.0 * single_area);
    try std.testing.expect(result.total_area > single_area); // But more than one atom
}
