// Native DSSP secondary structure assignment for ztraj.
//
// Entry point: compute(allocator, topology, frame, config) → DsspResult
//
// Uses ztraj's own Topology and Frame types. Atom coordinates are read from
// Frame on demand via indices stored in DsspResidue — no copying.

const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;
const ztraj_types = @import("../../types.zig");
const types = @import("types.zig");
const backbone = @import("backbone.zig");
const hbond_mod = @import("hbond.zig");
const beta_sheet = @import("beta_sheet.zig");
const helix_mod = @import("helix.zig");

const Topology = ztraj_types.Topology;
const Frame = ztraj_types.Frame;
const DsspResidue = types.DsspResidue;
const DsspConfig = types.DsspConfig;
const DsspResult = types.DsspResult;
const Vec3f32 = types.Vec3f32;

// Re-export public types for callers who import this file
pub const DsspResidueT = DsspResidue;
pub const DsspConfigT = DsspConfig;
pub const DsspResultT = DsspResult;

// ============================================================================
// Geometry helpers (inline — use atom indices from Frame)
// ============================================================================

fn getPos(frame: Frame, idx: u32) Vec3f32 {
    return .{ .x = frame.x[idx], .y = frame.y[idx], .z = frame.z[idx] };
}

/// Dihedral angle between four points (degrees). Returns 360 when degenerate.
fn dihedralAngle(p1: Vec3f32, p2: Vec3f32, p3: Vec3f32, p4: Vec3f32) ?f32 {
    const v12 = p1.sub(p2);
    const v43 = p4.sub(p3);
    const z = p2.sub(p3);

    const p = z.cross(v12);
    const x = z.cross(v43);
    const y = z.cross(x);

    const u_sq = x.dot(x);
    const v_sq = y.dot(y);

    if (u_sq > 0 and v_sq > 0) {
        const u = p.dot(x) / @sqrt(u_sq);
        const v = p.dot(y) / @sqrt(v_sq);
        if (u != 0 or v != 0) {
            return math.atan2(v, u) * (180.0 / math.pi);
        }
    }
    return null;
}

/// Cosine of the angle between vectors (p1-p2) and (p3-p4). Returns null when degenerate.
fn cosinusAngle(p1: Vec3f32, p2: Vec3f32, p3: Vec3f32, p4: Vec3f32) ?f32 {
    const v12 = p1.sub(p2);
    const v34 = p3.sub(p4);
    const denom = v12.dot(v12) * v34.dot(v34);
    if (denom > 0) return v12.dot(v34) / @sqrt(denom);
    return null;
}

/// Virtual bond angle at CA(i) defined by CA(i-2), CA(i), CA(i+2) (degrees).
fn kappaAngle(ca_prev2: Vec3f32, ca: Vec3f32, ca_next2: Vec3f32) ?f32 {
    const cosine = cosinusAngle(ca, ca_prev2, ca_next2, ca) orelse return null;
    const clamped = math.clamp(cosine, @as(f32, -1.0), @as(f32, 1.0));
    return math.acos(clamped) * (180.0 / math.pi);
}

// ============================================================================
// Backbone geometry calculation
// ============================================================================

fn calculateGeometry(residues: []DsspResidue, frame: Frame) void {
    const n = residues.len;
    if (n == 0) return;

    for (residues, 0..) |*res, i| {
        // phi: C(i-1) - N(i) - CA(i) - C(i)
        if (i > 0 and res.chain_break == .none) {
            const prev = &residues[i - 1];
            res.phi = dihedralAngle(
                getPos(frame, prev.c_idx),
                getPos(frame, res.n_idx),
                getPos(frame, res.ca_idx),
                getPos(frame, res.c_idx),
            );
        }

        // psi: N(i) - CA(i) - C(i) - N(i+1)
        if (i + 1 < n and residues[i + 1].chain_break == .none) {
            const next = &residues[i + 1];
            res.psi = dihedralAngle(
                getPos(frame, res.n_idx),
                getPos(frame, res.ca_idx),
                getPos(frame, res.c_idx),
                getPos(frame, next.n_idx),
            );
        }

        // omega: CA(i) - C(i) - N(i+1) - CA(i+1)
        if (i + 1 < n and residues[i + 1].chain_break == .none) {
            const next = &residues[i + 1];
            res.omega = dihedralAngle(
                getPos(frame, res.ca_idx),
                getPos(frame, res.c_idx),
                getPos(frame, next.n_idx),
                getPos(frame, next.ca_idx),
            );
        }

        // alpha: CA(i-1) - CA(i) - CA(i+1) - CA(i+2)
        if (i > 0 and i + 2 < n) {
            if (backbone.noChainBreak(residues, @intCast(i - 1), @intCast(i + 2))) {
                res.alpha = dihedralAngle(
                    getPos(frame, residues[i - 1].ca_idx),
                    getPos(frame, res.ca_idx),
                    getPos(frame, residues[i + 1].ca_idx),
                    getPos(frame, residues[i + 2].ca_idx),
                );
            }
        }

        // kappa: virtual bond angle CA(i-2) - CA(i) - CA(i+2)
        if (i >= 2 and i + 2 < n) {
            if (backbone.noChainBreak(residues, @intCast(i - 2), @intCast(i + 2))) {
                res.kappa = kappaAngle(
                    getPos(frame, residues[i - 2].ca_idx),
                    getPos(frame, res.ca_idx),
                    getPos(frame, residues[i + 2].ca_idx),
                );
            }
        }

        // tco: cosine of angle between C=O(i) and C=O(i-1)
        if (i > 0 and res.chain_break == .none) {
            const prev = &residues[i - 1];
            res.tco = cosinusAngle(
                getPos(frame, res.c_idx),
                getPos(frame, res.o_idx),
                getPos(frame, prev.c_idx),
                getPos(frame, prev.o_idx),
            );
        }
    }
}

// ============================================================================
// Main entry point
// ============================================================================

/// Run the full DSSP algorithm on a ztraj Topology + Frame.
///
/// Orchestration order:
///  1. Extract backbone (atom indices for N, CA, C, O)
///  2. Filter to complete residues
///  3. Detect chain breaks
///  4. Number residues sequentially
///  5. Assign hydrogens
///  6. Calculate backbone geometry (phi, psi, omega, kappa, alpha, tco)
///  7. Find near pairs (CA–CA < 9 Å), sort
///  8. Calculate H-bond energies
///  9. Calculate beta sheets
/// 10. Calculate alpha/3-10/pi helices
/// 11. Calculate PP-II helices
///
/// Caller must call DsspResult.deinit() when done.
pub fn compute(
    allocator: Allocator,
    topology: Topology,
    frame: Frame,
    config: DsspConfig,
) !DsspResult {
    // Validate topology/frame consistency
    if (topology.atoms.len != frame.nAtoms()) return error.TopologyFrameMismatch;

    // Step 1: Extract backbone atom indices
    const all_residues = try backbone.extractBackbone(allocator, topology, frame);
    defer allocator.free(all_residues);

    // Step 2: Filter to complete residues only
    var complete_count: usize = 0;
    for (all_residues) |res| {
        if (res.complete) complete_count += 1;
    }

    if (complete_count == 0) return error.NoCompleteResidues;

    const residues = try allocator.alloc(DsspResidue, complete_count);
    errdefer allocator.free(residues);

    var wi: usize = 0;
    for (all_residues) |res| {
        if (res.complete) {
            residues[wi] = res;
            wi += 1;
        }
    }

    // Step 3: Detect chain breaks
    backbone.detectChainBreaks(residues, topology, frame);

    // Step 4: Number residues sequentially
    for (residues, 0..) |*res, i| {
        res.number = @intCast(i);
    }

    // Step 5: Assign hydrogens
    backbone.assignHydrogen(residues, frame);

    // Step 6: Calculate backbone geometry
    calculateGeometry(residues, frame);

    // Step 7: Find near pairs and sort
    const near_pairs = try hbond_mod.findNearPairsFast(residues, frame, allocator);
    defer allocator.free(near_pairs);

    const PairCmp = struct {
        fn lessThan(_: @This(), a: [2]u32, b: [2]u32) bool {
            if (a[0] != b[0]) return a[0] < b[0];
            return a[1] < b[1];
        }
    };
    std.mem.sort([2]u32, near_pairs, PairCmp{}, PairCmp.lessThan);

    // Step 8: Calculate H-bond energies
    hbond_mod.calculateHBondEnergies(residues, frame, near_pairs);

    // Step 9: Calculate beta sheets
    try beta_sheet.calculateBetaSheets(residues, near_pairs, allocator);

    // Step 10: Calculate alpha / 3-10 / pi helices
    helix_mod.calculateAlphaHelices(residues, config.prefer_pi_helices);

    // Step 11: Calculate PP-II helices
    helix_mod.calculatePPHelices(residues, config.pp_stretch);

    return DsspResult{
        .residues = residues,
        .allocator = allocator,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "compute - minimal two-residue chain" {
    const allocator = std.testing.allocator;

    // Build topology: 1 chain, 2 residues, 8 atoms (N CA C O × 2)
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

    // Residue 0: N=0, CA=1.5, C=2.5, O=(2.5,1.2)
    frame.x[0] = 0.0;
    frame.y[0] = 0.0;
    frame.z[0] = 0.0; // N
    frame.x[1] = 1.5;
    frame.y[1] = 0.0;
    frame.z[1] = 0.0; // CA
    frame.x[2] = 2.5;
    frame.y[2] = 0.0;
    frame.z[2] = 0.0; // C
    frame.x[3] = 2.5;
    frame.y[3] = 1.2;
    frame.z[3] = 0.0; // O

    // Residue 1: N=3.5, CA=5.0, C=6.0, O=(6.0,1.2)
    frame.x[4] = 3.5;
    frame.y[4] = 0.0;
    frame.z[4] = 0.0; // N
    frame.x[5] = 5.0;
    frame.y[5] = 0.0;
    frame.z[5] = 0.0; // CA
    frame.x[6] = 6.0;
    frame.y[6] = 0.0;
    frame.z[6] = 0.0; // C
    frame.x[7] = 6.0;
    frame.y[7] = 1.2;
    frame.z[7] = 0.0; // O

    var result = try compute(allocator, topo, frame, .{});
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 2), result.residues.len);
    try std.testing.expect(result.residues[0].complete);
    try std.testing.expect(result.residues[1].complete);
}

test "compute - chain break detected" {
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
        .chain_index = 1,
        .atom_range = .{ .start = 4, .len = 4 },
        .resid = 1,
    };

    var frame = try Frame.init(allocator, 8);
    defer frame.deinit();

    // Chain A residue
    frame.x[0] = 0.0;
    frame.x[1] = 1.5;
    frame.x[2] = 2.5;
    frame.x[3] = 2.5;
    frame.y[3] = 1.2;
    // Chain B residue (far away)
    frame.x[4] = 30.0;
    frame.x[5] = 31.5;
    frame.x[6] = 32.5;
    frame.x[7] = 32.5;
    frame.y[7] = 1.2;

    var result = try compute(allocator, topo, frame, .{});
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 2), result.residues.len);
    // Second residue should have chain break
    try std.testing.expect(result.residues[1].chain_break != .none);
}

test "compute - incomplete residue filtered out" {
    const allocator = std.testing.allocator;

    // 2 residues: first has all 4 atoms, second only 3 (missing O)
    var topo = try Topology.init(allocator, .{
        .n_atoms = 7,
        .n_residues = 2,
        .n_chains = 1,
        .n_bonds = 0,
    });
    defer topo.deinit();

    topo.chains[0] = .{
        .name = ztraj_types.FixedString(4).fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 2 },
    };

    const names4 = [_][]const u8{ "N", "CA", "C", "O" };
    for (topo.atoms[0..4], 0..) |*atom, i| {
        atom.* = .{
            .name = ztraj_types.FixedString(4).fromSlice(names4[i]),
            .element = .C,
            .residue_index = 0,
        };
    }
    const names3 = [_][]const u8{ "N", "CA", "C" };
    for (topo.atoms[4..7], 0..) |*atom, i| {
        atom.* = .{
            .name = ztraj_types.FixedString(4).fromSlice(names3[i]),
            .element = .C,
            .residue_index = 1,
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
        .atom_range = .{ .start = 4, .len = 3 },
        .resid = 2,
    };

    var frame = try Frame.init(allocator, 7);
    defer frame.deinit();

    var result = try compute(allocator, topo, frame, .{});
    defer result.deinit();

    // Second residue was incomplete; only 1 residue should remain
    try std.testing.expectEqual(@as(usize, 1), result.residues.len);
}

test "dihedralAngle - coplanar points give 0 degrees" {
    const p1 = Vec3f32{ .x = 1.0, .y = 0.0, .z = 0.0 };
    const p2 = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const p3 = Vec3f32{ .x = 0.0, .y = 1.0, .z = 0.0 };
    const p4 = Vec3f32{ .x = 1.0, .y = 1.0, .z = 0.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), dihedralAngle(p1, p2, p3, p4).?, 0.01);
}

test "dihedralAngle - 90 degrees" {
    const p1 = Vec3f32{ .x = 1.0, .y = 0.0, .z = 0.0 };
    const p2 = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const p3 = Vec3f32{ .x = 0.0, .y = 1.0, .z = 0.0 };
    const p4 = Vec3f32{ .x = 0.0, .y = 1.0, .z = -1.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 90.0), dihedralAngle(p1, p2, p3, p4).?, 0.01);
}

test "cosinusAngle - parallel vectors" {
    const p1 = Vec3f32{ .x = 2.0, .y = 0.0, .z = 0.0 };
    const p2 = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const p3 = Vec3f32{ .x = 3.0, .y = 0.0, .z = 0.0 };
    const p4 = Vec3f32{ .x = 1.0, .y = 0.0, .z = 0.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cosinusAngle(p1, p2, p3, p4).?, 1e-6);
}

test "kappaAngle - straight chain gives 0 degrees" {
    const ca_prev2 = Vec3f32{ .x = -2.0, .y = 0.0, .z = 0.0 };
    const ca = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 };
    const ca_next2 = Vec3f32{ .x = 2.0, .y = 0.0, .z = 0.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), kappaAngle(ca_prev2, ca, ca_next2).?, 0.01);
}
