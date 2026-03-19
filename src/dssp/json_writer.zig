const std = @import("std");
const Allocator = std.mem.Allocator;
const types = @import("types.zig");
const residue_mod = @import("residue.zig");
const dssp_mod = @import("dssp.zig");

const Residue = residue_mod.Residue;
const StructureType = types.StructureType;
const DsspResult = dssp_mod.DsspResult;
const Statistics = dssp_mod.Statistics;

// Threshold margin for warning (kcal/mol)
// Energies within this range of -0.5 may produce different results from mkdssp.
// This is intentionally conservative: we flag energies on BOTH sides of the threshold
// (-0.51 to -0.49) because:
// 1. Energies slightly above -0.5 (e.g., -0.49) could be computed as below -0.5 by mkdssp
// 2. Energies slightly below -0.5 (e.g., -0.51) could be computed as above -0.5 by mkdssp
// The 0.01 margin (~2 ULPs at f32 precision near -0.5) covers typical floating-point variance.
const kThresholdMargin: f32 = 0.01;

/// Calculate minimum distance from H-bond threshold (-0.5 kcal/mol).
/// Returns null if no valid H-bonds exist.
fn getThresholdMargin(res: *const Residue) ?f32 {
    const threshold = types.kMaxHBondEnergyF32; // -0.5
    var min_margin: ?f32 = null;

    // Check all H-bond slots
    const hbonds = [_]types.HBond{
        res.hbond_donor[0],
        res.hbond_donor[1],
        res.hbond_acceptor[0],
        res.hbond_acceptor[1],
    };

    for (hbonds) |hb| {
        // Only consider valid H-bonds with meaningful energy
        if (hb.residue_index != null and hb.energy != 0.0) {
            const margin = @abs(hb.energy - threshold);
            if (min_margin == null or margin < min_margin.?) {
                min_margin = margin;
            }
        }
    }

    return min_margin;
}

/// Check if residue has any H-bond energy near the threshold.
fn isNearThreshold(res: *const Residue) bool {
    if (getThresholdMargin(res)) |margin| {
        return margin < kThresholdMargin;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Output format
// ---------------------------------------------------------------------------

pub const OutputFormat = enum {
    json,
    legacy,
};

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

/// Write DSSP result as JSON.
pub fn writeJson(writer: anytype, result: *const DsspResult) !void {
    try writer.writeAll("{\n");

    // Statistics
    try writer.writeAll("  \"statistics\": {\n");
    try writer.print("    \"total_residues\": {d},\n", .{result.statistics.total_residues});
    try writer.print("    \"complete_residues\": {d},\n", .{result.statistics.complete_residues});
    try writer.print("    \"chain_breaks\": {d},\n", .{result.statistics.chain_breaks});
    try writer.print("    \"hbond_count\": {d},\n", .{result.statistics.hbond_count});
    try writer.print("    \"ss_bond_count\": {d}\n", .{result.statistics.ss_bond_count});
    try writer.writeAll("  },\n");

    // Collect near-threshold residues for warnings
    var near_threshold_count: u32 = 0;
    for (result.residues) |res| {
        if (isNearThreshold(&res)) {
            near_threshold_count += 1;
        }
    }

    // Warnings section (only if there are near-threshold residues)
    if (near_threshold_count > 0) {
        try writer.writeAll("  \"warnings\": {\n");
        try writer.print("    \"near_threshold_count\": {d},\n", .{near_threshold_count});
        try writer.writeAll("    \"near_threshold_residues\": [");
        var first = true;
        for (result.residues) |res| {
            if (isNearThreshold(&res)) {
                if (!first) try writer.writeAll(", ");
                try writer.print("\"{s}:{d}\"", .{ res.getChainId(), res.seq_id });
                first = false;
            }
        }
        try writer.writeAll("],\n");
        try writer.writeAll("    \"message\": \"These residues have H-bond energies within 0.01 kcal/mol of the -0.5 threshold. Secondary structure assignments may differ from mkdssp due to floating-point precision.\"\n");
        try writer.writeAll("  },\n");
    }

    // Residues
    try writer.writeAll("  \"residues\": [\n");
    for (result.residues, 0..) |res, i| {
        if (i > 0) try writer.writeAll(",\n");
        try writeResidueJson(writer, &res);
    }
    try writer.writeAll("\n  ]\n");
    try writer.writeAll("}\n");
}

fn writeResidueJson(writer: anytype, res: *const Residue) !void {
    try writer.writeAll("    {\n");
    try writer.print("      \"chain_id\": \"{s}\",\n", .{res.getChainId()});
    try writer.print("      \"seq_id\": {d},\n", .{res.seq_id});
    try writer.print("      \"compound_id\": \"{s}\",\n", .{res.getCompoundId()});
    try writer.print("      \"residue_type\": \"{c}\",\n", .{res.residue_type.toChar()});
    try writer.print("      \"secondary_structure\": \"{c}\",\n", .{res.secondary_structure.toChar()});
    try writer.print("      \"accessibility\": {d:.1},\n", .{res.accessibility});

    // Angles
    try writer.writeAll("      \"angles\": {\n");
    try writeOptionalAngle(writer, "phi", res.phi, true);
    try writeOptionalAngle(writer, "psi", res.psi, true);
    try writeOptionalAngle(writer, "omega", res.omega, true);
    try writeOptionalAngle(writer, "kappa", res.kappa, true);
    try writeOptionalAngle(writer, "alpha", res.alpha, true);
    try writeOptionalAngle(writer, "tco", res.tco, false);
    try writer.writeAll("\n      },\n");

    // H-bonds
    try writer.writeAll("      \"hbonds\": {\n");
    try writeHBond(writer, "donor_0", res.hbond_donor[0], true);
    try writeHBond(writer, "donor_1", res.hbond_donor[1], true);
    try writeHBond(writer, "acceptor_0", res.hbond_acceptor[0], true);
    try writeHBond(writer, "acceptor_1", res.hbond_acceptor[1], false);
    try writer.writeAll("\n      },\n");

    // Beta sheet info
    try writer.print("      \"sheet\": {d},\n", .{res.sheet});
    try writer.print("      \"strand\": {d},\n", .{res.strand});

    // Threshold warning fields
    const near_threshold = isNearThreshold(res);
    const near_threshold_str: []const u8 = if (near_threshold) "true" else "false";
    try writer.print("      \"near_threshold\": {s},\n", .{near_threshold_str});
    if (getThresholdMargin(res)) |margin| {
        try writer.print("      \"threshold_margin\": {d:.4},\n", .{margin});
    } else {
        try writer.writeAll("      \"threshold_margin\": null,\n");
    }

    // Complete flag
    const complete_str: []const u8 = if (res.complete) "true" else "false";
    try writer.print("      \"complete\": {s}\n", .{complete_str});

    try writer.writeAll("    }");
}

fn writeOptionalAngle(writer: anytype, name: []const u8, value: ?f32, trailing_comma: bool) !void {
    try writer.print("        \"{s}\": ", .{name});
    if (value) |v| {
        try writer.print("{d:.1}", .{v});
    } else {
        try writer.writeAll("null");
    }
    if (trailing_comma) try writer.writeAll(",");
}

fn writeHBond(writer: anytype, name: []const u8, hb: types.HBond, trailing_comma: bool) !void {
    try writer.print("        \"{s}\": {{\"residue\": ", .{name});
    if (hb.residue_index) |idx| {
        try writer.print("{d}", .{idx});
    } else {
        try writer.writeAll("null");
    }
    try writer.print(", \"energy\": {d:.3}}}", .{hb.energy});
    if (trailing_comma) try writer.writeAll(",");
}

// ---------------------------------------------------------------------------
// Legacy DSSP text output
// ---------------------------------------------------------------------------

/// Write DSSP result in legacy text format (fixed-width columns).
/// Matches C++ DSSP output format from dssp-io.cpp.
pub fn writeLegacy(writer: anytype, result: *const DsspResult) !void {
    // Header line
    try writer.writeAll("  #  RESIDUE AA STRUCTURE BP1 BP2  ACC     N-H-->O    O-->H-N    N-H-->O    O-->H-N    TCO  KAPPA ALPHA  PHI   PSI    X-CA   Y-CA   Z-CA\n");

    for (result.residues, 0..) |res, i| {
        // Columns: nr (5), seq_num (5), ins_code (1), strand_id (1), space, code (1)
        // seq_id is signed but print without sign prefix (just like C++)
        const seq_id_unsigned: u32 = if (res.seq_id < 0) 0 else @intCast(res.seq_id);
        try writer.print("{:>5}{:>5} {s} ", .{ i + 1, seq_id_unsigned, res.getChainId() });

        // Residue type - cysteines with SS-bridges get lowercase letters
        const code: u8 = if (res.residue_type == .cys and res.ss_bridge_nr > 0)
            'a' + @as(u8, @intCast((res.ss_bridge_nr - 1) % 26))
        else
            res.residue_type.toChar();
        try writer.print("{c}  ", .{code});

        // SS type character
        try writer.print("{c}", .{res.secondary_structure.toChar()});

        // Helix flags in C++ order: PP (helix[3]), 3-10 (helix[0]), alpha (helix[1]), pi (helix[2])
        const helix_order = [_]struct { idx: usize, mid: u8 }{
            .{ .idx = 3, .mid = 'P' }, // PP-II
            .{ .idx = 0, .mid = '3' }, // 3-10 helix
            .{ .idx = 1, .mid = '4' }, // alpha helix
            .{ .idx = 2, .mid = '5' }, // pi helix
        };
        for (helix_order) |ht| {
            const flag = res.helix_flags[ht.idx];
            const c: u8 = switch (flag) {
                .none => ' ',
                .start => '>',
                .end => '<',
                .start_and_end => 'X',
                .middle => ht.mid,
            };
            try writer.print("{c}", .{c});
        }

        // Bend character
        const bend_c: u8 = if (res.bend) 'S' else ' ';
        try writer.print("{c}", .{bend_c});

        // Chirality character (from alpha angle sign)
        const chirality: u8 = if (res.alpha) |alpha| (if (alpha < 0) '-' else '+') else ' ';
        try writer.print("{c}", .{chirality});

        // Bridge labels
        for (res.beta_partner) |bp| {
            if (bp.residue_index != null and bp.ladder > 0) {
                const label: u8 = if (bp.parallel)
                    'a' + @as(u8, @intCast((bp.ladder - 1) % 26))
                else
                    'A' + @as(u8, @intCast((bp.ladder - 1) % 26));
                try writer.print("{c}", .{label});
            } else {
                try writer.writeAll(" ");
            }
        }

        // Bridge partners (BP1, BP2)
        for (res.beta_partner) |bp| {
            if (bp.residue_index) |idx| {
                try writer.print("{d:>4}", .{(idx + 1) % 10000});
            } else {
                try writer.writeAll("   0");
            }
        }

        // Sheet label
        const sheet_c: u8 = if (res.sheet > 0 and res.sheet <= 26)
            @as(u8, 'A') + @as(u8, @intCast((res.sheet - 1) % 26))
        else
            ' ';
        try writer.print("{c}", .{sheet_c});

        // Accessibility (rounded to integer, no sign)
        const acc_int: u32 = @intFromFloat(@floor(res.accessibility + 0.5));
        try writer.print("{:>4}", .{acc_int});

        // H-bonds in C++ order: NHO[0] (acceptor), ONH[0] (donor), NHO[1] (acceptor), ONH[1] (donor)
        // Each preceded by a space, then 11-char right-aligned string
        try writer.writeByte(' ');
        try writeHBondAcceptor(writer, res, 0, i);
        try writeHBondDonor(writer, res, 0, i);
        try writeHBondAcceptor(writer, res, 1, i);
        try writeHBondDonor(writer, res, 1, i);

        // Angles: TCO (3 decimals), kappa/alpha/phi/psi (1 decimal)
        try writeAngleLegacy(writer, res.tco, 3);
        try writeAngleLegacy(writer, res.kappa, 1);
        try writeAngleLegacy(writer, res.alpha, 1);
        try writeAngleLegacy(writer, res.phi, 1);
        try writeAngleLegacy(writer, res.psi, 1);

        // CA coordinates
        try writer.print(" {:>6.1} {:>6.1} {:>6.1}", .{ res.ca.x, res.ca.y, res.ca.z });
        try writer.writeAll("\n");
    }
}

/// Write N-H-->O bond (where this residue's NH donates to) in legacy format.
/// C++ format: {:>11s} containing "{:d},{:3.1f}" e.g. "    -4,-2.0"
fn writeHBondAcceptor(writer: anytype, res: Residue, slot: usize, self_idx: usize) !void {
    const hb = res.hbond_acceptor[slot];
    if (hb.residue_index) |idx| {
        const offset: i32 = @as(i32, @intCast(idx)) - @as(i32, @intCast(self_idx));
        // Format "{offset},{energy:.1}" then right-pad to 11 chars
        var buf: [16]u8 = undefined;
        const len = (std.fmt.bufPrint(&buf, "{d},{d:.1}", .{ offset, hb.energy }) catch unreachable).len;
        // Right-align in 11 chars
        const padding = 11 -| len;
        for (0..padding) |_| try writer.writeByte(' ');
        try writer.writeAll(buf[0..len]);
    } else {
        try writer.writeAll("     0, 0.0");
    }
}

/// Write O-->H-N bond (where this residue's CO accepts from) in legacy format.
fn writeHBondDonor(writer: anytype, res: Residue, slot: usize, self_idx: usize) !void {
    const hb = res.hbond_donor[slot];
    if (hb.residue_index) |idx| {
        const offset: i32 = @as(i32, @intCast(idx)) - @as(i32, @intCast(self_idx));
        var buf: [16]u8 = undefined;
        const len = (std.fmt.bufPrint(&buf, "{d},{d:.1}", .{ offset, hb.energy }) catch unreachable).len;
        const padding = 11 -| len;
        for (0..padding) |_| try writer.writeByte(' ');
        try writer.writeAll(buf[0..len]);
    } else {
        try writer.writeAll("     0, 0.0");
    }
}

fn writeAngleLegacy(writer: anytype, value: ?f32, precision: usize) !void {
    if (value) |v| {
        switch (precision) {
            3 => try writer.print("  {:>6.3}", .{v}),
            else => try writer.print("{:>6.1}", .{v}),
        }
    } else {
        // C++ uses 0.0 for undefined TCO (precision=3), 360.0 for other angles
        if (precision == 3) {
            try writer.writeAll("   0.000");
        } else {
            try writer.writeAll(" 360.0");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "writeJson - basic output" {
    const allocator = std.testing.allocator;

    var residues = [_]Residue{
        .{
            .chain_id = .{ 'A', ' ', ' ', ' ' },
            .chain_id_len = 1,
            .seq_id = 1,
            .compound_id = .{ 'A', 'L', 'A', ' ' },
            .compound_id_len = 3,
            .residue_type = .ala,
            .secondary_structure = .alpha_helix,
            .complete = true,
            .phi = -57.0,
            .psi = -47.0,
        },
    };

    const result = DsspResult{
        .residues = &residues,
        .ss_bonds = &.{},
        .statistics = .{ .total_residues = 1, .complete_residues = 1 },
        .side_chain_storage = &.{},
        .near_pairs = &.{},
        .allocator = allocator,
    };

    var buf: std.ArrayListAligned(u8, null) = .empty;
    defer buf.deinit(allocator);

    try writeJson(buf.writer(allocator), &result);

    const output = buf.items;
    try std.testing.expect(std.mem.indexOf(u8, output, "\"chain_id\": \"A\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "\"secondary_structure\": \"H\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "\"total_residues\": 1") != null);
}

test "writeLegacy - produces output" {
    const allocator = std.testing.allocator;

    var residues = [_]Residue{
        .{
            .chain_id = .{ 'A', ' ', ' ', ' ' },
            .chain_id_len = 1,
            .seq_id = 1,
            .compound_id = .{ 'A', 'L', 'A', ' ' },
            .compound_id_len = 3,
            .residue_type = .ala,
            .secondary_structure = .loop,
            .complete = true,
            .ca = types.Vec3f32{ .x = 1.5, .y = 2.0, .z = 3.0 },
        },
    };

    const result = DsspResult{
        .residues = &residues,
        .ss_bonds = &.{},
        .statistics = .{ .total_residues = 1, .complete_residues = 1 },
        .side_chain_storage = &.{},
        .near_pairs = &.{},
        .allocator = allocator,
    };

    var buf: std.ArrayListAligned(u8, null) = .empty;
    defer buf.deinit(allocator);

    try writeLegacy(buf.writer(allocator), &result);

    const output = buf.items;
    try std.testing.expect(std.mem.indexOf(u8, output, "RESIDUE") != null);
    try std.testing.expect(output.len > 100);
}

test "writeJson - no warnings when no near-threshold residues" {
    const allocator = std.testing.allocator;

    var residues = [_]Residue{
        // Residue with H-bond far from threshold (-2.0)
        .{
            .chain_id = .{ 'A', ' ', ' ', ' ' },
            .chain_id_len = 1,
            .seq_id = 1,
            .compound_id = .{ 'A', 'L', 'A', ' ' },
            .compound_id_len = 3,
            .residue_type = .ala,
            .secondary_structure = .alpha_helix,
            .complete = true,
            .hbond_acceptor = .{
                .{ .residue_index = 0, .energy = -2.0 },
                .{ .residue_index = null, .energy = 0.0 },
            },
        },
    };

    const result = DsspResult{
        .residues = &residues,
        .ss_bonds = &.{},
        .statistics = .{ .total_residues = 1, .complete_residues = 1 },
        .side_chain_storage = &.{},
        .near_pairs = &.{},
        .allocator = allocator,
    };

    var buf: std.ArrayListAligned(u8, null) = .empty;
    defer buf.deinit(allocator);

    try writeJson(buf.writer(allocator), &result);

    const output = buf.items;

    // Should NOT have warnings section when no near-threshold residues
    try std.testing.expect(std.mem.indexOf(u8, output, "\"warnings\":") == null);
    // Should still have near_threshold field in residue
    try std.testing.expect(std.mem.indexOf(u8, output, "\"near_threshold\": false") != null);
}

test "writeJson - near_threshold detection" {
    const allocator = std.testing.allocator;

    var residues = [_]Residue{
        // Residue 0: No H-bonds - should NOT be near threshold
        .{
            .chain_id = .{ 'A', ' ', ' ', ' ' },
            .chain_id_len = 1,
            .seq_id = 1,
            .compound_id = .{ 'A', 'L', 'A', ' ' },
            .compound_id_len = 3,
            .residue_type = .ala,
            .secondary_structure = .alpha_helix,
            .complete = true,
        },
        // Residue 1: H-bond at threshold (-0.500) - should be near threshold
        .{
            .chain_id = .{ 'A', ' ', ' ', ' ' },
            .chain_id_len = 1,
            .seq_id = 2,
            .compound_id = .{ 'G', 'L', 'Y', ' ' },
            .compound_id_len = 3,
            .residue_type = .gly,
            .secondary_structure = .alpha_helix,
            .complete = true,
            .hbond_acceptor = .{
                .{ .residue_index = 0, .energy = -0.500 },
                .{ .residue_index = null, .energy = 0.0 },
            },
        },
        // Residue 2: H-bond far from threshold (-1.5) - should NOT be near threshold
        .{
            .chain_id = .{ 'A', ' ', ' ', ' ' },
            .chain_id_len = 1,
            .seq_id = 3,
            .compound_id = .{ 'S', 'E', 'R', ' ' },
            .compound_id_len = 3,
            .residue_type = .ser,
            .secondary_structure = .strand,
            .complete = true,
            .hbond_donor = .{
                .{ .residue_index = 1, .energy = -1.5 },
                .{ .residue_index = null, .energy = 0.0 },
            },
        },
    };

    const result = DsspResult{
        .residues = &residues,
        .ss_bonds = &.{},
        .statistics = .{ .total_residues = 3, .complete_residues = 3 },
        .side_chain_storage = &.{},
        .near_pairs = &.{},
        .allocator = allocator,
    };

    var buf: std.ArrayListAligned(u8, null) = .empty;
    defer buf.deinit(allocator);

    try writeJson(buf.writer(allocator), &result);

    const output = buf.items;

    // Should have warnings section with 1 near-threshold residue
    try std.testing.expect(std.mem.indexOf(u8, output, "\"warnings\":") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "\"near_threshold_count\": 1") != null);

    // Check that warnings list uses chain_id:seq_id format
    try std.testing.expect(std.mem.indexOf(u8, output, "\"A:2\"") != null);

    // Check individual residue near_threshold fields
    try std.testing.expect(std.mem.indexOf(u8, output, "\"near_threshold\": true") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "\"near_threshold\": false") != null);
}

test "getThresholdMargin - edge cases" {
    // No H-bonds
    const res_no_hbonds = Residue{
        .chain_id = .{ 'A', ' ', ' ', ' ' },
        .chain_id_len = 1,
        .seq_id = 1,
        .compound_id = .{ 'A', 'L', 'A', ' ' },
        .compound_id_len = 3,
        .residue_type = .ala,
        .secondary_structure = .loop,
        .complete = true,
    };
    try std.testing.expect(getThresholdMargin(&res_no_hbonds) == null);
    try std.testing.expect(!isNearThreshold(&res_no_hbonds));

    // H-bond exactly at threshold (-0.5)
    var res_at_threshold = Residue{
        .chain_id = .{ 'A', ' ', ' ', ' ' },
        .chain_id_len = 1,
        .seq_id = 2,
        .compound_id = .{ 'G', 'L', 'Y', ' ' },
        .compound_id_len = 3,
        .residue_type = .gly,
        .secondary_structure = .alpha_helix,
        .complete = true,
    };
    res_at_threshold.hbond_acceptor[0] = .{ .residue_index = 0, .energy = -0.500 };
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), getThresholdMargin(&res_at_threshold).?, 1e-6);
    try std.testing.expect(isNearThreshold(&res_at_threshold));

    // H-bond near threshold (-0.505)
    var res_near_threshold = Residue{
        .chain_id = .{ 'A', ' ', ' ', ' ' },
        .chain_id_len = 1,
        .seq_id = 3,
        .compound_id = .{ 'S', 'E', 'R', ' ' },
        .compound_id_len = 3,
        .residue_type = .ser,
        .secondary_structure = .strand,
        .complete = true,
    };
    res_near_threshold.hbond_donor[0] = .{ .residue_index = 1, .energy = -0.505 };
    try std.testing.expectApproxEqAbs(@as(f32, 0.005), getThresholdMargin(&res_near_threshold).?, 1e-6);
    try std.testing.expect(isNearThreshold(&res_near_threshold));

    // H-bond far from threshold (-2.0)
    var res_far_threshold = Residue{
        .chain_id = .{ 'A', ' ', ' ', ' ' },
        .chain_id_len = 1,
        .seq_id = 4,
        .compound_id = .{ 'L', 'E', 'U', ' ' },
        .compound_id_len = 3,
        .residue_type = .leu,
        .secondary_structure = .alpha_helix,
        .complete = true,
    };
    res_far_threshold.hbond_acceptor[0] = .{ .residue_index = 2, .energy = -2.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), getThresholdMargin(&res_far_threshold).?, 1e-6);
    try std.testing.expect(!isNearThreshold(&res_far_threshold));
}
