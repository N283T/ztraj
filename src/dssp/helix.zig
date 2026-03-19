const std = @import("std");
const math = std.math;
const types = @import("types.zig");
const residue_mod = @import("residue.zig");
const hbond_mod = @import("hbond.zig");

const Residue = residue_mod.Residue;
const HelixType = types.HelixType;
const HelixPositionType = types.HelixPositionType;
const StructureType = types.StructureType;

// ---------------------------------------------------------------------------
// Alpha, 3/10, and Pi helix detection (dssp.cpp:1133-1248)
// ---------------------------------------------------------------------------

/// Calculate alpha, 3/10, and pi helices.
///
/// For each helix type (stride 3, 4, 5):
/// 1. Mark helix start/end/middle based on H-bond from i+stride to i
/// 2. Assign secondary structure with priority:
///    sheets > alpha helix > 3/10 helix > pi helix > turn > bend
pub fn calculateAlphaHelices(residues: []Residue, prefer_pi_helices: bool) void {
    const n = residues.len;
    if (n < 4) return;

    // Step 1: Mark helix flags for each helix type
    const helix_types = [_]HelixType{ .helix_3_10, .alpha, .pi };
    for (helix_types) |ht| {
        const stride = ht.stride();

        for (0..n) |i| {
            if (i + stride >= n) break;

            // Check chain continuity and H-bond from i+stride back to i
            if (!residue_mod.noChainBreak(residues, @intCast(i), @intCast(i + stride))) continue;
            if (!hbond_mod.testBond(residues, @intCast(i + stride), @intCast(i))) continue;

            // Mark end position
            const end_flag = residues[i + stride].getHelixFlag(ht);
            if (end_flag == .start) {
                residues[i + stride].setHelixFlag(ht, .start_and_end);
            } else {
                residues[i + stride].setHelixFlag(ht, .end);
            }

            // Mark middle positions
            for (i + 1..i + stride) |j| {
                if (residues[j].getHelixFlag(ht) == .none) {
                    residues[j].setHelixFlag(ht, .middle);
                }
            }

            // Mark start position
            const start_flag = residues[i].getHelixFlag(ht);
            if (start_flag == .end) {
                residues[i].setHelixFlag(ht, .start_and_end);
            } else {
                residues[i].setHelixFlag(ht, .start);
            }
        }
    }

    // Step 1b: Set bend flags (C++ dssp.cpp:1162-1166 - after helix flags, before SS assignment)
    for (residues) |*res| {
        if (res.kappa) |k| {
            res.bend = k > 70.0;
        }
    }

    // Step 2: Assign alpha helices (need 2 consecutive starts → 4 residues)
    // dssp.cpp:1168: for (uint32_t i = 1; i + 4 < inResidues.size(); ++i)
    if (n >= 5) {
        for (1..n) |i| {
            if (i + 4 >= n) break; // Match dssp.cpp boundary
            if (residues[i].isHelixStart(.alpha) and residues[i - 1].isHelixStart(.alpha)) {
                for (i..i + 4) |j| {
                    residues[j].secondary_structure = .alpha_helix;
                }
            }
        }
    }

    // Step 3: Assign 3/10 helices (only if not already assigned to sheets or alpha)
    // dssp.cpp:1177: for (uint32_t i = 1; i + 3 < inResidues.size(); ++i)
    if (n >= 4) {
        for (1..n) |i| {
            if (i + 3 >= n) break; // Match dssp.cpp boundary
            if (residues[i].isHelixStart(.helix_3_10) and residues[i - 1].isHelixStart(.helix_3_10)) {
                var can_assign = true;
                for (i..i + 3) |j| {
                    const ss = residues[j].secondary_structure;
                    if (ss != .loop and ss != .helix_3) {
                        can_assign = false;
                        break;
                    }
                }
                if (can_assign) {
                    for (i..i + 3) |j| {
                        residues[j].secondary_structure = .helix_3;
                    }
                }
            }
        }
    }

    // Step 4: Assign pi helices (only if not already assigned, unless prefer_pi)
    // dssp.cpp:1191: for (uint32_t i = 1; i + 5 < inResidues.size(); ++i)
    if (n >= 6) {
        for (1..n) |i| {
            if (i + 5 >= n) break; // Match dssp.cpp boundary
            if (residues[i].isHelixStart(.pi) and residues[i - 1].isHelixStart(.pi)) {
                var can_assign = true;
                for (i..i + 5) |j| {
                    const ss = residues[j].secondary_structure;
                    if (ss == .loop or ss == .helix_5) {
                        continue;
                    }
                    if (prefer_pi_helices and ss == .alpha_helix) {
                        continue;
                    }
                    can_assign = false;
                    break;
                }
                if (can_assign) {
                    for (i..i + 5) |j| {
                        residues[j].secondary_structure = .helix_5;
                    }
                }
            }
        }
    }

    // Step 5: Assign turns and bends (dssp.cpp:1208-1224)
    for (1..n) |i| {
        if (i + 1 >= n) break;
        if (residues[i].secondary_structure != .loop) continue;

        var is_turn = false;
        for (helix_types) |ht| {
            const stride = ht.stride();
            for (1..stride) |k| {
                if (i >= k and residues[i - k].isHelixStart(ht)) {
                    is_turn = true;
                    break;
                }
            }
            if (is_turn) break;
        }

        if (is_turn) {
            residues[i].secondary_structure = .turn;
        } else if (residues[i].bend) {
            residues[i].secondary_structure = .bend;
        }
    }
}

// ---------------------------------------------------------------------------
// PP-II helix detection (dssp.cpp:1252-1385)
// ---------------------------------------------------------------------------

/// Calculate polyproline II helices based on phi/psi angles.
///
/// PP-II region: phi = -75° ± 29°, psi = 145° ± 29°
/// Minimum stretch: 2 or 3 consecutive residues (configurable).
pub fn calculatePPHelices(residues: []Residue, stretch_length: u32) void {
    const n = residues.len;
    if (n < 3) return;

    const epsilon: f32 = 29.0;
    const phi_min: f32 = -75.0 - epsilon;
    const phi_max: f32 = -75.0 + epsilon;
    const psi_min: f32 = 145.0 - epsilon;
    const psi_max: f32 = 145.0 + epsilon;

    for (1..n) |i| {
        if (i + stretch_length >= n) break;

        var all_in_region = true;
        for (0..stretch_length) |k| {
            const idx = i + k;
            const phi = residues[idx].phi orelse 360.0;
            const psi = residues[idx].psi orelse 360.0;

            if (phi < phi_min or phi > phi_max or psi < psi_min or psi > psi_max) {
                all_in_region = false;
                break;
            }
        }

        if (!all_in_region) continue;

        // Assign PP-II helix flags and structure type
        for (0..stretch_length) |k| {
            const idx = i + k;

            // Set helix flag
            if (k == 0) {
                const flag = residues[idx].getHelixFlag(.pp);
                if (flag == .end) {
                    residues[idx].setHelixFlag(.pp, .start_and_end);
                } else {
                    residues[idx].setHelixFlag(.pp, .start);
                }
            } else if (k == stretch_length - 1) {
                const flag = residues[idx].getHelixFlag(.pp);
                if (flag == .start) {
                    residues[idx].setHelixFlag(.pp, .start_and_end);
                } else {
                    residues[idx].setHelixFlag(.pp, .end);
                }
            } else {
                if (residues[idx].getHelixFlag(.pp) == .none) {
                    residues[idx].setHelixFlag(.pp, .middle);
                }
            }

            // Only assign to loop residues
            if (residues[idx].secondary_structure == .loop) {
                residues[idx].secondary_structure = .helix_pp2;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "calculateAlphaHelices - marks turns for loop residues near helix starts" {
    // Create a small chain with helix flags pre-set
    var residues: [6]Residue = undefined;
    for (&residues, 0..) |*r, idx| {
        r.* = Residue{
            .number = @intCast(idx),
            .complete = true,
            .secondary_structure = .loop,
        };
    }

    // Pre-set helix starts for alpha helix at positions 0, 1
    residues[0].setHelixFlag(.alpha, .start);
    residues[1].setHelixFlag(.alpha, .start);

    // Manually set up H-bond records so testBond works
    // For alpha helix: bond from i+4 to i
    residues[4].hbond_acceptor[0] = .{ .residue_index = 0, .energy = -2.0 };
    residues[5].hbond_acceptor[0] = .{ .residue_index = 1, .energy = -2.0 };

    calculateAlphaHelices(&residues, false);

    // Residues 1-4 should be alpha helix
    try std.testing.expectEqual(StructureType.alpha_helix, residues[1].secondary_structure);
    try std.testing.expectEqual(StructureType.alpha_helix, residues[2].secondary_structure);
    try std.testing.expectEqual(StructureType.alpha_helix, residues[3].secondary_structure);
    try std.testing.expectEqual(StructureType.alpha_helix, residues[4].secondary_structure);
}

test "calculatePPHelices - assigns PP-II to qualifying residues" {
    var residues: [5]Residue = undefined;
    for (&residues, 0..) |*r, idx| {
        r.* = Residue{
            .number = @intCast(idx),
            .complete = true,
            .secondary_structure = .loop,
        };
    }

    // Set phi/psi in PP-II region for residues 1 and 2
    residues[1].phi = -75.0;
    residues[1].psi = 145.0;
    residues[2].phi = -70.0;
    residues[2].psi = 150.0;

    calculatePPHelices(&residues, 2);

    try std.testing.expectEqual(StructureType.helix_pp2, residues[1].secondary_structure);
    try std.testing.expectEqual(StructureType.helix_pp2, residues[2].secondary_structure);
    // Others should remain loop
    try std.testing.expectEqual(StructureType.loop, residues[0].secondary_structure);
    try std.testing.expectEqual(StructureType.loop, residues[3].secondary_structure);
}

test "calculatePPHelices - does not overwrite non-loop" {
    var residues: [5]Residue = undefined;
    for (&residues, 0..) |*r, idx| {
        r.* = Residue{
            .number = @intCast(idx),
            .complete = true,
            .secondary_structure = .loop,
        };
    }

    // Set phi/psi in PP-II region
    residues[1].phi = -75.0;
    residues[1].psi = 145.0;
    residues[2].phi = -70.0;
    residues[2].psi = 150.0;

    // But residue 1 is already alpha helix
    residues[1].secondary_structure = .alpha_helix;

    calculatePPHelices(&residues, 2);

    // Should not overwrite alpha helix
    try std.testing.expectEqual(StructureType.alpha_helix, residues[1].secondary_structure);
    // Residue 2 should still get PP-II
    try std.testing.expectEqual(StructureType.helix_pp2, residues[2].secondary_structure);
}
