const std = @import("std");
const types = @import("types.zig");
const geometry = @import("geometry.zig");

const Vec3f32 = types.Vec3f32;
const StructureType = types.StructureType;
const HelixType = types.HelixType;
const HelixPositionType = types.HelixPositionType;
const BridgeType = types.BridgeType;
const ChainBreakType = types.ChainBreakType;
const ResidueType = types.ResidueType;
const HBond = types.HBond;
const BridgePartner = types.BridgePartner;

// ---------------------------------------------------------------------------
// Side-chain atom
// ---------------------------------------------------------------------------

pub const SideChainAtom = struct {
    name: [4]u8 = .{ ' ', ' ', ' ', ' ' },
    name_len: u8 = 0,
    pos: Vec3f32 = Vec3f32.zero,

    pub fn getName(self: SideChainAtom) []const u8 {
        return self.name[0..self.name_len];
    }
};

// ---------------------------------------------------------------------------
// Residue
// ---------------------------------------------------------------------------

pub const Residue = struct {
    // -- Identity --
    chain_id: [4]u8 = .{ ' ', ' ', ' ', ' ' },
    chain_id_len: u8 = 0,
    seq_id: i32 = 0,
    compound_id: [4]u8 = .{ ' ', ' ', ' ', ' ' },
    compound_id_len: u8 = 0,
    residue_type: ResidueType = .unknown,
    number: u32 = 0, // internal sequential number

    // -- Backbone atom coordinates (f32, matching C++ `point`) --
    ca: Vec3f32 = Vec3f32.zero,
    c: Vec3f32 = Vec3f32.zero,
    n: Vec3f32 = Vec3f32.zero,
    o: Vec3f32 = Vec3f32.zero,
    h: Vec3f32 = Vec3f32.zero,

    // -- Side chain atoms --
    side_chain: []SideChainAtom = &.{},

    // -- Backbone dihedral angles (null = undefined, sentinel 360 in C++) --
    phi: ?f32 = null,
    psi: ?f32 = null,
    omega: ?f32 = null,
    kappa: ?f32 = null,
    alpha: ?f32 = null,
    tco: ?f32 = null,

    // -- H-bonds (2 best donors, 2 best acceptors) --
    hbond_donor: [2]HBond = .{ .{}, .{} },
    hbond_acceptor: [2]HBond = .{ .{}, .{} },

    // -- Secondary structure --
    secondary_structure: StructureType = .loop,
    helix_flags: [4]HelixPositionType = .{ .none, .none, .none, .none },
    bend: bool = false,

    // -- Beta sheet --
    beta_partner: [2]BridgePartner = .{ .{}, .{} },
    sheet: u32 = 0,
    strand: u32 = 0,

    // -- Other --
    accessibility: f32 = 0.0,
    chain_break: ChainBreakType = .none,
    ss_bridge_nr: u16 = 0,
    complete: bool = false,

    // -- Helpers --

    pub fn getChainId(self: *const Residue) []const u8 {
        return self.chain_id[0..self.chain_id_len];
    }

    pub fn getCompoundId(self: *const Residue) []const u8 {
        return self.compound_id[0..self.compound_id_len];
    }

    pub fn getHelixFlag(self: *const Residue, ht: HelixType) HelixPositionType {
        return self.helix_flags[@intFromEnum(ht)];
    }

    pub fn setHelixFlag(self: *Residue, ht: HelixType, pos: HelixPositionType) void {
        self.helix_flags[@intFromEnum(ht)] = pos;
    }

    pub fn isHelixStart(self: *const Residue, ht: HelixType) bool {
        const f = self.getHelixFlag(ht);
        return f == .start or f == .start_and_end;
    }

    pub fn isProline(self: *const Residue) bool {
        return self.residue_type == .pro;
    }
};

// ---------------------------------------------------------------------------
// Hydrogen placement (dssp.cpp:388-403)
// ---------------------------------------------------------------------------

/// Place the backbone amide hydrogen using the C=O of the previous residue.
/// H is placed along the bisector of the N-C(prev) bond, 1.0 Å from N.
///
/// For the first residue (or after a chain break), H is undefined (zero).
pub fn assignHydrogen(residues: []Residue) void {
    for (residues, 0..) |*res, i| {
        if (i == 0 or res.chain_break != .none or res.isProline()) {
            res.h = Vec3f32.zero;
            continue;
        }
        const prev = &residues[i - 1];
        // Direction: opposite to C=O of previous residue
        const co = prev.o.sub(prev.c);
        const co_len = co.length();
        if (co_len < 1e-6) {
            res.h = Vec3f32.zero;
            continue;
        }
        // H is 1.0 Å from N, along the direction opposite to C=O(prev)
        const unit_co = co.scale(1.0 / co_len);
        res.h = res.n.sub(unit_co);
    }
}

// ---------------------------------------------------------------------------
// Backbone geometry calculation (dssp.cpp:1500-1554)
// ---------------------------------------------------------------------------

/// Calculate backbone dihedral angles for all residues.
///
/// - phi: dihedral(C_prev, N, CA, C)
/// - psi: dihedral(N, CA, C, N_next)
/// - omega: dihedral(CA, C, N_next, CA_next)
/// - alpha: dihedral(CA_prev, CA, CA_next, CA_next+1)  -- at residues i-1
/// - kappa: virtual bond angle CA(i-2), CA(i), CA(i+2)
/// - tco: cosine of angle between C=O(i) and C=O(i-1)
pub fn calculateGeometry(residues: []Residue) void {
    const n = residues.len;
    if (n == 0) return;

    for (residues, 0..) |*res, i| {
        // phi: C(i-1) - N(i) - CA(i) - C(i)
        // res.chain_break indicates break *before* this residue, so only check res.chain_break
        if (i > 0 and res.chain_break == .none) {
            res.phi = geometry.dihedralAngle(residues[i - 1].c, res.n, res.ca, res.c);
        }

        // psi: N(i) - CA(i) - C(i) - N(i+1)
        // residues[i+1].chain_break indicates break before residues[i+1] (i.e., between i and i+1)
        if (i + 1 < n and residues[i + 1].chain_break == .none) {
            res.psi = geometry.dihedralAngle(res.n, res.ca, res.c, residues[i + 1].n);
        }

        // omega: CA(i) - C(i) - N(i+1) - CA(i+1)
        // Same condition as psi: check if there's a break before next residue
        if (i + 1 < n and residues[i + 1].chain_break == .none) {
            res.omega = geometry.dihedralAngle(res.ca, res.c, residues[i + 1].n, residues[i + 1].ca);
        }

        // alpha: dihedral CA(i-1) - CA(i) - CA(i+1) - CA(i+2)
        if (i > 0 and i + 2 < n) {
            if (noChainBreak(residues, @intCast(i - 1), @intCast(i + 2))) {
                res.alpha = geometry.dihedralAngle(
                    residues[i - 1].ca,
                    res.ca,
                    residues[i + 1].ca,
                    residues[i + 2].ca,
                );
            }
        }

        // kappa: virtual bond angle at CA(i), defined by CA(i-2), CA(i), CA(i+2)
        // C++ DSSP also requires: prevPrev.mSeqID + 4 == nextNext.mSeqID
        if (i >= 2 and i + 2 < n) {
            if (noChainBreak(residues, @intCast(i - 2), @intCast(i + 2)) and
                residues[i - 2].seq_id + 4 == residues[i + 2].seq_id)
            {
                res.kappa = geometry.kappaAngle(
                    residues[i - 2].ca,
                    res.ca,
                    residues[i + 2].ca,
                );
            }
        }

        // tco: cosine of angle between C=O(i) and C=O(i-1)
        if (i > 0 and res.chain_break == .none) {
            res.tco = geometry.cosinusAngle(res.c, res.o, residues[i - 1].c, residues[i - 1].o);
        }
    }
}

// ---------------------------------------------------------------------------
// Chain break detection (dssp.cpp:802-818)
// ---------------------------------------------------------------------------

/// Detect chain breaks by checking peptide bond length and chain identity.
pub fn detectChainBreaks(residues: []Residue) void {
    if (residues.len == 0) return;

    for (residues[1..], 1..) |*res, i| {
        const prev = &residues[i - 1];

        // Different chain → new chain break
        if (!std.mem.eql(u8, res.getChainId(), prev.getChainId())) {
            res.chain_break = .new_chain;
            continue;
        }

        // C(i-1) to N(i) distance > threshold → gap
        const dist = prev.c.distance(res.n);
        if (dist > types.kMaxPeptideBondLength) {
            res.chain_break = .gap;
        }
    }
}

/// Check that there is no chain break between residue indices `from` and `to` (inclusive).
pub fn noChainBreak(residues: []const Residue, from: u32, to: u32) bool {
    if (from >= to) return true;
    const start = from + 1;
    const end = @min(to + 1, @as(u32, @intCast(residues.len)));
    for (residues[start..end]) |res| {
        if (res.chain_break != .none) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "Residue - chain id / compound id" {
    var res = Residue{};
    res.chain_id[0] = 'A';
    res.chain_id_len = 1;
    res.compound_id[0] = 'A';
    res.compound_id[1] = 'L';
    res.compound_id[2] = 'A';
    res.compound_id_len = 3;

    try std.testing.expectEqualStrings("A", res.getChainId());
    try std.testing.expectEqualStrings("ALA", res.getCompoundId());
}

test "Residue - helix flags" {
    var res = Residue{};
    try std.testing.expectEqual(HelixPositionType.none, res.getHelixFlag(.alpha));

    res.setHelixFlag(.alpha, .start);
    try std.testing.expectEqual(HelixPositionType.start, res.getHelixFlag(.alpha));
    try std.testing.expect(res.isHelixStart(.alpha));
    try std.testing.expect(!res.isHelixStart(.helix_3_10));
}

test "noChainBreak - no breaks" {
    var residues = [_]Residue{
        .{},
        .{},
        .{},
    };
    try std.testing.expect(noChainBreak(&residues, 0, 2));
}

test "noChainBreak - with break" {
    var residues = [_]Residue{
        .{},
        .{ .chain_break = .new_chain },
        .{},
    };
    try std.testing.expect(!noChainBreak(&residues, 0, 2));
}

test "detectChainBreaks - peptide bond too long" {
    var residues = [_]Residue{
        .{
            .c = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .chain_id = .{ 'A', ' ', ' ', ' ' },
            .chain_id_len = 1,
        },
        .{
            .n = Vec3f32{ .x = 5.0, .y = 0.0, .z = 0.0 }, // 5.0 > 2.5 threshold
            .chain_id = .{ 'A', ' ', ' ', ' ' },
            .chain_id_len = 1,
        },
    };
    detectChainBreaks(&residues);
    try std.testing.expectEqual(ChainBreakType.gap, residues[1].chain_break);
}

test "detectChainBreaks - different chain" {
    var residues = [_]Residue{
        .{
            .chain_id = .{ 'A', ' ', ' ', ' ' },
            .chain_id_len = 1,
        },
        .{
            .chain_id = .{ 'B', ' ', ' ', ' ' },
            .chain_id_len = 1,
        },
    };
    detectChainBreaks(&residues);
    try std.testing.expectEqual(ChainBreakType.new_chain, residues[1].chain_break);
}

test "assignHydrogen - basic placement" {
    // Two residues: H on second should be ~1 Å from N, opposite C=O direction
    var residues = [_]Residue{
        .{
            .c = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .o = Vec3f32{ .x = 0.0, .y = 0.0, .z = 1.2 },
            .chain_id = .{ 'A', ' ', ' ', ' ' },
            .chain_id_len = 1,
        },
        .{
            .n = Vec3f32{ .x = 1.5, .y = 0.0, .z = 0.0 },
            .residue_type = .ala,
            .chain_id = .{ 'A', ' ', ' ', ' ' },
            .chain_id_len = 1,
        },
    };
    assignHydrogen(&residues);
    // H should be shifted from N in the -z direction (opposite to C=O)
    const h = residues[1].h;
    try std.testing.expect(h.z < residues[1].n.z);
    // Distance from N to H should be ~1 Å
    const dist = residues[1].n.distance(h);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), dist, 0.01);
}

test "assignHydrogen - proline gets zero" {
    var residues = [_]Residue{
        .{
            .c = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .o = Vec3f32{ .x = 0.0, .y = 0.0, .z = 1.2 },
        },
        .{
            .n = Vec3f32{ .x = 1.5, .y = 0.0, .z = 0.0 },
            .residue_type = .pro,
        },
    };
    assignHydrogen(&residues);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), residues[1].h.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), residues[1].h.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), residues[1].h.z, 1e-6);
}

test "calculateGeometry - two residues phi/psi/omega" {
    // Create two linked residues with known coordinates
    var residues = [_]Residue{
        .{
            .ca = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .c = Vec3f32{ .x = 1.5, .y = 0.0, .z = 0.0 },
            .n = Vec3f32{ .x = -1.0, .y = 0.0, .z = 0.0 },
            .o = Vec3f32{ .x = 1.5, .y = 1.2, .z = 0.0 },
            .complete = true,
        },
        .{
            .ca = Vec3f32{ .x = 3.0, .y = 1.0, .z = 0.0 },
            .c = Vec3f32{ .x = 4.5, .y = 1.0, .z = 0.0 },
            .n = Vec3f32{ .x = 2.0, .y = 0.5, .z = 0.0 },
            .o = Vec3f32{ .x = 4.5, .y = 2.2, .z = 0.0 },
            .complete = true,
        },
    };
    calculateGeometry(&residues);

    // Second residue should have phi defined
    try std.testing.expect(residues[1].phi != null);
    // First residue should have psi defined
    try std.testing.expect(residues[0].psi != null);
}

test "calculateGeometry - phi at chain start" {
    // Chain A: residue 0
    // Chain B: residue 1 (new_chain break) - phi should be null
    // Chain B: residue 2 - phi should be calculated using residue 1
    var residues = [_]Residue{
        .{
            .ca = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .c = Vec3f32{ .x = 1.5, .y = 0.0, .z = 0.0 },
            .n = Vec3f32{ .x = -1.0, .y = 0.0, .z = 0.0 },
            .o = Vec3f32{ .x = 1.5, .y = 1.2, .z = 0.0 },
            .chain_id = .{ 'A', ' ', ' ', ' ' },
            .chain_id_len = 1,
            .complete = true,
        },
        .{
            .ca = Vec3f32{ .x = 10.0, .y = 0.0, .z = 0.0 },
            .c = Vec3f32{ .x = 11.5, .y = 0.0, .z = 0.0 },
            .n = Vec3f32{ .x = 9.0, .y = 0.0, .z = 0.0 },
            .o = Vec3f32{ .x = 11.5, .y = 1.2, .z = 0.0 },
            .chain_id = .{ 'B', ' ', ' ', ' ' },
            .chain_id_len = 1,
            .chain_break = .new_chain,
            .complete = true,
        },
        .{
            .ca = Vec3f32{ .x = 13.0, .y = 1.0, .z = 0.0 },
            .c = Vec3f32{ .x = 14.5, .y = 1.0, .z = 0.0 },
            .n = Vec3f32{ .x = 12.0, .y = 0.5, .z = 0.0 },
            .o = Vec3f32{ .x = 14.5, .y = 2.2, .z = 0.0 },
            .chain_id = .{ 'B', ' ', ' ', ' ' },
            .chain_id_len = 1,
            .complete = true,
        },
    };
    calculateGeometry(&residues);

    // residue 0: phi should be null (first residue)
    try std.testing.expect(residues[0].phi == null);
    // residue 1: phi should be null (chain break before it)
    try std.testing.expect(residues[1].phi == null);
    // residue 2: phi should be defined (no chain break before it)
    try std.testing.expect(residues[2].phi != null);

    // residue 0: psi should be null (chain break after it, i.e., before residue 1)
    try std.testing.expect(residues[0].psi == null);
    // residue 1: psi should be defined (no chain break after it)
    try std.testing.expect(residues[1].psi != null);
}
