// Native DSSP types for ztraj.
//
// Uses atom indices (u32) into Frame's x/y/z arrays instead of copying
// coordinates into the residue struct. The computed hydrogen position is the
// only exception: it is stored as h_x/h_y/h_z because there is no atom in
// the Frame for it.

const std = @import("std");
const math = std.math;

// ============================================================================
// Vec3f32 — single-precision 3-component vector
// ============================================================================

pub const Vec3f32 = struct {
    x: f32,
    y: f32,
    z: f32,

    const Self = @This();

    pub const zero = Self{ .x = 0, .y = 0, .z = 0 };

    pub fn add(self: Self, other: Self) Self {
        return .{ .x = self.x + other.x, .y = self.y + other.y, .z = self.z + other.z };
    }

    pub fn sub(self: Self, other: Self) Self {
        return .{ .x = self.x - other.x, .y = self.y - other.y, .z = self.z - other.z };
    }

    pub fn scale(self: Self, s: f32) Self {
        return .{ .x = self.x * s, .y = self.y * s, .z = self.z * s };
    }

    pub fn dot(self: Self, other: Self) f32 {
        return self.x * other.x + self.y * other.y + self.z * other.z;
    }

    pub fn cross(self: Self, other: Self) Self {
        return .{
            .x = self.y * other.z - self.z * other.y,
            .y = self.z * other.x - self.x * other.z,
            .z = self.x * other.y - self.y * other.x,
        };
    }

    pub fn lengthSq(self: Self) f32 {
        return self.dot(self);
    }

    pub fn length(self: Self) f32 {
        return @sqrt(self.lengthSq());
    }

    pub fn distance(self: Self, other: Self) f32 {
        return self.sub(other).length();
    }

    pub fn distanceSq(self: Self, other: Self) f32 {
        return self.sub(other).lengthSq();
    }

    pub fn normalize(self: Self) Self {
        const len = self.length();
        if (len < 1e-6) return Self.zero;
        return self.scale(1.0 / len);
    }
};

// ============================================================================
// Constants (from Kabsch-Sander / mkdssp)
// ============================================================================

/// Maximum CA-CA distance for considering residue pairs (Å)
pub const kMinimalCADistance: f32 = 9.0;

/// Minimum valid distance for H-bond calculation (Å)
pub const kMinimalDistance: f32 = 0.5;

/// Electrostatic coupling constant: -332 * 0.42 * 0.2 (kcal/mol·Å)
pub const kCouplingConstantF32: f32 = -27.888;

/// Minimum (worst) H-bond energy (kcal/mol)
pub const kMinHBondEnergyF32: f32 = -9.9;

/// Threshold for an H-bond to exist (kcal/mol)
pub const kMaxHBondEnergyF32: f32 = -0.5;

/// Max C(i-1)–N(i) distance for peptide bond continuity (Å)
pub const kMaxPeptideBondLength: f32 = 2.5;

// ============================================================================
// Enums
// ============================================================================

/// Secondary structure type (DSSP single-character codes)
pub const StructureType = enum(u8) {
    loop = ' ',
    alpha_helix = 'H',
    beta_bridge = 'B',
    strand = 'E',
    helix_3 = 'G',
    helix_5 = 'I',
    helix_pp2 = 'P',
    turn = 'T',
    bend = 'S',

    pub fn toChar(self: StructureType) u8 {
        return @intFromEnum(self);
    }
};

/// Helix type; stride = @intFromEnum(t) + 3
pub const HelixType = enum(u2) {
    helix_3_10 = 0, // stride 3
    alpha = 1, // stride 4
    pi = 2, // stride 5
    pp = 3, // polyproline II

    pub fn stride(self: HelixType) u32 {
        return @as(u32, @intFromEnum(self)) + 3;
    }
};

/// Position within a helix
pub const HelixPositionType = enum(u3) {
    none = 0,
    start = 1,
    end = 2,
    start_and_end = 3,
    middle = 4,
};

/// Beta-bridge type
pub const BridgeType = enum(u2) {
    none = 0,
    parallel = 1,
    anti_parallel = 2,
};

/// Chain break type
pub const ChainBreakType = enum(u2) {
    none = 0,
    new_chain = 1,
    gap = 2,
};

/// Single-letter residue type codes
pub const ResidueType = enum(u8) {
    ala = 'A',
    arg = 'R',
    asn = 'N',
    asp = 'D',
    cys = 'C',
    gln = 'Q',
    glu = 'E',
    gly = 'G',
    his = 'H',
    ile = 'I',
    leu = 'L',
    lys = 'K',
    met = 'M',
    phe = 'F',
    pro = 'P',
    ser = 'S',
    thr = 'T',
    trp = 'W',
    tyr = 'Y',
    val = 'V',
    unknown = 'X',

    pub fn fromCompoundId(compound_id: []const u8) ResidueType {
        const map = std.StaticStringMap(ResidueType).initComptime(.{
            .{ "ALA", .ala },
            .{ "ARG", .arg },
            .{ "ASN", .asn },
            .{ "ASP", .asp },
            .{ "CYS", .cys },
            .{ "GLN", .gln },
            .{ "GLU", .glu },
            .{ "GLY", .gly },
            .{ "HIS", .his },
            .{ "ILE", .ile },
            .{ "LEU", .leu },
            .{ "LYS", .lys },
            .{ "MET", .met },
            .{ "PHE", .phe },
            .{ "PRO", .pro },
            .{ "SER", .ser },
            .{ "THR", .thr },
            .{ "TRP", .trp },
            .{ "TYR", .tyr },
            .{ "VAL", .val },
        });
        return map.get(compound_id) orelse .unknown;
    }

    pub fn toChar(self: ResidueType) u8 {
        return @intFromEnum(self);
    }
};

// ============================================================================
// H-Bond record
// ============================================================================

pub const HBond = struct {
    residue_index: ?u32 = null,
    energy: f32 = 0.0,
};

// ============================================================================
// Beta-bridge partner
// ============================================================================

pub const BridgePartner = struct {
    residue_index: ?u32 = null,
    ladder: u32 = 0,
    parallel: bool = false,
};

// ============================================================================
// DsspResidue — backbone atom indices + computed state
// ============================================================================

/// A residue representation for the DSSP algorithm.
///
/// Backbone atom positions are not copied; instead this struct stores atom
/// indices (n_idx, ca_idx, c_idx, o_idx) into the Frame's x/y/z arrays.
/// The caller reads coordinates on-demand via getPos(frame, idx).
///
/// The computed amide hydrogen (H) has no corresponding atom in the Frame, so
/// its position is stored inline as h_x/h_y/h_z.
pub const DsspResidue = struct {
    // Backbone atom indices into Frame arrays (undefined when complete=false)
    n_idx: u32 = 0,
    ca_idx: u32 = 0,
    c_idx: u32 = 0,
    o_idx: u32 = 0,

    // Computed amide hydrogen position (placed by assignHydrogen)
    h_x: f32 = 0,
    h_y: f32 = 0,
    h_z: f32 = 0,

    // Identity
    chain_index: u32 = 0,
    residue_index: u32 = 0, // index into topology.residues
    residue_type: ResidueType = .unknown,
    number: u32 = 0, // sequential number after filtering

    // Backbone dihedral angles (null = undefined)
    phi: ?f32 = null,
    psi: ?f32 = null,
    omega: ?f32 = null,
    kappa: ?f32 = null,
    alpha: ?f32 = null,
    tco: ?f32 = null,

    // H-bonds (2 best donors, 2 best acceptors)
    hbond_donor: [2]HBond = .{ .{}, .{} },
    hbond_acceptor: [2]HBond = .{ .{}, .{} },

    // Secondary structure
    secondary_structure: StructureType = .loop,
    helix_flags: [4]HelixPositionType = .{ .none, .none, .none, .none },
    bend: bool = false,

    // Beta sheet
    beta_partner: [2]BridgePartner = .{ .{}, .{} },
    sheet: u32 = 0,
    strand: u32 = 0,

    // Misc
    chain_break: ChainBreakType = .none,
    complete: bool = false,

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    pub fn isProline(self: *const DsspResidue) bool {
        return self.residue_type == .pro;
    }

    pub fn getHelixFlag(self: *const DsspResidue, ht: HelixType) HelixPositionType {
        return self.helix_flags[@intFromEnum(ht)];
    }

    pub fn setHelixFlag(self: *DsspResidue, ht: HelixType, pos: HelixPositionType) void {
        self.helix_flags[@intFromEnum(ht)] = pos;
    }

    pub fn isHelixStart(self: *const DsspResidue, ht: HelixType) bool {
        const f = self.getHelixFlag(ht);
        return f == .start or f == .start_and_end;
    }

    /// Get the stored hydrogen position as a Vec3f32.
    pub fn getH(self: *const DsspResidue) Vec3f32 {
        return .{ .x = self.h_x, .y = self.h_y, .z = self.h_z };
    }
};

// ============================================================================
// Configuration and Result
// ============================================================================

pub const DsspConfig = struct {
    /// Prefer pi helices over alpha helices (matches mkdssp default)
    prefer_pi_helices: bool = true,
    /// PP-II helix minimum stretch length (2 or 3, default 3)
    pp_stretch: u32 = 3,
    /// Number of threads (reserved for future parallel use)
    n_threads: usize = 0,
};

pub const DsspResult = struct {
    residues: []DsspResidue,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *DsspResult) void {
        self.allocator.free(self.residues);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "StructureType character codes" {
    try std.testing.expectEqual(@as(u8, 'H'), StructureType.alpha_helix.toChar());
    try std.testing.expectEqual(@as(u8, 'E'), StructureType.strand.toChar());
    try std.testing.expectEqual(@as(u8, ' '), StructureType.loop.toChar());
    try std.testing.expectEqual(@as(u8, 'G'), StructureType.helix_3.toChar());
    try std.testing.expectEqual(@as(u8, 'I'), StructureType.helix_5.toChar());
    try std.testing.expectEqual(@as(u8, 'P'), StructureType.helix_pp2.toChar());
}

test "HelixType stride" {
    try std.testing.expectEqual(@as(u32, 3), HelixType.helix_3_10.stride());
    try std.testing.expectEqual(@as(u32, 4), HelixType.alpha.stride());
    try std.testing.expectEqual(@as(u32, 5), HelixType.pi.stride());
    try std.testing.expectEqual(@as(u32, 6), HelixType.pp.stride());
}

test "ResidueType fromCompoundId" {
    try std.testing.expectEqual(ResidueType.ala, ResidueType.fromCompoundId("ALA"));
    try std.testing.expectEqual(ResidueType.gly, ResidueType.fromCompoundId("GLY"));
    try std.testing.expectEqual(ResidueType.pro, ResidueType.fromCompoundId("PRO"));
    try std.testing.expectEqual(ResidueType.unknown, ResidueType.fromCompoundId("UNK"));
}

test "DsspResidue helix flags" {
    var res = DsspResidue{};
    try std.testing.expectEqual(HelixPositionType.none, res.getHelixFlag(.alpha));
    res.setHelixFlag(.alpha, .start);
    try std.testing.expectEqual(HelixPositionType.start, res.getHelixFlag(.alpha));
    try std.testing.expect(res.isHelixStart(.alpha));
    try std.testing.expect(!res.isHelixStart(.helix_3_10));
}

test "DsspResidue isProline" {
    var res = DsspResidue{};
    res.residue_type = .pro;
    try std.testing.expect(res.isProline());
    res.residue_type = .ala;
    try std.testing.expect(!res.isProline());
}

test "DsspResidue getH" {
    var res = DsspResidue{};
    res.h_x = 1.0;
    res.h_y = 2.0;
    res.h_z = 3.0;
    const h = res.getH();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), h.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), h.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), h.z, 1e-6);
}

test "Vec3f32 operations" {
    const v1 = Vec3f32{ .x = 1.0, .y = 2.0, .z = 3.0 };
    const v2 = Vec3f32{ .x = 4.0, .y = 5.0, .z = 6.0 };

    const sum = v1.add(v2);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), sum.x, 1e-6);

    const diff = v2.sub(v1);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), diff.x, 1e-6);

    const scaled = v1.scale(2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), scaled.x, 1e-6);

    const d = v1.dot(v2);
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), d, 1e-6);
}

test "Vec3f32 length and distance" {
    const v = Vec3f32{ .x = 3.0, .y = 4.0, .z = 0.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), v.length(), 1e-6);

    const origin = Vec3f32.zero;
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), origin.distance(v), 1e-6);
}

test "Vec3f32 normalize" {
    const v = Vec3f32{ .x = 3.0, .y = 0.0, .z = 0.0 };
    const n = v.normalize();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), n.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), n.y, 1e-6);
}
