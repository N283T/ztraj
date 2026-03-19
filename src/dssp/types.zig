const std = @import("std");
const math = std.math;

// ---------------------------------------------------------------------------
// Vec3Gen – generic 3-component vector
// ---------------------------------------------------------------------------

pub fn Vec3Gen(comptime T: type) type {
    return struct {
        const Self = @This();

        x: T,
        y: T,
        z: T,

        pub const zero = Self{ .x = 0, .y = 0, .z = 0 };

        pub fn add(self: Self, other: Self) Self {
            return .{
                .x = self.x + other.x,
                .y = self.y + other.y,
                .z = self.z + other.z,
            };
        }

        pub fn sub(self: Self, other: Self) Self {
            return .{
                .x = self.x - other.x,
                .y = self.y - other.y,
                .z = self.z - other.z,
            };
        }

        pub fn scale(self: Self, scalar: T) Self {
            return .{
                .x = self.x * scalar,
                .y = self.y * scalar,
                .z = self.z * scalar,
            };
        }

        pub fn dot(self: Self, other: Self) T {
            return self.x * other.x + self.y * other.y + self.z * other.z;
        }

        pub fn cross(self: Self, other: Self) Self {
            return .{
                .x = self.y * other.z - self.z * other.y,
                .y = self.z * other.x - self.x * other.z,
                .z = self.x * other.y - self.y * other.x,
            };
        }

        pub fn lengthSq(self: Self) T {
            return self.dot(self);
        }

        pub fn length(self: Self) T {
            return @sqrt(self.lengthSq());
        }

        pub fn distance(self: Self, other: Self) T {
            return self.sub(other).length();
        }

        pub fn distanceSq(self: Self, other: Self) T {
            return self.sub(other).lengthSq();
        }

        pub fn normalize(self: Self) Self {
            const len = self.length();
            if (len < Epsilon(T).default) return Self.zero;
            return self.scale(1.0 / len);
        }

        pub fn fromF64(v: Vec3Gen(f64)) Self {
            if (T == f64) return v;
            return .{
                .x = @floatCast(v.x),
                .y = @floatCast(v.y),
                .z = @floatCast(v.z),
            };
        }

        pub fn toF64(self: Self) Vec3Gen(f64) {
            if (T == f64) return self;
            return .{
                .x = @floatCast(self.x),
                .y = @floatCast(self.y),
                .z = @floatCast(self.z),
            };
        }
    };
}

/// Default high-precision vector (for energy calculations)
pub const Vec3 = Vec3Gen(f64);
/// Single-precision vector (for coordinate storage, matching C++ `point`)
pub const Vec3f32 = Vec3Gen(f32);

// ---------------------------------------------------------------------------
// Epsilon – floating-point comparison tolerances
// ---------------------------------------------------------------------------

pub fn Epsilon(comptime T: type) type {
    return struct {
        pub const default: T = if (T == f32) 1e-6 else 1e-10;
        pub const trig: T = if (T == f32) 1e-7 else 1e-10;
    };
}

// ---------------------------------------------------------------------------
// Constants (from dssp.cpp lines 242-258)
// ---------------------------------------------------------------------------

/// Maximum CA-CA distance for considering residue pairs (Å)
pub const kMinimalCADistance: f32 = 9.0;

/// Minimum valid distance for H-bond calculation (Å)
pub const kMinimalDistance: f32 = 0.5;

// H-bond energy constants (f32 to match C++ mkdssp floating-point behavior)
// C++ uses float for coordinates and distance(), and kCouplingConstant is
// declared as float (-27.888f). Using f32 ensures identical rounding behavior.

/// Electrostatic coupling constant: -332 * 0.42 * 0.2 (kcal/mol * Å)
pub const kCouplingConstantF32: f32 = -27.888;

/// Minimum (worst) H-bond energy (kcal/mol)
pub const kMinHBondEnergyF32: f32 = -9.9;

/// Threshold for an H-bond to exist (kcal/mol)
pub const kMaxHBondEnergyF32: f32 = -0.5;

/// Max C-N distance for peptide bond continuity (Å)
pub const kMaxPeptideBondLength: f32 = 2.5;

/// Atomic radii for surface accessibility (Å)
pub const kRadiusN: f32 = 1.65;
pub const kRadiusCA: f32 = 1.87;
pub const kRadiusC: f32 = 1.76;
pub const kRadiusO: f32 = 1.4;
pub const kRadiusSideAtom: f32 = 1.8;
pub const kRadiusWater: f32 = 1.4;

/// Number of Fibonacci sphere points (N in [-N..N] gives 2N+1 points)
pub const kFibonacciN: i32 = 200;

/// Histogram bin count for statistics
pub const kHistogramSize: usize = 30;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Secondary structure type (matches DSSP single-character codes)
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

/// Helix type (stride = @intFromEnum(t) + 3)
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

// ---------------------------------------------------------------------------
// H-Bond record
// ---------------------------------------------------------------------------

pub const HBond = struct {
    residue_index: ?u32 = null,
    energy: f32 = 0.0,
};

// ---------------------------------------------------------------------------
// Beta-bridge partner
// ---------------------------------------------------------------------------

pub const BridgePartner = struct {
    residue_index: ?u32 = null,
    ladder: u32 = 0,
    parallel: bool = false,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "Vec3f32 add/sub" {
    const v1 = Vec3f32{ .x = 1.0, .y = 2.0, .z = 3.0 };
    const v2 = Vec3f32{ .x = 4.0, .y = 5.0, .z = 6.0 };

    const sum = v1.add(v2);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), sum.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), sum.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), sum.z, 1e-6);

    const diff = v2.sub(v1);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), diff.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), diff.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), diff.z, 1e-6);
}

test "Vec3f32 dot/cross" {
    const v1 = Vec3f32{ .x = 1.0, .y = 0.0, .z = 0.0 };
    const v2 = Vec3f32{ .x = 0.0, .y = 1.0, .z = 0.0 };

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), v1.dot(v2), 1e-6);

    const c = v1.cross(v2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), c.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), c.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), c.z, 1e-6);
}

test "Vec3f32 length/distance" {
    const v = Vec3f32{ .x = 3.0, .y = 4.0, .z = 0.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), v.length(), 1e-6);

    const origin = Vec3f32.zero;
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), origin.distance(v), 1e-6);
}

test "Vec3f32 normalize" {
    const v = Vec3f32{ .x = 3.0, .y = 0.0, .z = 0.0 };
    const n = v.normalize();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), n.length(), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), n.x, 1e-6);
}

test "Vec3 f64 precision" {
    const v1 = Vec3{ .x = 1e-8, .y = 2e-8, .z = 3e-8 };
    const v2 = Vec3{ .x = 1e-8, .y = 2e-8, .z = 3e-8 };
    const sum = v1.add(v2);
    try std.testing.expectApproxEqAbs(@as(f64, 2e-8), sum.x, 1e-15);
}

test "Vec3f32 conversion roundtrip" {
    const v64 = Vec3{ .x = 1.5, .y = 2.5, .z = 3.5 };
    const v32 = Vec3f32.fromF64(v64);
    const back = v32.toF64();
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), back.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), back.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 3.5), back.z, 1e-6);
}

test "StructureType character codes" {
    try std.testing.expectEqual(@as(u8, 'H'), StructureType.alpha_helix.toChar());
    try std.testing.expectEqual(@as(u8, 'E'), StructureType.strand.toChar());
    try std.testing.expectEqual(@as(u8, ' '), StructureType.loop.toChar());
}

test "HelixType stride" {
    try std.testing.expectEqual(@as(u32, 3), HelixType.helix_3_10.stride());
    try std.testing.expectEqual(@as(u32, 4), HelixType.alpha.stride());
    try std.testing.expectEqual(@as(u32, 5), HelixType.pi.stride());
}

test "ResidueType fromCompoundId" {
    try std.testing.expectEqual(ResidueType.ala, ResidueType.fromCompoundId("ALA"));
    try std.testing.expectEqual(ResidueType.gly, ResidueType.fromCompoundId("GLY"));
    try std.testing.expectEqual(ResidueType.pro, ResidueType.fromCompoundId("PRO"));
    try std.testing.expectEqual(ResidueType.unknown, ResidueType.fromCompoundId("UNK"));
}
