const std = @import("std");
const json = std.json;
const Allocator = std.mem.Allocator;
const types = @import("types.zig");
const residue_mod = @import("residue.zig");

const Vec3f32 = types.Vec3f32;
const ResidueType = types.ResidueType;
const Residue = residue_mod.Residue;
const SideChainAtom = residue_mod.SideChainAtom;

// ---------------------------------------------------------------------------
// JSON schema types (for std.json parsing)
// ---------------------------------------------------------------------------

const JsonAtomPos = struct {
    N: ?[3]f64 = null,
    CA: ?[3]f64 = null,
    C: ?[3]f64 = null,
    O: ?[3]f64 = null,
};

const JsonSideChainAtom = struct {
    name: []const u8,
    pos: [3]f64,
};

const JsonResidue = struct {
    chain_id: []const u8,
    seq_id: i64,
    compound_id: []const u8,
    atoms: JsonAtomPos,
    side_chain: ?[]const JsonSideChainAtom = null,
};

const JsonSSBond = struct {
    chain1: []const u8,
    seq1: i64,
    chain2: []const u8,
    seq2: i64,
};

const JsonInput = struct {
    residues: []const JsonResidue,
    ss_bonds: ?[]const JsonSSBond = null,
};

// ---------------------------------------------------------------------------
// SS-bond record (returned to caller)
// ---------------------------------------------------------------------------

pub const SSBond = struct {
    chain1: [4]u8 = .{ ' ', ' ', ' ', ' ' },
    chain1_len: u8 = 0,
    seq1: i32 = 0,
    chain2: [4]u8 = .{ ' ', ' ', ' ', ' ' },
    chain2_len: u8 = 0,
    seq2: i32 = 0,
};

// ---------------------------------------------------------------------------
// Parse result
// ---------------------------------------------------------------------------

pub const ParseResult = struct {
    residues: []Residue,
    ss_bonds: []SSBond,
    side_chain_storage: []SideChainAtom,
    allocator: Allocator,

    pub fn deinit(self: *ParseResult) void {
        self.allocator.free(self.residues);
        self.allocator.free(self.ss_bonds);
        self.allocator.free(self.side_chain_storage);
    }
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub const ParseError = error{
    InvalidJson,
    MissingResidues,
    MissingBackboneAtom,
    OutOfMemory,
};

/// Parse JSON input data into a slice of `Residue` structs.
pub fn parseJsonInput(allocator: Allocator, json_data: []const u8) ParseError!ParseResult {
    const parsed = json.parseFromSlice(JsonInput, allocator, json_data, .{
        .ignore_unknown_fields = true,
    }) catch return ParseError.InvalidJson;
    defer parsed.deinit();

    const input = parsed.value;

    if (input.residues.len == 0) {
        return ParseError.MissingResidues;
    }

    // Count total side chain atoms for single allocation
    var total_side_chain: usize = 0;
    for (input.residues) |jr| {
        if (jr.side_chain) |sc| {
            total_side_chain += sc.len;
        }
    }

    const residues = allocator.alloc(Residue, input.residues.len) catch return ParseError.OutOfMemory;
    errdefer allocator.free(residues);

    const side_chain_storage = allocator.alloc(SideChainAtom, total_side_chain) catch return ParseError.OutOfMemory;
    errdefer allocator.free(side_chain_storage);

    var sc_offset: usize = 0;

    for (input.residues, 0..) |jr, i| {
        var res = Residue{};

        // Chain ID
        const cid_len = @min(jr.chain_id.len, 4);
        @memcpy(res.chain_id[0..cid_len], jr.chain_id[0..cid_len]);
        res.chain_id_len = @intCast(cid_len);

        // Sequence ID - use safe cast to handle potential overflow
        res.seq_id = std.math.cast(i32, jr.seq_id) orelse return ParseError.InvalidJson;

        // Compound ID
        const cpd_len = @min(jr.compound_id.len, 4);
        @memcpy(res.compound_id[0..cpd_len], jr.compound_id[0..cpd_len]);
        res.compound_id_len = @intCast(cpd_len);

        // Residue type
        res.residue_type = ResidueType.fromCompoundId(jr.compound_id);

        // Sequential number
        res.number = @intCast(i);

        // Backbone atoms
        var atom_count: u8 = 0;

        if (jr.atoms.N) |pos| {
            res.n = toVec3f32(pos);
            atom_count += 1;
        }
        if (jr.atoms.CA) |pos| {
            res.ca = toVec3f32(pos);
            atom_count += 1;
        }
        if (jr.atoms.C) |pos| {
            res.c = toVec3f32(pos);
            atom_count += 1;
        }
        if (jr.atoms.O) |pos| {
            res.o = toVec3f32(pos);
            atom_count += 1;
        }

        res.complete = (atom_count == 4);

        // Side chain atoms
        if (jr.side_chain) |sc| {
            const sc_start = sc_offset;
            for (sc) |sa| {
                var atom = SideChainAtom{};
                const name_len = @min(sa.name.len, 4);
                @memcpy(atom.name[0..name_len], sa.name[0..name_len]);
                atom.name_len = @intCast(name_len);
                atom.pos = toVec3f32(sa.pos);
                side_chain_storage[sc_offset] = atom;
                sc_offset += 1;
            }
            res.side_chain = side_chain_storage[sc_start..sc_offset];
        }

        residues[i] = res;
    }

    // Parse SS bonds
    const ss_bond_count = if (input.ss_bonds) |bonds| bonds.len else 0;
    const ss_bonds = allocator.alloc(SSBond, ss_bond_count) catch return ParseError.OutOfMemory;
    errdefer allocator.free(ss_bonds);

    if (input.ss_bonds) |bonds| {
        for (bonds, 0..) |jb, i| {
            var bond = SSBond{};
            const c1_len = @min(jb.chain1.len, 4);
            @memcpy(bond.chain1[0..c1_len], jb.chain1[0..c1_len]);
            bond.chain1_len = @intCast(c1_len);
            bond.seq1 = std.math.cast(i32, jb.seq1) orelse return ParseError.InvalidJson;

            const c2_len = @min(jb.chain2.len, 4);
            @memcpy(bond.chain2[0..c2_len], jb.chain2[0..c2_len]);
            bond.chain2_len = @intCast(c2_len);
            bond.seq2 = std.math.cast(i32, jb.seq2) orelse return ParseError.InvalidJson;

            ss_bonds[i] = bond;
        }
    }

    return ParseResult{
        .residues = residues,
        .ss_bonds = ss_bonds,
        .side_chain_storage = side_chain_storage,
        .allocator = allocator,
    };
}

fn toVec3f32(pos: [3]f64) Vec3f32 {
    return .{
        .x = @floatCast(pos[0]),
        .y = @floatCast(pos[1]),
        .z = @floatCast(pos[2]),
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "parseJsonInput - minimal two residues" {
    const input =
        \\{
        \\  "residues": [
        \\    {
        \\      "chain_id": "A",
        \\      "seq_id": 1,
        \\      "compound_id": "MET",
        \\      "atoms": {
        \\        "N": [1.0, 2.0, 3.0],
        \\        "CA": [4.0, 5.0, 6.0],
        \\        "C": [7.0, 8.0, 9.0],
        \\        "O": [10.0, 11.0, 12.0]
        \\      }
        \\    },
        \\    {
        \\      "chain_id": "A",
        \\      "seq_id": 2,
        \\      "compound_id": "ALA",
        \\      "atoms": {
        \\        "N": [13.0, 14.0, 15.0],
        \\        "CA": [16.0, 17.0, 18.0],
        \\        "C": [19.0, 20.0, 21.0],
        \\        "O": [22.0, 23.0, 24.0]
        \\      }
        \\    }
        \\  ]
        \\}
    ;

    const allocator = std.testing.allocator;
    var result = try parseJsonInput(allocator, input);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 2), result.residues.len);
    try std.testing.expectEqualStrings("A", result.residues[0].getChainId());
    try std.testing.expectEqual(@as(i32, 1), result.residues[0].seq_id);
    try std.testing.expectEqualStrings("MET", result.residues[0].getCompoundId());
    try std.testing.expectEqual(ResidueType.met, result.residues[0].residue_type);
    try std.testing.expect(result.residues[0].complete);

    // Check backbone coordinates
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result.residues[0].n.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result.residues[0].ca.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), result.residues[0].c.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), result.residues[0].o.x, 1e-6);
}

test "parseJsonInput - with side chain and ss_bonds" {
    const input =
        \\{
        \\  "residues": [
        \\    {
        \\      "chain_id": "A",
        \\      "seq_id": 1,
        \\      "compound_id": "CYS",
        \\      "atoms": {
        \\        "N": [0.0, 0.0, 0.0],
        \\        "CA": [1.0, 0.0, 0.0],
        \\        "C": [2.0, 0.0, 0.0],
        \\        "O": [2.0, 1.0, 0.0]
        \\      },
        \\      "side_chain": [
        \\        {"name": "CB", "pos": [1.0, 1.0, 0.0]},
        \\        {"name": "SG", "pos": [1.0, 2.0, 0.0]}
        \\      ]
        \\    }
        \\  ],
        \\  "ss_bonds": [
        \\    {"chain1": "A", "seq1": 6, "chain2": "A", "seq2": 127}
        \\  ]
        \\}
    ;

    const allocator = std.testing.allocator;
    var result = try parseJsonInput(allocator, input);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 1), result.residues.len);
    try std.testing.expectEqual(@as(usize, 2), result.residues[0].side_chain.len);
    try std.testing.expectEqualStrings("CB", result.residues[0].side_chain[0].getName());
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result.residues[0].side_chain[0].pos.x, 1e-6);

    try std.testing.expectEqual(@as(usize, 1), result.ss_bonds.len);
    try std.testing.expectEqual(@as(i32, 6), result.ss_bonds[0].seq1);
    try std.testing.expectEqual(@as(i32, 127), result.ss_bonds[0].seq2);
}

test "parseJsonInput - incomplete residue" {
    const input =
        \\{
        \\  "residues": [
        \\    {
        \\      "chain_id": "A",
        \\      "seq_id": 1,
        \\      "compound_id": "GLY",
        \\      "atoms": {
        \\        "N": [0.0, 0.0, 0.0],
        \\        "CA": [1.0, 0.0, 0.0]
        \\      }
        \\    }
        \\  ]
        \\}
    ;

    const allocator = std.testing.allocator;
    var result = try parseJsonInput(allocator, input);
    defer result.deinit();

    try std.testing.expect(!result.residues[0].complete);
}

test "parseJsonInput - empty residues" {
    const input =
        \\{
        \\  "residues": []
        \\}
    ;

    const allocator = std.testing.allocator;
    const result = parseJsonInput(allocator, input);
    try std.testing.expectError(ParseError.MissingResidues, result);
}

test "parseJsonInput - invalid json" {
    const allocator = std.testing.allocator;
    const result = parseJsonInput(allocator, "not valid json");
    try std.testing.expectError(ParseError.InvalidJson, result);
}
