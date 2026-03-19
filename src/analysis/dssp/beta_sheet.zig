// Beta-sheet detection for native DSSP.
//
// Operates entirely on DsspResidue H-bond records; no Frame coordinates
// needed. Ported from src/dssp/beta_sheet.zig.

const std = @import("std");
const Allocator = std.mem.Allocator;
const types = @import("types.zig");
const backbone = @import("backbone.zig");
const hbond_mod = @import("hbond.zig");

const DsspResidue = types.DsspResidue;
const BridgeType = types.BridgeType;
const BridgePartner = types.BridgePartner;
const StructureType = types.StructureType;

// ============================================================================
// Bridge detection
// ============================================================================

/// Test for a beta bridge between residues at indices i and j.
///
/// Parallel bridge:
///   Pattern I:  bond(i+1, j) AND bond(j, i-1)
///   Pattern II: bond(j+1, i) AND bond(i, j-1)
///
/// Antiparallel bridge:
///   Pattern III: bond(i+1, j-1) AND bond(j+1, i-1)
///   Pattern IV:  bond(j, i)     AND bond(i, j)
pub fn testBridge(residues: []const DsspResidue, i: u32, j: u32) BridgeType {
    const n: u32 = @intCast(residues.len);
    if (i == 0 or j == 0 or i + 1 >= n or j + 1 >= n) return .none;

    if (!backbone.noChainBreak(residues, i - 1, i + 1)) return .none;
    if (!backbone.noChainBreak(residues, j - 1, j + 1)) return .none;

    // Parallel patterns
    if ((hbond_mod.testBond(residues, i + 1, j) and hbond_mod.testBond(residues, j, i - 1)) or
        (hbond_mod.testBond(residues, j + 1, i) and hbond_mod.testBond(residues, i, j - 1)))
    {
        return .parallel;
    }

    // Antiparallel patterns
    if ((hbond_mod.testBond(residues, i + 1, j - 1) and hbond_mod.testBond(residues, j + 1, i - 1)) or
        (hbond_mod.testBond(residues, j, i) and hbond_mod.testBond(residues, i, j)))
    {
        return .anti_parallel;
    }

    return .none;
}

// ============================================================================
// Internal bridge/ladder data structure
// ============================================================================

const Bridge = struct {
    bridge_type: BridgeType,
    sheet: u32 = 0,
    ladder: u32 = 0,
    chain_index: u32 = 0,
    i_indices: std.ArrayListAligned(u32, null) = .empty,
    j_indices: std.ArrayListAligned(u32, null) = .empty,

    fn deinit(self: *Bridge, allocator: Allocator) void {
        self.i_indices.deinit(allocator);
        self.j_indices.deinit(allocator);
    }

    fn iBegin(self: *const Bridge) u32 {
        return if (self.i_indices.items.len > 0) self.i_indices.items[0] else 0;
    }

    fn iEnd(self: *const Bridge) u32 {
        return if (self.i_indices.items.len > 0)
            self.i_indices.items[self.i_indices.items.len - 1]
        else
            0;
    }

    fn jBegin(self: *const Bridge) u32 {
        return if (self.j_indices.items.len > 0) self.j_indices.items[0] else 0;
    }

    fn jEnd(self: *const Bridge) u32 {
        return if (self.j_indices.items.len > 0)
            self.j_indices.items[self.j_indices.items.len - 1]
        else
            0;
    }

    /// Sort: chain_index → iBegin → size (desc) → jBegin
    fn lessThan(_: void, a: Bridge, b: Bridge) bool {
        if (a.chain_index != b.chain_index) return a.chain_index < b.chain_index;
        if (a.iBegin() != b.iBegin()) return a.iBegin() < b.iBegin();
        if (a.i_indices.items.len != b.i_indices.items.len)
            return a.i_indices.items.len > b.i_indices.items.len;
        return a.jBegin() < b.jBegin();
    }
};

// ============================================================================
// Sheet calculation
// ============================================================================

/// Calculate beta sheets from H-bond data.
///
/// Steps:
///  1. Find all bridges (H-bond pattern matching)
///  2. Extend bridges into ladders (consecutive bridges)
///  3. Merge ladders with bulges
///  4. Group ladders into sheets (flood-fill)
///  5. Assign B (beta_bridge) and E (strand) to residues
///  6. Assign strand numbers within each sheet
pub fn calculateBetaSheets(
    residues: []DsspResidue,
    pairs: [][2]u32,
    allocator: Allocator,
) !void {
    // Step 1: Find all bridges
    var bridges: std.ArrayListAligned(Bridge, null) = .empty;
    defer {
        for (bridges.items) |*b| b.deinit(allocator);
        bridges.deinit(allocator);
    }

    for (pairs) |pair| {
        const i = pair[0];
        const j = pair[1];

        const bt = testBridge(residues, i, j);
        if (bt == .none) continue;

        var extended = false;
        for (bridges.items) |*bridge| {
            if (bridge.bridge_type != bt) continue;
            const bi_end = bridge.iEnd();
            if (i != bi_end + 1) continue;

            if (bt == .parallel) {
                const bj_end = bridge.jEnd();
                if (j == bj_end + 1) {
                    try bridge.i_indices.append(allocator, i);
                    try bridge.j_indices.append(allocator, j);
                    extended = true;
                    break;
                }
            } else {
                // antiparallel: j decreases
                if (j + 1 == bridge.jBegin()) {
                    try bridge.i_indices.append(allocator, i);
                    try bridge.j_indices.insert(allocator, 0, j);
                    extended = true;
                    break;
                }
            }
        }

        if (!extended) {
            var new_bridge = Bridge{
                .bridge_type = bt,
                .chain_index = residues[i].chain_index,
            };
            try new_bridge.i_indices.append(allocator, i);
            try new_bridge.j_indices.append(allocator, j);
            try bridges.append(allocator, new_bridge);
        }
    }

    // Step 2: Sort bridges
    std.mem.sort(Bridge, bridges.items, {}, Bridge.lessThan);

    // Step 3: Merge bridges with bulges
    var bi: usize = 0;
    while (bi < bridges.items.len) : (bi += 1) {
        var bj: usize = bi + 1;
        while (bj < bridges.items.len) {
            if (bridges.items[bi].bridge_type != bridges.items[bj].bridge_type) {
                bj += 1;
                continue;
            }

            const ibi = bridges.items[bi].iBegin();
            const iei = bridges.items[bi].iEnd();
            const ibj = bridges.items[bj].iBegin();
            const iej = bridges.items[bj].iEnd();

            if (ibj > iei and ibj - iei >= 6) {
                bj += 1;
                continue;
            }
            if (iei >= ibj and ibi <= iej) {
                bj += 1;
                continue;
            }

            const jbi = bridges.items[bi].jBegin();
            const jei = bridges.items[bi].jEnd();
            const jbj = bridges.items[bj].jBegin();
            const jej = bridges.items[bj].jEnd();

            if (!backbone.noChainBreak(residues, @min(ibi, ibj), @max(iei, iej))) {
                bj += 1;
                continue;
            }
            if (!backbone.noChainBreak(residues, @min(jbi, jbj), @max(jei, jej))) {
                bj += 1;
                continue;
            }

            var is_bulge = false;
            if (bridges.items[bi].bridge_type == .parallel) {
                is_bulge = (jbj >= jei and jbj - jei < 6 and ibj > iei and ibj - iei < 3) or
                    (jbj >= jei and jbj - jei < 3);
            } else {
                is_bulge = (jbi >= jej and jbi - jej < 6 and ibj > iei and ibj - iei < 3) or
                    (jbi >= jej and jbi - jej < 3);
            }

            if (is_bulge) {
                for (bridges.items[bj].i_indices.items) |idx| {
                    try bridges.items[bi].i_indices.append(allocator, idx);
                }
                if (bridges.items[bi].bridge_type == .parallel) {
                    for (bridges.items[bj].j_indices.items) |idx| {
                        try bridges.items[bi].j_indices.append(allocator, idx);
                    }
                } else {
                    var insert_pos: usize = 0;
                    for (bridges.items[bj].j_indices.items) |idx| {
                        try bridges.items[bi].j_indices.insert(allocator, insert_pos, idx);
                        insert_pos += 1;
                    }
                }
                bridges.items[bj].deinit(allocator);
                _ = bridges.orderedRemove(bj);
            } else {
                bj += 1;
            }
        }
    }

    // Step 4: Group ladders into sheets (flood-fill)
    var ladder_id: u32 = 0;
    var sheet_id: u32 = 1;

    const assigned = try allocator.alloc(bool, bridges.items.len);
    defer allocator.free(assigned);
    @memset(assigned, false);

    for (0..bridges.items.len) |start| {
        if (assigned[start]) continue;

        var queue: std.ArrayListAligned(usize, null) = .empty;
        defer queue.deinit(allocator);
        try queue.append(allocator, start);
        assigned[start] = true;

        var qi: usize = 0;
        while (qi < queue.items.len) : (qi += 1) {
            const a = queue.items[qi];
            for (0..bridges.items.len) |b_idx| {
                if (assigned[b_idx]) continue;
                if (bridgesLinked(&bridges.items[a], &bridges.items[b_idx])) {
                    assigned[b_idx] = true;
                    try queue.append(allocator, b_idx);
                }
            }
        }

        for (queue.items) |idx| {
            bridges.items[idx].sheet = sheet_id;
            bridges.items[idx].ladder = ladder_id;
            ladder_id += 1;
        }
        sheet_id += 1;
    }

    // Step 5: Assign secondary structure to residues
    for (bridges.items) |*bridge| {
        const is_ladder = bridge.i_indices.items.len > 1;
        const ss: StructureType = if (is_ladder) .strand else .beta_bridge;

        const j_len = bridge.j_indices.items.len;
        for (bridge.i_indices.items, 0..) |idx, k| {
            const j_partner: ?u32 = if (j_len > k) blk: {
                const j_idx = if (bridge.bridge_type == .parallel) k else j_len - 1 - k;
                break :blk bridge.j_indices.items[j_idx];
            } else null;

            if (residues[idx].beta_partner[0].residue_index == null) {
                residues[idx].beta_partner[0] = .{
                    .residue_index = j_partner,
                    .ladder = bridge.ladder,
                    .parallel = bridge.bridge_type == .parallel,
                };
            } else {
                residues[idx].beta_partner[1] = .{
                    .residue_index = j_partner,
                    .ladder = bridge.ladder,
                    .parallel = bridge.bridge_type == .parallel,
                };
            }
        }

        const i_begin = bridge.iBegin();
        const i_end = bridge.iEnd();
        var idx: u32 = i_begin;
        while (idx <= i_end) : (idx += 1) {
            if (residues[idx].secondary_structure != .strand) {
                residues[idx].secondary_structure = ss;
            }
            residues[idx].sheet = bridge.sheet;
        }

        const i_len = bridge.i_indices.items.len;
        for (bridge.j_indices.items, 0..) |jdx, k| {
            const i_partner: ?u32 = if (i_len > k) blk: {
                const i_idx = if (bridge.bridge_type == .parallel) k else i_len - 1 - k;
                break :blk bridge.i_indices.items[i_idx];
            } else null;

            if (residues[jdx].beta_partner[0].residue_index == null) {
                residues[jdx].beta_partner[0] = .{
                    .residue_index = i_partner,
                    .ladder = bridge.ladder,
                    .parallel = bridge.bridge_type == .parallel,
                };
            } else {
                residues[jdx].beta_partner[1] = .{
                    .residue_index = i_partner,
                    .ladder = bridge.ladder,
                    .parallel = bridge.bridge_type == .parallel,
                };
            }
        }

        const j_begin = bridge.jBegin();
        const j_end = bridge.jEnd();
        var jdx: u32 = @min(j_begin, j_end);
        const j_max: u32 = @max(j_begin, j_end);
        while (jdx <= j_max) : (jdx += 1) {
            if (residues[jdx].secondary_structure != .strand) {
                residues[jdx].secondary_structure = ss;
            }
            residues[jdx].sheet = bridge.sheet;
        }
    }

    // Step 6: Assign strand numbers within each sheet
    var strand_nr: u32 = 0;
    var s: u32 = 1;
    while (s < sheet_id) : (s += 1) {
        var last_nr: i64 = -2;
        for (residues) |*res| {
            if (res.sheet != s) continue;
            if (@as(i64, @intCast(res.number)) != last_nr + 1) {
                strand_nr += 1;
            }
            res.strand = strand_nr;
            last_nr = @intCast(res.number);
        }
    }
}

fn bridgesLinked(a: *const Bridge, b: *const Bridge) bool {
    return slicesShareElement(a.i_indices.items, b.i_indices.items) or
        slicesShareElement(a.i_indices.items, b.j_indices.items) or
        slicesShareElement(a.j_indices.items, b.i_indices.items) or
        slicesShareElement(a.j_indices.items, b.j_indices.items);
}

fn slicesShareElement(a: []const u32, b: []const u32) bool {
    for (a) |va| {
        for (b) |vb| {
            if (va == vb) return true;
        }
    }
    return false;
}

// ============================================================================
// Tests
// ============================================================================

test "testBridge - no bridge without H-bonds" {
    var residues: [5]DsspResidue = undefined;
    for (&residues, 0..) |*r, idx| {
        r.* = DsspResidue{ .number = @intCast(idx), .complete = true };
    }
    try std.testing.expectEqual(BridgeType.none, testBridge(&residues, 2, 4));
}

test "testBridge - parallel bridge pattern I" {
    var residues: [7]DsspResidue = undefined;
    for (&residues, 0..) |*r, idx| {
        r.* = DsspResidue{ .number = @intCast(idx), .complete = true };
    }
    // i=2, j=4 → bond(i+1=3, j=4) AND bond(j=4, i-1=1)
    residues[3].hbond_acceptor[0] = .{ .residue_index = 4, .energy = -2.0 };
    residues[4].hbond_acceptor[0] = .{ .residue_index = 1, .energy = -2.0 };

    try std.testing.expectEqual(BridgeType.parallel, testBridge(&residues, 2, 4));
}

test "testBridge - antiparallel bridge pattern IV" {
    var residues: [5]DsspResidue = undefined;
    for (&residues, 0..) |*r, idx| {
        r.* = DsspResidue{ .number = @intCast(idx), .complete = true };
    }
    // i=1, j=3 → bond(j=3, i=1) AND bond(i=1, j=3)
    residues[3].hbond_acceptor[0] = .{ .residue_index = 1, .energy = -2.0 };
    residues[1].hbond_acceptor[0] = .{ .residue_index = 3, .energy = -2.0 };

    try std.testing.expectEqual(BridgeType.anti_parallel, testBridge(&residues, 1, 3));
}

test "testBridge - boundary check at i=0" {
    var residues: [3]DsspResidue = undefined;
    for (&residues, 0..) |*r, idx| {
        r.* = DsspResidue{ .number = @intCast(idx), .complete = true };
    }
    try std.testing.expectEqual(BridgeType.none, testBridge(&residues, 0, 2));
}

test "calculateBetaSheets - no pairs produces no sheets" {
    const allocator = std.testing.allocator;
    var residues: [4]DsspResidue = undefined;
    for (&residues, 0..) |*r, idx| {
        r.* = DsspResidue{ .number = @intCast(idx), .complete = true };
    }
    const pairs: [][2]u32 = &.{};
    try calculateBetaSheets(&residues, pairs, allocator);

    for (residues) |res| {
        try std.testing.expectEqual(StructureType.loop, res.secondary_structure);
    }
}

test "calculateBetaSheets - antiparallel bridge assigned" {
    const allocator = std.testing.allocator;

    var residues: [5]DsspResidue = undefined;
    for (&residues, 0..) |*r, idx| {
        r.* = DsspResidue{ .number = @intCast(idx), .complete = true };
    }

    // Set antiparallel H-bonds: i=1, j=3 (pattern IV)
    residues[3].hbond_acceptor[0] = .{ .residue_index = 1, .energy = -2.0 };
    residues[1].hbond_acceptor[0] = .{ .residue_index = 3, .energy = -2.0 };

    // Only pair (1,3) is needed
    var pairs_buf = [_][2]u32{.{ 1, 3 }};
    try calculateBetaSheets(&residues, &pairs_buf, allocator);

    // Both residues 1 and 3 should get beta_bridge (single bridge)
    try std.testing.expectEqual(StructureType.beta_bridge, residues[1].secondary_structure);
    try std.testing.expectEqual(StructureType.beta_bridge, residues[3].secondary_structure);
}

test "bridgesLinked - shared index" {
    const allocator = std.testing.allocator;

    var a = Bridge{ .bridge_type = .parallel };
    defer a.deinit(allocator);
    try a.i_indices.append(allocator, 5);
    try a.j_indices.append(allocator, 10);

    var b = Bridge{ .bridge_type = .parallel };
    defer b.deinit(allocator);
    try b.i_indices.append(allocator, 10);
    try b.j_indices.append(allocator, 15);

    try std.testing.expect(bridgesLinked(&a, &b));
}

test "bridgesLinked - no shared index" {
    const allocator = std.testing.allocator;

    var a = Bridge{ .bridge_type = .parallel };
    defer a.deinit(allocator);
    try a.i_indices.append(allocator, 5);
    try a.j_indices.append(allocator, 10);

    var b = Bridge{ .bridge_type = .parallel };
    defer b.deinit(allocator);
    try b.i_indices.append(allocator, 20);
    try b.j_indices.append(allocator, 25);

    try std.testing.expect(!bridgesLinked(&a, &b));
}
