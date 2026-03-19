const std = @import("std");
const Allocator = std.mem.Allocator;
const types = @import("types.zig");
const residue_mod = @import("residue.zig");
const hbond_mod = @import("hbond.zig");

const Residue = residue_mod.Residue;
const BridgeType = types.BridgeType;
const BridgePartner = types.BridgePartner;
const StructureType = types.StructureType;

// ---------------------------------------------------------------------------
// Bridge detection (dssp.cpp:836-858)
// ---------------------------------------------------------------------------

/// Test for a beta bridge between residues at indices i and j.
///
/// Parallel bridge:
///   Pattern I: bond(i+1, j) AND bond(j, i-1)
///   Pattern II: bond(j+1, i) AND bond(i, j-1)
///
/// Antiparallel bridge:
///   Pattern III: bond(i+1, j-1) AND bond(j+1, i-1)
///   Pattern IV:  bond(j, i) AND bond(i, j)
pub fn testBridge(residues: []const Residue, i: u32, j: u32) BridgeType {
    const n: u32 = @intCast(residues.len);
    if (i == 0 or j == 0 or i + 1 >= n or j + 1 >= n) {
        return .none;
    }

    // Check chain continuity around both residues
    if (!residue_mod.noChainBreak(residues, i - 1, i + 1)) {
        return .none;
    }
    if (!residue_mod.noChainBreak(residues, j - 1, j + 1)) {
        return .none;
    }

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

// ---------------------------------------------------------------------------
// Internal bridge/ladder data structure
// ---------------------------------------------------------------------------

const Bridge = struct {
    bridge_type: BridgeType,
    sheet: u32 = 0,
    ladder: u32 = 0,
    // Chain ID for sorting (copied from first residue)
    chain_id: [4]u8 = .{ ' ', ' ', ' ', ' ' },
    chain_id_len: u8 = 0,
    // Residue indices on the i-side and j-side of the bridge
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
        return if (self.i_indices.items.len > 0) self.i_indices.items[self.i_indices.items.len - 1] else 0;
    }

    fn jBegin(self: *const Bridge) u32 {
        return if (self.j_indices.items.len > 0) self.j_indices.items[0] else 0;
    }

    fn jEnd(self: *const Bridge) u32 {
        return if (self.j_indices.items.len > 0) self.j_indices.items[self.j_indices.items.len - 1] else 0;
    }

    /// Comparison for sorting bridges.
    ///
    /// Sort order: chain_id → iBegin → size (descending) → jBegin
    ///
    /// Note: C++ dssp only uses (chain, iBegin) with no tiebreaker, making the sort
    /// order undefined for bridges with equal iBegin. This implementation adds
    /// deterministic tiebreakers to ensure consistent results:
    ///   - Larger bridges first (more likely to form extended ladders)
    ///   - jBegin as final tiebreaker for complete determinism
    fn lessThan(_: void, a: Bridge, b: Bridge) bool {
        const a_chain = a.chain_id[0..a.chain_id_len];
        const b_chain = b.chain_id[0..b.chain_id_len];
        const chain_cmp = std.mem.order(u8, a_chain, b_chain);
        if (chain_cmp != .eq) {
            return chain_cmp == .lt;
        }
        if (a.iBegin() != b.iBegin()) {
            return a.iBegin() < b.iBegin();
        }
        // Tiebreaker 1: larger bridges first (prefer merging bigger structures)
        if (a.i_indices.items.len != b.i_indices.items.len) {
            return a.i_indices.items.len > b.i_indices.items.len;
        }
        // Tiebreaker 2: jBegin for complete determinism
        return a.jBegin() < b.jBegin();
    }
};

// ---------------------------------------------------------------------------
// Sheet calculation (dssp.cpp:836-1129)
// ---------------------------------------------------------------------------

/// Calculate beta sheets from H-bond data.
///
/// Steps:
/// 1. Find all bridges (pairs of residues with H-bond patterns)
/// 2. Extend bridges into ladders (consecutive bridges)
/// 3. Merge ladders with bulges
/// 4. Group ladders into sheets (via shared residues)
/// 5. Assign B (isolated bridge) and E (strand) to residues
pub fn calculateBetaSheets(residues: []Residue, pairs: [][2]u32, allocator: Allocator) !void {
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

        // Try to extend an existing bridge/ladder
        var extended = false;
        for (bridges.items) |*bridge| {
            if (bridge.bridge_type != bt) continue;

            const bj_end = bridge.jEnd();

            const bi_end = bridge.iEnd();
            if (i != bi_end + 1) continue; // i must be consecutive (dssp.cpp:899)

            if (bt == .parallel) {
                // Extend parallel ladder: both i and j consecutive (dssp.cpp:899-908)
                if (j == bj_end + 1) {
                    try bridge.i_indices.append(allocator, i);
                    try bridge.j_indices.append(allocator, j);
                    extended = true;
                    break;
                }
            } else {
                // Extend antiparallel ladder: both i and j consecutive (dssp.cpp:899,910-916)
                if (j + 1 == bridge.jBegin()) {
                    try bridge.i_indices.append(allocator, i);
                    // Insert at front for antiparallel
                    try bridge.j_indices.insert(allocator, 0, j);
                    extended = true;
                    break;
                }
            }
        }

        if (!extended) {
            // Create new bridge
            var new_bridge = Bridge{ .bridge_type = bt };
            // Copy chain_id from the first residue on i-side
            const cid = residues[i].getChainId();
            const cid_len = @min(cid.len, 4);
            @memcpy(new_bridge.chain_id[0..cid_len], cid[0..cid_len]);
            new_bridge.chain_id_len = @intCast(cid_len);
            try new_bridge.i_indices.append(allocator, i);
            try new_bridge.j_indices.append(allocator, j);
            try bridges.append(allocator, new_bridge);
        }
    }

    // Step 2: Sort bridges by chain and i-index (dssp.cpp:934)
    std.mem.sort(Bridge, bridges.items, {}, Bridge.lessThan);

    // Step 3: Merge bridges with bulges (dssp.cpp:933-975)
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

            // Skip if too far apart
            if (ibj > iei and ibj - iei >= 6) {
                bj += 1;
                continue;
            }

            // Skip if overlapping on i-side (dssp.cpp:952-953)
            // Using non-strict inequality (>= and <=) to match dssp.cpp behavior.
            // When two bridges share an i-endpoint (e.g., iei=3381 and ibj=3381),
            // only the first one can be merged; the second is skipped.
            if (iei >= ibj and ibi <= iej) {
                bj += 1;
                continue;
            }

            const jbi = bridges.items[bi].jBegin();
            const jei = bridges.items[bi].jEnd();
            const jbj = bridges.items[bj].jBegin();
            const jej = bridges.items[bj].jEnd();

            // Chain break check (dssp.cpp:950-951)
            // Skip if there's a chain break in the merge range
            if (!residue_mod.noChainBreak(residues, @min(ibi, ibj), @max(iei, iej))) {
                bj += 1;
                continue;
            }
            if (!residue_mod.noChainBreak(residues, @min(jbi, jbj), @max(jei, jej))) {
                bj += 1;
                continue;
            }

            var is_bulge = false;
            if (bridges.items[bi].bridge_type == .parallel) {
                // mkdssp: bulge = ((jbj - jei < 6 and ibj - iei < 3) or (jbj - jei < 3))
                // Note: jbj >= jei handles the case where both bridges end at same j-index
                is_bulge = (jbj >= jei and jbj - jei < 6 and ibj > iei and ibj - iei < 3) or
                    (jbj >= jei and jbj - jei < 3);
            } else {
                // mkdssp: bulge = ((jbi - jej < 6 and ibj - iei < 3) or (jbi - jej < 3))
                is_bulge = (jbi >= jej and jbi - jej < 6 and ibj > iei and ibj - iei < 3) or
                    (jbi >= jej and jbi - jej < 3);
            }

            if (is_bulge) {
                // Merge bridge bj into bi
                for (bridges.items[bj].i_indices.items) |idx| {
                    try bridges.items[bi].i_indices.append(allocator, idx);
                }

                if (bridges.items[bi].bridge_type == .parallel) {
                    for (bridges.items[bj].j_indices.items) |idx| {
                        try bridges.items[bi].j_indices.append(allocator, idx);
                    }
                } else {
                    // Antiparallel: insert at beginning
                    var insert_pos: usize = 0;
                    for (bridges.items[bj].j_indices.items) |idx| {
                        try bridges.items[bi].j_indices.insert(allocator, insert_pos, idx);
                        insert_pos += 1;
                    }
                }

                bridges.items[bj].deinit(allocator);
                _ = bridges.orderedRemove(bj);
                // Don't increment bj
            } else {
                bj += 1;
            }
        }
    }

    // Step 4: Group ladders into sheets (dssp.cpp:977-1039)
    var ladder_id: u32 = 0;
    var sheet_id: u32 = 1;

    // Mark which bridges have been assigned to a sheet
    const assigned = try allocator.alloc(bool, bridges.items.len);
    defer allocator.free(assigned);
    @memset(assigned, false);

    for (0..bridges.items.len) |start| {
        if (assigned[start]) continue;

        // BFS/flood-fill to find all connected ladders
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

        // Assign sheet/ladder IDs
        for (queue.items) |idx| {
            bridges.items[idx].sheet = sheet_id;
            bridges.items[idx].ladder = ladder_id;
            ladder_id += 1;
        }
        sheet_id += 1;
    }

    // Step 5: Assign secondary structure to residues (dssp.cpp:1066-1107)
    // Note: mkdssp assigns to ALL residues from front to back, filling gaps
    for (bridges.items) |*bridge| {
        const is_ladder = bridge.i_indices.items.len > 1;
        const ss: StructureType = if (is_ladder) .strand else .beta_bridge;

        // Assign beta partners for i-side (only actual indices get partners)
        // For parallel: i[k] pairs with j[k]
        // For antiparallel: i[k] pairs with j[len-1-k] (reverse order, dssp.cpp:1086)
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

        // Assign SS to ALL residues from iBegin to iEnd (filling gaps)
        const i_begin = bridge.iBegin();
        const i_end = bridge.iEnd();
        var idx: u32 = i_begin;
        while (idx <= i_end) : (idx += 1) {
            if (residues[idx].secondary_structure != .strand) {
                residues[idx].secondary_structure = ss;
            }
            residues[idx].sheet = bridge.sheet;
        }

        // Assign beta partners for j-side (only actual indices get partners)
        // For parallel: j[k] pairs with i[k]
        // For antiparallel: j[k] pairs with i[len-1-k] (reverse order, dssp.cpp:1090)
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

        // Assign SS to ALL residues from jBegin to jEnd (filling gaps)
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

/// Check if two bridges share any residue indices (dssp.cpp:862-868).
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "testBridge - no bridge without H-bonds" {
    var residues: [5]Residue = undefined;
    for (&residues, 0..) |*r, idx| {
        r.* = Residue{
            .number = @intCast(idx),
            .complete = true,
        };
    }
    const result = testBridge(&residues, 2, 4);
    try std.testing.expectEqual(BridgeType.none, result);
}

test "testBridge - parallel bridge" {
    var residues: [7]Residue = undefined;
    for (&residues, 0..) |*r, idx| {
        r.* = Residue{
            .number = @intCast(idx),
            .complete = true,
        };
    }

    // Set up H-bonds for parallel pattern I: bond(i+1, j) AND bond(j, i-1)
    // i=2, j=4 → bond(3, 4) AND bond(4, 1)
    residues[3].hbond_acceptor[0] = .{ .residue_index = 4, .energy = -2.0 };
    residues[4].hbond_acceptor[0] = .{ .residue_index = 1, .energy = -2.0 };

    const result = testBridge(&residues, 2, 4);
    try std.testing.expectEqual(BridgeType.parallel, result);
}

test "testBridge - antiparallel bridge" {
    var residues: [5]Residue = undefined;
    for (&residues, 0..) |*r, idx| {
        r.* = Residue{
            .number = @intCast(idx),
            .complete = true,
        };
    }

    // Antiparallel pattern IV: bond(j, i) AND bond(i, j)
    // i=1, j=3 → bond(3, 1) AND bond(1, 3)
    residues[3].hbond_acceptor[0] = .{ .residue_index = 1, .energy = -2.0 };
    residues[1].hbond_acceptor[0] = .{ .residue_index = 3, .energy = -2.0 };

    const result = testBridge(&residues, 1, 3);
    try std.testing.expectEqual(BridgeType.anti_parallel, result);
}

test "bridgesLinked - shared index" {
    const allocator = std.testing.allocator;

    var a = Bridge{ .bridge_type = .parallel };
    defer a.deinit(allocator);
    try a.i_indices.append(allocator, 5);
    try a.j_indices.append(allocator, 10);

    var b = Bridge{ .bridge_type = .parallel };
    defer b.deinit(allocator);
    try b.i_indices.append(allocator, 10); // shares with a.j_indices
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

test "Bridge.lessThan - equal iBegin uses jBegin as tiebreaker" {
    const allocator = std.testing.allocator;

    var a = Bridge{ .bridge_type = .parallel };
    defer a.deinit(allocator);
    try a.i_indices.append(allocator, 72);
    try a.j_indices.append(allocator, 94);

    var b = Bridge{ .bridge_type = .parallel };
    defer b.deinit(allocator);
    try b.i_indices.append(allocator, 72);
    try b.j_indices.append(allocator, 91);

    // Same iBegin (72) and same size (1), so use jBegin as tiebreaker
    // b.jBegin (91) < a.jBegin (94), so b comes before a
    try std.testing.expect(!Bridge.lessThan({}, a, b)); // a is NOT less than b
    try std.testing.expect(Bridge.lessThan({}, b, a)); // b IS less than a
}

test "Bridge.lessThan - equal iBegin uses size as first tiebreaker" {
    const allocator = std.testing.allocator;

    // Bridge a: i=[72,73] (size=2), j=[94]
    var a = Bridge{ .bridge_type = .parallel };
    defer a.deinit(allocator);
    try a.i_indices.append(allocator, 72);
    try a.i_indices.append(allocator, 73);
    try a.j_indices.append(allocator, 94);

    // Bridge b: i=[72] (size=1), j=[91]
    var b = Bridge{ .bridge_type = .parallel };
    defer b.deinit(allocator);
    try b.i_indices.append(allocator, 72);
    try b.j_indices.append(allocator, 91);

    // Same iBegin (72), but a has larger size (2 > 1)
    // Larger bridges come first, so a < b
    try std.testing.expect(Bridge.lessThan({}, a, b)); // a IS less than b (larger first)
    try std.testing.expect(!Bridge.lessThan({}, b, a)); // b is NOT less than a
}

test "Bridge.lessThan - primary sort by iBegin" {
    const allocator = std.testing.allocator;

    // Bridge a: i=72, j=100
    var a = Bridge{ .bridge_type = .parallel };
    defer a.deinit(allocator);
    try a.i_indices.append(allocator, 72);
    try a.j_indices.append(allocator, 100);

    // Bridge b: i=74, j=50 (higher i, lower j)
    var b = Bridge{ .bridge_type = .parallel };
    defer b.deinit(allocator);
    try b.i_indices.append(allocator, 74);
    try b.j_indices.append(allocator, 50);

    // a should come before b because iBegin(72) < iBegin(74), regardless of j
    try std.testing.expect(Bridge.lessThan({}, a, b));
    try std.testing.expect(!Bridge.lessThan({}, b, a));
}

test "Bridge.lessThan - chain_id takes precedence over all other keys" {
    const allocator = std.testing.allocator;

    // Bridge a: chain='B', i=10
    var a = Bridge{ .bridge_type = .parallel };
    defer a.deinit(allocator);
    a.chain_id[0] = 'B';
    a.chain_id_len = 1;
    try a.i_indices.append(allocator, 10);
    try a.j_indices.append(allocator, 20);

    // Bridge b: chain='A', i=10 (same iBegin, but lower chain)
    var b = Bridge{ .bridge_type = .parallel };
    defer b.deinit(allocator);
    b.chain_id[0] = 'A';
    b.chain_id_len = 1;
    try b.i_indices.append(allocator, 10);
    try b.j_indices.append(allocator, 20);

    // Chain 'A' < 'B', so b comes before a regardless of other keys
    try std.testing.expect(!Bridge.lessThan({}, a, b)); // a is NOT less than b
    try std.testing.expect(Bridge.lessThan({}, b, a)); // b IS less than a
}
