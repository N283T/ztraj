//! mmCIF Parser for DSSP.
//!
//! This module provides an mmCIF parser that extracts atom coordinates
//! and groups them into residues for DSSP secondary structure calculation.
//!
//! ## Extracted Data
//!
//! From `_atom_site` category:
//! - Backbone atoms: N, CA, C, O (required for DSSP)
//! - Side chain atoms: all non-backbone, non-hydrogen atoms
//!
//! From `_struct_conn` category:
//! - Disulfide bonds (conn_type_id = "disulf")
//!
//! ## Usage
//!
//! ```zig
//! const parser = @import("mmcif_parser.zig");
//!
//! var mmcif = parser.MmcifParser.init(allocator);
//! const result = try mmcif.parseFile("structure.cif");
//! defer result.deinit();
//! ```

const std = @import("std");
const Allocator = std.mem.Allocator;
const cif = @import("cif.zig");
const types = @import("types.zig");
const residue_mod = @import("residue.zig");
const json_parser = @import("json_parser.zig");

const Vec3f32 = types.Vec3f32;
const ResidueType = types.ResidueType;
const Residue = residue_mod.Residue;
const SideChainAtom = residue_mod.SideChainAtom;
const SSBond = json_parser.SSBond;
const ParseResult = json_parser.ParseResult;

/// Maximum file size for reading (256 MB)
const MAX_FILE_SIZE = 256 * 1024 * 1024;

/// Error types for mmCIF parsing
pub const ParseError = error{
    /// No atom_site loop found in the file
    NoAtomSiteLoop,
    /// Missing required coordinate field (Cartn_x, Cartn_y, or Cartn_z)
    MissingCoordinateField,
    /// Invalid coordinate value (not a valid number)
    InvalidCoordinate,
    /// File read error
    FileReadError,
    /// Memory allocation failed
    OutOfMemory,
};

/// Column indices for atom_site fields
const AtomSiteColumns = struct {
    cartn_x: ?usize = null,
    cartn_y: ?usize = null,
    cartn_z: ?usize = null,
    type_symbol: ?usize = null,
    label_atom_id: ?usize = null,
    auth_atom_id: ?usize = null,
    label_comp_id: ?usize = null,
    auth_comp_id: ?usize = null,
    label_asym_id: ?usize = null,
    auth_asym_id: ?usize = null,
    label_seq_id: ?usize = null,
    auth_seq_id: ?usize = null,
    pdbx_pdb_ins_code: ?usize = null,
    group_pdb: ?usize = null,
    label_alt_id: ?usize = null,
    pdbx_pdb_model_num: ?usize = null,

    /// Check if required coordinate fields are present
    fn hasRequiredFields(self: AtomSiteColumns) bool {
        return self.cartn_x != null and self.cartn_y != null and self.cartn_z != null;
    }

    /// Get atom name column (prefer label over auth)
    fn getAtomNameCol(self: AtomSiteColumns) ?usize {
        return self.label_atom_id orelse self.auth_atom_id;
    }

    /// Get residue name column (prefer label over auth)
    fn getResNameCol(self: AtomSiteColumns) ?usize {
        return self.label_comp_id orelse self.auth_comp_id;
    }

    /// Get chain ID column (prefer label over auth)
    fn getChainCol(self: AtomSiteColumns) ?usize {
        return self.label_asym_id orelse self.auth_asym_id;
    }

    /// Get residue sequence number column (prefer label over auth)
    fn getResSeqCol(self: AtomSiteColumns) ?usize {
        return self.label_seq_id orelse self.auth_seq_id;
    }
};

/// Column indices for struct_conn fields
const StructConnColumns = struct {
    conn_type_id: ?usize = null,
    ptnr1_label_asym_id: ?usize = null,
    ptnr1_auth_asym_id: ?usize = null,
    ptnr1_label_seq_id: ?usize = null,
    ptnr1_auth_seq_id: ?usize = null,
    ptnr2_label_asym_id: ?usize = null,
    ptnr2_auth_asym_id: ?usize = null,
    ptnr2_label_seq_id: ?usize = null,
    ptnr2_auth_seq_id: ?usize = null,

    fn getChain1Col(self: StructConnColumns) ?usize {
        return self.ptnr1_label_asym_id orelse self.ptnr1_auth_asym_id;
    }

    fn getChain2Col(self: StructConnColumns) ?usize {
        return self.ptnr2_label_asym_id orelse self.ptnr2_auth_asym_id;
    }

    fn getSeq1Col(self: StructConnColumns) ?usize {
        return self.ptnr1_label_seq_id orelse self.ptnr1_auth_seq_id;
    }

    fn getSeq2Col(self: StructConnColumns) ?usize {
        return self.ptnr2_label_seq_id orelse self.ptnr2_auth_seq_id;
    }
};

/// Per-residue tracking during streaming parse
const ResidueParseInfo = struct {
    sc_start: usize,
    backbone_count: u8 = 0,
};

/// mmCIF Parser
pub const MmcifParser = struct {
    allocator: Allocator,
    /// Filter to include only ATOM records (exclude HETATM)
    atom_only: bool = true,
    /// Filter to include only first alternate location
    first_alt_loc_only: bool = true,
    /// Model number to extract (null = first model)
    model_num: ?u32 = 1,

    pub fn init(allocator: Allocator) MmcifParser {
        return .{ .allocator = allocator };
    }

    /// Parse mmCIF from a string
    /// Streams atom data directly into Residue structs without intermediate TempAtom buffer
    pub fn parse(self: *MmcifParser, source: []const u8) ParseError!ParseResult {
        var residues = std.ArrayListUnmanaged(Residue){};
        defer residues.deinit(self.allocator);

        var side_chains = std.ArrayListUnmanaged(SideChainAtom){};
        defer side_chains.deinit(self.allocator);

        var res_info = std.ArrayListUnmanaged(ResidueParseInfo){};
        defer res_info.deinit(self.allocator);

        // Pre-allocate based on file size estimate (~80 bytes per atom line,
        // ~8 atoms/residue, ~4 side chain atoms per residue)
        const estimated_atoms = source.len / 80;
        side_chains.ensureTotalCapacity(self.allocator, estimated_atoms / 2) catch {};
        residues.ensureTotalCapacity(self.allocator, estimated_atoms / 8) catch {};
        res_info.ensureTotalCapacity(self.allocator, estimated_atoms / 8) catch {};

        var ss_bonds = std.ArrayListUnmanaged(SSBond){};
        defer ss_bonds.deinit(self.allocator);

        // Parse atom_site (streaming into residues)
        var tokenizer = cif.Tokenizer.init(source);
        try self.parseAtomSite(&tokenizer, &residues, &side_chains, &res_info);

        // Save position after atom_site parsing
        const pos_after_atom_site = tokenizer.pos;

        // Try to find struct_conn from current position (common case: it comes after atom_site)
        try self.parseStructConn(&tokenizer, &ss_bonds);

        // If struct_conn not found after atom_site, try from beginning
        // (handles rare case where struct_conn appears before atom_site)
        if (ss_bonds.items.len == 0 and pos_after_atom_site > 0) {
            tokenizer = cif.Tokenizer.init(source);
            try self.parseStructConn(&tokenizer, &ss_bonds);
        }

        if (residues.items.len == 0) {
            return ParseError.NoAtomSiteLoop;
        }

        // Convert to owned slices
        const owned_residues = residues.toOwnedSlice(self.allocator) catch return ParseError.OutOfMemory;
        errdefer self.allocator.free(owned_residues);

        const sc_storage = side_chains.toOwnedSlice(self.allocator) catch return ParseError.OutOfMemory;
        errdefer self.allocator.free(sc_storage);

        // Fix up side_chain slices and complete flag
        for (owned_residues, 0..) |*res, i| {
            const info = res_info.items[i];
            const sc_end = if (i + 1 < res_info.items.len) res_info.items[i + 1].sc_start else sc_storage.len;
            res.side_chain = sc_storage[info.sc_start..sc_end];
            res.complete = (info.backbone_count == 4);
        }

        // Transfer SS bonds ownership
        const owned_ss = ss_bonds.toOwnedSlice(self.allocator) catch return ParseError.OutOfMemory;

        return ParseResult{
            .residues = owned_residues,
            .ss_bonds = owned_ss,
            .side_chain_storage = sc_storage,
            .allocator = self.allocator,
        };
    }

    /// Parse mmCIF from a file
    pub fn parseFile(self: *MmcifParser, path: []const u8) ParseError!ParseResult {
        const file = std.fs.cwd().openFile(path, .{}) catch {
            return ParseError.FileReadError;
        };
        defer file.close();

        const source = file.readToEndAlloc(self.allocator, MAX_FILE_SIZE) catch |err| {
            return if (err == error.OutOfMemory) ParseError.OutOfMemory else ParseError.FileReadError;
        };
        defer self.allocator.free(source);

        return self.parse(source);
    }

    /// Parse atom_site loop, streaming directly into Residue structs
    fn parseAtomSite(
        self: *MmcifParser,
        tokenizer: *cif.Tokenizer,
        residues: *std.ArrayListUnmanaged(Residue),
        side_chains: *std.ArrayListUnmanaged(SideChainAtom),
        res_info: *std.ArrayListUnmanaged(ResidueParseInfo),
    ) ParseError!void {
        // Find atom_site loop
        const loop_info = self.findLoop(tokenizer, "_atom_site.") orelse return;
        defer self.allocator.free(loop_info.tags);
        const columns = self.parseAtomSiteColumns(loop_info.tags);

        if (!columns.hasRequiredFields()) {
            return ParseError.MissingCoordinateField;
        }

        // Parse atom data
        var row_values = self.allocator.alloc([]const u8, loop_info.num_cols) catch return ParseError.OutOfMemory;
        defer self.allocator.free(row_values);

        var first_alt_loc: ?u8 = null;
        var col: usize = 0;
        var prev_chain: []const u8 = "";
        var prev_seq: i32 = std.math.minInt(i32);

        while (true) {
            const token = tokenizer.next();

            switch (token) {
                .value => |value| {
                    // Defensive bounds check - skip if malformed data exceeds expected columns
                    if (col >= loop_info.num_cols) {
                        col = 0;
                        continue;
                    }
                    row_values[col] = value;
                    col += 1;

                    if (col >= loop_info.num_cols) {
                        // Complete row - process it
                        if (self.shouldIncludeAtom(row_values, columns, &first_alt_loc)) {
                            try self.processAtomRow(row_values, columns, residues, side_chains, res_info, &prev_chain, &prev_seq);
                        }
                        col = 0;
                    }
                },
                .eof, .loop, .data_block => break,
                .tag => break,
                else => {},
            }
        }
    }

    /// Process a single atom row, streaming directly into the current Residue
    fn processAtomRow(
        self: *MmcifParser,
        row_values: []const []const u8,
        columns: AtomSiteColumns,
        residues: *std.ArrayListUnmanaged(Residue),
        side_chains: *std.ArrayListUnmanaged(SideChainAtom),
        res_info: *std.ArrayListUnmanaged(ResidueParseInfo),
        prev_chain: *[]const u8,
        prev_seq: *i32,
    ) ParseError!void {
        // Parse coordinates first - skip row silently if invalid
        const x = parseFloat(row_values[columns.cartn_x.?]) catch return;
        const y = parseFloat(row_values[columns.cartn_y.?]) catch return;
        const z = parseFloat(row_values[columns.cartn_z.?]) catch return;

        var chain_id: []const u8 = "";
        if (columns.getChainCol()) |col| {
            const v = row_values[col];
            if (!cif.isNull(v)) chain_id = v;
        }

        var seq_id: i32 = 0;
        if (columns.getResSeqCol()) |col| {
            const v = row_values[col];
            if (!cif.isNull(v)) {
                seq_id = std.fmt.parseInt(i32, v, 10) catch 0;
            }
        }

        // Start new residue if key changed
        if (!std.mem.eql(u8, chain_id, prev_chain.*) or seq_id != prev_seq.*) {
            var res = Residue{};

            const cid_len = @min(chain_id.len, 4);
            @memcpy(res.chain_id[0..cid_len], chain_id[0..cid_len]);
            res.chain_id_len = @intCast(cid_len);
            res.seq_id = seq_id;

            var comp_id: []const u8 = "UNK";
            if (columns.getResNameCol()) |col| {
                const v = row_values[col];
                if (!cif.isNull(v)) comp_id = v;
            }
            const cpd_len = @min(comp_id.len, 4);
            @memcpy(res.compound_id[0..cpd_len], comp_id[0..cpd_len]);
            res.compound_id_len = @intCast(cpd_len);
            res.residue_type = ResidueType.fromCompoundId(comp_id);
            res.number = std.math.cast(u32, residues.items.len) orelse return ParseError.OutOfMemory;

            residues.append(self.allocator, res) catch return ParseError.OutOfMemory;
            res_info.append(self.allocator, .{ .sc_start = side_chains.items.len }) catch return ParseError.OutOfMemory;

            prev_chain.* = chain_id;
            prev_seq.* = seq_id;
        }

        // Apply atom to current residue
        var atom_name: []const u8 = "";
        if (columns.getAtomNameCol()) |col| {
            const v = row_values[col];
            if (!cif.isNull(v)) atom_name = v;
        }

        const pos = Vec3f32{
            .x = @floatCast(x),
            .y = @floatCast(y),
            .z = @floatCast(z),
        };

        const res = &residues.items[residues.items.len - 1];
        const info = &res_info.items[res_info.items.len - 1];

        if (std.mem.eql(u8, atom_name, "N")) {
            res.n = pos;
            info.backbone_count += 1;
        } else if (std.mem.eql(u8, atom_name, "CA")) {
            res.ca = pos;
            info.backbone_count += 1;
        } else if (std.mem.eql(u8, atom_name, "C")) {
            res.c = pos;
            info.backbone_count += 1;
        } else if (std.mem.eql(u8, atom_name, "O")) {
            res.o = pos;
            info.backbone_count += 1;
        } else {
            var sc_atom = SideChainAtom{};
            const name_len = @min(atom_name.len, 4);
            @memcpy(sc_atom.name[0..name_len], atom_name[0..name_len]);
            sc_atom.name_len = @intCast(name_len);
            sc_atom.pos = pos;
            side_chains.append(self.allocator, sc_atom) catch return ParseError.OutOfMemory;
        }
    }

    /// Parse struct_conn loop for disulfide bonds
    fn parseStructConn(self: *MmcifParser, tokenizer: *cif.Tokenizer, ss_bonds: *std.ArrayListUnmanaged(SSBond)) ParseError!void {
        // Find struct_conn loop
        const loop_info = self.findLoop(tokenizer, "_struct_conn.") orelse return;
        defer self.allocator.free(loop_info.tags);
        const columns = self.parseStructConnColumns(loop_info.tags);

        if (columns.conn_type_id == null) return;

        // Parse bond data
        var row_values = self.allocator.alloc([]const u8, loop_info.num_cols) catch return ParseError.OutOfMemory;
        defer self.allocator.free(row_values);

        var col: usize = 0;

        while (true) {
            const token = tokenizer.next();

            switch (token) {
                .value => |value| {
                    // Defensive bounds check - skip if malformed data exceeds expected columns
                    if (col >= loop_info.num_cols) {
                        col = 0;
                        continue;
                    }
                    row_values[col] = value;
                    col += 1;

                    if (col >= loop_info.num_cols) {
                        // Complete row - check if disulfide
                        if (columns.conn_type_id) |type_col| {
                            if (std.ascii.eqlIgnoreCase(row_values[type_col], "disulf")) {
                                if (self.parseSSBondRow(row_values, columns)) |bond| {
                                    ss_bonds.append(self.allocator, bond) catch return ParseError.OutOfMemory;
                                }
                            }
                        }
                        col = 0;
                    }
                },
                .eof, .loop, .data_block => break,
                .tag => break,
                else => {},
            }
        }
    }

    /// Loop info returned by findLoop
    const LoopInfo = struct {
        tags: []const []const u8,
        num_cols: usize,
    };

    /// Find a loop with the given tag prefix
    fn findLoop(self: *MmcifParser, tokenizer: *cif.Tokenizer, prefix: []const u8) ?LoopInfo {
        var tags = std.ArrayListUnmanaged([]const u8){};
        defer tags.deinit(self.allocator);

        var in_target_loop = false;

        while (true) {
            const saved_pos = tokenizer.pos;
            const saved_line = tokenizer.line;
            const saved_col = tokenizer.col;

            const token = tokenizer.next();

            switch (token) {
                .eof => {
                    if (in_target_loop and tags.items.len > 0) {
                        const num_cols = tags.items.len;
                        return LoopInfo{
                            .tags = tags.toOwnedSlice(self.allocator) catch return null,
                            .num_cols = num_cols,
                        };
                    }
                    return null;
                },
                .loop => {
                    in_target_loop = false;
                    tags.clearRetainingCapacity();
                },
                .tag => |tag| {
                    if (std.ascii.startsWithIgnoreCase(tag, prefix)) {
                        in_target_loop = true;
                        tags.append(self.allocator, tag) catch return null;
                    } else if (in_target_loop) {
                        // Different category - done with target loop
                        tokenizer.pos = saved_pos;
                        tokenizer.line = saved_line;
                        tokenizer.col = saved_col;
                        const num_cols = tags.items.len;
                        return LoopInfo{
                            .tags = tags.toOwnedSlice(self.allocator) catch return null,
                            .num_cols = num_cols,
                        };
                    }
                },
                .value => {
                    if (in_target_loop) {
                        // Hit values - columns are complete
                        tokenizer.pos = saved_pos;
                        tokenizer.line = saved_line;
                        tokenizer.col = saved_col;
                        const num_cols = tags.items.len;
                        return LoopInfo{
                            .tags = tags.toOwnedSlice(self.allocator) catch return null,
                            .num_cols = num_cols,
                        };
                    }
                },
                .data_block => {
                    if (in_target_loop and tags.items.len > 0) {
                        tokenizer.pos = saved_pos;
                        tokenizer.line = saved_line;
                        tokenizer.col = saved_col;
                        const num_cols = tags.items.len;
                        return LoopInfo{
                            .tags = tags.toOwnedSlice(self.allocator) catch return null,
                            .num_cols = num_cols,
                        };
                    }
                },
                else => {},
            }
        }
    }

    /// Parse atom_site column indices from tags
    fn parseAtomSiteColumns(self: *MmcifParser, tags: []const []const u8) AtomSiteColumns {
        _ = self;
        var columns = AtomSiteColumns{};
        const prefix = "_atom_site.";

        for (tags, 0..) |tag, i| {
            // Skip tags that don't have the expected prefix length
            if (tag.len <= prefix.len) continue;
            const field = tag[prefix.len..];

            if (std.ascii.eqlIgnoreCase(field, "Cartn_x")) {
                columns.cartn_x = i;
            } else if (std.ascii.eqlIgnoreCase(field, "Cartn_y")) {
                columns.cartn_y = i;
            } else if (std.ascii.eqlIgnoreCase(field, "Cartn_z")) {
                columns.cartn_z = i;
            } else if (std.ascii.eqlIgnoreCase(field, "type_symbol")) {
                columns.type_symbol = i;
            } else if (std.ascii.eqlIgnoreCase(field, "label_atom_id")) {
                columns.label_atom_id = i;
            } else if (std.ascii.eqlIgnoreCase(field, "auth_atom_id")) {
                columns.auth_atom_id = i;
            } else if (std.ascii.eqlIgnoreCase(field, "label_comp_id")) {
                columns.label_comp_id = i;
            } else if (std.ascii.eqlIgnoreCase(field, "auth_comp_id")) {
                columns.auth_comp_id = i;
            } else if (std.ascii.eqlIgnoreCase(field, "label_asym_id")) {
                columns.label_asym_id = i;
            } else if (std.ascii.eqlIgnoreCase(field, "auth_asym_id")) {
                columns.auth_asym_id = i;
            } else if (std.ascii.eqlIgnoreCase(field, "label_seq_id")) {
                columns.label_seq_id = i;
            } else if (std.ascii.eqlIgnoreCase(field, "auth_seq_id")) {
                columns.auth_seq_id = i;
            } else if (std.ascii.eqlIgnoreCase(field, "pdbx_PDB_ins_code")) {
                columns.pdbx_pdb_ins_code = i;
            } else if (std.ascii.eqlIgnoreCase(field, "group_PDB")) {
                columns.group_pdb = i;
            } else if (std.ascii.eqlIgnoreCase(field, "label_alt_id")) {
                columns.label_alt_id = i;
            } else if (std.ascii.eqlIgnoreCase(field, "pdbx_PDB_model_num")) {
                columns.pdbx_pdb_model_num = i;
            }
        }

        return columns;
    }

    /// Parse struct_conn column indices from tags
    fn parseStructConnColumns(self: *MmcifParser, tags: []const []const u8) StructConnColumns {
        _ = self;
        var columns = StructConnColumns{};
        const prefix = "_struct_conn.";

        for (tags, 0..) |tag, i| {
            // Skip tags that don't have the expected prefix length
            if (tag.len <= prefix.len) continue;
            const field = tag[prefix.len..];

            if (std.ascii.eqlIgnoreCase(field, "conn_type_id")) {
                columns.conn_type_id = i;
            } else if (std.ascii.eqlIgnoreCase(field, "ptnr1_label_asym_id")) {
                columns.ptnr1_label_asym_id = i;
            } else if (std.ascii.eqlIgnoreCase(field, "ptnr1_auth_asym_id")) {
                columns.ptnr1_auth_asym_id = i;
            } else if (std.ascii.eqlIgnoreCase(field, "ptnr1_label_seq_id")) {
                columns.ptnr1_label_seq_id = i;
            } else if (std.ascii.eqlIgnoreCase(field, "ptnr1_auth_seq_id")) {
                columns.ptnr1_auth_seq_id = i;
            } else if (std.ascii.eqlIgnoreCase(field, "ptnr2_label_asym_id")) {
                columns.ptnr2_label_asym_id = i;
            } else if (std.ascii.eqlIgnoreCase(field, "ptnr2_auth_asym_id")) {
                columns.ptnr2_auth_asym_id = i;
            } else if (std.ascii.eqlIgnoreCase(field, "ptnr2_label_seq_id")) {
                columns.ptnr2_label_seq_id = i;
            } else if (std.ascii.eqlIgnoreCase(field, "ptnr2_auth_seq_id")) {
                columns.ptnr2_auth_seq_id = i;
            }
        }

        return columns;
    }

    /// Check if an atom should be included based on filters
    fn shouldIncludeAtom(self: *MmcifParser, row_values: []const []const u8, columns: AtomSiteColumns, first_alt_loc: *?u8) bool {
        // Check group_PDB filter (ATOM vs HETATM)
        if (self.atom_only) {
            if (columns.group_pdb) |col| {
                if (!std.mem.eql(u8, row_values[col], "ATOM")) {
                    return false;
                }
            }
        }

        // Check model number filter
        if (self.model_num) |target_model| {
            if (columns.pdbx_pdb_model_num) |col| {
                const model_str = row_values[col];
                if (!cif.isNull(model_str)) {
                    const model = std.fmt.parseInt(u32, model_str, 10) catch 1;
                    if (model != target_model) {
                        return false;
                    }
                }
            }
        }

        // Check alternate location filter
        if (self.first_alt_loc_only) {
            if (columns.label_alt_id) |col| {
                const alt_id = row_values[col];
                if (!cif.isNull(alt_id) and alt_id.len > 0) {
                    const alt_char = alt_id[0];
                    if (first_alt_loc.*) |first| {
                        if (alt_char != first) {
                            return false;
                        }
                    } else {
                        first_alt_loc.* = alt_char;
                    }
                }
            }
        }

        // Skip hydrogen atoms
        if (columns.type_symbol) |col| {
            const symbol = row_values[col];
            if (std.mem.eql(u8, symbol, "H") or std.mem.eql(u8, symbol, "D")) {
                return false;
            }
        }

        return true;
    }

    /// Parse a single SS bond row
    fn parseSSBondRow(self: *MmcifParser, row_values: []const []const u8, columns: StructConnColumns) ?SSBond {
        _ = self;
        var bond = SSBond{};

        // Partner 1
        if (columns.getChain1Col()) |col| {
            const c = row_values[col];
            if (!cif.isNull(c)) {
                const len = @min(c.len, 4);
                @memcpy(bond.chain1[0..len], c[0..len]);
                bond.chain1_len = @intCast(len);
            }
        }

        if (columns.getSeq1Col()) |col| {
            const s = row_values[col];
            if (!cif.isNull(s)) {
                bond.seq1 = std.fmt.parseInt(i32, s, 10) catch return null;
            }
        }

        // Partner 2
        if (columns.getChain2Col()) |col| {
            const c = row_values[col];
            if (!cif.isNull(c)) {
                const len = @min(c.len, 4);
                @memcpy(bond.chain2[0..len], c[0..len]);
                bond.chain2_len = @intCast(len);
            }
        }

        if (columns.getSeq2Col()) |col| {
            const s = row_values[col];
            if (!cif.isNull(s)) {
                bond.seq2 = std.fmt.parseInt(i32, s, 10) catch return null;
            }
        }

        return bond;
    }
};

/// Parse a float from a string, handling CIF null values and uncertainty notation
fn parseFloat(s: []const u8) !f64 {
    if (cif.isNull(s)) {
        return ParseError.InvalidCoordinate;
    }

    // Handle parenthetical uncertainty notation: "1.234(5)" -> "1.234"
    var end = s.len;
    for (s, 0..) |c, i| {
        if (c == '(') {
            end = i;
            break;
        }
    }

    return std.fmt.parseFloat(f64, s[0..end]) catch {
        return ParseError.InvalidCoordinate;
    };
}

// ============================================================================
// Tests
// ============================================================================

test "parse simple mmCIF" {
    const source =
        \\data_TEST
        \\#
        \\loop_
        \\_atom_site.id
        \\_atom_site.type_symbol
        \\_atom_site.label_atom_id
        \\_atom_site.label_comp_id
        \\_atom_site.label_asym_id
        \\_atom_site.label_seq_id
        \\_atom_site.Cartn_x
        \\_atom_site.Cartn_y
        \\_atom_site.Cartn_z
        \\_atom_site.group_PDB
        \\_atom_site.pdbx_PDB_model_num
        \\1 N N ALA A 1 10.000 20.000 30.000 ATOM 1
        \\2 C CA ALA A 1 11.000 21.000 31.000 ATOM 1
        \\3 C C ALA A 1 12.000 22.000 32.000 ATOM 1
        \\4 O O ALA A 1 13.000 23.000 33.000 ATOM 1
        \\5 C CB ALA A 1 11.500 20.500 30.500 ATOM 1
        \\#
    ;

    var parser = MmcifParser.init(std.testing.allocator);
    var result = try parser.parse(source);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 1), result.residues.len);
    try std.testing.expectEqualStrings("A", result.residues[0].getChainId());
    try std.testing.expectEqual(@as(i32, 1), result.residues[0].seq_id);
    try std.testing.expectEqualStrings("ALA", result.residues[0].getCompoundId());
    try std.testing.expect(result.residues[0].complete);

    // Check backbone coordinates
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), result.residues[0].n.x, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), result.residues[0].ca.x, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), result.residues[0].c.x, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 13.0), result.residues[0].o.x, 0.001);

    // Check side chain
    try std.testing.expectEqual(@as(usize, 1), result.residues[0].side_chain.len);
    try std.testing.expectEqualStrings("CB", result.residues[0].side_chain[0].getName());
}

test "parse multiple residues" {
    const source =
        \\data_TEST
        \\loop_
        \\_atom_site.id
        \\_atom_site.type_symbol
        \\_atom_site.label_atom_id
        \\_atom_site.label_comp_id
        \\_atom_site.label_asym_id
        \\_atom_site.label_seq_id
        \\_atom_site.Cartn_x
        \\_atom_site.Cartn_y
        \\_atom_site.Cartn_z
        \\_atom_site.group_PDB
        \\_atom_site.pdbx_PDB_model_num
        \\1 N N ALA A 1 10.0 20.0 30.0 ATOM 1
        \\2 C CA ALA A 1 11.0 21.0 31.0 ATOM 1
        \\3 C C ALA A 1 12.0 22.0 32.0 ATOM 1
        \\4 O O ALA A 1 13.0 23.0 33.0 ATOM 1
        \\5 N N GLY A 2 14.0 24.0 34.0 ATOM 1
        \\6 C CA GLY A 2 15.0 25.0 35.0 ATOM 1
        \\7 C C GLY A 2 16.0 26.0 36.0 ATOM 1
        \\8 O O GLY A 2 17.0 27.0 37.0 ATOM 1
        \\#
    ;

    var parser = MmcifParser.init(std.testing.allocator);
    var result = try parser.parse(source);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 2), result.residues.len);
    try std.testing.expectEqualStrings("ALA", result.residues[0].getCompoundId());
    try std.testing.expectEqualStrings("GLY", result.residues[1].getCompoundId());
    try std.testing.expectEqual(@as(i32, 1), result.residues[0].seq_id);
    try std.testing.expectEqual(@as(i32, 2), result.residues[1].seq_id);
}

test "parse with SS bonds" {
    const source =
        \\data_TEST
        \\loop_
        \\_atom_site.id
        \\_atom_site.type_symbol
        \\_atom_site.label_atom_id
        \\_atom_site.label_comp_id
        \\_atom_site.label_asym_id
        \\_atom_site.label_seq_id
        \\_atom_site.Cartn_x
        \\_atom_site.Cartn_y
        \\_atom_site.Cartn_z
        \\_atom_site.group_PDB
        \\_atom_site.pdbx_PDB_model_num
        \\1 N N CYS A 6 10.0 20.0 30.0 ATOM 1
        \\2 C CA CYS A 6 11.0 21.0 31.0 ATOM 1
        \\3 C C CYS A 6 12.0 22.0 32.0 ATOM 1
        \\4 O O CYS A 6 13.0 23.0 33.0 ATOM 1
        \\#
        \\loop_
        \\_struct_conn.id
        \\_struct_conn.conn_type_id
        \\_struct_conn.ptnr1_label_asym_id
        \\_struct_conn.ptnr1_label_seq_id
        \\_struct_conn.ptnr2_label_asym_id
        \\_struct_conn.ptnr2_label_seq_id
        \\1 disulf A 6 A 127
        \\#
    ;

    var parser = MmcifParser.init(std.testing.allocator);
    var result = try parser.parse(source);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 1), result.ss_bonds.len);
    try std.testing.expectEqual(@as(i32, 6), result.ss_bonds[0].seq1);
    try std.testing.expectEqual(@as(i32, 127), result.ss_bonds[0].seq2);
}

test "missing required fields" {
    const source =
        \\data_TEST
        \\loop_
        \\_atom_site.id
        \\_atom_site.type_symbol
        \\_atom_site.Cartn_x
        \\_atom_site.Cartn_y
        \\1 C 10.0 20.0
        \\#
    ;

    var parser = MmcifParser.init(std.testing.allocator);
    const result = parser.parse(source);
    try std.testing.expectError(ParseError.MissingCoordinateField, result);
}

test "no atom_site loop" {
    const source =
        \\data_TEST
        \\loop_
        \\_cell.length_a
        \\_cell.length_b
        \\10.0 20.0
        \\#
    ;

    var parser = MmcifParser.init(std.testing.allocator);
    const result = parser.parse(source);
    try std.testing.expectError(ParseError.NoAtomSiteLoop, result);
}
