//! CIF (Crystallographic Information File) Tokenizer.
//!
//! This module provides a simple tokenizer for parsing CIF/mmCIF files.
//! It handles the basic CIF syntax elements needed for atom_site extraction.
//!
//! CIF Syntax Elements:
//! - `data_<name>` - Data block declaration
//! - `loop_` - Loop block declaration
//! - `_category.field` - Tag names
//! - Values: unquoted, single-quoted, double-quoted, or semicolon-delimited
//! - Comments: lines starting with #
//!
//! Reference: https://www.iucr.org/resources/cif/spec/version1.1/cifsyntax

const std = @import("std");
const char_table = @import("char_table.zig");
const types = @import("types.zig");

const CharType = char_table.CharType;
const charType = char_table.charType;
const isWhitespace = char_table.isWhitespace;

const Token = types.Token;

/// CIF Tokenizer
pub const Tokenizer = struct {
    source: []const u8,
    pos: usize,
    line: usize,
    col: usize,

    /// Initialize tokenizer with source text
    pub fn init(source: []const u8) Tokenizer {
        return .{
            .source = source,
            .pos = 0,
            .line = 1,
            .col = 1,
        };
    }

    /// Get the next token
    /// Uses character type lookup table for fast dispatch
    pub fn next(self: *Tokenizer) Token {
        self.skipWhitespaceAndComments();

        if (self.pos >= self.source.len) {
            return .eof;
        }

        const c = self.source[self.pos];
        const ctype = charType(c);

        // Dispatch based on character type (using lookup table)
        switch (ctype) {
            .semicolon => {
                // Semicolon text block (must be at beginning of line)
                if (self.col == 1) {
                    return self.readSemicolonText();
                }
                // Otherwise treat as ordinary value
                return self.readUnquotedValue();
            },
            .quote => return self.readQuotedString(c),
            .underscore => return self.readTag(),
            .ordinary => {
                // Check for keywords: data_, loop_, save_
                const cl = c | 0x20; // lowercase
                if (cl == 'd' and self.matchKeyword("data_")) {
                    return self.readDataBlock();
                }
                if (cl == 'l' and self.matchKeyword("loop_")) {
                    self.pos += 5;
                    self.col += 5;
                    return .loop;
                }
                if (cl == 's' and self.matchKeyword("save_")) {
                    return self.readSaveFrame();
                }
                // Read as unquoted value
                return self.readUnquotedValue();
            },
            else => {
                // Invalid or unexpected character - try to read as value anyway
                return self.readUnquotedValue();
            },
        }
    }

    /// Skip whitespace and comments
    /// Uses character type lookup table for faster classification
    fn skipWhitespaceAndComments(self: *Tokenizer) void {
        while (self.pos < self.source.len) {
            const c = self.source[self.pos];
            const ctype = charType(c);

            if (ctype == .whitespace) {
                if (c == '\n') {
                    self.line += 1;
                    self.col = 1;
                } else {
                    self.col += 1;
                }
                self.pos += 1;
            } else if (ctype == .hash) {
                // Skip comment until end of line
                while (self.pos < self.source.len and self.source[self.pos] != '\n') {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    /// Check if remaining source matches a keyword (case-insensitive)
    /// Uses bit manipulation for efficient ASCII case-insensitive comparison:
    /// For ASCII letters, (c | 0x20) converts to lowercase without branching
    fn matchKeyword(self: *Tokenizer, keyword: []const u8) bool {
        if (self.pos + keyword.len > self.source.len) {
            return false;
        }

        // Fast case-insensitive comparison using bit operations
        // For ASCII: 'A'|0x20 = 'a', 'a'|0x20 = 'a'
        for (keyword, 0..) |kc, i| {
            const sc = self.source[self.pos + i];
            // Compare with case bit masked
            if ((sc | 0x20) != (kc | 0x20)) {
                return false;
            }
        }
        return true;
    }

    /// Read a data block name: data_<name>
    fn readDataBlock(self: *Tokenizer) Token {
        self.pos += 5; // skip "data_"
        self.col += 5;

        const start = self.pos;
        while (self.pos < self.source.len) {
            const c = self.source[self.pos];
            if (isWhitespace(c)) break;
            self.pos += 1;
            self.col += 1;
        }

        return .{ .data_block = self.source[start..self.pos] };
    }

    /// Read a save frame: save_<name> or save_ (end)
    fn readSaveFrame(self: *Tokenizer) Token {
        self.pos += 5; // skip "save_"
        self.col += 5;

        const start = self.pos;
        while (self.pos < self.source.len) {
            const c = self.source[self.pos];
            if (isWhitespace(c)) break;
            self.pos += 1;
            self.col += 1;
        }

        if (start == self.pos) {
            return .save_end;
        }
        return .{ .save_begin = self.source[start..self.pos] };
    }

    /// Read a tag: _category.field
    fn readTag(self: *Tokenizer) Token {
        const start = self.pos;
        self.pos += 1; // skip '_'
        self.col += 1;

        while (self.pos < self.source.len) {
            const c = self.source[self.pos];
            if (isWhitespace(c)) break;
            self.pos += 1;
            self.col += 1;
        }

        return .{ .tag = self.source[start..self.pos] };
    }

    /// Read a quoted string (single or double quotes)
    fn readQuotedString(self: *Tokenizer, quote: u8) Token {
        self.pos += 1; // skip opening quote
        self.col += 1;

        const start = self.pos;

        while (self.pos < self.source.len) {
            const c = self.source[self.pos];

            if (c == quote) {
                // Check if quote is at end of token (followed by whitespace or EOF)
                const next_pos = self.pos + 1;
                if (next_pos >= self.source.len or isWhitespace(self.source[next_pos])) {
                    const value = self.source[start..self.pos];
                    self.pos += 1; // skip closing quote
                    self.col += 1;
                    return .{ .value = value };
                }
            }

            if (c == '\n') {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
            self.pos += 1;
        }

        // Unterminated string
        return .{ .err = "Unterminated quoted string" };
    }

    /// Read semicolon-delimited text block
    fn readSemicolonText(self: *Tokenizer) Token {
        self.pos += 1; // skip opening semicolon
        self.col += 1;

        // Skip to end of first line
        while (self.pos < self.source.len and self.source[self.pos] != '\n') {
            self.pos += 1;
        }
        if (self.pos < self.source.len) {
            self.pos += 1; // skip newline
            self.line += 1;
            self.col = 1;
        }

        const start = self.pos;

        // Find closing semicolon at beginning of line
        while (self.pos < self.source.len) {
            if (self.col == 1 and self.source[self.pos] == ';') {
                const value = self.source[start..self.pos];
                self.pos += 1; // skip closing semicolon
                self.col += 1;
                return .{ .value = value };
            }

            if (self.source[self.pos] == '\n') {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
            self.pos += 1;
        }

        // Unterminated text block
        return .{ .err = "Unterminated semicolon text block" };
    }

    /// Read an unquoted value
    fn readUnquotedValue(self: *Tokenizer) Token {
        const start = self.pos;

        while (self.pos < self.source.len) {
            const c = self.source[self.pos];
            if (isWhitespace(c)) break;
            self.pos += 1;
            self.col += 1;
        }

        return .{ .value = self.source[start..self.pos] };
    }

    /// Get current line number (1-based)
    pub fn getLine(self: *const Tokenizer) usize {
        return self.line;
    }

    /// Get current column number (1-based)
    pub fn getColumn(self: *const Tokenizer) usize {
        return self.col;
    }
};

/// Check if a value is the CIF "inapplicable" marker
pub fn isInapplicable(value: []const u8) bool {
    return std.mem.eql(u8, value, ".");
}

/// Check if a value is the CIF "unknown" marker
pub fn isUnknown(value: []const u8) bool {
    return std.mem.eql(u8, value, "?");
}

/// Check if a value is null (either . or ?)
pub fn isNull(value: []const u8) bool {
    return isInapplicable(value) or isUnknown(value);
}

// ============================================================================
// Tests
// ============================================================================

test "tokenize data block" {
    var tok = Tokenizer.init("data_1ABC");
    const token = tok.next();
    try std.testing.expectEqual(Token.data_block, std.meta.activeTag(token));
    try std.testing.expectEqualStrings("1ABC", token.data_block);
    try std.testing.expectEqual(Token.eof, std.meta.activeTag(tok.next()));
}

test "tokenize loop" {
    var tok = Tokenizer.init("loop_");
    try std.testing.expectEqual(Token.loop, tok.next());
    try std.testing.expectEqual(Token.eof, std.meta.activeTag(tok.next()));
}

test "tokenize tag" {
    var tok = Tokenizer.init("_atom_site.Cartn_x");
    const token = tok.next();
    try std.testing.expectEqual(Token.tag, std.meta.activeTag(token));
    try std.testing.expectEqualStrings("_atom_site.Cartn_x", token.tag);
}

test "tokenize unquoted value" {
    var tok = Tokenizer.init("123.456");
    const token = tok.next();
    try std.testing.expectEqual(Token.value, std.meta.activeTag(token));
    try std.testing.expectEqualStrings("123.456", token.value);
}

test "tokenize single quoted string" {
    var tok = Tokenizer.init("'hello world'");
    const token = tok.next();
    try std.testing.expectEqual(Token.value, std.meta.activeTag(token));
    try std.testing.expectEqualStrings("hello world", token.value);
}

test "tokenize double quoted string" {
    var tok = Tokenizer.init("\"test value\"");
    const token = tok.next();
    try std.testing.expectEqual(Token.value, std.meta.activeTag(token));
    try std.testing.expectEqualStrings("test value", token.value);
}

test "tokenize semicolon text" {
    const source =
        \\;first line
        \\second line
        \\;
    ;
    var tok = Tokenizer.init(source);
    const token = tok.next();
    try std.testing.expectEqual(Token.value, std.meta.activeTag(token));
    try std.testing.expectEqualStrings("second line\n", token.value);
}

test "skip comments" {
    var tok = Tokenizer.init("# comment\ndata_TEST");
    const token = tok.next();
    try std.testing.expectEqual(Token.data_block, std.meta.activeTag(token));
    try std.testing.expectEqualStrings("TEST", token.data_block);
}

test "tokenize multiple tokens" {
    var tok = Tokenizer.init(
        \\data_TEST
        \\loop_
        \\_atom_site.id
        \\_atom_site.type_symbol
        \\1 C
        \\2 N
    );

    // data_TEST
    var token = tok.next();
    try std.testing.expectEqual(Token.data_block, std.meta.activeTag(token));
    try std.testing.expectEqualStrings("TEST", token.data_block);

    // loop_
    try std.testing.expectEqual(Token.loop, tok.next());

    // _atom_site.id
    token = tok.next();
    try std.testing.expectEqual(Token.tag, std.meta.activeTag(token));
    try std.testing.expectEqualStrings("_atom_site.id", token.tag);

    // _atom_site.type_symbol
    token = tok.next();
    try std.testing.expectEqual(Token.tag, std.meta.activeTag(token));
    try std.testing.expectEqualStrings("_atom_site.type_symbol", token.tag);

    // Values: 1 C 2 N
    token = tok.next();
    try std.testing.expectEqualStrings("1", token.value);
    token = tok.next();
    try std.testing.expectEqualStrings("C", token.value);
    token = tok.next();
    try std.testing.expectEqualStrings("2", token.value);
    token = tok.next();
    try std.testing.expectEqualStrings("N", token.value);

    try std.testing.expectEqual(Token.eof, std.meta.activeTag(tok.next()));
}

test "special values" {
    try std.testing.expect(isInapplicable("."));
    try std.testing.expect(!isInapplicable("?"));
    try std.testing.expect(isUnknown("?"));
    try std.testing.expect(!isUnknown("."));
    try std.testing.expect(isNull("."));
    try std.testing.expect(isNull("?"));
    try std.testing.expect(!isNull("value"));
}

test "case insensitive keywords" {
    var tok = Tokenizer.init("DATA_test LOOP_");
    const token = tok.next();
    try std.testing.expectEqual(Token.data_block, std.meta.activeTag(token));
    try std.testing.expectEqualStrings("test", token.data_block);
    try std.testing.expectEqual(Token.loop, tok.next());
}

test "quoted string with embedded quote char" {
    // In CIF, quotes inside quoted strings work if not followed by whitespace
    var tok = Tokenizer.init("'it''s ok'");
    const token = tok.next();
    try std.testing.expectEqual(Token.value, std.meta.activeTag(token));
    // The embedded '' is kept as-is
    try std.testing.expectEqualStrings("it''s ok", token.value);
}
