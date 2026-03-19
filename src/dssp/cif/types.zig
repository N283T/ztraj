//! Token types for CIF parsing.

/// Token types for CIF parsing
pub const Token = union(enum) {
    /// Data block declaration: data_<name>
    data_block: []const u8,
    /// Loop declaration: loop_
    loop,
    /// Save frame begin: save_<name> (not commonly used in mmCIF)
    save_begin: []const u8,
    /// Save frame end: save_
    save_end,
    /// Tag name: _category.field
    tag: []const u8,
    /// Value (string, number, or special value like . or ?)
    value: []const u8,
    /// End of file
    eof,
    /// Error token with message
    err: []const u8,

    /// Check if this is a value token
    pub fn isValue(self: Token) bool {
        return self == .value;
    }

    /// Check if this is a tag token
    pub fn isTag(self: Token) bool {
        return self == .tag;
    }

    /// Get the string value if this is a value or tag token
    pub fn getString(self: Token) ?[]const u8 {
        return switch (self) {
            .value => |v| v,
            .tag => |t| t,
            .data_block => |d| d,
            .save_begin => |s| s,
            else => null,
        };
    }
};
