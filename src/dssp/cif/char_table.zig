//! Character classification table for CIF parsing (Gemmi-style optimization).
//!
//! Pre-computed 256-byte lookup table for O(1) character type classification.

/// Character types for CIF parsing
/// 0 = Invalid/control character
/// 1 = Ordinary character (can appear in unquoted values)
/// 2 = Whitespace (space, tab, newline, carriage return)
/// 3 = Special: quote characters (' or ")
/// 4 = Special: hash (#) - comment start
/// 5 = Special: underscore (_) - tag start
/// 6 = Special: semicolon (;) - text block delimiter
pub const CharType = enum(u8) {
    invalid = 0,
    ordinary = 1,
    whitespace = 2,
    quote = 3,
    hash = 4,
    underscore = 5,
    semicolon = 6,
};

/// Lookup table for character classification (256 bytes)
/// Pre-computed for O(1) character type lookup
pub const CHAR_TABLE: [256]CharType = blk: {
    var table: [256]CharType = .{.invalid} ** 256;

    // Whitespace characters
    table[' '] = .whitespace;
    table['\t'] = .whitespace;
    table['\n'] = .whitespace;
    table['\r'] = .whitespace;

    // Ordinary characters: printable ASCII except special chars
    // Numbers 0-9
    for ('0'..'9' + 1) |c| {
        table[c] = .ordinary;
    }
    // Uppercase A-Z
    for ('A'..'Z' + 1) |c| {
        table[c] = .ordinary;
    }
    // Lowercase a-z
    for ('a'..'z' + 1) |c| {
        table[c] = .ordinary;
    }

    // Ordinary punctuation (valid in unquoted values)
    table['.'] = .ordinary;
    table['-'] = .ordinary;
    table['+'] = .ordinary;
    table['('] = .ordinary;
    table[')'] = .ordinary;
    table['['] = .ordinary;
    table[']'] = .ordinary;
    table['{'] = .ordinary;
    table['}'] = .ordinary;
    table['/'] = .ordinary;
    table['\\'] = .ordinary;
    table['?'] = .ordinary;
    table['!'] = .ordinary;
    table['@'] = .ordinary;
    table['$'] = .ordinary;
    table['%'] = .ordinary;
    table['^'] = .ordinary;
    table['&'] = .ordinary;
    table['*'] = .ordinary;
    table['<'] = .ordinary;
    table['>'] = .ordinary;
    table['='] = .ordinary;
    table['~'] = .ordinary;
    table['`'] = .ordinary;
    table[':'] = .ordinary;
    table[','] = .ordinary;

    // Special characters
    table['\''] = .quote;
    table['"'] = .quote;
    table['#'] = .hash;
    table['_'] = .underscore;
    table[';'] = .semicolon;

    break :blk table;
};

/// Get character type from lookup table (O(1))
pub inline fn charType(c: u8) CharType {
    return CHAR_TABLE[c];
}

/// Check if character is whitespace using lookup table
pub inline fn isWhitespace(c: u8) bool {
    return CHAR_TABLE[c] == .whitespace;
}

/// Check if character is ordinary (valid in unquoted values)
pub inline fn isOrdinary(c: u8) bool {
    return CHAR_TABLE[c] == .ordinary or CHAR_TABLE[c] == .underscore;
}
