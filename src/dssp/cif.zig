//! CIF (Crystallographic Information File) parsing modules.
//!
//! Re-exports the tokenizer, types, and character classification table
//! for convenient access by consumers.

pub const char_table = @import("cif/char_table.zig");
pub const types = @import("cif/types.zig");
pub const tokenizer_mod = @import("cif/tokenizer.zig");

// Re-export main types for convenience
pub const Tokenizer = tokenizer_mod.Tokenizer;
pub const Token = types.Token;
pub const isNull = tokenizer_mod.isNull;
pub const isInapplicable = tokenizer_mod.isInapplicable;
pub const isUnknown = tokenizer_mod.isUnknown;

test {
    @import("std").testing.refAllDecls(@This());
}
