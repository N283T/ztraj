//! ztraj — molecular dynamics trajectory analysis library.
//!
//! Public API surface. Import this module to access all ztraj types and
//! analysis functions.

pub const types = @import("types.zig");
pub const element = @import("element.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
