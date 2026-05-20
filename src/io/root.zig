//! ztraj I/O namespace.
//!
//! High-level parsers and trajectory readers/writers live here. Low-level
//! format-native helpers are only exported when they are useful for internal
//! composition or advanced Zig consumers.

pub const pdb = @import("pdb.zig");
pub const mmcif = @import("mmcif.zig");
pub const cif_tokenizer = @import("cif_tokenizer.zig");
pub const xtc = @import("xtc.zig");
pub const trr = @import("trr.zig");
pub const dcd = @import("dcd.zig");
pub const gro = @import("gro.zig");
pub const prmtop = @import("prmtop.zig");
pub const nc = @import("nc.zig");
