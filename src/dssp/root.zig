//! DSSP secondary structure assignment.
//!
//! Adapted from zdssp (https://github.com/N283T/zdssp).
//! Will be replaced with an import dependency once zdssp is released.

pub const dssp = @import("dssp.zig");
pub const types = @import("types.zig");
pub const residue = @import("residue.zig");
pub const geometry = @import("geometry.zig");
pub const hbond = @import("hbond.zig");
pub const helix = @import("helix.zig");
pub const beta_sheet = @import("beta_sheet.zig");
pub const accessibility = @import("accessibility.zig");
pub const pdb_parser = @import("pdb_parser.zig");
pub const mmcif_parser = @import("mmcif_parser.zig");
pub const json_parser = @import("json_parser.zig");
pub const json_writer = @import("json_writer.zig");
pub const neighbor_list = @import("neighbor_list.zig");
pub const batch = @import("batch.zig");
pub const gzip = @import("gzip.zig");

// Re-export key types
pub const DsspResult = dssp.DsspResult;
pub const DsspConfig = dssp.DsspConfig;
pub const calculate = dssp.calculate;
pub const calculateFromParseResult = dssp.calculateFromParseResult;
