//! ztraj-io — standalone structure and trajectory I/O.
//!
//! Exposed to package consumers as the `ztraj-io` module. It contains the
//! file-format readers/writers under `src/io/` plus the core types and element
//! data they depend on, and nothing that requires the `zsasa` dependency.
//!
//! Note: these declarations come from a separate module instance than the full
//! `ztraj` module, so `ztraj-io` types are not interchangeable with `ztraj`
//! types. Import one module or the other, not both.

pub const types = @import("types.zig");
pub const element = @import("element.zig");

pub const pdb = io.pdb;
pub const mmcif = io.mmcif;
pub const cif_tokenizer = io.cif_tokenizer;
pub const xtc = io.xtc;
pub const trr = io.trr;
pub const dcd = io.dcd;
pub const gro = io.gro;
pub const prmtop = io.prmtop;
pub const nc = io.nc;
pub const xdr = io.xdr;

/// Same namespace as `ztraj.io`.
pub const io = @import("io/root.zig");

test {
    // Compile-check the standalone module (build.zig compiles but does not
    // run this test binary). `pdb` is left out because its tests `@embedFile`
    // data outside the module root (same restriction as in `root.zig`).
    _ = types;
    _ = element;
    _ = mmcif;
    _ = cif_tokenizer;
    _ = xtc;
    _ = trr;
    _ = dcd;
    _ = gro;
    _ = prmtop;
    _ = nc;
    _ = xdr;
}
