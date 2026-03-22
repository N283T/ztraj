//! ztraj — molecular dynamics trajectory analysis library.
//!
//! Public API surface. Import this module to access all ztraj types and
//! analysis functions.

pub const types = @import("types.zig");
pub const element = @import("element.zig");
pub const simd = @import("simd.zig");
pub const thread_pool = @import("thread_pool.zig");
pub const neighbor_list = @import("neighbor_list.zig");
pub const mmap_reader = @import("mmap_reader.zig");

pub const geometry = struct {
    pub const distances = @import("geometry/distances.zig");
    pub const angles = @import("geometry/angles.zig");
    pub const dihedrals = @import("geometry/dihedrals.zig");
    pub const center = @import("geometry/center.zig");
    pub const rg = @import("geometry/rg.zig");
    pub const inertia = @import("geometry/inertia.zig");
    pub const rmsd = @import("geometry/rmsd.zig");
    pub const rmsf = @import("geometry/rmsf.zig");
    pub const protein_dihedrals = @import("geometry/protein_dihedrals.zig");
    pub const superpose = @import("geometry/superpose.zig");
    pub const pbc = @import("geometry/pbc.zig");
};

pub const analysis = struct {
    pub const hbonds = @import("analysis/hbonds.zig");
    pub const contacts = @import("analysis/contacts.zig");
    pub const rdf = @import("analysis/rdf.zig");
    pub const sasa = @import("analysis/sasa.zig");
    pub const native_contacts = @import("analysis/native_contacts.zig");
    pub const msd = @import("analysis/msd.zig");
    pub const pca = @import("analysis/pca.zig");
};

pub const io = struct {
    pub const pdb = @import("io/pdb.zig");
    pub const mmcif = @import("io/mmcif.zig");
    pub const cif_tokenizer = @import("io/cif_tokenizer.zig");
    pub const xtc = @import("io/xtc.zig");
    pub const trr = @import("io/trr.zig");
    pub const dcd = @import("io/dcd.zig");
    pub const gro = @import("io/gro.zig");
};

pub const select = @import("select.zig");
pub const output = @import("output.zig");

/// DSSP secondary structure assignment (native implementation).
pub const dssp = @import("analysis/dssp/dssp.zig");

test {
    // Use refAllDecls (non-recursive) to avoid pulling dssp's test suite
    // and pdb.zig's @embedFile path issues
    @import("std").testing.refAllDecls(@This());
    // Explicitly pull in GRO parser tests (not reached by non-recursive refAllDecls)
    _ = io.gro;
}
