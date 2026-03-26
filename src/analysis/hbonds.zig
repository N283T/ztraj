//! Hydrogen bond detection using Baker-Hubbard criteria.
//!
//! A hydrogen bond D-H...A is detected when:
//! - D (donor) is N or O covalently bonded to H
//! - A (acceptor) is N, O, or S
//! - H...A distance < dist_cutoff (default 2.5 Å)
//! - D-H...A angle > angle_cutoff (default 120°)
//!
//! Reference: Baker & Hubbard (1984) Prog. Biophys. Mol. Biol. 44, 97–179.

const std = @import("std");
const types = @import("../types.zig");

// ============================================================================
// Spatial cell list for acceptor lookup
// ============================================================================

/// Cell list data for spatial acceleration of acceptor lookup.
const CellList = struct {
    /// Sorted acceptor atom indices (indices into the original atom array).
    sorted_indices: []u32,
    /// cell_offsets[i] is the start index in sorted_indices for cell i.
    /// cell_offsets[n_cells] is the total number of acceptors (sentinel).
    cell_offsets: []u32,
    /// Grid dimensions.
    nx: u32,
    ny: u32,
    nz: u32,
    /// Bounding box minimum (with padding).
    min_x: f32,
    min_y: f32,
    min_z: f32,
    /// Inverse cell size for coordinate-to-cell conversion.
    inv_cell_size: f32,

    fn deinit(self: CellList, allocator: std.mem.Allocator) void {
        allocator.free(self.sorted_indices);
        allocator.free(self.cell_offsets);
    }

    /// Return the cell index for a position, clamped to grid bounds.
    inline fn cellIndex(self: CellList, px: f32, py: f32, pz: f32) u32 {
        const cx = @min(@as(u32, @intFromFloat(@max(0.0, (px - self.min_x) * self.inv_cell_size))), self.nx - 1);
        const cy = @min(@as(u32, @intFromFloat(@max(0.0, (py - self.min_y) * self.inv_cell_size))), self.ny - 1);
        const cz = @min(@as(u32, @intFromFloat(@max(0.0, (pz - self.min_z) * self.inv_cell_size))), self.nz - 1);
        return cx * self.ny * self.nz + cy * self.nz + cz;
    }
};

/// Build a cell list of acceptor atoms (N, O, S) for spatial lookup.
fn buildAcceptorCellList(
    allocator: std.mem.Allocator,
    frame: types.Frame,
    topology: types.Topology,
    cell_size: f32,
) !CellList {
    const n_atoms = topology.atoms.len;

    // Pre-filter: collect acceptor indices.
    var acc_count: u32 = 0;
    for (0..n_atoms) |i| {
        const elem = topology.atoms[i].element;
        if (elem == .N or elem == .O or elem == .S) acc_count += 1;
    }

    if (acc_count == 0) {
        const offsets = try allocator.alloc(u32, 2);
        offsets[0] = 0;
        offsets[1] = 0;
        return CellList{
            .sorted_indices = try allocator.alloc(u32, 0),
            .cell_offsets = offsets,
            .nx = 1,
            .ny = 1,
            .nz = 1,
            .min_x = 0,
            .min_y = 0,
            .min_z = 0,
            .inv_cell_size = 1.0 / cell_size,
        };
    }

    // Compute bounding box from ALL atom positions.
    var bmin_x: f32 = frame.x[0];
    var bmin_y: f32 = frame.y[0];
    var bmin_z: f32 = frame.z[0];
    var bmax_x: f32 = frame.x[0];
    var bmax_y: f32 = frame.y[0];
    var bmax_z: f32 = frame.z[0];
    for (1..n_atoms) |i| {
        bmin_x = @min(bmin_x, frame.x[i]);
        bmin_y = @min(bmin_y, frame.y[i]);
        bmin_z = @min(bmin_z, frame.z[i]);
        bmax_x = @max(bmax_x, frame.x[i]);
        bmax_y = @max(bmax_y, frame.y[i]);
        bmax_z = @max(bmax_z, frame.z[i]);
    }

    // Add padding of cell_size.
    bmin_x -= cell_size;
    bmin_y -= cell_size;
    bmin_z -= cell_size;
    bmax_x += cell_size;
    bmax_y += cell_size;
    bmax_z += cell_size;

    const inv_cs = 1.0 / cell_size;
    const nx: u32 = @max(1, @as(u32, @intFromFloat(@ceil((bmax_x - bmin_x) * inv_cs))));
    const ny: u32 = @max(1, @as(u32, @intFromFloat(@ceil((bmax_y - bmin_y) * inv_cs))));
    const nz: u32 = @max(1, @as(u32, @intFromFloat(@ceil((bmax_z - bmin_z) * inv_cs))));
    const n_cells: u32 = nx * ny * nz;

    // Counting sort: first pass — count atoms per cell.
    const counts = try allocator.alloc(u32, n_cells + 1);
    defer allocator.free(counts);
    @memset(counts, 0);

    // Temporary array to hold acceptor indices and their cell assignments.
    const acc_indices = try allocator.alloc(u32, acc_count);
    defer allocator.free(acc_indices);
    const acc_cells = try allocator.alloc(u32, acc_count);
    defer allocator.free(acc_cells);

    var ai: u32 = 0;
    for (0..n_atoms) |i| {
        const elem = topology.atoms[i].element;
        if (elem == .N or elem == .O or elem == .S) {
            const idx: u32 = @intCast(i);
            acc_indices[ai] = idx;
            const cx = @min(@as(u32, @intFromFloat(@max(0.0, (frame.x[i] - bmin_x) * inv_cs))), nx - 1);
            const cy = @min(@as(u32, @intFromFloat(@max(0.0, (frame.y[i] - bmin_y) * inv_cs))), ny - 1);
            const cz = @min(@as(u32, @intFromFloat(@max(0.0, (frame.z[i] - bmin_z) * inv_cs))), nz - 1);
            const cell = cx * ny * nz + cy * nz + cz;
            acc_cells[ai] = cell;
            counts[cell] += 1;
            ai += 1;
        }
    }

    // Prefix sum to get offsets.
    const cell_offsets = try allocator.alloc(u32, n_cells + 1);
    cell_offsets[0] = 0;
    for (1..n_cells + 1) |c| {
        cell_offsets[c] = cell_offsets[c - 1] + counts[c - 1];
    }

    // Second pass: place acceptors into sorted order.
    const sorted = try allocator.alloc(u32, acc_count);

    // Reset counts for placement.
    @memset(counts[0..n_cells], 0);
    for (0..acc_count) |j| {
        const cell = acc_cells[j];
        sorted[cell_offsets[cell] + counts[cell]] = acc_indices[j];
        counts[cell] += 1;
    }

    return CellList{
        .sorted_indices = sorted,
        .cell_offsets = cell_offsets,
        .nx = nx,
        .ny = ny,
        .nz = nz,
        .min_x = bmin_x,
        .min_y = bmin_y,
        .min_z = bmin_z,
        .inv_cell_size = inv_cs,
    };
}

// ============================================================================
// Public types
// ============================================================================

/// A detected hydrogen bond.
pub const HBond = struct {
    /// Index of the donor heavy atom (N or O) in the topology.
    donor: u32,
    /// Index of the hydrogen atom in the topology.
    hydrogen: u32,
    /// Index of the acceptor atom (N, O, or S) in the topology.
    acceptor: u32,
    /// H...A distance in Angstroms.
    distance: f32,
    /// D-H...A angle in degrees.
    angle: f32,
};

/// Detection parameters.
pub const Config = struct {
    /// Maximum H...A distance to consider (Å). Default: 2.5.
    dist_cutoff: f32 = 2.5,
    /// Minimum D-H...A angle to consider (degrees). Default: 120.0.
    angle_cutoff: f32 = 120.0,
};

// ============================================================================
// Detection
// ============================================================================

/// Detect hydrogen bonds in a single frame using the Baker-Hubbard criteria.
///
/// Iterates over all covalent bonds in `topology` looking for D-H pairs where
/// D is N or O, then tests every potential acceptor (N, O, S) against the
/// distance and angle thresholds.
///
/// The returned slice is owned by the caller; free with `allocator.free()`.
pub fn detect(
    allocator: std.mem.Allocator,
    topology: types.Topology,
    frame: types.Frame,
    config: Config,
) ![]HBond {
    var result = std.ArrayList(HBond){};
    errdefer result.deinit(allocator);

    const n_atoms = topology.atoms.len;
    if (n_atoms == 0) return result.toOwnedSlice(allocator);

    // Build spatial cell list of acceptor atoms for O(1) neighbor lookup.
    var cell_list = try buildAcceptorCellList(allocator, frame, topology, config.dist_cutoff);
    defer cell_list.deinit(allocator);

    try detectWithCellList(allocator, topology, frame, config, &cell_list, &result);

    return result.toOwnedSlice(allocator);
}

/// Core detection logic using a pre-built cell list.
/// Iterates topology bonds, finds D-H pairs, and checks nearby acceptors via the cell list.
fn detectWithCellList(
    allocator: std.mem.Allocator,
    topology: types.Topology,
    frame: types.Frame,
    config: Config,
    cell_list: *const CellList,
    result: *std.ArrayList(HBond),
) !void {
    for (topology.bonds) |bond| {
        const a1 = topology.atoms[bond.atom_i];
        const a2 = topology.atoms[bond.atom_j];

        // Identify which atom is H and which is the donor heavy atom.
        var donor_idx: u32 = undefined;
        var h_idx: u32 = undefined;
        const is_dh: bool = blk: {
            if (a1.element == .H and (a2.element == .N or a2.element == .O)) {
                h_idx = bond.atom_i;
                donor_idx = bond.atom_j;
                break :blk true;
            }
            if (a2.element == .H and (a1.element == .N or a1.element == .O)) {
                h_idx = bond.atom_j;
                donor_idx = bond.atom_i;
                break :blk true;
            }
            break :blk false;
        };

        if (!is_dh) continue;

        // Pre-fetch H position in f64 to avoid repeated conversions.
        const hx: f64 = @floatCast(frame.x[h_idx]);
        const hy: f64 = @floatCast(frame.y[h_idx]);
        const hz: f64 = @floatCast(frame.z[h_idx]);

        // D-H vector (donor minus hydrogen).
        const v1x: f64 = @as(f64, frame.x[donor_idx]) - hx;
        const v1y: f64 = @as(f64, frame.y[donor_idx]) - hy;
        const v1z: f64 = @as(f64, frame.z[donor_idx]) - hz;
        const mag1 = @sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
        if (mag1 < 1e-10) continue;

        // Determine the cell of the hydrogen atom and iterate 27 neighboring cells.
        const h_cell_x = @min(@as(u32, @intFromFloat(@max(0.0, (frame.x[h_idx] - cell_list.min_x) * cell_list.inv_cell_size))), cell_list.nx - 1);
        const h_cell_y = @min(@as(u32, @intFromFloat(@max(0.0, (frame.y[h_idx] - cell_list.min_y) * cell_list.inv_cell_size))), cell_list.ny - 1);
        const h_cell_z = @min(@as(u32, @intFromFloat(@max(0.0, (frame.z[h_idx] - cell_list.min_z) * cell_list.inv_cell_size))), cell_list.nz - 1);

        const cx_lo: u32 = if (h_cell_x > 0) h_cell_x - 1 else 0;
        const cx_hi: u32 = @min(h_cell_x + 1, cell_list.nx - 1);
        const cy_lo: u32 = if (h_cell_y > 0) h_cell_y - 1 else 0;
        const cy_hi: u32 = @min(h_cell_y + 1, cell_list.ny - 1);
        const cz_lo: u32 = if (h_cell_z > 0) h_cell_z - 1 else 0;
        const cz_hi: u32 = @min(h_cell_z + 1, cell_list.nz - 1);

        var cx: u32 = cx_lo;
        while (cx <= cx_hi) : (cx += 1) {
            var cy: u32 = cy_lo;
            while (cy <= cy_hi) : (cy += 1) {
                var cz: u32 = cz_lo;
                while (cz <= cz_hi) : (cz += 1) {
                    const cell = cx * cell_list.ny * cell_list.nz + cy * cell_list.nz + cz;
                    const start = cell_list.cell_offsets[cell];
                    const end = cell_list.cell_offsets[cell + 1];

                    for (cell_list.sorted_indices[start..end]) |acc_idx| {
                        if (acc_idx == donor_idx or acc_idx == h_idx) continue;

                        // H...A vector.
                        const v2x: f64 = @as(f64, frame.x[acc_idx]) - hx;
                        const v2y: f64 = @as(f64, frame.y[acc_idx]) - hy;
                        const v2z: f64 = @as(f64, frame.z[acc_idx]) - hz;

                        // H...A distance.
                        const dist_sq = v2x * v2x + v2y * v2y + v2z * v2z;
                        const dist: f32 = @floatCast(@sqrt(dist_sq));
                        if (dist > config.dist_cutoff) continue;

                        const mag2 = @sqrt(dist_sq);
                        if (mag2 < 1e-10) continue;

                        // D-H...A angle via dot product.
                        const dot_val = v1x * v2x + v1y * v2y + v1z * v2z;
                        const cos_angle = std.math.clamp(dot_val / (mag1 * mag2), -1.0, 1.0);
                        const angle_rad = std.math.acos(cos_angle);
                        const angle_deg: f32 = @floatCast(angle_rad * (180.0 / std.math.pi));

                        if (angle_deg >= config.angle_cutoff) {
                            try result.append(allocator, .{
                                .donor = donor_idx,
                                .hydrogen = h_idx,
                                .acceptor = acc_idx,
                                .distance = dist,
                                .angle = angle_deg,
                            });
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Parallel implementation
// ============================================================================

/// A pre-scanned donor-hydrogen bond pair.
const DHBond = struct {
    donor_idx: u32,
    h_idx: u32,
};

/// Worker function for parallel hydrogen bond detection.
/// Each worker processes a slice of the pre-scanned D-H bond list using a shared cell list.
fn hbondWorker(
    dh_bonds: []const DHBond,
    frame: types.Frame,
    config: Config,
    cell_list: *const CellList,
    result: *std.ArrayList(HBond),
    allocator: std.mem.Allocator,
    had_oom: *bool,
) void {
    for (dh_bonds) |dh| {
        const donor_idx = dh.donor_idx;
        const h_idx = dh.h_idx;

        // Pre-fetch H position in f64.
        const hx: f64 = @floatCast(frame.x[h_idx]);
        const hy: f64 = @floatCast(frame.y[h_idx]);
        const hz: f64 = @floatCast(frame.z[h_idx]);

        // D-H vector (donor minus hydrogen).
        const v1x: f64 = @as(f64, frame.x[donor_idx]) - hx;
        const v1y: f64 = @as(f64, frame.y[donor_idx]) - hy;
        const v1z: f64 = @as(f64, frame.z[donor_idx]) - hz;
        const mag1 = @sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
        if (mag1 < 1e-10) continue;

        // Determine the cell of the hydrogen atom and iterate 27 neighboring cells.
        const h_cell_x = @min(@as(u32, @intFromFloat(@max(0.0, (frame.x[h_idx] - cell_list.min_x) * cell_list.inv_cell_size))), cell_list.nx - 1);
        const h_cell_y = @min(@as(u32, @intFromFloat(@max(0.0, (frame.y[h_idx] - cell_list.min_y) * cell_list.inv_cell_size))), cell_list.ny - 1);
        const h_cell_z = @min(@as(u32, @intFromFloat(@max(0.0, (frame.z[h_idx] - cell_list.min_z) * cell_list.inv_cell_size))), cell_list.nz - 1);

        const cx_lo: u32 = if (h_cell_x > 0) h_cell_x - 1 else 0;
        const cx_hi: u32 = @min(h_cell_x + 1, cell_list.nx - 1);
        const cy_lo: u32 = if (h_cell_y > 0) h_cell_y - 1 else 0;
        const cy_hi: u32 = @min(h_cell_y + 1, cell_list.ny - 1);
        const cz_lo: u32 = if (h_cell_z > 0) h_cell_z - 1 else 0;
        const cz_hi: u32 = @min(h_cell_z + 1, cell_list.nz - 1);

        var cx: u32 = cx_lo;
        while (cx <= cx_hi) : (cx += 1) {
            var cy: u32 = cy_lo;
            while (cy <= cy_hi) : (cy += 1) {
                var cz: u32 = cz_lo;
                while (cz <= cz_hi) : (cz += 1) {
                    const cell = cx * cell_list.ny * cell_list.nz + cy * cell_list.nz + cz;
                    const start = cell_list.cell_offsets[cell];
                    const end = cell_list.cell_offsets[cell + 1];

                    for (cell_list.sorted_indices[start..end]) |acc_idx| {
                        if (acc_idx == donor_idx or acc_idx == h_idx) continue;

                        // H...A vector.
                        const v2x: f64 = @as(f64, frame.x[acc_idx]) - hx;
                        const v2y: f64 = @as(f64, frame.y[acc_idx]) - hy;
                        const v2z: f64 = @as(f64, frame.z[acc_idx]) - hz;

                        // H...A distance.
                        const dist_sq = v2x * v2x + v2y * v2y + v2z * v2z;
                        const dist: f32 = @floatCast(@sqrt(dist_sq));
                        if (dist > config.dist_cutoff) continue;

                        const mag2 = @sqrt(dist_sq);
                        if (mag2 < 1e-10) continue;

                        // D-H...A angle via dot product.
                        const dot_val = v1x * v2x + v1y * v2y + v1z * v2z;
                        const cos_angle = std.math.clamp(dot_val / (mag1 * mag2), -1.0, 1.0);
                        const angle_rad = std.math.acos(cos_angle);
                        const angle_deg: f32 = @floatCast(angle_rad * (180.0 / std.math.pi));

                        if (angle_deg >= config.angle_cutoff) {
                            result.append(allocator, .{
                                .donor = donor_idx,
                                .hydrogen = h_idx,
                                .acceptor = acc_idx,
                                .distance = dist,
                                .angle = angle_deg,
                            }) catch {
                                had_oom.* = true;
                                return;
                            };
                        }
                    }
                }
            }
        }
    }
}

/// Multi-threaded version of `detect`.
///
/// Pre-scans topology bonds to build a list of D-H pairs, then distributes
/// them across threads. Each thread independently scans all atoms for
/// acceptors. Falls back to single-threaded `detect` when `n_threads <= 1`
/// or the bond count is too small.
///
/// Note: The provided `allocator` must be thread-safe (e.g. the default general-
/// purpose allocator, `page_allocator`, or a thread-safe arena). Using a non-
/// thread-safe allocator will cause data races.
///
/// The returned slice is owned by the caller; free with `allocator.free()`.
pub fn detectParallel(
    allocator: std.mem.Allocator,
    topology: types.Topology,
    frame: types.Frame,
    config: Config,
    n_threads: usize,
) ![]HBond {
    // Fallback to single-threaded for small workloads.
    if (n_threads <= 1 or topology.bonds.len < 16) {
        return detect(allocator, topology, frame, config);
    }

    const cpu_count = std.Thread.getCpuCount() catch {
        return detect(allocator, topology, frame, config);
    };
    const actual_threads = @min(n_threads, cpu_count);

    // Pre-scan bonds to collect D-H pairs.
    var dh_list = std.ArrayList(DHBond){};
    defer dh_list.deinit(allocator);

    for (topology.bonds) |bond| {
        const a1 = topology.atoms[bond.atom_i];
        const a2 = topology.atoms[bond.atom_j];

        if (a1.element == .H and (a2.element == .N or a2.element == .O)) {
            try dh_list.append(allocator, .{ .donor_idx = bond.atom_j, .h_idx = bond.atom_i });
        } else if (a2.element == .H and (a1.element == .N or a1.element == .O)) {
            try dh_list.append(allocator, .{ .donor_idx = bond.atom_i, .h_idx = bond.atom_j });
        }
    }

    const dh_bonds = dh_list.items;
    if (dh_bonds.len == 0) {
        return allocator.alloc(HBond, 0);
    }

    // Build spatial cell list of acceptor atoms (shared across all workers).
    var cell_list = try buildAcceptorCellList(allocator, frame, topology, config.dist_cutoff);
    defer cell_list.deinit(allocator);

    // Don't use more threads than D-H bonds.
    const thread_count = @min(actual_threads, dh_bonds.len);

    // Per-thread OOM flags.
    const oom_flags = try allocator.alloc(bool, thread_count);
    defer allocator.free(oom_flags);
    for (0..thread_count) |t| {
        oom_flags[t] = false;
    }

    // Thread-local ArrayLists (zero-initialized = safe for defer).
    const tl_lists = try allocator.alloc(std.ArrayList(HBond), thread_count);
    defer allocator.free(tl_lists);
    for (0..thread_count) |t| {
        tl_lists[t] = std.ArrayList(HBond){};
    }
    defer for (0..thread_count) |t| {
        tl_lists[t].deinit(allocator);
    };

    // Partition D-H bonds across threads.
    const chunk_size = dh_bonds.len / thread_count;
    const remainder = dh_bonds.len % thread_count;

    // Spawn threads.
    const threads = try allocator.alloc(std.Thread, thread_count);
    defer allocator.free(threads);

    var spawned: usize = 0;
    errdefer for (threads[0..spawned]) |thread| {
        thread.join();
    };

    var offset: usize = 0;
    for (0..thread_count) |t| {
        const this_chunk = chunk_size + @as(usize, if (t < remainder) 1 else 0);
        threads[t] = try std.Thread.spawn(.{}, hbondWorker, .{
            dh_bonds[offset..][0..this_chunk],
            frame,
            config,
            &cell_list,
            &tl_lists[t],
            allocator,
            &oom_flags[t],
        });
        spawned += 1;
        offset += this_chunk;
    }

    // Join all threads.
    for (threads[0..spawned]) |thread| {
        thread.join();
    }
    spawned = 0;

    // Check for OOM in any worker.
    for (0..thread_count) |t| {
        if (oom_flags[t]) return error.OutOfMemory;
    }

    // Count total and concatenate.
    var total: usize = 0;
    for (0..thread_count) |t| {
        total += tl_lists[t].items.len;
    }

    const result = try allocator.alloc(HBond, total);
    errdefer allocator.free(result);

    var concat_offset: usize = 0;
    for (0..thread_count) |t| {
        const items = tl_lists[t].items;
        if (items.len > 0) {
            @memcpy(result[concat_offset..][0..items.len], items);
            concat_offset += items.len;
        }
    }

    return result;
}

// ============================================================================
// Tests
// ============================================================================

/// Build a minimal Topology + Frame for testing.
/// Layout:
///   atom 0: donor N at origin
///   atom 1: H bonded to donor
///   atom 2: acceptor O
/// Bond: 0-1 (N-H).
fn makeTestSystem(
    allocator: std.mem.Allocator,
    hx: f32,
    hy: f32,
    hz: f32,
    ax: f32,
    ay: f32,
    az: f32,
) !struct { topo: types.Topology, frame: types.Frame } {
    var topo = try types.Topology.init(allocator, .{
        .n_atoms = 3,
        .n_residues = 1,
        .n_chains = 1,
        .n_bonds = 1,
    });
    errdefer topo.deinit();

    const FS4 = types.FixedString(4);
    const FS5 = types.FixedString(5);

    topo.atoms[0] = .{ .name = FS4.fromSlice("N"), .element = .N, .residue_index = 0 };
    topo.atoms[1] = .{ .name = FS4.fromSlice("H"), .element = .H, .residue_index = 0 };
    topo.atoms[2] = .{ .name = FS4.fromSlice("O"), .element = .O, .residue_index = 0 };

    topo.residues[0] = .{
        .name = FS5.fromSlice("ALA"),
        .chain_index = 0,
        .atom_range = .{ .start = 0, .len = 3 },
        .resid = 1,
    };
    topo.chains[0] = .{
        .name = FS4.fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 1 },
    };
    topo.bonds[0] = .{ .atom_i = 0, .atom_j = 1 }; // N-H bond

    var frame = try types.Frame.init(allocator, 3);
    errdefer frame.deinit();

    // Donor N at origin.
    frame.x[0] = 0.0;
    frame.y[0] = 0.0;
    frame.z[0] = 0.0;
    // Hydrogen.
    frame.x[1] = hx;
    frame.y[1] = hy;
    frame.z[1] = hz;
    // Acceptor O.
    frame.x[2] = ax;
    frame.y[2] = ay;
    frame.z[2] = az;

    return .{ .topo = topo, .frame = frame };
}

test "hbonds: detects valid N-H...O bond" {
    // Linear geometry: N at (0,0,0), H at (1,0,0), O at (2.5,0,0).
    // H...O distance = 1.5 Å, D-H...A angle = 180°.
    const allocator = std.testing.allocator;

    var sys = try makeTestSystem(allocator, 1.0, 0.0, 0.0, 2.5, 0.0, 0.0);
    defer sys.topo.deinit();
    defer sys.frame.deinit();

    const bonds = try detect(allocator, sys.topo, sys.frame, .{});
    defer allocator.free(bonds);

    try std.testing.expectEqual(@as(usize, 1), bonds.len);
    try std.testing.expectEqual(@as(u32, 0), bonds[0].donor);
    try std.testing.expectEqual(@as(u32, 1), bonds[0].hydrogen);
    try std.testing.expectEqual(@as(u32, 2), bonds[0].acceptor);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), bonds[0].distance, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 180.0), bonds[0].angle, 1e-2);
}

test "hbonds: H...A distance too large — no bond" {
    // H...O = 3.5 Å (> default cutoff of 2.5).
    const allocator = std.testing.allocator;

    var sys = try makeTestSystem(allocator, 1.0, 0.0, 0.0, 4.5, 0.0, 0.0);
    defer sys.topo.deinit();
    defer sys.frame.deinit();

    const bonds = try detect(allocator, sys.topo, sys.frame, .{});
    defer allocator.free(bonds);

    try std.testing.expectEqual(@as(usize, 0), bonds.len);
}

test "hbonds: angle too small — no bond" {
    // N at origin, H at (1,0,0), O at (1,2,0).
    // D-H...A angle ~ 63.4° (< 120° threshold).
    const allocator = std.testing.allocator;

    var sys = try makeTestSystem(allocator, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0);
    defer sys.topo.deinit();
    defer sys.frame.deinit();

    const bonds = try detect(allocator, sys.topo, sys.frame, .{});
    defer allocator.free(bonds);

    try std.testing.expectEqual(@as(usize, 0), bonds.len);
}

test "hbonds: non-donor atom bonded to H — no bond detected" {
    // Carbon bonded to H should not be treated as a donor.
    const allocator = std.testing.allocator;

    var topo = try types.Topology.init(allocator, .{
        .n_atoms = 3,
        .n_residues = 1,
        .n_chains = 1,
        .n_bonds = 1,
    });
    defer topo.deinit();

    const FS4 = types.FixedString(4);
    const FS5 = types.FixedString(5);

    topo.atoms[0] = .{ .name = FS4.fromSlice("C"), .element = .C, .residue_index = 0 };
    topo.atoms[1] = .{ .name = FS4.fromSlice("H"), .element = .H, .residue_index = 0 };
    topo.atoms[2] = .{ .name = FS4.fromSlice("O"), .element = .O, .residue_index = 0 };
    topo.residues[0] = .{
        .name = FS5.fromSlice("ALA"),
        .chain_index = 0,
        .atom_range = .{ .start = 0, .len = 3 },
        .resid = 1,
    };
    topo.chains[0] = .{
        .name = FS4.fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 1 },
    };
    topo.bonds[0] = .{ .atom_i = 0, .atom_j = 1 }; // C-H bond

    var frame = try types.Frame.init(allocator, 3);
    defer frame.deinit();
    frame.x[0] = 0.0;
    frame.y[0] = 0.0;
    frame.z[0] = 0.0; // C
    frame.x[1] = 1.0;
    frame.y[1] = 0.0;
    frame.z[1] = 0.0; // H
    frame.x[2] = 2.3;
    frame.y[2] = 0.0;
    frame.z[2] = 0.0; // O

    const bonds = try detect(allocator, topo, frame, .{});
    defer allocator.free(bonds);

    try std.testing.expectEqual(@as(usize, 0), bonds.len);
}

test "hbonds: sulfur acceptor is detected" {
    // N-H...S should trigger on a sulfur acceptor.
    const allocator = std.testing.allocator;

    var topo = try types.Topology.init(allocator, .{
        .n_atoms = 3,
        .n_residues = 1,
        .n_chains = 1,
        .n_bonds = 1,
    });
    defer topo.deinit();

    const FS4 = types.FixedString(4);
    const FS5 = types.FixedString(5);

    topo.atoms[0] = .{ .name = FS4.fromSlice("N"), .element = .N, .residue_index = 0 };
    topo.atoms[1] = .{ .name = FS4.fromSlice("H"), .element = .H, .residue_index = 0 };
    topo.atoms[2] = .{ .name = FS4.fromSlice("S"), .element = .S, .residue_index = 0 };
    topo.residues[0] = .{
        .name = FS5.fromSlice("MET"),
        .chain_index = 0,
        .atom_range = .{ .start = 0, .len = 3 },
        .resid = 1,
    };
    topo.chains[0] = .{
        .name = FS4.fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 1 },
    };
    topo.bonds[0] = .{ .atom_i = 0, .atom_j = 1 };

    var frame = try types.Frame.init(allocator, 3);
    defer frame.deinit();
    frame.x[0] = 0.0;
    frame.y[0] = 0.0;
    frame.z[0] = 0.0;
    frame.x[1] = 1.0;
    frame.y[1] = 0.0;
    frame.z[1] = 0.0;
    frame.x[2] = 2.3;
    frame.y[2] = 0.0;
    frame.z[2] = 0.0;

    const bonds = try detect(allocator, topo, frame, .{});
    defer allocator.free(bonds);

    try std.testing.expectEqual(@as(usize, 1), bonds.len);
    try std.testing.expectEqual(@as(u32, 2), bonds[0].acceptor);
}

test "hbonds: detectParallel matches single-threaded detect" {
    const allocator = std.testing.allocator;

    // Linear geometry: N at (0,0,0), H at (1,0,0), O at (2.5,0,0).
    var sys = try makeTestSystem(allocator, 1.0, 0.0, 0.0, 2.5, 0.0, 0.0);
    defer sys.topo.deinit();
    defer sys.frame.deinit();

    // Single-threaded.
    const st_bonds = try detect(allocator, sys.topo, sys.frame, .{});
    defer allocator.free(st_bonds);

    // Multi-threaded (falls back because bonds < 16, so result must match).
    const mt_bonds = try detectParallel(allocator, sys.topo, sys.frame, .{}, 4);
    defer allocator.free(mt_bonds);

    try std.testing.expectEqual(st_bonds.len, mt_bonds.len);
    for (st_bonds, mt_bonds) |st, mt| {
        try std.testing.expectEqual(st.donor, mt.donor);
        try std.testing.expectEqual(st.hydrogen, mt.hydrogen);
        try std.testing.expectEqual(st.acceptor, mt.acceptor);
        try std.testing.expectApproxEqAbs(st.distance, mt.distance, 1e-4);
        try std.testing.expectApproxEqAbs(st.angle, mt.angle, 1e-2);
    }
}

test "hbonds: no bonds in empty topology" {
    const allocator = std.testing.allocator;

    var topo = try types.Topology.init(allocator, .{
        .n_atoms = 0,
        .n_residues = 0,
        .n_chains = 0,
        .n_bonds = 0,
    });
    defer topo.deinit();

    var frame = try types.Frame.init(allocator, 0);
    defer frame.deinit();

    const bonds = try detect(allocator, topo, frame, .{});
    defer allocator.free(bonds);

    try std.testing.expectEqual(@as(usize, 0), bonds.len);
}

test "hbonds: cell list produces correct results with multiple acceptors" {
    // 10-atom system with mixed elements, multiple D-H pairs and acceptors.
    // Tests that the cell list spatial optimization finds the same bonds
    // as a brute-force scan would.
    const allocator = std.testing.allocator;

    const n_atoms: u32 = 10;
    const n_bonds: u32 = 2;

    var topo = try types.Topology.init(allocator, .{
        .n_atoms = n_atoms,
        .n_residues = 1,
        .n_chains = 1,
        .n_bonds = n_bonds,
    });
    defer topo.deinit();

    const FS4 = types.FixedString(4);
    const FS5 = types.FixedString(5);

    // Atom layout:
    //  0: N  donor1     (0, 0, 0)
    //  1: H  bonded to 0 (1, 0, 0)
    //  2: O  acceptor   (2.3, 0, 0)   -- within cutoff of H1 (dist=1.3), angle~180
    //  3: C  non-acceptor (3, 0, 0)
    //  4: N  acceptor   (10, 0, 0)     -- far away, should NOT be detected
    //  5: O  donor2     (0, 5, 0)
    //  6: H  bonded to 5 (0, 6, 0)
    //  7: S  acceptor   (0, 7.2, 0)   -- within cutoff of H6 (dist=1.2), angle~180
    //  8: C  non-acceptor (0, 8, 0)
    //  9: O  acceptor   (0, 6.5, 2.0) -- within cutoff of H6 (dist~2.06) but angle too small

    topo.atoms[0] = .{ .name = FS4.fromSlice("N"), .element = .N, .residue_index = 0 };
    topo.atoms[1] = .{ .name = FS4.fromSlice("H1"), .element = .H, .residue_index = 0 };
    topo.atoms[2] = .{ .name = FS4.fromSlice("O"), .element = .O, .residue_index = 0 };
    topo.atoms[3] = .{ .name = FS4.fromSlice("C"), .element = .C, .residue_index = 0 };
    topo.atoms[4] = .{ .name = FS4.fromSlice("N2"), .element = .N, .residue_index = 0 };
    topo.atoms[5] = .{ .name = FS4.fromSlice("O2"), .element = .O, .residue_index = 0 };
    topo.atoms[6] = .{ .name = FS4.fromSlice("H2"), .element = .H, .residue_index = 0 };
    topo.atoms[7] = .{ .name = FS4.fromSlice("S"), .element = .S, .residue_index = 0 };
    topo.atoms[8] = .{ .name = FS4.fromSlice("C2"), .element = .C, .residue_index = 0 };
    topo.atoms[9] = .{ .name = FS4.fromSlice("O3"), .element = .O, .residue_index = 0 };

    topo.residues[0] = .{
        .name = FS5.fromSlice("TST"),
        .chain_index = 0,
        .atom_range = .{ .start = 0, .len = n_atoms },
        .resid = 1,
    };
    topo.chains[0] = .{
        .name = FS4.fromSlice("A"),
        .residue_range = .{ .start = 0, .len = 1 },
    };

    topo.bonds[0] = .{ .atom_i = 0, .atom_j = 1 }; // N-H bond (donor1)
    topo.bonds[1] = .{ .atom_i = 5, .atom_j = 6 }; // O-H bond (donor2)

    var frame = try types.Frame.init(allocator, n_atoms);
    defer frame.deinit();

    // Positions (Angstroms).
    frame.x[0] = 0.0;
    frame.y[0] = 0.0;
    frame.z[0] = 0.0; // N donor1
    frame.x[1] = 1.0;
    frame.y[1] = 0.0;
    frame.z[1] = 0.0; // H1
    frame.x[2] = 2.3;
    frame.y[2] = 0.0;
    frame.z[2] = 0.0; // O acceptor (near H1)
    frame.x[3] = 3.0;
    frame.y[3] = 0.0;
    frame.z[3] = 0.0; // C (not acceptor)
    frame.x[4] = 10.0;
    frame.y[4] = 0.0;
    frame.z[4] = 0.0; // N2 acceptor (far away)
    frame.x[5] = 0.0;
    frame.y[5] = 5.0;
    frame.z[5] = 0.0; // O donor2
    frame.x[6] = 0.0;
    frame.y[6] = 6.0;
    frame.z[6] = 0.0; // H2
    frame.x[7] = 0.0;
    frame.y[7] = 7.2;
    frame.z[7] = 0.0; // S acceptor (near H2)
    frame.x[8] = 0.0;
    frame.y[8] = 8.0;
    frame.z[8] = 0.0; // C2 (not acceptor)
    frame.x[9] = 0.0;
    frame.y[9] = 6.5;
    frame.z[9] = 2.0; // O3 acceptor (near H2 but bad angle)

    const hbonds = try detect(allocator, topo, frame, .{});
    defer allocator.free(hbonds);

    // Expect exactly 2 H-bonds:
    // 1) N(0)-H(1)...O(2): dist=1.3, angle=180
    // 2) O(5)-H(6)...S(7): dist=1.2, angle=180
    try std.testing.expectEqual(@as(usize, 2), hbonds.len);

    // Verify both bonds are present (order may vary due to cell iteration order).
    var found_nh_o = false;
    var found_oh_s = false;
    for (hbonds) |hb| {
        if (hb.donor == 0 and hb.hydrogen == 1 and hb.acceptor == 2) {
            found_nh_o = true;
            try std.testing.expectApproxEqAbs(@as(f32, 1.3), hb.distance, 1e-4);
            try std.testing.expectApproxEqAbs(@as(f32, 180.0), hb.angle, 1e-2);
        }
        if (hb.donor == 5 and hb.hydrogen == 6 and hb.acceptor == 7) {
            found_oh_s = true;
            try std.testing.expectApproxEqAbs(@as(f32, 1.2), hb.distance, 1e-4);
            try std.testing.expectApproxEqAbs(@as(f32, 180.0), hb.angle, 1e-2);
        }
    }
    try std.testing.expect(found_nh_o);
    try std.testing.expect(found_oh_s);
}
