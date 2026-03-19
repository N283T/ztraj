const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;
const types = @import("types.zig");
const residue_mod = @import("residue.zig");
const neighbor_list_mod = @import("neighbor_list.zig");
const simd = @import("simd.zig");
const thread_pool_mod = @import("thread_pool.zig");

const Vec3f32 = types.Vec3f32;
const Residue = residue_mod.Residue;
const SideChainAtom = residue_mod.SideChainAtom;

// ---------------------------------------------------------------------------
// AABB (Axis-Aligned Bounding Box) for early rejection
// Pattern from C++ DSSP (dssp.cpp:471-479)
// ---------------------------------------------------------------------------

const AABB = struct {
    min: Vec3f32,
    max: Vec3f32,

    /// Check if a sphere intersects this bounding box
    fn intersectsSphere(self: AABB, center: Vec3f32, radius: f32) bool {
        return center.x + radius >= self.min.x and
            center.x - radius <= self.max.x and
            center.y + radius >= self.min.y and
            center.y - radius <= self.max.y and
            center.z + radius >= self.min.z and
            center.z - radius <= self.max.z;
    }

    /// Compute AABB for a residue (backbone + side chain)
    fn fromResidue(res: *const Residue) AABB {
        var min_x = @min(@min(res.n.x, res.ca.x), @min(res.c.x, res.o.x));
        var max_x = @max(@max(res.n.x, res.ca.x), @max(res.c.x, res.o.x));
        var min_y = @min(@min(res.n.y, res.ca.y), @min(res.c.y, res.o.y));
        var max_y = @max(@max(res.n.y, res.ca.y), @max(res.c.y, res.o.y));
        var min_z = @min(@min(res.n.z, res.ca.z), @min(res.c.z, res.o.z));
        var max_z = @max(@max(res.n.z, res.ca.z), @max(res.c.z, res.o.z));

        // Include side chain atoms
        for (res.side_chain) |sc| {
            min_x = @min(min_x, sc.pos.x);
            max_x = @max(max_x, sc.pos.x);
            min_y = @min(min_y, sc.pos.y);
            max_y = @max(max_y, sc.pos.y);
            min_z = @min(min_z, sc.pos.z);
            max_z = @max(max_z, sc.pos.z);
        }

        // Extend by max atom radius + water radius
        const extend = types.kRadiusCA + types.kRadiusWater; // ~3.3 Å
        return .{
            .min = .{ .x = min_x - extend, .y = min_y - extend, .z = min_z - extend },
            .max = .{ .x = max_x + extend, .y = max_y + extend, .z = max_z + extend },
        };
    }
};

// ---------------------------------------------------------------------------
// Fibonacci sphere surface dots (dssp.cpp:627-650)
// ---------------------------------------------------------------------------

const kN = types.kFibonacciN; // 200
const kPointCount = 2 * kN + 1; // 401

/// Pre-computed Fibonacci sphere points and per-point solid angle weight.
const SurfaceDots = struct {
    points: [kPointCount]Vec3f32,
    weight: f32,

    fn init() SurfaceDots {
        @setEvalBranchQuota(10000);
        const golden_ratio: f32 = (1.0 + @sqrt(@as(f32, 5.0))) / 2.0;
        const point_count_f: f32 = @floatFromInt(kPointCount);
        const w = 4.0 * math.pi / point_count_f;

        var dots = SurfaceDots{
            .points = undefined,
            .weight = w,
        };

        for (0..kPointCount) |idx| {
            const i_int: i32 = @as(i32, @intCast(idx)) - kN;
            const i: f32 = @floatFromInt(i_int);

            // Latitude
            const lat = math.asin(2.0 * i / point_count_f);

            // Longitude (golden spiral)
            const lon: f32 = @floatCast(@mod(@as(f64, @floatCast(i)), @as(f64, @floatCast(golden_ratio))) *
                2.0 * math.pi / @as(f64, @floatCast(golden_ratio)));

            dots.points[idx] = Vec3f32{
                .x = @sin(lon) * @cos(lat),
                .y = @cos(lon) * @cos(lat),
                .z = @sin(lat),
            };
        }

        return dots;
    }
};

const surface_dots = SurfaceDots.init();

// ---------------------------------------------------------------------------
// Neighbour atom record for surface calculation
// ---------------------------------------------------------------------------

const NeighbourAtom = struct {
    location: Vec3f32,
    radius_sq: f32, // (radius_atom + radius_water)^2
    distance: f32, // distance from reference atom
};

// ---------------------------------------------------------------------------
// Surface accessibility (dssp.cpp:610-714)
// ---------------------------------------------------------------------------

/// Calculate the accessible surface area for a single atom.
///
/// Uses Fibonacci sphere test points to determine which fraction
/// of the atom's surface is not occluded by neighbouring atoms.
fn calculateAtomSurface(
    atom_radius: f32,
    neighbours: []const NeighbourAtom,
) f32 {
    const radius = atom_radius + types.kRadiusWater;
    var surface: f32 = 0.0;

    for (surface_dots.points) |dot| {
        const test_point = dot.scale(radius);

        var free = true;
        for (neighbours) |nb| {
            const dx = test_point.x - nb.location.x;
            const dy = test_point.y - nb.location.y;
            const dz = test_point.z - nb.location.z;
            const dist_sq = dx * dx + dy * dy + dz * dz;
            if (dist_sq < nb.radius_sq) {
                free = false;
                break;
            }
        }

        if (free) {
            surface += surface_dots.weight;
        }
    }

    return surface * radius * radius;
}

/// SIMD-optimized calculation of accessible surface area for a single atom.
///
/// Uses Fibonacci sphere test points with SIMD batch processing of neighbors.
/// Processes neighbors in batches of 8 using @Vector(8, f32) for parallel
/// distance calculation and collision detection.
///
/// Performance: Processes 8 neighbors simultaneously vs. 1 in scalar version.
/// Expected speedup: ~4-7x for proteins with many neighbors (>16 per atom).
///
/// CPU feature detection: Uses compile-time detection in simd.zig to select
/// optimal vector width (AVX-512: 16, AVX2/NEON: 8, SSE: 4).
fn calculateAtomSurfaceSimd(
    atom_radius: f32,
    neighbours: []const NeighbourAtom,
) f32 {
    const radius = atom_radius + types.kRadiusWater;
    var surface: f32 = 0.0;

    // Use CPU-optimal vector width detected at compile time
    const batch_size = simd.vec_len;
    const n_full_batches = neighbours.len / batch_size;

    for (surface_dots.points) |dot| {
        const test_point = dot.scale(radius);

        var free = true;

        // Process neighbors in batches using SIMD (width determined by CPU features)
        var batch_idx: usize = 0;
        while (batch_idx < n_full_batches) : (batch_idx += 1) {
            const batch_start = batch_idx * batch_size;

            // Build batched coordinate arrays and radii
            var nb_xs: [batch_size]f32 = undefined;
            var nb_ys: [batch_size]f32 = undefined;
            var nb_zs: [batch_size]f32 = undefined;
            var radii_sq: [batch_size]f32 = undefined;

            inline for (0..batch_size) |i| {
                const nb = neighbours[batch_start + i];
                nb_xs[i] = nb.location.x;
                nb_ys[i] = nb.location.y;
                nb_zs[i] = nb.location.z;
                radii_sq[i] = nb.radius_sq;
            }

            // Check if test point is buried by any neighbor in this batch
            if (simd.isPointBuriedBatch(
                batch_size,
                test_point.x,
                test_point.y,
                test_point.z,
                nb_xs,
                nb_ys,
                nb_zs,
                radii_sq,
            )) {
                free = false;
                break;
            }
        }

        // Handle remainder with scalar code
        if (free) {
            const remainder_start = n_full_batches * batch_size;
            for (neighbours[remainder_start..]) |nb| {
                const dx = test_point.x - nb.location.x;
                const dy = test_point.y - nb.location.y;
                const dz = test_point.z - nb.location.z;
                const dist_sq = dx * dx + dy * dy + dz * dz;
                if (dist_sq < nb.radius_sq) {
                    free = false;
                    break;
                }
            }
        }

        if (free) {
            surface += surface_dots.weight;
        }
    }

    return surface * radius * radius;
}

/// Collect neighbouring atoms relative to a reference atom.
/// Returns atoms from nearby residues whose extended radius overlaps.
fn collectNeighbours(
    atom: Vec3f32,
    atom_radius: f32,
    residues: []const Residue,
    allocator: Allocator,
) ![]NeighbourAtom {
    var neighbours: std.ArrayListAligned(NeighbourAtom, null) = .empty;
    errdefer neighbours.deinit(allocator);

    const probe = atom_radius + types.kRadiusWater;

    for (residues) |res| {
        if (!res.complete) continue;

        // Quick bounding-box check using CA distance
        // Max atom radius ~2Å + water 1.4Å + probe ~3.3Å ≈ generous cutoff
        const ca_dist = atom.distance(res.ca);
        if (ca_dist > 12.0) continue;

        // Add backbone atoms
        try addNeighbour(&neighbours, allocator, atom, probe, res.n, types.kRadiusN);
        try addNeighbour(&neighbours, allocator, atom, probe, res.ca, types.kRadiusCA);
        try addNeighbour(&neighbours, allocator, atom, probe, res.c, types.kRadiusC);
        try addNeighbour(&neighbours, allocator, atom, probe, res.o, types.kRadiusO);

        // Add side chain atoms
        for (res.side_chain) |sc| {
            try addNeighbour(&neighbours, allocator, atom, probe, sc.pos, types.kRadiusSideAtom);
        }
    }

    // Sort by distance (nearest first for early exit optimisation)
    std.mem.sort(NeighbourAtom, neighbours.items, {}, struct {
        fn lessThan(_: void, a: NeighbourAtom, b: NeighbourAtom) bool {
            return a.distance < b.distance;
        }
    }.lessThan);

    return neighbours.toOwnedSlice(allocator);
}

fn addNeighbour(
    list: *std.ArrayListAligned(NeighbourAtom, null),
    allocator: Allocator,
    atom: Vec3f32,
    probe: f32,
    nb_pos: Vec3f32,
    nb_radius: f32,
) !void {
    const nb_extended = nb_radius + types.kRadiusWater;
    const dist_sq = atom.distanceSq(nb_pos);

    // Skip the same atom position (squared distance < 0.0001, matching C++ DSSP)
    if (dist_sq < 0.0001) return;

    const dist = @sqrt(dist_sq);
    if (dist < probe + nb_extended) {
        // Store position relative to atom centre (so test points are also relative)
        try list.append(allocator, .{
            .location = nb_pos.sub(atom),
            .radius_sq = nb_extended * nb_extended,
            .distance = dist,
        });
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Calculate surface accessibility for all residues.
///
/// Each residue's accessibility is the sum of per-atom accessible surface
/// areas (N, CA, C, O, plus side chain atoms).
pub fn calculateAccessibilities(residues: []Residue, allocator: Allocator) !void {
    for (residues) |*res| {
        if (!res.complete) continue;

        var total: f32 = 0.0;

        // Backbone atoms
        const backbone_atoms = [_]struct { pos: Vec3f32, radius: f32 }{
            .{ .pos = res.n, .radius = types.kRadiusN },
            .{ .pos = res.ca, .radius = types.kRadiusCA },
            .{ .pos = res.c, .radius = types.kRadiusC },
            .{ .pos = res.o, .radius = types.kRadiusO },
        };

        for (backbone_atoms) |ba| {
            const neighbours = try collectNeighbours(ba.pos, ba.radius, residues, allocator);
            defer allocator.free(neighbours);
            total += calculateAtomSurfaceSimd(ba.radius, neighbours);
        }

        // Side chain atoms
        for (res.side_chain) |sc| {
            const neighbours = try collectNeighbours(sc.pos, types.kRadiusSideAtom, residues, allocator);
            defer allocator.free(neighbours);
            total += calculateAtomSurfaceSimd(types.kRadiusSideAtom, neighbours);
        }

        res.accessibility = total;
    }
}

// ---------------------------------------------------------------------------
// Optimized accessibility using spatial hash grid
// ---------------------------------------------------------------------------

/// Collect neighbours using a pre-built residue neighbor list.
///
/// Instead of scanning all residues (O(N)), only checks residues that
/// are known to be nearby via the spatial hash grid (O(nearby)).
fn collectNeighboursFromList(
    atom: Vec3f32,
    atom_radius: f32,
    residues: []const Residue,
    self_residue: *const Residue,
    residue_neighbors: []const u32,
    allocator: Allocator,
) ![]NeighbourAtom {
    var neighbours: std.ArrayListAligned(NeighbourAtom, null) = .empty;
    errdefer neighbours.deinit(allocator);

    const probe = atom_radius + types.kRadiusWater;

    // Add self-residue atoms (the spatial grid neighbor list does not include self)
    try addNeighbour(&neighbours, allocator, atom, probe, self_residue.n, types.kRadiusN);
    try addNeighbour(&neighbours, allocator, atom, probe, self_residue.ca, types.kRadiusCA);
    try addNeighbour(&neighbours, allocator, atom, probe, self_residue.c, types.kRadiusC);
    try addNeighbour(&neighbours, allocator, atom, probe, self_residue.o, types.kRadiusO);
    for (self_residue.side_chain) |sc| {
        try addNeighbour(&neighbours, allocator, atom, probe, sc.pos, types.kRadiusSideAtom);
    }

    for (residue_neighbors) |j| {
        const res = &residues[j];
        if (!res.complete) continue;

        // AABB early rejection: skip residue if sphere doesn't intersect bounding box
        const aabb = AABB.fromResidue(res);
        if (!aabb.intersectsSphere(atom, probe)) continue;

        // Add backbone atoms
        try addNeighbour(&neighbours, allocator, atom, probe, res.n, types.kRadiusN);
        try addNeighbour(&neighbours, allocator, atom, probe, res.ca, types.kRadiusCA);
        try addNeighbour(&neighbours, allocator, atom, probe, res.c, types.kRadiusC);
        try addNeighbour(&neighbours, allocator, atom, probe, res.o, types.kRadiusO);

        // Add side chain atoms
        for (res.side_chain) |sc| {
            try addNeighbour(&neighbours, allocator, atom, probe, sc.pos, types.kRadiusSideAtom);
        }
    }

    // Sort by distance (nearest first for early exit optimisation)
    std.mem.sort(NeighbourAtom, neighbours.items, {}, struct {
        fn lessThan(_: void, a: NeighbourAtom, b: NeighbourAtom) bool {
            return a.distance < b.distance;
        }
    }.lessThan);

    return neighbours.toOwnedSlice(allocator);
}

/// Calculate surface accessibility using a spatial hash grid for neighbor finding.
///
/// This is O(N + nearby) per residue instead of O(N²) total, providing
/// significant speedup for large proteins (>100 residues).
/// Uses stack-allocated neighbor buffer to avoid heap allocation in hot loop.
pub fn calculateAccessibilitiesOptimized(residues: []Residue, allocator: Allocator) !void {
    const n = residues.len;
    if (n == 0) return;

    // Extract CA positions and completeness flags
    const ca_positions = try allocator.alloc(Vec3f32, n);
    defer allocator.free(ca_positions);
    const complete = try allocator.alloc(bool, n);
    defer allocator.free(complete);

    for (residues, 0..) |res, i| {
        ca_positions[i] = res.ca;
        complete[i] = res.complete;
    }

    // Build residue-level neighbor list with 12 Å cutoff
    const residue_neighbors = try neighbor_list_mod.buildResidueNeighborList(
        ca_positions,
        complete,
        12.0,
        allocator,
    );
    defer neighbor_list_mod.freeResidueNeighborList(residue_neighbors, allocator);

    // Pre-compute AABBs for all residues
    const residue_aabbs = try allocator.alloc(AABB, n);
    defer allocator.free(residue_aabbs);
    for (residues, 0..) |*res, i| {
        residue_aabbs[i] = AABB.fromResidue(res);
    }

    // Stack-allocated neighbor buffer (no heap allocation in hot loop)
    var neighbour_buffer: [512]NeighbourAtom = undefined;

    // Calculate per-residue accessibility using the neighbor list
    for (residues, 0..) |*res, i| {
        if (!res.complete) continue;

        var total: f32 = 0.0;

        const backbone_atoms = [_]struct { pos: Vec3f32, radius: f32 }{
            .{ .pos = res.n, .radius = types.kRadiusN },
            .{ .pos = res.ca, .radius = types.kRadiusCA },
            .{ .pos = res.c, .radius = types.kRadiusC },
            .{ .pos = res.o, .radius = types.kRadiusO },
        };

        for (backbone_atoms) |ba| {
            const n_neighbours = collectNeighboursNoAlloc(
                ba.pos, ba.radius, residues, res, residue_neighbors[i],
                residue_aabbs, &neighbour_buffer,
            );
            total += calculateAtomSurfaceSimd(ba.radius, neighbour_buffer[0..n_neighbours]);
        }

        for (res.side_chain) |sc| {
            const n_neighbours = collectNeighboursNoAlloc(
                sc.pos, types.kRadiusSideAtom, residues, res, residue_neighbors[i],
                residue_aabbs, &neighbour_buffer,
            );
            total += calculateAtomSurfaceSimd(types.kRadiusSideAtom, neighbour_buffer[0..n_neighbours]);
        }

        res.accessibility = total;
    }
}

// ---------------------------------------------------------------------------
// Parallel accessibility calculation using thread pool
// ---------------------------------------------------------------------------

/// Context passed to each worker thread
///
/// Thread safety: All fields except `results` are read-only from workers' perspective.
/// The `results` slice is partitioned so each worker writes to a disjoint range,
/// eliminating the need for synchronization. Workers never read from `results`.
const ParallelContext = struct {
    residues: []const Residue,
    residue_neighbors: []const []const u32,
    residue_aabbs: []const AABB, // Pre-computed AABBs for all residues
    results: []f32, // Each worker writes to disjoint range [start..end]
    allocator: Allocator,
};

/// Worker function: calculate accessibility for a range of residues
fn parallelWorker(ctx: ParallelContext, start: usize, end: usize) void {
    // Pre-allocate neighbor buffer to avoid repeated allocations
    // Typical protein has ~20-50 neighbors per atom after spatial filtering
    var neighbour_buffer: [512]NeighbourAtom = undefined;

    for (start..end) |i| {
        const res = &ctx.residues[i];
        if (!res.complete) {
            ctx.results[i] = 0.0;
            continue;
        }

        var total: f32 = 0.0;

        // Process backbone atoms
        const backbone_atoms = [_]struct { pos: Vec3f32, radius: f32 }{
            .{ .pos = res.n, .radius = types.kRadiusN },
            .{ .pos = res.ca, .radius = types.kRadiusCA },
            .{ .pos = res.c, .radius = types.kRadiusC },
            .{ .pos = res.o, .radius = types.kRadiusO },
        };

        for (backbone_atoms) |ba| {
            const n_neighbours = collectNeighboursNoAlloc(
                ba.pos, ba.radius, ctx.residues, res, ctx.residue_neighbors[i],
                ctx.residue_aabbs, &neighbour_buffer,
            );
            total += calculateAtomSurfaceSimd(ba.radius, neighbour_buffer[0..n_neighbours]);
        }

        // Process side chain atoms
        for (res.side_chain) |sc| {
            const n_neighbours = collectNeighboursNoAlloc(
                sc.pos, types.kRadiusSideAtom, ctx.residues, res, ctx.residue_neighbors[i],
                ctx.residue_aabbs, &neighbour_buffer,
            );
            total += calculateAtomSurfaceSimd(types.kRadiusSideAtom, neighbour_buffer[0..n_neighbours]);
        }

        ctx.results[i] = total;
    }
}

/// Fully non-allocating version using pre-computed AABBs and stack buffer
/// Returns the number of neighbours written to the buffer
fn collectNeighboursNoAlloc(
    atom: Vec3f32,
    atom_radius: f32,
    residues: []const Residue,
    self_residue: *const Residue,
    residue_neighbors: []const u32,
    residue_aabbs: []const AABB,
    buffer: []NeighbourAtom,
) usize {
    var count: usize = 0;
    const probe = atom_radius + types.kRadiusWater;

    // Add self-residue atoms (inline for speed)
    count = addNeighbourToBuffer(buffer, count, atom, probe, self_residue.n, types.kRadiusN);
    count = addNeighbourToBuffer(buffer, count, atom, probe, self_residue.ca, types.kRadiusCA);
    count = addNeighbourToBuffer(buffer, count, atom, probe, self_residue.c, types.kRadiusC);
    count = addNeighbourToBuffer(buffer, count, atom, probe, self_residue.o, types.kRadiusO);
    for (self_residue.side_chain) |sc| {
        count = addNeighbourToBuffer(buffer, count, atom, probe, sc.pos, types.kRadiusSideAtom);
    }

    for (residue_neighbors) |j| {
        const res = &residues[j];
        if (!res.complete) continue;

        // Use pre-computed AABB for early rejection
        if (!residue_aabbs[j].intersectsSphere(atom, probe)) continue;

        count = addNeighbourToBuffer(buffer, count, atom, probe, res.n, types.kRadiusN);
        count = addNeighbourToBuffer(buffer, count, atom, probe, res.ca, types.kRadiusCA);
        count = addNeighbourToBuffer(buffer, count, atom, probe, res.c, types.kRadiusC);
        count = addNeighbourToBuffer(buffer, count, atom, probe, res.o, types.kRadiusO);

        for (res.side_chain) |sc| {
            count = addNeighbourToBuffer(buffer, count, atom, probe, sc.pos, types.kRadiusSideAtom);
        }
    }

    // Sort by distance (nearest first for early exit)
    std.mem.sort(NeighbourAtom, buffer[0..count], {}, struct {
        fn lessThan(_: void, a: NeighbourAtom, b: NeighbourAtom) bool {
            return a.distance < b.distance;
        }
    }.lessThan);

    return count;
}

/// Add neighbour to pre-allocated buffer (no allocation)
/// Buffer size of 512 is sufficient for typical proteins (~20-50 neighbors per atom).
fn addNeighbourToBuffer(
    buffer: []NeighbourAtom,
    count: usize,
    atom: Vec3f32,
    probe: f32,
    nb_pos: Vec3f32,
    nb_radius: f32,
) usize {
    if (count >= buffer.len) {
        // Buffer full - this indicates an unusually dense region
        // Panic in all builds to ensure we don't silently produce incorrect results
        @panic("NeighbourAtom buffer overflow - increase buffer size or use dynamic allocation");
    }

    const nb_extended = nb_radius + types.kRadiusWater;
    const dist_sq = atom.distanceSq(nb_pos);

    // Skip the same atom position
    if (dist_sq < 0.0001) return count;

    const dist = @sqrt(dist_sq);
    if (dist < probe + nb_extended) {
        buffer[count] = .{
            .location = nb_pos.sub(atom),
            .radius_sq = nb_extended * nb_extended,
            .distance = dist,
        };
        return count + 1;
    }
    return count;
}

/// Calculate surface accessibility for all residues using parallel threads.
/// n_threads: 0 = auto-detect (default), 1 = single-threaded, N = use N threads
pub fn calculateAccessibilitiesParallel(residues: []Residue, allocator: Allocator, n_threads_config: usize) !void {
    const n = residues.len;
    if (n == 0) return;

    const cpu_count = std.Thread.getCpuCount() catch 1;
    const requested_threads = if (n_threads_config == 0) cpu_count else n_threads_config;

    // Single-threaded for small proteins (threading overhead not worth it)
    if (n < 50 or requested_threads == 1) {
        return calculateAccessibilitiesOptimized(residues, allocator);
    }

    // Extract CA positions and completeness flags
    const ca_positions = try allocator.alloc(Vec3f32, n);
    defer allocator.free(ca_positions);
    const complete = try allocator.alloc(bool, n);
    defer allocator.free(complete);

    for (residues, 0..) |res, i| {
        ca_positions[i] = res.ca;
        complete[i] = res.complete;
    }

    // Build residue-level neighbor list
    const residue_neighbors = try neighbor_list_mod.buildResidueNeighborList(
        ca_positions,
        complete,
        12.0,
        allocator,
    );
    defer neighbor_list_mod.freeResidueNeighborList(residue_neighbors, allocator);

    // Pre-compute AABBs for all residues (avoids repeated calculation in hot loop)
    const residue_aabbs = try allocator.alloc(AABB, n);
    defer allocator.free(residue_aabbs);
    for (residues, 0..) |*res, i| {
        residue_aabbs[i] = AABB.fromResidue(res);
    }

    // Allocate results array
    const results = try allocator.alloc(f32, n);
    defer allocator.free(results);

    // Parallel execution via ThreadPool
    const n_threads = @min(requested_threads, cpu_count);
    const chunk_size = (n + n_threads - 1) / n_threads;

    const ctx = ParallelContext{
        .residues = residues,
        .residue_neighbors = residue_neighbors,
        .residue_aabbs = residue_aabbs,
        .results = results,
        .allocator = allocator,
    };

    // Worker writes directly to ctx.results; pool result type is unused
    const Unit = struct {};
    const work_fn = struct {
        fn call(c: ParallelContext, start: usize, end: usize) Unit {
            parallelWorker(c, start, end);
            return .{};
        }
    }.call;

    var pool = try thread_pool_mod.ThreadPool(ParallelContext, Unit).init(
        allocator, n_threads, work_fn, ctx, n, chunk_size,
    );
    defer pool.deinit();
    try pool.run();

    // Copy results back to residues
    for (residues, 0..) |*res, i| {
        res.accessibility = results[i];
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "SurfaceDots - correct count and unit sphere" {
    try std.testing.expectEqual(@as(usize, 401), surface_dots.points.len);

    // All points should be on unit sphere (length ≈ 1.0)
    for (surface_dots.points) |p| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), p.length(), 0.01);
    }

    // Weight should be 4π/401 ≈ 0.0313
    try std.testing.expectApproxEqAbs(@as(f32, 4.0 * math.pi / 401.0), surface_dots.weight, 0.001);
}

test "calculateAtomSurface - isolated atom" {
    // An atom with no neighbours should have full surface area = 4πr²
    const radius: f32 = types.kRadiusCA; // 1.87
    const total_radius = radius + types.kRadiusWater; // 1.87 + 1.4 = 3.27
    const expected = 4.0 * math.pi * total_radius * total_radius;

    const neighbours = &[_]NeighbourAtom{};

    const result = calculateAtomSurface(radius, neighbours);
    // Should be approximately 4πr² (within ~1% due to finite point count)
    try std.testing.expectApproxEqRel(expected, result, 0.02);
}

test "calculateAccessibilities - single isolated residue" {
    const allocator = std.testing.allocator;

    var residues = [_]Residue{
        .{
            .n = Vec3f32{ .x = 0.0, .y = 0.0, .z = 0.0 },
            .ca = Vec3f32{ .x = 1.5, .y = 0.0, .z = 0.0 },
            .c = Vec3f32{ .x = 2.5, .y = 0.0, .z = 0.0 },
            .o = Vec3f32{ .x = 2.5, .y = 1.2, .z = 0.0 },
            .complete = true,
        },
    };

    try calculateAccessibilities(&residues, allocator);
    // Single residue should have non-zero accessibility
    try std.testing.expect(residues[0].accessibility > 0.0);
}

test "calculateAtomSurfaceSimd - matches scalar version" {
    // Create test neighbors with various distances
    const neighbours = [_]NeighbourAtom{
        .{ .location = Vec3f32{ .x = 2.0, .y = 0.0, .z = 0.0 }, .radius_sq = 4.0, .distance = 2.0 },
        .{ .location = Vec3f32{ .x = 0.0, .y = 3.0, .z = 0.0 }, .radius_sq = 9.0, .distance = 3.0 },
        .{ .location = Vec3f32{ .x = 0.0, .y = 0.0, .z = 4.0 }, .radius_sq = 16.0, .distance = 4.0 },
        .{ .location = Vec3f32{ .x = 1.0, .y = 1.0, .z = 1.0 }, .radius_sq = 3.0, .distance = 1.732 },
        .{ .location = Vec3f32{ .x = -2.0, .y = 0.0, .z = 0.0 }, .radius_sq = 4.0, .distance = 2.0 },
        .{ .location = Vec3f32{ .x = 0.0, .y = -3.0, .z = 0.0 }, .radius_sq = 9.0, .distance = 3.0 },
        .{ .location = Vec3f32{ .x = 0.0, .y = 0.0, .z = -4.0 }, .radius_sq = 16.0, .distance = 4.0 },
        .{ .location = Vec3f32{ .x = -1.0, .y = -1.0, .z = -1.0 }, .radius_sq = 3.0, .distance = 1.732 },
        .{ .location = Vec3f32{ .x = 5.0, .y = 0.0, .z = 0.0 }, .radius_sq = 25.0, .distance = 5.0 },
    };

    const radius: f32 = types.kRadiusCA;

    const scalar_result = calculateAtomSurface(radius, &neighbours);
    const simd_result = calculateAtomSurfaceSimd(radius, &neighbours);

    // SIMD version should produce nearly identical results (within floating-point precision)
    try std.testing.expectApproxEqRel(scalar_result, simd_result, 0.0001);
}

test "calculateAtomSurfaceSimd - isolated atom" {
    // An atom with no neighbours should have full surface area = 4πr²
    const radius: f32 = types.kRadiusCA; // 1.87
    const total_radius = radius + types.kRadiusWater; // 1.87 + 1.4 = 3.27
    const expected = 4.0 * math.pi * total_radius * total_radius;

    const neighbours = &[_]NeighbourAtom{};

    const result = calculateAtomSurfaceSimd(radius, neighbours);
    // Should be approximately 4πr² (within ~1% due to finite point count)
    try std.testing.expectApproxEqRel(expected, result, 0.02);
}

test "calculateAtomSurfaceSimd - small batch (< 8 neighbors)" {
    // Test with fewer than 8 neighbors to verify remainder handling
    const neighbours = [_]NeighbourAtom{
        .{ .location = Vec3f32{ .x = 2.0, .y = 0.0, .z = 0.0 }, .radius_sq = 4.0, .distance = 2.0 },
        .{ .location = Vec3f32{ .x = 0.0, .y = 3.0, .z = 0.0 }, .radius_sq = 9.0, .distance = 3.0 },
        .{ .location = Vec3f32{ .x = 0.0, .y = 0.0, .z = 4.0 }, .radius_sq = 16.0, .distance = 4.0 },
    };

    const radius: f32 = types.kRadiusCA;

    const scalar_result = calculateAtomSurface(radius, &neighbours);
    const simd_result = calculateAtomSurfaceSimd(radius, &neighbours);

    try std.testing.expectApproxEqRel(scalar_result, simd_result, 0.0001);
}
