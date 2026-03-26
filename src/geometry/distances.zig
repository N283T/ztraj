//! Pairwise atom distance calculation.

const std = @import("std");
const simd = @import("../simd.zig");
const thread_pool = @import("../thread_pool.zig");

const vec_len = simd.optimal_vector_width.f64_width;

/// Compute Euclidean distances for given atom index pairs.
///
/// For each pair (i, j) in `pairs`, computes:
///   result[k] = sqrt((x[j]-x[i])^2 + (y[j]-y[i])^2 + (z[j]-z[i])^2)
///
/// Uses f64 intermediate precision to minimize floating-point error.
/// SIMD vectorized: processes `vec_len` pairs at a time with scalar-to-vector packing.
/// `result` must have the same length as `pairs`.
pub fn compute(
    x: []const f32,
    y: []const f32,
    z: []const f32,
    pairs: []const [2]u32,
    result: []f32,
) void {
    std.debug.assert(result.len == pairs.len);

    const V = vec_len;
    var k: usize = 0;

    // SIMD main loop: process V pairs at a time
    while (k + V <= pairs.len) : (k += V) {
        var dx_arr: [V]f64 = undefined;
        var dy_arr: [V]f64 = undefined;
        var dz_arr: [V]f64 = undefined;
        inline for (0..V) |vi| {
            const pair = pairs[k + vi];
            dx_arr[vi] = @as(f64, x[pair[1]]) - @as(f64, x[pair[0]]);
            dy_arr[vi] = @as(f64, y[pair[1]]) - @as(f64, y[pair[0]]);
            dz_arr[vi] = @as(f64, z[pair[1]]) - @as(f64, z[pair[0]]);
        }
        const vdx: @Vector(V, f64) = dx_arr;
        const vdy: @Vector(V, f64) = dy_arr;
        const vdz: @Vector(V, f64) = dz_arr;
        const dist_sq = vdx * vdx + vdy * vdy + vdz * vdz;
        const dist = @sqrt(dist_sq);
        const dist_f32: @Vector(V, f32) = @floatCast(dist);
        result[k..][0..V].* = dist_f32;
    }

    // Scalar tail for remainder
    while (k < pairs.len) : (k += 1) {
        const pair = pairs[k];
        const i = pair[0];
        const j = pair[1];
        const dx: f64 = @as(f64, x[j]) - @as(f64, x[i]);
        const dy: f64 = @as(f64, y[j]) - @as(f64, y[i]);
        const dz: f64 = @as(f64, z[j]) - @as(f64, z[i]);
        result[k] = @floatCast(@sqrt(dx * dx + dy * dy + dz * dz));
    }
}

/// Compute Euclidean distances in parallel across multiple threads.
///
/// Falls back to single-threaded `compute` when `n_threads <= 1` or
/// `pairs.len < 256`. Each thread writes to a non-overlapping slice of
/// `result`, so no synchronization is needed.
pub fn computeParallel(
    allocator: std.mem.Allocator,
    x: []const f32,
    y: []const f32,
    z: []const f32,
    pairs: []const [2]u32,
    result: []f32,
    n_threads: usize,
) !void {
    std.debug.assert(result.len == pairs.len);

    // Fall back to single-threaded for small workloads or single thread
    if (n_threads <= 1 or pairs.len < 256) {
        compute(x, y, z, pairs, result);
        return;
    }

    const Context = struct {
        x: []const f32,
        y: []const f32,
        z: []const f32,
        pairs: []const [2]u32,
        result: []f32,
    };

    const work_fn = struct {
        fn call(ctx: Context, chunk_start: usize, chunk_end: usize) u0 {
            compute(
                ctx.x,
                ctx.y,
                ctx.z,
                ctx.pairs[chunk_start..chunk_end],
                ctx.result[chunk_start..chunk_end],
            );
            return 0;
        }
    }.call;

    const reduce_fn = struct {
        fn call(_: []const u0) u0 {
            return 0;
        }
    }.call;

    const chunk_size = @max(256, pairs.len / (n_threads * 4));

    _ = try thread_pool.parallelFor(
        Context,
        u0,
        allocator,
        n_threads,
        work_fn,
        Context{
            .x = x,
            .y = y,
            .z = z,
            .pairs = pairs,
            .result = result,
        },
        pairs.len,
        chunk_size,
        reduce_fn,
    );
}

// ============================================================================
// Tests
// ============================================================================

test "distances: 3-4-5 right triangle" {
    // atoms at (0,0,0) and (3,4,0) -> distance = 5.0
    const x = [_]f32{ 0.0, 3.0 };
    const y = [_]f32{ 0.0, 4.0 };
    const z = [_]f32{ 0.0, 0.0 };
    const pairs = [_][2]u32{.{ 0, 1 }};
    var result = [_]f32{0.0};

    compute(&x, &y, &z, &pairs, &result);

    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result[0], 1e-5);
}

test "distances: coincident atoms" {
    const x = [_]f32{ 1.5, 1.5 };
    const y = [_]f32{ 2.3, 2.3 };
    const z = [_]f32{ -0.7, -0.7 };
    const pairs = [_][2]u32{.{ 0, 1 }};
    var result = [_]f32{0.0};

    compute(&x, &y, &z, &pairs, &result);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 1e-6);
}

test "distances: multiple pairs" {
    // 3 atoms: origin, (1,0,0), (0,2,0)
    const x = [_]f32{ 0.0, 1.0, 0.0 };
    const y = [_]f32{ 0.0, 0.0, 2.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0 };

    const pairs = [_][2]u32{
        .{ 0, 1 }, // distance = 1.0
        .{ 0, 2 }, // distance = 2.0
        .{ 1, 2 }, // distance = sqrt(1+4) = sqrt(5)
    };
    var result = [_]f32{ 0.0, 0.0, 0.0 };

    compute(&x, &y, &z, &pairs, &result);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, @sqrt(5.0)), result[2], 1e-5);
}

test "distances: axis-aligned 3D" {
    // atoms at (0,0,0) and (0,0,7) -> distance = 7.0
    const x = [_]f32{ 0.0, 0.0 };
    const y = [_]f32{ 0.0, 0.0 };
    const z = [_]f32{ 0.0, 7.0 };
    const pairs = [_][2]u32{.{ 0, 1 }};
    var result = [_]f32{0.0};

    compute(&x, &y, &z, &pairs, &result);

    try std.testing.expectApproxEqAbs(@as(f32, 7.0), result[0], 1e-5);
}

test "distances: zero pairs is no-op" {
    const x = [_]f32{1.0};
    const y = [_]f32{1.0};
    const z = [_]f32{1.0};
    const pairs = [_][2]u32{};
    var result = [_]f32{};

    compute(&x, &y, &z, &pairs, &result); // must not crash
}

test "distances: large batch exercises SIMD path" {
    // 50 atoms along x-axis at positions 0,1,2,...,49
    // 49 consecutive pairs: (0,1),(1,2),...,(48,49) — odd count to test scalar tail
    const n_atoms = 50;
    const n_pairs = 49;

    var x: [n_atoms]f32 = undefined;
    var y: [n_atoms]f32 = undefined;
    var z: [n_atoms]f32 = undefined;
    for (0..n_atoms) |i| {
        x[i] = @floatFromInt(i);
        y[i] = 0.0;
        z[i] = 0.0;
    }

    var pairs: [n_pairs][2]u32 = undefined;
    for (0..n_pairs) |i| {
        pairs[i] = .{ @intCast(i), @intCast(i + 1) };
    }

    var result: [n_pairs]f32 = undefined;
    compute(&x, &y, &z, &pairs, &result);

    // Each consecutive pair is 1.0 apart along x-axis
    for (0..n_pairs) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[i], 1e-5);
    }
}

test "distances: multi-threaded matches single-threaded" {
    const allocator = std.testing.allocator;

    // Generate enough pairs to exceed the 256 threshold
    const n_atoms = 100;
    const n_pairs = 512;

    var x: [n_atoms]f32 = undefined;
    var y: [n_atoms]f32 = undefined;
    var z: [n_atoms]f32 = undefined;
    for (0..n_atoms) |i| {
        const fi: f32 = @floatFromInt(i);
        x[i] = fi * 1.1;
        y[i] = fi * 0.7;
        z[i] = fi * 0.3;
    }

    // Create pairs cycling through atom indices
    var pairs: [n_pairs][2]u32 = undefined;
    for (0..n_pairs) |i| {
        pairs[i] = .{
            @intCast(i % n_atoms),
            @intCast((i + 1) % n_atoms),
        };
    }

    // Single-threaded reference
    var result_st: [n_pairs]f32 = undefined;
    compute(&x, &y, &z, &pairs, &result_st);

    // Multi-threaded (4 threads)
    var result_mt: [n_pairs]f32 = undefined;
    try computeParallel(allocator, &x, &y, &z, &pairs, &result_mt, 4);

    // Results must be identical (same algorithm, same floating-point ops)
    for (0..n_pairs) |i| {
        try std.testing.expectEqual(result_st[i], result_mt[i]);
    }
}
