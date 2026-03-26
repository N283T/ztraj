//! Structural superposition using the QCP algorithm.
//!
//! Computes the optimal rotation + translation to align a mobile structure
//! onto a reference structure, minimizing RMSD. Returns the transformed
//! coordinates (not in-place — creates new coordinate arrays).
//!
//! The rotation quaternion is obtained from the 4x4 matrix K (Theobald 2005)
//! after finding the largest eigenvalue via Newton-Raphson (same as rmsd.zig).

const std = @import("std");
const simd_mod = @import("../simd.zig");

const vec_len = simd_mod.optimal_vector_width.f64_width;

/// Superposition result: aligned coordinates and RMSD.
pub const SuperposeResult = struct {
    /// Aligned coordinates in SOA layout (same length as input).
    x: []f32,
    y: []f32,
    z: []f32,
    /// RMSD after alignment.
    rmsd: f64,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *SuperposeResult) void {
        self.allocator.free(self.x);
        self.allocator.free(self.y);
        self.allocator.free(self.z);
    }
};

/// Align `mobile` coordinates onto `reference` by optimal rotation + translation.
///
/// If `atom_indices` is non-null, the rotation is computed using only those atoms,
/// but the transformation is applied to ALL atoms. This is common for backbone-only
/// fitting with full structure output.
///
/// Returns new coordinate arrays — the input is not modified.
pub fn superpose(
    allocator: std.mem.Allocator,
    ref_x: []const f32,
    ref_y: []const f32,
    ref_z: []const f32,
    mob_x: []const f32,
    mob_y: []const f32,
    mob_z: []const f32,
    atom_indices: ?[]const u32,
) !SuperposeResult {
    const n_all = mob_x.len;
    const n_fit: usize = if (atom_indices) |idx| idx.len else n_all;
    if (n_fit == 0) {
        // No atoms to fit — return copy of mobile
        const out_x = try allocator.alloc(f32, n_all);
        errdefer allocator.free(out_x);
        const out_y = try allocator.alloc(f32, n_all);
        errdefer allocator.free(out_y);
        const out_z = try allocator.alloc(f32, n_all);
        errdefer allocator.free(out_z);
        @memcpy(out_x, mob_x);
        @memcpy(out_y, mob_y);
        @memcpy(out_z, mob_z);
        return .{ .x = out_x, .y = out_y, .z = out_z, .rmsd = 0.0, .allocator = allocator };
    }

    const n_f: f64 = @floatFromInt(n_fit);

    // Step 1: Compute centers of the fitting atoms
    var ref_cx: f64 = 0.0;
    var ref_cy: f64 = 0.0;
    var ref_cz: f64 = 0.0;
    var mob_cx: f64 = 0.0;
    var mob_cy: f64 = 0.0;
    var mob_cz: f64 = 0.0;

    if (atom_indices) |indices| {
        for (indices) |idx| {
            ref_cx += @as(f64, ref_x[idx]);
            ref_cy += @as(f64, ref_y[idx]);
            ref_cz += @as(f64, ref_z[idx]);
            mob_cx += @as(f64, mob_x[idx]);
            mob_cy += @as(f64, mob_y[idx]);
            mob_cz += @as(f64, mob_z[idx]);
        }
    } else {
        const V = vec_len;
        var v_ref_cx: @Vector(V, f64) = @splat(0.0);
        var v_ref_cy: @Vector(V, f64) = @splat(0.0);
        var v_ref_cz: @Vector(V, f64) = @splat(0.0);
        var v_mob_cx: @Vector(V, f64) = @splat(0.0);
        var v_mob_cy: @Vector(V, f64) = @splat(0.0);
        var v_mob_cz: @Vector(V, f64) = @splat(0.0);

        var i: usize = 0;
        while (i + V <= n_all) : (i += V) {
            const vrx: @Vector(V, f32) = ref_x[i..][0..V].*;
            const vry: @Vector(V, f32) = ref_y[i..][0..V].*;
            const vrz: @Vector(V, f32) = ref_z[i..][0..V].*;
            const vmx: @Vector(V, f32) = mob_x[i..][0..V].*;
            const vmy: @Vector(V, f32) = mob_y[i..][0..V].*;
            const vmz: @Vector(V, f32) = mob_z[i..][0..V].*;

            v_ref_cx += @as(@Vector(V, f64), @floatCast(vrx));
            v_ref_cy += @as(@Vector(V, f64), @floatCast(vry));
            v_ref_cz += @as(@Vector(V, f64), @floatCast(vrz));
            v_mob_cx += @as(@Vector(V, f64), @floatCast(vmx));
            v_mob_cy += @as(@Vector(V, f64), @floatCast(vmy));
            v_mob_cz += @as(@Vector(V, f64), @floatCast(vmz));
        }

        ref_cx = @reduce(.Add, v_ref_cx);
        ref_cy = @reduce(.Add, v_ref_cy);
        ref_cz = @reduce(.Add, v_ref_cz);
        mob_cx = @reduce(.Add, v_mob_cx);
        mob_cy = @reduce(.Add, v_mob_cy);
        mob_cz = @reduce(.Add, v_mob_cz);

        // Scalar tail
        while (i < n_all) : (i += 1) {
            ref_cx += @as(f64, ref_x[i]);
            ref_cy += @as(f64, ref_y[i]);
            ref_cz += @as(f64, ref_z[i]);
            mob_cx += @as(f64, mob_x[i]);
            mob_cy += @as(f64, mob_y[i]);
            mob_cz += @as(f64, mob_z[i]);
        }
    }
    ref_cx /= n_f;
    ref_cy /= n_f;
    ref_cz /= n_f;
    mob_cx /= n_f;
    mob_cy /= n_f;
    mob_cz /= n_f;

    // Step 2: Build inner product matrix S and G values
    var sxx: f64 = 0.0;
    var sxy: f64 = 0.0;
    var sxz: f64 = 0.0;
    var syx: f64 = 0.0;
    var syy: f64 = 0.0;
    var syz: f64 = 0.0;
    var szx: f64 = 0.0;
    var szy: f64 = 0.0;
    var szz: f64 = 0.0;
    var g_ref: f64 = 0.0;
    var g_mob: f64 = 0.0;

    // S = Ref^T * Mobile (same convention as rmsd.zig)
    if (atom_indices) |indices| {
        for (indices) |idx| {
            const rx = @as(f64, ref_x[idx]) - ref_cx;
            const ry = @as(f64, ref_y[idx]) - ref_cy;
            const rz = @as(f64, ref_z[idx]) - ref_cz;
            const mx = @as(f64, mob_x[idx]) - mob_cx;
            const my = @as(f64, mob_y[idx]) - mob_cy;
            const mz = @as(f64, mob_z[idx]) - mob_cz;
            g_ref += rx * rx + ry * ry + rz * rz;
            g_mob += mx * mx + my * my + mz * mz;
            sxx += rx * mx;
            sxy += rx * my;
            sxz += rx * mz;
            syx += ry * mx;
            syy += ry * my;
            syz += ry * mz;
            szx += rz * mx;
            szy += rz * my;
            szz += rz * mz;
        }
    } else {
        const V = vec_len;
        // Broadcast center values
        const splat_ref_cx: @Vector(V, f64) = @splat(ref_cx);
        const splat_ref_cy: @Vector(V, f64) = @splat(ref_cy);
        const splat_ref_cz: @Vector(V, f64) = @splat(ref_cz);
        const splat_mob_cx: @Vector(V, f64) = @splat(mob_cx);
        const splat_mob_cy: @Vector(V, f64) = @splat(mob_cy);
        const splat_mob_cz: @Vector(V, f64) = @splat(mob_cz);

        // 11 SIMD accumulators: 9 S-matrix + 2 G values
        var v_sxx: @Vector(V, f64) = @splat(0.0);
        var v_sxy: @Vector(V, f64) = @splat(0.0);
        var v_sxz: @Vector(V, f64) = @splat(0.0);
        var v_syx: @Vector(V, f64) = @splat(0.0);
        var v_syy: @Vector(V, f64) = @splat(0.0);
        var v_syz: @Vector(V, f64) = @splat(0.0);
        var v_szx: @Vector(V, f64) = @splat(0.0);
        var v_szy: @Vector(V, f64) = @splat(0.0);
        var v_szz: @Vector(V, f64) = @splat(0.0);
        var v_g_ref: @Vector(V, f64) = @splat(0.0);
        var v_g_mob: @Vector(V, f64) = @splat(0.0);

        var i: usize = 0;
        while (i + V <= n_all) : (i += V) {
            // Load f32 coords and widen to f64
            const vrx: @Vector(V, f64) = @as(@Vector(V, f64), @floatCast(@as(@Vector(V, f32), ref_x[i..][0..V].*))) - splat_ref_cx;
            const vry: @Vector(V, f64) = @as(@Vector(V, f64), @floatCast(@as(@Vector(V, f32), ref_y[i..][0..V].*))) - splat_ref_cy;
            const vrz: @Vector(V, f64) = @as(@Vector(V, f64), @floatCast(@as(@Vector(V, f32), ref_z[i..][0..V].*))) - splat_ref_cz;
            const vmx: @Vector(V, f64) = @as(@Vector(V, f64), @floatCast(@as(@Vector(V, f32), mob_x[i..][0..V].*))) - splat_mob_cx;
            const vmy: @Vector(V, f64) = @as(@Vector(V, f64), @floatCast(@as(@Vector(V, f32), mob_y[i..][0..V].*))) - splat_mob_cy;
            const vmz: @Vector(V, f64) = @as(@Vector(V, f64), @floatCast(@as(@Vector(V, f32), mob_z[i..][0..V].*))) - splat_mob_cz;

            // G values
            v_g_ref += vrx * vrx + vry * vry + vrz * vrz;
            v_g_mob += vmx * vmx + vmy * vmy + vmz * vmz;

            // S-matrix cross products
            v_sxx += vrx * vmx;
            v_sxy += vrx * vmy;
            v_sxz += vrx * vmz;
            v_syx += vry * vmx;
            v_syy += vry * vmy;
            v_syz += vry * vmz;
            v_szx += vrz * vmx;
            v_szy += vrz * vmy;
            v_szz += vrz * vmz;
        }

        // Reduce all 11 accumulators
        sxx = @reduce(.Add, v_sxx);
        sxy = @reduce(.Add, v_sxy);
        sxz = @reduce(.Add, v_sxz);
        syx = @reduce(.Add, v_syx);
        syy = @reduce(.Add, v_syy);
        syz = @reduce(.Add, v_syz);
        szx = @reduce(.Add, v_szx);
        szy = @reduce(.Add, v_szy);
        szz = @reduce(.Add, v_szz);
        g_ref = @reduce(.Add, v_g_ref);
        g_mob = @reduce(.Add, v_g_mob);

        // Scalar tail
        while (i < n_all) : (i += 1) {
            const rx = @as(f64, ref_x[i]) - ref_cx;
            const ry = @as(f64, ref_y[i]) - ref_cy;
            const rz = @as(f64, ref_z[i]) - ref_cz;
            const mx = @as(f64, mob_x[i]) - mob_cx;
            const my = @as(f64, mob_y[i]) - mob_cy;
            const mz = @as(f64, mob_z[i]) - mob_cz;

            g_ref += rx * rx + ry * ry + rz * rz;
            g_mob += mx * mx + my * my + mz * mz;

            sxx += rx * mx;
            sxy += rx * my;
            sxz += rx * mz;
            syx += ry * mx;
            syy += ry * my;
            syz += ry * mz;
            szx += rz * mx;
            szy += rz * my;
            szz += rz * mz;
        }
    }

    // Step 3: Build K matrix and find largest eigenvalue
    const k00 = sxx + syy + szz;
    const k01 = syz - szy;
    const k02 = szx - sxz;
    const k03 = sxy - syx;
    const k11 = sxx - syy - szz;
    const k12 = sxy + syx;
    const k13 = szx + sxz;
    const k22 = -sxx + syy - szz;
    const k23 = syz + szy;
    const k33 = -sxx - syy + szz;

    // Newton-Raphson for largest eigenvalue (same as rmsd.zig)
    const sxx_2 = sxx * sxx;
    const syy_2 = syy * syy;
    const szz_2 = szz * szz;
    const sxy_2 = sxy * sxy;
    const syz_2 = syz * syz;
    const sxz_2 = sxz * sxz;
    const syx_2 = syx * syx;
    const szy_2 = szy * szy;
    const szx_2 = szx * szx;

    const c2 = -2.0 * (sxx_2 + syy_2 + szz_2 + sxy_2 + syz_2 + sxz_2 + syx_2 + szy_2 + szx_2);
    const det_s = sxx * (syy * szz - syz * szy) - sxy * (syx * szz - syz * szx) + sxz * (syx * szy - syy * szx);
    const c1 = -8.0 * det_s;
    const c0 = k00 * (k11 * (k22 * k33 - k23 * k23) - k12 * (k12 * k33 - k23 * k13) + k13 * (k12 * k23 - k22 * k13)) -
        k01 * (k01 * (k22 * k33 - k23 * k23) - k12 * (k02 * k33 - k23 * k03) + k13 * (k02 * k23 - k22 * k03)) +
        k02 * (k01 * (k12 * k33 - k23 * k13) - k11 * (k02 * k33 - k23 * k03) + k13 * (k02 * k13 - k12 * k03)) -
        k03 * (k01 * (k12 * k23 - k22 * k13) - k11 * (k02 * k23 - k22 * k03) + k12 * (k02 * k13 - k12 * k03));

    const g = (g_ref + g_mob) / 2.0;
    var lam = g;
    for (0..50) |_| {
        const lam2 = lam * lam;
        const lam3 = lam2 * lam;
        const lam4 = lam3 * lam;
        const p = lam4 + c2 * lam2 + c1 * lam + c0;
        const dp = 4.0 * lam3 + 2.0 * c2 * lam + c1;
        const delta = p / dp;
        lam -= delta;
        if (@abs(delta) < 1e-12) break;
    }

    const rmsd_inner = (g_ref + g_mob - 2.0 * lam) / n_f;
    const rmsd = @sqrt(@max(0.0, rmsd_inner));

    // Step 4: Find quaternion (eigenvector of K for eigenvalue lam)
    // Solve (K - lam*I) * q = 0 using adjugate method
    // Build the shifted matrix
    const a11 = k11 - lam;
    const a22 = k22 - lam;
    const a33 = k33 - lam;

    // Compute cofactors of first row to get quaternion
    // q0 = cofactor(0,0), q1 = -cofactor(0,1), q2 = cofactor(0,2), q3 = -cofactor(0,3)
    const q0 = a11 * (a22 * a33 - k23 * k23) - k12 * (k12 * a33 - k23 * k13) + k13 * (k12 * k23 - a22 * k13);
    const q1 = -(k01 * (a22 * a33 - k23 * k23) - k12 * (k02 * a33 - k23 * k03) + k13 * (k02 * k23 - a22 * k03));
    const q2 = k01 * (k12 * a33 - k23 * k13) - a11 * (k02 * a33 - k23 * k03) + k13 * (k02 * k13 - k12 * k03);
    const q3 = -(k01 * (k12 * k23 - a22 * k13) - a11 * (k02 * k23 - a22 * k03) + k12 * (k02 * k13 - k12 * k03));

    // Normalize quaternion
    const qlen = @sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    const nq0 = if (qlen > 1e-15) q0 / qlen else 1.0;
    const nq1 = if (qlen > 1e-15) q1 / qlen else 0.0;
    const nq2 = if (qlen > 1e-15) q2 / qlen else 0.0;
    const nq3 = if (qlen > 1e-15) q3 / qlen else 0.0;

    // Step 5: Convert quaternion to 3x3 rotation matrix
    const r00 = 1.0 - 2.0 * (nq2 * nq2 + nq3 * nq3);
    const r01 = 2.0 * (nq1 * nq2 - nq0 * nq3);
    const r02 = 2.0 * (nq1 * nq3 + nq0 * nq2);
    const r10 = 2.0 * (nq1 * nq2 + nq0 * nq3);
    const r11 = 1.0 - 2.0 * (nq1 * nq1 + nq3 * nq3);
    const r12 = 2.0 * (nq2 * nq3 - nq0 * nq1);
    const r20 = 2.0 * (nq1 * nq3 - nq0 * nq2);
    const r21 = 2.0 * (nq2 * nq3 + nq0 * nq1);
    const r22 = 1.0 - 2.0 * (nq1 * nq1 + nq2 * nq2);

    // Step 6: Apply rotation to ALL mobile atoms (not just fitting atoms)
    // Transform: x_aligned = R * (x_mobile - center_mobile) + center_ref
    const out_x = try allocator.alloc(f32, n_all);
    errdefer allocator.free(out_x);
    const out_y = try allocator.alloc(f32, n_all);
    errdefer allocator.free(out_y);
    const out_z = try allocator.alloc(f32, n_all);
    errdefer allocator.free(out_z);

    // SIMD rotation application
    {
        const V = vec_len;
        // Broadcast rotation matrix elements and centers
        const s_r00: @Vector(V, f64) = @splat(r00);
        const s_r01: @Vector(V, f64) = @splat(r01);
        const s_r02: @Vector(V, f64) = @splat(r02);
        const s_r10: @Vector(V, f64) = @splat(r10);
        const s_r11: @Vector(V, f64) = @splat(r11);
        const s_r12: @Vector(V, f64) = @splat(r12);
        const s_r20: @Vector(V, f64) = @splat(r20);
        const s_r21: @Vector(V, f64) = @splat(r21);
        const s_r22: @Vector(V, f64) = @splat(r22);
        const s_mob_cx: @Vector(V, f64) = @splat(mob_cx);
        const s_mob_cy: @Vector(V, f64) = @splat(mob_cy);
        const s_mob_cz: @Vector(V, f64) = @splat(mob_cz);
        const s_ref_cx: @Vector(V, f64) = @splat(ref_cx);
        const s_ref_cy: @Vector(V, f64) = @splat(ref_cy);
        const s_ref_cz: @Vector(V, f64) = @splat(ref_cz);

        var i: usize = 0;
        while (i + V <= n_all) : (i += V) {
            // Load f32 mob coords, widen to f64, subtract mob center
            const cx: @Vector(V, f64) = @as(@Vector(V, f64), @floatCast(@as(@Vector(V, f32), mob_x[i..][0..V].*))) - s_mob_cx;
            const cy: @Vector(V, f64) = @as(@Vector(V, f64), @floatCast(@as(@Vector(V, f32), mob_y[i..][0..V].*))) - s_mob_cy;
            const cz: @Vector(V, f64) = @as(@Vector(V, f64), @floatCast(@as(@Vector(V, f32), mob_z[i..][0..V].*))) - s_mob_cz;

            // 3 multiply-add chains for rotation + translation
            const nx = s_r00 * cx + s_r01 * cy + s_r02 * cz + s_ref_cx;
            const ny = s_r10 * cx + s_r11 * cy + s_r12 * cz + s_ref_cy;
            const nz = s_r20 * cx + s_r21 * cy + s_r22 * cz + s_ref_cz;

            // Narrow f64→f32 and store
            out_x[i..][0..V].* = @floatCast(nx);
            out_y[i..][0..V].* = @floatCast(ny);
            out_z[i..][0..V].* = @floatCast(nz);
        }

        // Scalar tail
        while (i < n_all) : (i += 1) {
            const cx = @as(f64, mob_x[i]) - mob_cx;
            const cy = @as(f64, mob_y[i]) - mob_cy;
            const cz = @as(f64, mob_z[i]) - mob_cz;

            out_x[i] = @floatCast(r00 * cx + r01 * cy + r02 * cz + ref_cx);
            out_y[i] = @floatCast(r10 * cx + r11 * cy + r12 * cz + ref_cy);
            out_z[i] = @floatCast(r20 * cx + r21 * cy + r22 * cz + ref_cz);
        }
    }

    return .{ .x = out_x, .y = out_y, .z = out_z, .rmsd = rmsd, .allocator = allocator };
}

// ============================================================================
// Tests
// ============================================================================

test "superpose: identical structures → RMSD 0, coords unchanged" {
    const allocator = std.testing.allocator;
    const x = [_]f32{ 1.0, 2.0, 3.0 };
    const y = [_]f32{ 0.0, 1.0, 0.0 };
    const z = [_]f32{ 0.5, 0.5, 1.5 };

    var result = try superpose(allocator, &x, &y, &z, &x, &y, &z, null);
    defer result.deinit();

    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result.rmsd, 1e-8);
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(x[i], result.x[i], 1e-5);
        try std.testing.expectApproxEqAbs(y[i], result.y[i], 1e-5);
        try std.testing.expectApproxEqAbs(z[i], result.z[i], 1e-5);
    }
}

test "superpose: pure translation → aligned to reference center" {
    const allocator = std.testing.allocator;
    const ref_x = [_]f32{ 0.0, 1.0, 0.0 };
    const ref_y = [_]f32{ 0.0, 0.0, 1.0 };
    const ref_z = [_]f32{ 0.0, 0.0, 0.0 };

    // Translated by (10, 20, 30)
    const mob_x = [_]f32{ 10.0, 11.0, 10.0 };
    const mob_y = [_]f32{ 20.0, 20.0, 21.0 };
    const mob_z = [_]f32{ 30.0, 30.0, 30.0 };

    var result = try superpose(allocator, &ref_x, &ref_y, &ref_z, &mob_x, &mob_y, &mob_z, null);
    defer result.deinit();

    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result.rmsd, 1e-6);
    // After alignment, coords should match reference
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(ref_x[i], result.x[i], 1e-4);
        try std.testing.expectApproxEqAbs(ref_y[i], result.y[i], 1e-4);
        try std.testing.expectApproxEqAbs(ref_z[i], result.z[i], 1e-4);
    }
}

test "superpose: with atom_indices — fit subset, transform all" {
    const allocator = std.testing.allocator;
    // 4 atoms, fit on first 3 only
    const ref_x = [_]f32{ 0.0, 1.0, 0.0, 5.0 };
    const ref_y = [_]f32{ 0.0, 0.0, 1.0, 5.0 };
    const ref_z = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    // Mobile: same triangle shifted, 4th atom at different position
    const mob_x = [_]f32{ 10.0, 11.0, 10.0, 15.0 };
    const mob_y = [_]f32{ 20.0, 20.0, 21.0, 25.0 };
    const mob_z = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    const indices = [_]u32{ 0, 1, 2 };

    var result = try superpose(allocator, &ref_x, &ref_y, &ref_z, &mob_x, &mob_y, &mob_z, &indices);
    defer result.deinit();

    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result.rmsd, 1e-6);
    // First 3 atoms should be aligned to reference
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(ref_x[i], result.x[i], 1e-4);
        try std.testing.expectApproxEqAbs(ref_y[i], result.y[i], 1e-4);
    }
    // 4th atom should also be transformed (not same as ref)
    try std.testing.expect(result.x.len == 4);
}

test "superpose: 90-degree rotation around z-axis" {
    const allocator = std.testing.allocator;
    // Reference: centered triangle in xy-plane
    const ref_x = [_]f32{ 1.0, -1.0, 0.0 };
    const ref_y = [_]f32{ 0.0, 0.0, 1.0 };
    const ref_z = [_]f32{ 0.0, 0.0, 0.0 };
    // Mobile: same triangle rotated 90° around z (x→y, y→-x)
    const mob_x = [_]f32{ 0.0, 0.0, -1.0 };
    const mob_y = [_]f32{ 1.0, -1.0, 0.0 };
    const mob_z = [_]f32{ 0.0, 0.0, 0.0 };

    var result = try superpose(allocator, &ref_x, &ref_y, &ref_z, &mob_x, &mob_y, &mob_z, null);
    defer result.deinit();

    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result.rmsd, 1e-5);
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(ref_x[i], result.x[i], 1e-3);
        try std.testing.expectApproxEqAbs(ref_y[i], result.y[i], 1e-3);
        try std.testing.expectApproxEqAbs(ref_z[i], result.z[i], 1e-3);
    }
}

test "superpose: large array exercises SIMD rotation path" {
    const allocator = std.testing.allocator;
    const N = 100;
    var ref_x_arr: [N]f32 = undefined;
    var ref_y_arr: [N]f32 = undefined;
    var ref_z_arr: [N]f32 = undefined;
    var mob_x_arr: [N]f32 = undefined;
    var mob_y_arr: [N]f32 = undefined;
    var mob_z_arr: [N]f32 = undefined;

    for (0..N) |j| {
        const fj: f32 = @floatFromInt(j);
        // Reference: helix-like pattern
        ref_x_arr[j] = fj * 1.5;
        ref_y_arr[j] = @sin(fj * 0.1) * 5.0;
        ref_z_arr[j] = @cos(fj * 0.1) * 5.0;
        // Mobile: same but translated by (10, -5, 3)
        mob_x_arr[j] = ref_x_arr[j] + 10.0;
        mob_y_arr[j] = ref_y_arr[j] - 5.0;
        mob_z_arr[j] = ref_z_arr[j] + 3.0;
    }

    var result = try superpose(allocator, &ref_x_arr, &ref_y_arr, &ref_z_arr, &mob_x_arr, &mob_y_arr, &mob_z_arr, null);
    defer result.deinit();

    // Pure translation => RMSD should be 0
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result.rmsd, 1e-6);

    // Aligned coords should match reference
    for (0..N) |j| {
        try std.testing.expectApproxEqAbs(ref_x_arr[j], result.x[j], 1e-3);
        try std.testing.expectApproxEqAbs(ref_y_arr[j], result.y[j], 1e-3);
        try std.testing.expectApproxEqAbs(ref_z_arr[j], result.z[j], 1e-3);
    }
}
