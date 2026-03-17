//! RMSD via the Quaternion Characteristic Polynomial (QCP) algorithm.
//! Reference: Theobald, D.L. (2005) Acta Crystallographica A61:478-480.

const std = @import("std");

/// Compute the minimum RMSD between two structures using the QCP algorithm.
///
/// The algorithm internally centers both structures and finds the optimal
/// rotation that minimises the RMSD, so a pure translation between the two
/// structures yields RMSD = 0.
///
/// If `atom_indices` is non-null, only the listed atom indices are used.
/// Otherwise all atoms (0..ref_x.len) are used.
/// All arithmetic is performed in f64.
pub fn compute(
    ref_x: []const f32,
    ref_y: []const f32,
    ref_z: []const f32,
    x: []const f32,
    y: []const f32,
    z: []const f32,
    atom_indices: ?[]const u32,
) f64 {
    // Collect the atoms we care about
    const n: f64 = if (atom_indices) |idx| @floatFromInt(idx.len) else @floatFromInt(ref_x.len);
    if (n == 0.0) return 0.0;

    // Step 1: Compute centres
    var ref_cx: f64 = 0.0;
    var ref_cy: f64 = 0.0;
    var ref_cz: f64 = 0.0;
    var tgt_cx: f64 = 0.0;
    var tgt_cy: f64 = 0.0;
    var tgt_cz: f64 = 0.0;

    if (atom_indices) |indices| {
        for (indices) |idx| {
            ref_cx += @as(f64, ref_x[idx]);
            ref_cy += @as(f64, ref_y[idx]);
            ref_cz += @as(f64, ref_z[idx]);
            tgt_cx += @as(f64, x[idx]);
            tgt_cy += @as(f64, y[idx]);
            tgt_cz += @as(f64, z[idx]);
        }
    } else {
        for (0..ref_x.len) |idx| {
            ref_cx += @as(f64, ref_x[idx]);
            ref_cy += @as(f64, ref_y[idx]);
            ref_cz += @as(f64, ref_z[idx]);
            tgt_cx += @as(f64, x[idx]);
            tgt_cy += @as(f64, y[idx]);
            tgt_cz += @as(f64, z[idx]);
        }
    }
    ref_cx /= n;
    ref_cy /= n;
    ref_cz /= n;
    tgt_cx /= n;
    tgt_cy /= n;
    tgt_cz /= n;

    // Step 2: Build inner product matrix S = Ref^T * Tgt and G values
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
    var g_tgt: f64 = 0.0;

    if (atom_indices) |indices| {
        for (indices) |idx| {
            const rx = @as(f64, ref_x[idx]) - ref_cx;
            const ry = @as(f64, ref_y[idx]) - ref_cy;
            const rz = @as(f64, ref_z[idx]) - ref_cz;
            const tx = @as(f64, x[idx]) - tgt_cx;
            const ty = @as(f64, y[idx]) - tgt_cy;
            const tz = @as(f64, z[idx]) - tgt_cz;

            g_ref += rx * rx + ry * ry + rz * rz;
            g_tgt += tx * tx + ty * ty + tz * tz;

            sxx += rx * tx;
            sxy += rx * ty;
            sxz += rx * tz;
            syx += ry * tx;
            syy += ry * ty;
            syz += ry * tz;
            szx += rz * tx;
            szy += rz * ty;
            szz += rz * tz;
        }
    } else {
        for (0..ref_x.len) |idx| {
            const rx = @as(f64, ref_x[idx]) - ref_cx;
            const ry = @as(f64, ref_y[idx]) - ref_cy;
            const rz = @as(f64, ref_z[idx]) - ref_cz;
            const tx = @as(f64, x[idx]) - tgt_cx;
            const ty = @as(f64, y[idx]) - tgt_cy;
            const tz = @as(f64, z[idx]) - tgt_cz;

            g_ref += rx * rx + ry * ry + rz * rz;
            g_tgt += tx * tx + ty * ty + tz * tz;

            sxx += rx * tx;
            sxy += rx * ty;
            sxz += rx * tz;
            syx += ry * tx;
            syy += ry * ty;
            syz += ry * tz;
            szx += rz * tx;
            szy += rz * ty;
            szz += rz * tz;
        }
    }

    // Step 3: Build coefficients of the characteristic polynomial
    // (Theobald 2005, supplementary equations)
    const sxx_2 = sxx * sxx;
    const syy_2 = syy * syy;
    const szz_2 = szz * szz;
    const sxy_2 = sxy * sxy;
    const syz_2 = syz * syz;
    const sxz_2 = sxz * sxz;
    const syx_2 = syx * syx;
    const szy_2 = szy * szy;
    const szx_2 = szx * szx;

    // c2 = -2 * trace(S^T * S)
    const c2 = -2.0 * (sxx_2 + syy_2 + szz_2 + sxy_2 + syz_2 + sxz_2 +
        syx_2 + szy_2 + szx_2);

    // c1 = -8 * det(S) (Theobald 2005, eq. S3)
    const det_s = sxx * (syy * szz - syz * szy) -
        sxy * (syx * szz - syz * szx) +
        sxz * (syx * szy - syy * szx);
    const c1 = -8.0 * det_s;

    // c0 = det(K) where K is the 4x4 symmetric matrix (Theobald eq. 10)
    // K built from S (see Theobald eq. 10):
    const k00 = sxx + syy + szz;
    const k11 = sxx - syy - szz;
    const k22 = -sxx + syy - szz;
    const k33 = -sxx - syy + szz;
    const k01 = syz - szy;
    const k02 = szx - sxz;
    const k03 = sxy - syx;
    const k12 = sxy + syx;
    const k13 = szx + sxz;
    const k23 = syz + szy;

    // det(K) via cofactor expansion
    const c0 = k00 * (k11 * (k22 * k33 - k23 * k23) -
        k12 * (k12 * k33 - k23 * k13) +
        k13 * (k12 * k23 - k22 * k13)) -
        k01 * (k01 * (k22 * k33 - k23 * k23) -
            k12 * (k02 * k33 - k23 * k03) +
            k13 * (k02 * k23 - k22 * k03)) +
        k02 * (k01 * (k12 * k33 - k23 * k13) -
            k11 * (k02 * k33 - k23 * k03) +
            k13 * (k02 * k13 - k12 * k03)) -
        k03 * (k01 * (k12 * k23 - k22 * k13) -
            k11 * (k02 * k23 - k22 * k03) +
            k12 * (k02 * k13 - k12 * k03));

    // Step 4: Newton-Raphson to find the largest eigenvalue
    // p(lambda) = lambda^4 + c2*lambda^2 + c1*lambda + c0
    const g = (g_ref + g_tgt) / 2.0;
    var lam = g; // Initial guess: upper bound

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

    // Step 5: RMSD = sqrt(max(0, (G_ref + G_tgt - 2*lambda_max) / n))
    const inner = (g_ref + g_tgt - 2.0 * lam) / n;
    return @sqrt(@max(0.0, inner));
}

// ============================================================================
// Tests
// ============================================================================

test "rmsd: identical structures" {
    const x = [_]f32{ 1.0, 2.0, 3.0 };
    const y = [_]f32{ 0.0, 1.0, 0.0 };
    const z = [_]f32{ 0.5, 0.5, 1.5 };

    const rmsd = compute(&x, &y, &z, &x, &y, &z, null);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rmsd, 1e-8);
}

test "rmsd: pure translation removed by alignment" {
    const x = [_]f32{ 1.0, 2.0, 3.0 };
    const y = [_]f32{ 0.0, 1.0, 0.0 };
    const z = [_]f32{ 0.5, 0.5, 1.5 };

    // Shift all atoms by (10, 5, -3)
    const x2 = [_]f32{ 11.0, 12.0, 13.0 };
    const y2 = [_]f32{ 5.0, 6.0, 5.0 };
    const z2 = [_]f32{ -2.5, -2.5, -1.5 };

    const rmsd = compute(&x, &y, &z, &x2, &y2, &z2, null);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rmsd, 1e-6);
}

test "rmsd: uniform displacement without alignment context" {
    // All atoms displaced by the same vector => after centering, still identical
    // So RMSD should be 0
    const x = [_]f32{ 0.0, 1.0, 2.0, 3.0 };
    const y = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const d: f32 = 3.0;
    const x2 = [_]f32{ 0.0 + d, 1.0 + d, 2.0 + d, 3.0 + d };
    const y2 = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const z2 = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    const rmsd = compute(&x, &y, &z, &x2, &y2, &z2, null);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rmsd, 1e-6);
}

test "rmsd: rotated structure" {
    // Rotation around z-axis by 90 degrees:
    // (1,0,0) -> (0,1,0), (-1,0,0) -> (0,-1,0), (0,1,0) -> (-1,0,0)
    const x = [_]f32{ 1.0, -1.0, 0.0 };
    const y = [_]f32{ 0.0, 0.0, 1.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0 };

    const x2 = [_]f32{ 0.0, 0.0, -1.0 };
    const y2 = [_]f32{ 1.0, -1.0, 0.0 };
    const z2 = [_]f32{ 0.0, 0.0, 0.0 };

    const rmsd = compute(&x, &y, &z, &x2, &y2, &z2, null);
    // Optimal alignment should give RMSD ≈ 0
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rmsd, 1e-5);
}

test "rmsd: with atom_indices subset" {
    // 4 atoms; use only first two which are identical, last two differ wildly
    const x = [_]f32{ 1.0, 2.0, 100.0, 200.0 };
    const y = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const x2 = [_]f32{ 1.0, 2.0, -100.0, -200.0 };
    const y2 = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const z2 = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    const indices = [_]u32{ 0, 1 };
    const rmsd = compute(&x, &y, &z, &x2, &y2, &z2, &indices);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), rmsd, 1e-6);
}

test "rmsd: zero atoms" {
    const x = [_]f32{};
    const y = [_]f32{};
    const z = [_]f32{};
    const rmsd = compute(&x, &y, &z, &x, &y, &z, null);
    try std.testing.expectEqual(@as(f64, 0.0), rmsd);
}
