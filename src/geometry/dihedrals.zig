//! Dihedral (torsion) angle calculation for atom quartets (i-j-k-l).

const std = @import("std");

/// Compute dihedral angles (in radians, range [-pi, pi]) for a list of atom
/// quartets.
///
/// For quartet (i, j, k, l):
///   b1 = pos[j] - pos[i]
///   b2 = pos[k] - pos[j]
///   b3 = pos[l] - pos[k]
///   n1 = cross(b1, b2)
///   n2 = cross(b2, b3)
///   m  = cross(normalize(b2), n1)
///   angle = atan2(dot(m, n2), dot(n1, n2))
///
/// Uses f64 intermediate precision.
/// `result` must have the same length as `quartets`. Values are in radians.
pub fn compute(
    x: []const f32,
    y: []const f32,
    z: []const f32,
    quartets: []const [4]u32,
    result: []f32,
) void {
    std.debug.assert(result.len == quartets.len);

    for (quartets, 0..) |q, idx| {
        const i = q[0];
        const j = q[1];
        const k = q[2];
        const l = q[3];

        // Bond vectors in f64
        const b1x: f64 = @as(f64, x[j]) - @as(f64, x[i]);
        const b1y: f64 = @as(f64, y[j]) - @as(f64, y[i]);
        const b1z: f64 = @as(f64, z[j]) - @as(f64, z[i]);

        const b2x: f64 = @as(f64, x[k]) - @as(f64, x[j]);
        const b2y: f64 = @as(f64, y[k]) - @as(f64, y[j]);
        const b2z: f64 = @as(f64, z[k]) - @as(f64, z[j]);

        const b3x: f64 = @as(f64, x[l]) - @as(f64, x[k]);
        const b3y: f64 = @as(f64, y[l]) - @as(f64, y[k]);
        const b3z: f64 = @as(f64, z[l]) - @as(f64, z[k]);

        // n1 = cross(b1, b2)
        const n1x = b1y * b2z - b1z * b2y;
        const n1y = b1z * b2x - b1x * b2z;
        const n1z = b1x * b2y - b1y * b2x;

        // n2 = cross(b2, b3)
        const n2x = b2y * b3z - b2z * b3y;
        const n2y = b2z * b3x - b2x * b3z;
        const n2z = b2x * b3y - b2y * b3x;

        // normalize b2
        const b2_len = @sqrt(b2x * b2x + b2y * b2y + b2z * b2z);
        const b2nx = b2x / b2_len;
        const b2ny = b2y / b2_len;
        const b2nz = b2z / b2_len;

        // m = cross(b2_normalized, n1)  [IUPAC sign convention]
        const mx = b2ny * n1z - b2nz * n1y;
        const my = b2nz * n1x - b2nx * n1z;
        const mz = b2nx * n1y - b2ny * n1x;

        const dot_m_n2 = mx * n2x + my * n2y + mz * n2z;
        const dot_n1_n2 = n1x * n2x + n1y * n2y + n1z * n2z;

        result[idx] = @floatCast(std.math.atan2(dot_m_n2, dot_n1_n2));
    }
}

// ============================================================================
// Tests
// ============================================================================

test "dihedrals: planar cis (0 degrees)" {
    // All atoms in the XY plane, cis arrangement -> 0 dihedral
    // i=(0,1,0), j=(0,0,0), k=(1,0,0), l=(1,1,0)
    // b1=(0,-1,0), b2=(1,0,0), b3=(0,1,0)
    // n1=cross(b1,b2)=(0,0,1), n2=cross(b2,b3)=(0,0,-1)... hmm let's use classic cis
    // Classic cis: i=(-1,0,0), j=(0,0,0), k=(1,0,0), l=(2,0,0) is collinear - bad
    // Use: i=(0,1,0), j=(0,0,0), k=(1,0,0), l=(1,0,1) for non-collinear
    // Actually let's construct a known 0-degree dihedral carefully:
    // cis: i=(0,1,0), j=(0,0,0), k=(1,0,0), l=(1,1,0)
    // b1=(0,-1,0), b2=(1,0,0), b3=(0,1,0)
    // n1=cross(b1,b2) = (-1*0-0*0, 0*1-0*0, 0*0-(-1)*1) = (0,0,1)
    // n2=cross(b2,b3) = (0*0-0*1, 0*0-1*0, 1*1-0*0) = (0,0,1)
    // angle = atan2(dot(m,n2), dot(n1,n2))
    // b2_norm=(1,0,0), m=cross(b2n,n1)=cross((1,0,0),(0,0,1))=(0*1-0*0, 0*0-1*1, 1*0-0*0)=(0,-1,0)
    // dot(m,n2)=dot((0,-1,0),(0,0,1))=0
    // dot(n1,n2)=dot((0,0,1),(0,0,1))=1
    // angle=atan2(0,1)=0  -> cis = 0
    const x = [_]f32{ 0.0, 0.0, 1.0, 1.0 };
    const y = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const quartets = [_][4]u32{.{ 0, 1, 2, 3 }};
    var result = [_]f32{0.0};

    compute(&x, &y, &z, &quartets, &result);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 1e-5);
}

test "dihedrals: planar trans (pi radians)" {
    // trans: i=(0,1,0), j=(0,0,0), k=(1,0,0), l=(1,-1,0)
    // b1=(0,-1,0), b2=(1,0,0), b3=(0,-1,0)
    // n1=cross(b1,b2)=(0,0,1) (same as before)
    // n2=cross(b2,b3)=cross((1,0,0),(0,-1,0))=(0*0-0*(-1), 0*0-1*0, 1*(-1)-0*0)=(0,0,-1)
    // b2_norm=(1,0,0), m=cross(b2n,n1)=(0,-1,0) (same as before)
    // dot(m,n2)=dot((0,-1,0),(0,0,-1))=0
    // dot(n1,n2)=dot((0,0,1),(0,0,-1))=-1
    // angle=atan2(0,-1)=pi
    const x = [_]f32{ 0.0, 0.0, 1.0, 1.0 };
    const y = [_]f32{ 1.0, 0.0, 0.0, -1.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const quartets = [_][4]u32{.{ 0, 1, 2, 3 }};
    var result = [_]f32{0.0};

    compute(&x, &y, &z, &quartets, &result);

    try std.testing.expectApproxEqAbs(
        @as(f32, std.math.pi),
        @abs(result[0]),
        1e-5,
    );
}

test "dihedrals: 90 degree dihedral" {
    // i=(0,1,0), j=(0,0,0), k=(1,0,0), l=(1,0,1)
    // b1=(0,-1,0), b2=(1,0,0), b3=(0,0,1)
    // n1=cross(b1,b2)=cross((0,-1,0),(1,0,0))=((-1)*0-0*0, 0*1-0*0, 0*0-(-1)*1)=(0,0,1)
    // n2=cross(b2,b3)=cross((1,0,0),(0,0,1))=(0*1-0*0, 0*0-1*1, 1*0-0*0)=(0,-1,0)
    // b2_norm=(1,0,0), m=cross(b2n,n1)=cross((1,0,0),(0,0,1))=(0,-1,0)  [IUPAC: cross(b2n,n1)]
    // dot(m,n2)=dot((0,-1,0),(0,-1,0))=1
    // dot(n1,n2)=dot((0,0,1),(0,-1,0))=0
    // angle=atan2(1,0)=pi/2
    const x = [_]f32{ 0.0, 0.0, 1.0, 1.0 };
    const y = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0, 1.0 };
    const quartets = [_][4]u32{.{ 0, 1, 2, 3 }};
    var result = [_]f32{0.0};

    compute(&x, &y, &z, &quartets, &result);

    try std.testing.expectApproxEqAbs(
        @as(f32, std.math.pi / 2.0),
        @abs(result[0]),
        1e-5,
    );
}

test "dihedrals: multiple quartets" {
    // Combine cis and trans from the tests above
    const x = [_]f32{ 0.0, 0.0, 1.0, 1.0, 1.0 };
    const y = [_]f32{ 1.0, 0.0, 0.0, 1.0, -1.0 };
    const z = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0 };
    const quartets = [_][4]u32{
        .{ 0, 1, 2, 3 }, // cis -> 0
        .{ 0, 1, 2, 4 }, // trans -> pi
    };
    var result = [_]f32{ 0.0, 0.0 };

    compute(&x, &y, &z, &quartets, &result);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, std.math.pi), @abs(result[1]), 1e-5);
}

test "dihedrals: zero quartets is no-op" {
    const x = [_]f32{1.0};
    const y = [_]f32{1.0};
    const z = [_]f32{1.0};
    const quartets = [_][4]u32{};
    var result = [_]f32{};

    compute(&x, &y, &z, &quartets, &result);
}
