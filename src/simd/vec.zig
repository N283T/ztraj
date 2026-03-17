const std = @import("std");
const builtin = @import("builtin");

// ============================================================================
// Local type definitions (Vec3Gen, Vec3, Vec3f32, Epsilon)
// ============================================================================

/// Generic 3D vector/point with x, y, z fields of type T.
pub fn Vec3Gen(comptime T: type) type {
    return struct {
        x: T,
        y: T,
        z: T,
    };
}

/// f64 3D vector (default precision).
pub const Vec3 = Vec3Gen(f64);

/// f32 3D vector.
pub const Vec3f32 = Vec3Gen(f32);

/// Epsilon values for floating-point comparisons.
pub fn Epsilon(comptime T: type) type {
    return struct {
        /// Stricter epsilon for trigonometric functions (e.g., atan2 near-zero)
        /// f32: 1e-7, f64: 1e-10
        pub const trig: T = if (T == f32) 1e-7 else 1e-10;
    };
}

// ============================================================================
// Compile-time CPU feature detection
// ============================================================================

/// Compile-time CPU feature detection for SIMD optimization.
pub const cpu_features = struct {
    /// AVX-512F support (512-bit vectors)
    pub const has_avx512f = if (builtin.cpu.arch == .x86_64)
        std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f)
    else
        false;

    /// AVX2 support (256-bit vectors)
    pub const has_avx2 = if (builtin.cpu.arch == .x86_64)
        std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)
    else
        false;

    /// ARM NEON support (128-bit vectors, available on all aarch64)
    pub const has_neon = builtin.cpu.arch == .aarch64;
};

/// Optimal vector widths based on detected CPU features.
/// - AVX-512: 16 f32s or 8 f64s per vector (512-bit)
/// - AVX2: 8 f32s or 4 f64s per vector (256-bit)
/// - NEON: 8 f32s or 2 f64s per vector (128-bit, but efficient for f32)
/// - Fallback: 4 f32s or 2 f64s per vector
pub const optimal_vector_width = struct {
    /// Optimal f32 vector width for this CPU
    pub const f32_width: comptime_int = if (cpu_features.has_avx512f) 16 else if (cpu_features.has_avx2 or cpu_features.has_neon) 8 else 4;

    /// Optimal f64 vector width for this CPU
    pub const f64_width: comptime_int = if (cpu_features.has_avx512f) 8 else if (cpu_features.has_avx2) 4 else 2;
};
