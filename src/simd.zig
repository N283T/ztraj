// Re-export sub-modules
pub const vec = @import("simd/vec.zig");
pub const trig = @import("simd/trig.zig");
pub const distance = @import("simd/distance.zig");
pub const lee_richards = @import("simd/lee_richards.zig");

// Re-export all public symbols at top level for backward compatibility

// From vec.zig
pub const Vec3Gen = vec.Vec3Gen;
pub const Vec3 = vec.Vec3;
pub const Vec3f32 = vec.Vec3f32;
pub const Epsilon = vec.Epsilon;
pub const cpu_features = vec.cpu_features;
pub const optimal_vector_width = vec.optimal_vector_width;

// From trig.zig
pub const fastAcos = trig.fastAcos;
pub const fastAtan2 = trig.fastAtan2;
pub const fastAcosGen = trig.fastAcosGen;
pub const fastAtan2Gen = trig.fastAtan2Gen;

// From distance.zig
pub const distanceSquaredBatch4 = distance.distanceSquaredBatch4;
pub const isPointBuriedBatch4 = distance.isPointBuriedBatch4;
pub const distanceSquaredBatch8 = distance.distanceSquaredBatch8;
pub const isPointBuriedBatch8 = distance.isPointBuriedBatch8;
pub const distanceSquaredBatch16 = distance.distanceSquaredBatch16;
pub const isPointBuriedBatch16 = distance.isPointBuriedBatch16;
pub const distanceSquaredBatch4Gen = distance.distanceSquaredBatch4Gen;
pub const isPointBuriedBatch4Gen = distance.isPointBuriedBatch4Gen;
pub const distanceSquaredBatch8Gen = distance.distanceSquaredBatch8Gen;
pub const isPointBuriedBatch8Gen = distance.isPointBuriedBatch8Gen;
pub const distanceSquaredBatch16Gen = distance.distanceSquaredBatch16Gen;
pub const isPointBuriedBatch16Gen = distance.isPointBuriedBatch16Gen;

// From lee_richards.zig
pub const xyDistanceBatch4 = lee_richards.xyDistanceBatch4;
pub const sliceRadiiBatch4 = lee_richards.sliceRadiiBatch4;
pub const circlesOverlapBatch4 = lee_richards.circlesOverlapBatch4;
pub const xyDistanceBatch8 = lee_richards.xyDistanceBatch8;
pub const sliceRadiiBatch8 = lee_richards.sliceRadiiBatch8;
pub const circlesOverlapBatch8 = lee_richards.circlesOverlapBatch8;
pub const xyDistanceBatch4Gen = lee_richards.xyDistanceBatch4Gen;
pub const sliceRadiiBatch4Gen = lee_richards.sliceRadiiBatch4Gen;
pub const circlesOverlapBatch4Gen = lee_richards.circlesOverlapBatch4Gen;
pub const xyDistanceBatch8Gen = lee_richards.xyDistanceBatch8Gen;
pub const sliceRadiiBatch8Gen = lee_richards.sliceRadiiBatch8Gen;
pub const circlesOverlapBatch8Gen = lee_richards.circlesOverlapBatch8Gen;

test {
    @import("std").testing.refAllDecls(@This());
}
