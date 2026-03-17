# ztraj Phase 2 Refactoring — File Structure Cleanup

## Goal

Split `main.zig` (1344 lines) and `simd.zig` (1514 lines) into focused sub-modules to meet the 800-line guideline and improve maintainability. Also deduplicate `Vec3Gen` between `simd.zig` and `neighbor_list.zig`.

## Scope

Structural changes only. No behavior changes, no new features, no test additions. All existing tests must continue to pass. The public API (`root.zig`) remains unchanged.

## Non-Goals

- Test coverage improvements (separate effort)
- Performance optimization for contacts/hbonds (separate effort)
- mdtraj validation (separate effort)

## Design

### main.zig → src/cli/ directory

| File | Responsibility | ~Lines |
|------|---------------|--------|
| `main.zig` | Entry point: version/help, parse args, dispatch subcommand. Imports `build_options` (injected by build system) for `--version`. | 80 |
| `cli/args.zig` | `Subcommand` enum, `Args` struct, `ParseArgsError`, `parseArgs`, `printUsage` | 200 |
| `cli/loader.zig` | `loadTopology`, `loadAllFrames`, file type detection (`isPdb`, `isCif`, `isXtc`, `isDcd`, `endsWithCI`) | 290 |
| `cli/parsers.zig` | `parsePairs`, `parseTriplets`, `parseQuartets`, `validateIndices`, `resolveSelection` | 100 |
| `cli/runners.zig` | All 11 `runXxx` functions, `flushOutput`, `writeScalarSeriesBuf` | 860 |

Note: `runners.zig` slightly exceeds 800 lines but each `runXxx` function is independent and self-contained. Further splitting (e.g. by geometry vs analysis) would scatter a uniform pattern across multiple files for minimal benefit. If individual runners grow, they can be extracted then.

Import graph (no cycles):
```
main.zig → cli/args.zig, cli/runners.zig
cli/runners.zig → cli/args.zig, cli/loader.zig, cli/parsers.zig, ztraj library
cli/loader.zig → ztraj io modules
cli/parsers.zig → ztraj select module
```

### simd.zig → src/simd/ directory

| File | Responsibility | ~Lines |
|------|---------------|--------|
| `simd.zig` | Re-export hub (see Re-export Contract below) | 30 |
| `simd/vec.zig` | `Vec3Gen`, `Vec3`, `Vec3f32`, `Epsilon`, CPU feature detection, `optimal_vector_width` | 70 |
| `simd/trig.zig` | `fastAcos`, `fastAtan2` (f64 + Gen variants) + co-located tests | 200 |
| `simd/distance.zig` | `distanceSquaredBatch{4,8,16}`, `isPointBuriedBatch{4,8,16}` (f64 + Gen) + co-located tests | 500 |
| `simd/lee_richards.zig` | `xyDistanceBatch{4,8}`, `sliceRadiiBatch{4,8}`, `circlesOverlapBatch{4,8}` (f64 + Gen) + co-located tests | 500 |

#### Re-export Contract

`simd.zig` must re-export the complete public API so that `root.zig` (`pub const simd = @import("simd.zig")`) remains unchanged. The re-export file must expose:

- **Types**: `Vec3Gen`, `Vec3`, `Vec3f32`, `Epsilon`
- **CPU**: `cpu_features`, `optimal_vector_width`
- **Trig**: `fastAcos`, `fastAtan2`, `fastAcosGen`, `fastAtan2Gen`
- **Distance**: `distanceSquaredBatch{4,8,16}`, `isPointBuriedBatch{4,8,16}`, plus all `*Gen` variants
- **Lee-Richards**: `xyDistanceBatch{4,8}`, `sliceRadiiBatch{4,8}`, `circlesOverlapBatch{4,8}`, plus all `*Gen` variants

#### Test Placement

Tests are co-located with their functions (Zig convention). Each sub-module file contains the tests for the functions it defines. `zig build test` discovers them via `root.zig → refAllDecls`.

### Vec3Gen deduplication

`neighbor_list.zig` currently defines its own `Vec3Gen`, `Vec3`, `Vec3f32`. After refactoring, it imports from `simd/vec.zig` instead, removing the duplicate definitions.

## Invariants

- `zig build test` passes before and after every commit
- `root.zig` public API surface unchanged
- `build.zig` unchanged — root source files (`src/main.zig` for exe, `src/root.zig` for lib) keep their paths; sub-module imports are resolved relative to the importing file
- No behavioral changes — pure structural refactoring

## Commit Strategy

One commit per logical unit:
1. Create `src/cli/` and split `main.zig`
2. Create `src/simd/` and split `simd.zig`
3. Deduplicate `Vec3Gen` in `neighbor_list.zig`
