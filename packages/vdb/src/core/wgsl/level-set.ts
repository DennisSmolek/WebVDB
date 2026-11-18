/**
 * The opt-in `'levelSet'` helper (K3.7, R1 §6 Step 7): a zero-crossing finder
 * for NanoVDB level sets on the sparse tree, and its gradient normal. Written
 * in the generated module's final names (public `vdb*` API plus the vendored
 * HDDA fields it reads), emitted only when `helpers` lists it.
 *
 * Portions derived from mjurczyk/openvdb (MIT), commit 75e7345
 * (`src/three-openvdb/levelset/webgpu/LevelSetSurfaceMaterial.js`: the sphere
 * trace with a minimum step, the secant step between the last outside and the
 * first inside sample, central-difference normals). Copyright (c)
 * mjurczyk/openvdb contributors, MIT (see THIRD_PARTY.md). Changes: traced on
 * the sparse NanoVDB tree (trilinear `vdbSample`, full resolution) instead of
 * his dense 8-bit distance texture; an HDDA walk skips cells whose trilinear
 * support is all tiles or background above zero; bisection before the secant
 * step; normals through the grid's affine. See DIVERGES.md.
 *
 * CPU twins: `tools` `levelSetHit`, `levelSetGradient`, `levelSetNormal`
 * (`tools/cpu/level-set.ts`), same walk, same sample positions.
 */

/**
 * The WGSL text of the `'levelSet'` helper (`march` profile).
 *
 * - `vdbLevelSetHit(g, o, d, t, sdfToIndex, maxSamples) -> bool`: the first
 *   `t` in `(*t).x .. (*t).y` (clipped to the trilinear support box
 *   `[min - 1, max + 1]`) where the trilinear SDF `vdbSample` goes below 0,
 *   along `o + d * t` in grid `g`'s index space (`d` need not be unit length).
 *   On a hit `(*t).x` is the crossing; a ray whose first sample is already
 *   inside hits at that sample. `sdfToIndex` converts SDF values to index
 *   voxels (1 / the grid's voxel size for a level set in its own world units);
 *   the step is `max(0.95 * v * sdfToIndex, 0.25)` voxels. `maxSamples` caps
 *   the samples (a ray that runs out misses).
 * - `vdbLevelSetGradient(acc, xyz, h) -> vec3f`: central differences of
 *   `vdbSample` at +/- `h` voxels, in SDF units per index voxel.
 * - `vdbLevelSetNormal(g, xyz, h) -> vec3f`: the outward unit normal in world
 *   space, `normalize(W^T * gradient)` with `W` the record's world-to-index
 *   linear part (correct under rotation and non-uniform scale).
 *
 * The walk: an HDDA over aligned cells of at least 8 voxels, the cell size
 * following `vdbGetDim` at the cell corner (8 inside leaves and lower tiles,
 * larger in upper and root tiles). A cell is skipped when the 2x2x2 block of
 * same-size cells from its corner (every tap a trilinear sample in the cell can
 * read) is all uniform (`vdbGetDim >= dim`) with values above 0: no sample in
 * it can be inside. A larger cell that fails the test is walked again at 8.
 * Other cells are sphere traced; a sample below 0 brackets the crossing with
 * the previous sample, which 8 bisections and a secant step refine.
 *
 * Every `vdbSample` call site inlines the type-dispatched trilinear reader,
 * so the gradient's six taps run in a loop (one site): with six sites a
 * module took 98-168 s to compile on SwiftShader, with one 10-19 s. The hit
 * keeps three sites (march, closing sample, bisection): folding them into one
 * moded loop compiled faster again (6-11 s) but ran 2x slower on Metal
 * (teapot frame 15.7 vs 8.0 ms). The sample sequence is the twin's.
 */
export function levelSetWgsl(): string[] {
  return [
    `fn vdb_ls_skip(acc: ptr<function, VdbAccessor>, c: vec3i, dim: u32) -> bool {
    for (var k = 0u; k < 8u; k++) {
        let p = c + vec3i(vec3u(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u)) * i32(dim);
        if vdbGetDim(acc, p) < dim {
            return false;
        }
        if !(vdbReadValue(acc, p) > 0.0) {
            return false;
        }
    }
    return true;
}`,
    `fn vdbLevelSetHit(g: u32, o: vec3f, d: vec3f, t: ptr<function, vec2f>, sdfToIndex: f32, maxSamples: u32) -> bool {
    if !vdb_live(g) {
        return false;
    }
    var t0 = (*t).x;
    var t1 = (*t).y;
    let clip = vdb_hdda_ray_clip(vec3f(vdbIndexMin(g)) - vec3f(1.0), vec3f(vdbIndexMax(g)) + vec3f(1.0), o, &t0, d, &t1);
    if !clip || !(t0 <= t1) {
        return false;
    }
    var acc = vdbAccessorInit(g);
    if !acc.ok {
        return false;
    }
    let inv = 1.0 / max(length(d), 1e-30);
    let minStep = 0.25 * inv;
    let k = 0.95 * sdfToIndex * inv;
    var h: VdbHdda;
    vdbHddaInit(&h, o, t0, d, t1, max(vdbGetDim(&acc, vec3i(floor(o + d * t0))), 8u));
    var tc = t0;
    var tp = t0;
    var vp = 0.0;
    var have = false;
    var n = 0u;
    var updates = 0u;
    var fineUntil = -3.0e38;
    var crossed = false;
    var vc = 0.0;
    for (var it = 0u; it < 8192u; it++) {
        var want = max(vdbGetDim(&acc, h.voxel), 8u);
        if h.tmin < fineUntil {
            want = 8u;
        }
        if want != u32(h.dim) && updates < 2u {
            vdbHddaUpdate(&h, o, d, want);
            updates++;
            continue;
        }
        updates = 0u;
        let te = min(min(h.next.x, h.next.y), min(h.next.z, h.tmax));
        if te >= tc {
            let dim = u32(h.dim);
            if vdb_ls_skip(&acc, h.voxel, dim) {
                tc = te;
            } else if dim > 8u && te > h.tmin {
                fineUntil = te;
                vdbHddaUpdate(&h, o, d, 8u);
                continue;
            } else {
                loop {
                    if tc > te || n >= maxSamples {
                        break;
                    }
                    vc = vdbSample(&acc, o + d * tc);
                    n++;
                    if vc < 0.0 {
                        crossed = true;
                        break;
                    }
                    tp = tc;
                    vp = vc;
                    have = true;
                    tc += max(vc * k, minStep);
                }
                if crossed || n >= maxSamples {
                    break;
                }
            }
        }
        if !vdbHddaStep(&h) {
            break;
        }
    }
    if !crossed && have && tp < t1 && n < maxSamples {
        tc = t1;
        vc = vdbSample(&acc, o + d * tc);
        crossed = vc < 0.0;
    }
    if !crossed {
        return false;
    }
    if !have {
        (*t).x = tc;
        return true;
    }
    var lo = tp;
    var hi = tc;
    var vlo = vp;
    var vhi = vc;
    for (var i = 0u; i < 8u; i++) {
        let m = 0.5 * (lo + hi);
        let vm = vdbSample(&acc, o + d * m);
        if vm < 0.0 {
            hi = m;
            vhi = vm;
        } else {
            lo = m;
            vlo = vm;
        }
    }
    (*t).x = lo + (hi - lo) * (vlo / (vlo - vhi));
    return true;
}`,
    `fn vdbLevelSetGradient(acc: ptr<function, VdbAccessor>, xyz: vec3f, h: f32) -> vec3f {
    var gr = vec3f(0.0);
    for (var i = 0u; i < 6u; i++) {
        let axis = i >> 1u;
        let sgn = select(-1.0, 1.0, (i & 1u) == 0u);
        var e = vec3f(0.0);
        e[axis] = sgn * h;
        gr[axis] += sgn * vdbSample(acc, xyz + e);
    }
    return gr / (2.0 * h);
}`,
    `fn vdbLevelSetNormal(g: u32, xyz: vec3f, h: f32) -> vec3f {
    var acc = vdbAccessorInit(g);
    let gi = vdbLevelSetGradient(&acc, xyz, h);
    let n = vec3f(dot(vdbWorldToIndexDir(g, vec3f(1.0, 0.0, 0.0)), gi), dot(vdbWorldToIndexDir(g, vec3f(0.0, 1.0, 0.0)), gi), dot(vdbWorldToIndexDir(g, vec3f(0.0, 0.0, 1.0)), gi));
    let l = length(n);
    if !(l > 0.0) {
        return vec3f(0.0, 1.0, 0.0);
    }
    return n / l;
}`,
  ];
}
