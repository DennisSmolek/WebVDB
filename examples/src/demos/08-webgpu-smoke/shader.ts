/**
 * The host half of demo 08's shader: a full-screen triangle and an
 * emission-absorption raymarch over every grid in the pool. It calls only the
 * public `vdb*` API of the generated module (`profile: 'march'`, helpers
 * `cellBoundTrilinear` and `hddaCellEnd`), which main.ts prepends. The module declares `vdbPool` at `@group(0) @binding(0)`; this
 * half adds the frame uniforms at binding 1.
 *
 * Per pixel: for each grid, move the world ray into that grid's index space
 * (`vdbWorldToIndex*` read the record, so instance transforms are free), clip
 * it to the grid's index box, and march the grids front to back. Inside a grid
 * an HDDA walks 8^3 cells (leaf-sized); cells whose bound is 0 are skipped,
 * the rest are sampled at a fixed step with `vdbSample` (trilinear).
 *
 * The grids are assumed not to overlap (the demo places them apart), so
 * marching them one after another in entry order composites correctly.
 */

/** Byte layout of `struct Frame` below; main.ts writes it with one `writeBuffer`. */
export const FRAME_FLOATS = 20;

export const HOST_WGSL = /* wgsl */ `
struct Frame {
    eye: vec3f,
    tanHalfFov: f32,
    right: vec3f,      // unit camera axes
    aspect: f32,
    up: vec3f,
    density: f32,      // extinction per world unit at value 1
    forward: vec3f,
    stepVoxels: f32,   // march step, in index voxels of the grid being marched
    resolution: vec2f,
    _pad: vec2f,
};
@group(0) @binding(1) var<uniform> frame: Frame;

const MAX_GRIDS = 4u;
const CELL = 8u;           // HDDA cell edge in voxels: one leaf node
const MAX_CELLS = 2048u;   // safety cap per grid

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn tint(g: u32) -> vec3f {
    return select(vec3f(1.0, 0.62, 0.32), vec3f(0.92, 0.95, 1.0), g == 0u);
}

// March grid g over the clipped interval t (world units, shared by every grid)
// and composite into (T, color).
fn marchGrid(g: u32, o: vec3f, d: vec3f, t: vec2f, T: ptr<function, f32>, color: ptr<function, vec3f>) {
    var acc = vdbAccessorInit(g);
    let dt = frame.stepVoxels / length(d);   // world distance of one step
    var ts = t.x + 0.5 * dt;                  // sample positions on a fixed lattice
    var h: VdbHdda;
    vdbHddaInit(&h, o, t.x, d, t.y, CELL);
    for (var n = 0u; n < MAX_CELLS; n++) {
        let cellEnd = vdbHddaCellEnd(&h);
        // Trilinear-safe skip: a tap reads one voxel past its base, into the
        // +x/+y/+z neighbour cells, so bound the 2x2x2 block (CLAUDE.md Traps).
        if vdbCellBoundTrilinear(&acc, h.voxel, CELL) > 0.0 {
            for (; ts < cellEnd; ts += dt) {
                let v = max(vdbSample(&acc, o + d * ts), 0.0);
                let alpha = 1.0 - exp(-v * frame.density * dt);
                *color += *T * alpha * tint(g) * mix(0.55, 1.0, saturate(v));
                *T *= 1.0 - alpha;
            }
            if *T < 0.01 {
                return;
            }
        } else if ts < cellEnd {
            ts += ceil((cellEnd - ts) / dt) * dt;   // skip the empty cell, stay on the lattice
        }
        if !vdbHddaStep(&h) {
            return;
        }
    }
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let uv = (pos.xy / frame.resolution) * 2.0 - 1.0;
    let rd = normalize(frame.forward
        + frame.right * (uv.x * frame.tanHalfFov * frame.aspect)
        - frame.up * (uv.y * frame.tanHalfFov));
    let ro = frame.eye;

    // Clip the ray against every live grid, in each grid's index space.
    var ids: array<u32, MAX_GRIDS>;
    var t0s: array<f32, MAX_GRIDS>;
    var t1s: array<f32, MAX_GRIDS>;
    var count = 0u;
    for (var g = 0u; g < min(vdbGridCount(), MAX_GRIDS); g++) {
        var t = vec2f(0.0, 1.0e9);
        if vdbRayClip(g, vdbWorldToIndex(g, ro), vdbWorldToIndexDir(g, rd), &t) {
            // insertion sort by entry distance: front to back
            var k = count;
            while k > 0u && t0s[k - 1u] > t.x {
                ids[k] = ids[k - 1u];
                t0s[k] = t0s[k - 1u];
                t1s[k] = t1s[k - 1u];
                k--;
            }
            ids[k] = g;
            t0s[k] = t.x;
            t1s[k] = t.y;
            count++;
        }
    }

    var T = 1.0;
    var color = vec3f(0.0);
    for (var k = 0u; k < count; k++) {
        let g = ids[k];
        marchGrid(g, vdbWorldToIndex(g, ro), vdbWorldToIndexDir(g, rd), vec2f(t0s[k], t1s[k]), &T, &color);
        if T < 0.01 {
            break;
        }
    }

    let sky = mix(vec3f(0.05, 0.065, 0.12), vec3f(0.16, 0.2, 0.3), 0.5 - 0.5 * uv.y);
    let c = color + T * sky;
    return vec4f(pow(c, vec3f(1.0 / 2.2)), 1.0);   // the targets are non-sRGB formats
}
`;
