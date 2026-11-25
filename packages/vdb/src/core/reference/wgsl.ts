/**
 * K6.R: the reference renderer's compute kernel (`vdbRefTraceCs`) and its uniform layout. The algorithm, the random
 * stream and the draw order are those of the CPU twin (`tools/reference/trace.ts`), which documents them; this file
 * is that code in WGSL. One VDB source at group 2 (`vdbSrc*`, the K6 group-2 contract), so the kernel binds at most
 * the pool's one storage buffer (D10); the accumulation lives in rgba32float textures (group 1).
 *
 * Compile-time care (the K3.7 trap: inlined sample sites cost SwiftShader minutes): the path is a state machine whose
 * every ray query (primary transmittance, path segment, light visibility) goes through ONE call of `vdbRefTrack`, and
 * the source is read at one call site per channel.
 *
 * Group 0: `vdbRefU` (VdbRefUniforms), `vdbRefBatch` (dynamic offset, 256 B stride). Group 1: accumulation in
 * (sum, sum of squares; sampled, unfilterable), out (storage, write), the equirect environment map.
 */

/** Uniform layout (bytes) of `VdbRefUniforms`; the packer is `core/reference/renderer.ts` `vdbReferencePackUniforms`. */
export const VDB_REF_UNIFORM_OFFSETS = Object.freeze({
  cameraToWorld: 0,
  projectionInverse: 64,
  size: 128,
  seed: 136,
  spp: 140,
  maxBounces: 144,
  rrStart: 148,
  volumeCount: 152,
  lightCount: 156,
  hemiCount: 160,
  flags: 164,
  envSize: 168,
  uniformEnv: 176,
  envRotation: 192,
  hemis: 240,
  volumes: 432,
  lights: 1712,
  size_: 2736,
});
export const VDB_REF_HEMI_BYTES = 48;
export const VDB_REF_VOLUME_BYTES = 160;
export const VDB_REF_LIGHT_BYTES = 64;
/** `flags` bits. */
export const VDB_REF_FLAG_BOX_FILTER = 1;
export const VDB_REF_FLAG_ORTHOGRAPHIC = 2;
export const VDB_REF_FLAG_EQUIRECT = 4;
/** Stride of the per-dispatch batch records (dynamic uniform offsets: 256-byte aligned). */
export const VDB_REF_BATCH_STRIDE = 256;

/**
 * The fail-before mutations (`?mutate=`): WGSL text patches, one per name, matching the CPU twin's
 * `VDBReferenceMutation`s. Each must make an analytic check fail.
 */
export const VDB_REFERENCE_MUTATIONS: Readonly<Record<string, readonly [string, string]>> = Object.freeze({
  "ratio-weight": ["trans = trans * (1.0 - sum / sbar);", "trans = trans * (1.0 - 0.9 * sum / sbar);"],
  "delta-accept": ["let u = vdbRefRand() * sbar;", "let u = -1.0;"],
  "phase-norm": ["return (1.0 - g * g) / (4.0 * VDB_REF_PI * d * sqrt(d));", "return (1.0 - g * g) / (d * sqrt(d));"],
  "nee-transmittance": ["rad = rad + beta * neeLi * hit.trans;", "rad = rad + beta * neeLi;"],
  "emission-albedo": ["rad = rad + beta * le;", "rad = rad + beta * le * albedo;"],
  "escape-depth": ["if (depth > 0u) {", "if (depth == 1u) {"],
});

/** The kernel's WGSL after the source's (`vdbSrc*`) and module text. */
export function vdbReferenceKernelWgsl(o: { maxVolumes: number; maxLights: number; maxHemispheres: number; maxSteps: number; ratioRr: number }): string {
  return /* wgsl */ `
// ---- K6.R reference renderer (core/reference/wgsl.ts): brute-force volumetric path tracing -------------------
const VDB_REF_PI: f32 = 3.14159265358979;
const VDB_REF_MAX_STEPS: u32 = ${o.maxSteps}u;
const VDB_REF_RATIO_RR: f32 = ${o.ratioRr.toFixed(6)};
const VDB_REF_FAR: f32 = 1e30;

struct VdbRefHemi {
  sky: vec4f,
  ground: vec4f,
  up: vec4f,
}

struct VdbRefVolume {
  worldToLocal: mat4x4f,
  params: vec4f,
  albedo: vec4f,
  phase: vec4f,
  emission: vec4f,
  temperature: vec4f,
  ids: vec4u,
}

struct VdbRefLight {
  position: vec4f,
  direction: vec4f,
  color: vec4f,
  extra: vec4f,
}

struct VdbRefUniforms {
  cameraToWorld: mat4x4f,
  projectionInverse: mat4x4f,
  size: vec2u,
  seed: u32,
  spp: u32,
  maxBounces: u32,
  rrStart: u32,
  volumeCount: u32,
  lightCount: u32,
  hemiCount: u32,
  flags: u32,
  envSize: vec2u,
  uniformEnv: vec4f,
  envRotation0: vec4f,
  envRotation1: vec4f,
  envRotation2: vec4f,
  hemis: array<VdbRefHemi, ${o.maxHemispheres}>,
  volumes: array<VdbRefVolume, ${o.maxVolumes}>,
  lights: array<VdbRefLight, ${o.maxLights}>,
}

struct VdbRefBatch {
  batch: u32,
  accumulate: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<uniform> vdbRefU: VdbRefUniforms;
@group(0) @binding(1) var<uniform> vdbRefBatch: VdbRefBatch;
@group(1) @binding(0) var vdbRefAccumIn0: texture_2d<f32>;
@group(1) @binding(1) var vdbRefAccumIn1: texture_2d<f32>;
@group(1) @binding(2) var vdbRefAccumOut0: texture_storage_2d<rgba32float, write>;
@group(1) @binding(3) var vdbRefAccumOut1: texture_storage_2d<rgba32float, write>;
@group(1) @binding(4) var vdbRefEnvMap: texture_2d<f32>;

// ---- random stream (tools/reference/rng.ts) ----
var<private> vdbRefState: u32;

fn vdbRefPcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn vdbRefSeed(seed: u32, sampleIndex: u32, pixel: u32) {
  vdbRefState = vdbRefPcg(pixel + vdbRefPcg(seed ^ vdbRefPcg(sampleIndex)));
}

fn vdbRefRand() -> f32 {
  vdbRefState = vdbRefState * 747796405u + 2891336453u;
  let s = vdbRefState;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return f32(((w >> 22u) ^ w) >> 8u) * 5.9604644775390625e-8;
}

// ---- phase (tools/reference/trace.ts vdbRefHenyeyGreenstein, vdbRefPhase, vdbRefSampleHgCos) ----
fn vdbRefHg(c: f32, g: f32) -> f32 {
  let d = 1.0 + g * g - 2.0 * g * c;
  return (1.0 - g * g) / (4.0 * VDB_REF_PI * d * sqrt(d));
}

fn vdbRefPhaseEval(c: f32, ph: vec4f) -> f32 {
  let a = vdbRefHg(c, ph.x);
  if (ph.z > 0.0) {
    return a + (vdbRefHg(c, ph.y) - a) * ph.z;
  }
  return a;
}

fn vdbRefSampleHgCos(g: f32, u: f32) -> f32 {
  if (abs(g) < 1e-3) {
    return 1.0 - 2.0 * u;
  }
  let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u);
  return clamp((1.0 + g * g - s * s) / (2.0 * g), -1.0, 1.0);
}

fn vdbRefDirectionAround(n: vec3f, cosTheta: f32, u: f32) -> vec3f {
  let sgn = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (sgn + n.z);
  let b = n.x * n.y * a;
  let b1 = vec3f(1.0 + sgn * n.x * n.x * a, sgn * b, -sgn * n.x);
  let b2 = vec3f(b, sgn + n.y * n.y * a, -n.y);
  let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
  let phi = 2.0 * VDB_REF_PI * u;
  return cos(phi) * sinTheta * b1 + sin(phi) * sinTheta * b2 + cosTheta * n;
}

fn vdbRefSmoothstep(e0: f32, e1: f32, x: f32) -> f32 {
  if (!(e1 > e0)) {
    return select(0.0, 1.0, x >= e0);
  }
  let t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

fn vdbRefMax3(v: vec3f) -> f32 {
  return max(v.x, max(v.y, v.z));
}

// ---- geometry (vdbRefVolumeInterval, vdbRefCameraRay) ----
fn vdbRefInterval(i: u32, o: vec3f, d: vec3f, tMax: f32) -> vec2f {
  let m = vdbRefU.volumes[i].worldToLocal;
  let lo = (m * vec4f(o, 1.0)).xyz;
  let ld = (m * vec4f(d, 0.0)).xyz;
  var t0 = -VDB_REF_FAR;
  var t1 = VDB_REF_FAR;
  if (vdbRefU.volumes[i].params.w > 0.5) {
    let a = dot(ld, ld);
    let b = dot(lo, ld);
    let c = dot(lo, lo) - 0.25;
    let disc = b * b - a * c;
    if (disc <= 0.0 || a <= 0.0) {
      return vec2f(1.0, 0.0);
    }
    let q = sqrt(disc);
    t0 = (-b - q) / a;
    t1 = (-b + q) / a;
  } else {
    for (var k = 0u; k < 3u; k = k + 1u) {
      if (abs(ld[k]) < 1e-12) {
        if (abs(lo[k]) > 0.5) {
          return vec2f(1.0, 0.0);
        }
      } else {
        let ta = (-0.5 - lo[k]) / ld[k];
        let tb = (0.5 - lo[k]) / ld[k];
        t0 = max(t0, min(ta, tb));
        t1 = min(t1, max(ta, tb));
      }
    }
  }
  t0 = max(t0, 0.0);
  t1 = min(t1, tMax);
  if (t0 < t1) {
    return vec2f(t0, t1);
  }
  return vec2f(1.0, 0.0);
}

fn vdbRefUnproject(ndc: vec3f) -> vec3f {
  let p = vdbRefU.projectionInverse * vec4f(ndc, 1.0);
  return p.xyz / p.w;
}

// ---- the medium (the renderer's sampleExtinction / sampleAlbedo / sampleEmission at level 0) ----
fn vdbRefUvw(i: u32, x: vec3f) -> vec3f {
  return (vdbRefU.volumes[i].worldToLocal * vec4f(x, 1.0)).xyz + vec3f(0.5);
}

fn vdbRefSigma(i: u32, x: vec3f) -> f32 {
  let uvw = vdbRefUvw(i, x);
  let v = vdbRefU.volumes[i].ids.x;
  let value = vdbSrcDensity(v, uvw, 0.0) * vdbSrcMask(v, uvw, 0.0);
  let p = vdbRefU.volumes[i].params;
  return select(0.0, value, value > p.z) * p.x;
}

// ---- tracking: delta (ratio false: the collision) or ratio (ratio true: T), piecewise majorants ----
struct VdbRefHit {
  escaped: bool,
  t: f32,
  vol: u32,
  trans: f32,
}

fn vdbRefTrack(o: vec3f, d: vec3f, tMax: f32, ratio: bool) -> VdbRefHit {
  var iv: array<vec2f, ${o.maxVolumes}>;
  let n = vdbRefU.volumeCount;
  for (var i = 0u; i < n; i = i + 1u) {
    iv[i] = vdbRefInterval(i, o, d, tMax);
  }
  var t = 0.0;
  var trans = 1.0;
  for (var step = 0u; step < VDB_REF_MAX_STEPS; step = step + 1u) {
    var sbar = 0.0;
    var tb = tMax;
    for (var i = 0u; i < n; i = i + 1u) {
      let w = iv[i];
      if (w.x <= t && t < w.y) {
        sbar = sbar + vdbRefU.volumes[i].params.y;
        tb = min(tb, w.y);
      } else if (w.x > t && w.x <= w.y) {
        tb = min(tb, w.x);
      }
    }
    if (!(sbar > 0.0)) {
      if (tb >= tMax) {
        return VdbRefHit(true, tMax, 0u, trans);
      }
      t = tb;
      continue;
    }
    let dt = -log(1.0 - vdbRefRand()) / sbar;
    if (t + dt >= tb) {
      t = tb;
      if (t >= tMax) {
        return VdbRefHit(true, tMax, 0u, trans);
      }
      continue;
    }
    t = t + dt;
    let x = o + t * d;
    var sig: array<f32, ${o.maxVolumes}>;
    var sum = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
      let w = iv[i];
      var s = 0.0;
      if (w.x <= t && t < w.y) {
        s = vdbRefSigma(i, x);
      }
      sig[i] = s;
      sum = sum + s;
    }
    if (!ratio) {
      let u = vdbRefRand() * sbar;
      var acc = 0.0;
      for (var i = 0u; i < n; i = i + 1u) {
        let w = iv[i];
        if (!(w.x <= t && t < w.y)) {
          continue;
        }
        acc = acc + sig[i];
        if (u < acc) {
          return VdbRefHit(false, t, i, trans);
        }
      }
    } else {
      trans = trans * (1.0 - sum / sbar);
      if (trans < VDB_REF_RATIO_RR) {
        if (vdbRefRand() >= 0.5) {
          return VdbRefHit(false, t, 0u, 0.0);
        }
        trans = trans * 2.0;
      }
    }
  }
  return VdbRefHit(true, t, 0u, select(0.0, trans, ratio));
}

// ---- lights and environment (vdbRefLightAt, vdbRefEnvironmentRadiance) ----
struct VdbRefNee {
  ok: bool,
  dir: vec3f,
  dist: f32,
  li: vec3f,
}

fn vdbRefLightAt(k: u32, x: vec3f) -> VdbRefNee {
  let l = vdbRefU.lights[k];
  let kind = u32(l.position.w + 0.5);
  if (kind == 0u) {
    return VdbRefNee(vdbRefMax3(l.color.xyz) > 0.0, l.direction.xyz, VDB_REF_FAR, l.color.xyz);
  }
  let to = l.position.xyz - x;
  let dist = length(to);
  if (!(dist > 0.0)) {
    return VdbRefNee(false, vec3f(0.0, 1.0, 0.0), 0.0, vec3f(0.0));
  }
  let dir = to / dist;
  var s = select(1.0, 1.0 / (dist * dist), l.extra.x > 0.5);
  if (kind == 2u) {
    s = s * vdbRefSmoothstep(l.direction.w, l.color.w, -dot(dir, l.direction.xyz));
  }
  let li = l.color.xyz * s;
  return VdbRefNee(s > 0.0 && vdbRefMax3(li) > 0.0, dir, dist, li);
}

fn vdbRefEnvironment(d: vec3f) -> vec3f {
  var e = vdbRefU.uniformEnv.xyz;
  for (var i = 0u; i < vdbRefU.hemiCount; i = i + 1u) {
    let h = vdbRefU.hemis[i];
    e = e + select(h.ground.xyz, h.sky.xyz, dot(d, h.up.xyz) >= 0.0);
  }
  if ((vdbRefU.flags & ${VDB_REF_FLAG_EQUIRECT}u) != 0u) {
    let m = vec3f(dot(vdbRefU.envRotation0.xyz, d), dot(vdbRefU.envRotation1.xyz, d), dot(vdbRefU.envRotation2.xyz, d));
    let u = atan2(m.z, m.x) / (2.0 * VDB_REF_PI) + 0.5;
    let v = asin(clamp(m.y, -1.0, 1.0)) / VDB_REF_PI + 0.5;
    let size = vec2i(vdbRefU.envSize);
    let px = clamp(i32(floor(u * f32(size.x))), 0, size.x - 1);
    let py = clamp(i32(floor((1.0 - v) * f32(size.y))), 0, size.y - 1);
    e = e + textureLoad(vdbRefEnvMap, vec2i(px, py), 0).xyz * vdbRefU.uniformEnv.w;
  }
  return e;
}

// ---- one sample: the state machine (mode 0 primary T, 1 path segment, 2 light visibility) ----
fn vdbRefSample(pixel: vec2u, sampleIndex: u32) -> vec4f {
  vdbRefSeed(vdbRefU.seed, sampleIndex, pixel.y * vdbRefU.size.x + pixel.x);
  var j = vec2f(0.5, 0.5);
  if ((vdbRefU.flags & ${VDB_REF_FLAG_BOX_FILTER}u) != 0u) {
    j.x = vdbRefRand();
    j.y = vdbRefRand();
  }
  let size = vec2f(vdbRefU.size);
  let ndc = vec2f((f32(pixel.x) + j.x) / size.x * 2.0 - 1.0, (1.0 - (f32(pixel.y) + j.y) / size.y) * 2.0 - 1.0);
  let pNear = vdbRefUnproject(vec3f(ndc, 0.0));
  let pFar = vdbRefUnproject(vec3f(ndc, 1.0));
  let dv = normalize(pFar - pNear);
  var ov = vec3f(0.0);
  if ((vdbRefU.flags & ${VDB_REF_FLAG_ORTHOGRAPHIC}u) != 0u) {
    ov = vdbRefUnproject(vec3f(ndc, -1.0));
  }
  var o = (vdbRefU.cameraToWorld * vec4f(ov, 1.0)).xyz;
  var d = normalize((vdbRefU.cameraToWorld * vec4f(dv, 0.0)).xyz);

  var rad = vec3f(0.0);
  var beta = vec3f(1.0);
  var primaryT = 1.0;
  var mode = 0u;
  var depth = 0u;
  var lightIndex = 0u;
  var x = o;
  var vol = 0u;
  var neeDir = vec3f(0.0, 1.0, 0.0);
  var neeDist = 0.0;
  var neeLi = vec3f(0.0);
  loop {
    var ro = o;
    var rd = d;
    var tMax = VDB_REF_FAR;
    if (mode == 2u) {
      ro = x;
      rd = neeDir;
      tMax = neeDist;
    }
    let hit = vdbRefTrack(ro, rd, tMax, mode != 1u);
    if (mode == 0u) {
      primaryT = hit.trans;
      mode = 1u;
      continue;
    }
    if (mode == 2u) {
      rad = rad + beta * neeLi * hit.trans;
      lightIndex = lightIndex + 1u;
    } else {
      if (hit.escaped) {
        if (depth > 0u) {
          rad = rad + beta * vdbRefEnvironment(d);
        }
        break;
      }
      depth = depth + 1u;
      vol = hit.vol;
      x = o + hit.t * d;
      let uvw = vdbRefUvw(vol, x);
      let rec = vdbRefU.volumes[vol];
      let v = rec.ids.x;
      let albedo = rec.albedo.xyz * vdbSrcAlbedo(v, uvw, 0.0);
      var le = vec3f(0.0);
      if (rec.albedo.w > 0.5) {
        let temperature = mix(rec.temperature.x, rec.temperature.y, vdbSrcEmission(v, uvw, 0.0));
        if (temperature > 0.0) {
          le = rec.emission.xyz * pow(temperature / max(rec.temperature.y, 1.0), 4.0);
        }
      }
      rad = rad + beta * le;
      beta = beta * albedo;
      if (!(vdbRefMax3(beta) > 0.0)) {
        break;
      }
      lightIndex = 0u;
    }
    // The next light that reaches x (dark ones draw nothing), else finish the collision.
    var found = false;
    loop {
      if (lightIndex >= vdbRefU.lightCount) {
        break;
      }
      let nee = vdbRefLightAt(lightIndex, x);
      if (nee.ok) {
        neeDir = nee.dir;
        neeDist = nee.dist;
        neeLi = nee.li * vdbRefPhaseEval(dot(d, nee.dir), vdbRefU.volumes[vol].phase);
        found = true;
        break;
      }
      lightIndex = lightIndex + 1u;
    }
    if (found) {
      mode = 2u;
      continue;
    }
    if (depth >= vdbRefU.maxBounces) {
      break;
    }
    if (depth >= vdbRefU.rrStart) {
      let q = min(vdbRefMax3(beta), 1.0);
      if (q < 1.0) {
        if (vdbRefRand() >= q) {
          break;
        }
        beta = beta / q;
      }
    }
    let ph = vdbRefU.volumes[vol].phase;
    let lobe = vdbRefRand();
    let u1 = vdbRefRand();
    let u2 = vdbRefRand();
    let g = select(ph.x, ph.y, ph.z > 0.0 && lobe < ph.z);
    d = normalize(vdbRefDirectionAround(d, vdbRefSampleHgCos(g, u1), u2));
    o = x;
    mode = 1u;
  }
  return vec4f(rad, primaryT);
}

@compute @workgroup_size(8, 8, 1)
fn vdbRefTraceCs(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= vdbRefU.size.x || gid.y >= vdbRefU.size.y) {
    return;
  }
  var m = vec4f(0.0);
  for (var s = 0u; s < vdbRefU.spp; s = s + 1u) {
    m = m + vdbRefSample(gid.xy, vdbRefBatch.batch * vdbRefU.spp + s);
  }
  m = m / f32(vdbRefU.spp);
  var s0 = m;
  var s1 = m * m;
  if (vdbRefBatch.accumulate != 0u) {
    s0 = s0 + textureLoad(vdbRefAccumIn0, gid.xy, 0);
    s1 = s1 + textureLoad(vdbRefAccumIn1, gid.xy, 0);
  }
  textureStore(vdbRefAccumOut0, gid.xy, s0);
  textureStore(vdbRefAccumOut1, gid.xy, s1);
}
`;
}
