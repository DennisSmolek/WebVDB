// The samples of docs/site/core/rendering-without-three.mdx: the core frame renderer on raw WebGPU (demo 17's frame
// loop, reduced). The frame renderer, the sparse source and the receiver WGSL are internal until 0.2.0 (K6.8), so this
// file imports them by path, as demo 17 does; the regions shown on the page start after the imports.
import { VDBGridPool } from '@pmndrs/vdb';
import {
    vdbCreateFrameRenderer,
    vdbEmissionProbeWgsl,
    vdbFrameIntegrateEntries,
    vdbFrameIntegrateWgsl,
    vdbSparseSourceGpu,
    vdbVolumeShadowWgsl,
    type VDBFrameInput,
    type VDBFrameRendererGpu,
    type VDBFrameUpdate,
} from '../../../packages/vdb/src/core/internal.js';

interface Camera {
    view: number[];
    projection: number[];
    near: number;
    far: number;
}
declare function drawOpaque(encoder: GPUCommandEncoder, u: VDBFrameUpdate): void;
declare function present(encoder: GPUCommandEncoder): void;

export async function setup(adapter: GPUAdapter, cloudImage: Uint32Array, fireDensity: Uint32Array, fireTemperature: Uint32Array) {
    // #region frame-setup
    // The device is yours; the pool and the renderer record into encoders you submit.
    const device = await adapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter) });
    const pool = await VDBGridPool.create(device);

    // One sparse source serves every volume of one pool: one bind group, one dispatch per frame.
    const source = vdbSparseSourceGpu(pool, { types: ['fp8', 'float'] });
    const cloudSlot = source.volume({ density: pool.add(cloudImage) }, { image: cloudImage });
    const fireSlot = source.volume({ density: pool.add(fireDensity), emission: pool.add(fireTemperature) });

    // The renderer's composite pass draws into your colour target and reads your depth.
    const frame = vdbCreateFrameRenderer(device, { composite: { colorFormat: 'rgba16float', depthFormat: 'depth32float' } });
    Object.assign(frame.settings, { lighting: 'frame', castShadows: true, width: 256, height: 256, depth: 24 });

    // matrixWorld maps the unit box [-0.5, 0.5]^3 to the world (column-major).
    const cloud = frame.addVolume(cloudSlot, { matrixWorld: [200, 0, 0, 0, 0, 120, 0, 0, 0, 0, 160, 0, 0, 140, 0, 1], extinction: 40 });
    const fire = frame.addVolume(fireSlot, {
        matrixWorld: [60, 0, 0, 0, 0, 80, 0, 0, 0, 0, 60, 0, -120, 40, 0, 1],
        temperatureRange: [0, 4500],
        emissiveIntensity: 150,
        emissionColor: { spectrum: 'planck' },
    });
    await frame.prepare(); // optional: builds every pipeline up front, with the async creators
    // #endregion
    return { device, pool, source, frame, cloud, fire };
}

export function frameLoop(
    device: GPUDevice,
    pool: VDBGridPool,
    source: ReturnType<typeof vdbSparseSourceGpu>,
    frame: VDBFrameRendererGpu,
    camera: Camera,
    size: [number, number],
    targets: { color: GPUTexture; sceneColor: GPUTexture; depth: GPUTexture },
    sky: GPUTexture,
    sunDepth: GPUTexture,
    sunViewProjection: number[],
    sunRevision: number,
    frameId: number,
) {
    // #region frame-loop
    const encoder = device.createCommandEncoder();
    pool.update(encoder); // 1. your pool first (uploads, growth)
    source.update(encoder); // 2. the core source after it: pyramids, the volume table (same encoder)

    // 3. the core: shadow grid, transmittance, environment, volume shadow maps, emission probes, froxels
    const input: VDBFrameInput = {
        frameId,
        camera: { view: camera.view, projection: camera.projection, near: camera.near, far: camera.far },
        viewport: { drawingBuffer: size, canvas: size },
        lights: [{ id: 'sun', kind: 'directional', direction: [-0.6, -0.7, -0.4], color: [2.4, 2.3, 2.1], shadow: { size: 1024, view: 'fit' } }],
        environment: { id: 'sky', revision: 1, texture: sky, projection: 'equirect', intensity: 1 },
        occluders: [{ light: 'sun', depth: sunDepth.createView({ dimension: '2d-array' }), viewProjection: [sunViewProjection], revision: sunRevision }],
        emissionLight: { resolution: 16 },
    };
    const u = frame.update(encoder, input);

    // 4. your opaque passes, reading u.outputs (volume shadow maps, emission probes) and writing colour + depth
    drawOpaque(encoder, u);

    // 5. the volumes over them: the core's composite pass (his sRGB blend reads a copy of the colour)
    encoder.copyTextureToTexture({ texture: targets.color }, { texture: targets.sceneColor }, size);
    frame.render(encoder, {
        color: targets.color.createView(),
        size,
        depth: targets.depth.createView(),
        sceneColor: targets.sceneColor.createView(),
        clampColors: false, // you tone map afterwards
    });
    present(encoder); // tone map + sRGB into the canvas
    device.queue.submit([encoder.finish()]); // one update() per submitted encoder
    // #endregion
}

// #region frame-receivers
// A receiver's WGSL: the core's two receiver functions (they declare their structs) and your bindings.
const receiverWgsl = /* wgsl */ `
${vdbVolumeShadowWgsl()}
${vdbEmissionProbeWgsl()}
@group(1) @binding(0) var volumeShadowMap: texture_2d_array<f32>;
@group(1) @binding(1) var<uniform> volumeShadowView: VdbVolumeShadowView;
@group(1) @binding(2) var linearSampler: sampler;
@group(1) @binding(3) var emissionProbes: texture_3d<f32>;
@group(1) @binding(4) var<uniform> emissionGrid: VdbEmissionGrid;

fn lit(world: vec3f, n: vec3f, albedo: vec3f, sun: vec3f, toSun: vec3f, viewDepth: f32, fireIntensity: f32) -> vec3f {
    // volume -> mesh: the Beer map, read with the view the core drew it with
    let shadow = vdbVolumeShadow(volumeShadowMap, linearSampler, volumeShadowView, world, viewDepth);
    // the volumes' emission as irradiance (a Lambert surface: albedo * E / pi)
    let fire = vdbEmissionIrradiance(emissionProbes, linearSampler, emissionGrid, world, n) * fireIntensity / 3.14159265;
    return albedo * (sun * max(dot(n, toSun), 0.0) * shadow + fire);
}`;
// #endregion

export function receiverGroup(device: GPUDevice, layout: GPUBindGroupLayout, u: VDBFrameUpdate, linear: GPUSampler): GPUBindGroup | null {
    // #region frame-receiver-bind
    // Rebuild when u.outputs.version moves (a texture is a new object). Before the first map: bind placeholders
    // (a zeroed view buffer reads 1, a zeroed grid 0).
    const map = u.outputs.volumeShadows.get('sun');
    const probes = u.outputs.emissionProbes;
    if (!map || !probes) return null;
    return device.createBindGroup({
        layout,
        entries: [
            { binding: 0, resource: map.texture.createView({ dimension: '2d-array' }) },
            { binding: 1, resource: { buffer: map.view } },
            { binding: 2, resource: linear },
            { binding: 3, resource: probes.texture.createView({ dimension: '3d' }) },
            { binding: 4, resource: { buffer: probes.grid } },
        ],
    });
    // #endregion
}

export function hostComposite(device: GPUDevice, u: VDBFrameUpdate, linear: GPUSampler) {
    // #region frame-host-composite
    // Composite in your own fragment shader instead of frame.render(): bind the integrated froxels at a slot of
    // your choice and call vdbFrameIntegrate with the pixel's screen uv (bottom-left) and its view depth.
    const code = /* wgsl */ `
    ${vdbFrameIntegrateWgsl({ slot: { group: 1, firstBinding: 0 } })}
    fn overScene(sceneColor: vec3f, screenUv: vec2f, viewDepth: f32) -> vec3f {
        let i = vdbFrameIntegrate(screenUv, viewDepth);
        if (i.transmittance.y >= 1.0) {
            return sceneColor; // no volume in front of this pixel
        }
        return vdbFrameComposite(i, vec4f(sceneColor, 1.0), true).rgb; // his sRGB blend
    }`;
    const layout = device.createBindGroupLayout({ entries: vdbFrameIntegrateEntries({ group: 1, firstBinding: 0 }) });
    const group = device.createBindGroup({
        layout,
        entries: [
            { binding: 0, resource: u.outputs.froxels!.integrated.createView({ dimension: '3d' }) },
            { binding: 1, resource: linear },
            { binding: 2, resource: { buffer: u.outputs.uniforms } },
        ],
    });
    // #endregion
    return { code, receiverWgsl, group };
}
