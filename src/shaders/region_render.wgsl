struct ScreenUniform {
    window_size_physical: vec2<f32>,
    window_size_logical: vec2<f32>,
    dpi_scale: f32,
    _pad0: f32,
}

struct TimeUniform {
    frame_time: f32,
    delta_time: f32,
}

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_position: vec4<f32>,
    xz_intersection: vec4<f32>,
    mouse_ray: vec4<f32>,
    zoom: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct TerrainParametersUniform {
    cells_per_chunk_axis: u32,
    cells_per_region_axis: u32,

    heights_per_chunk_axis: u32,
    heights_per_region_axis: u32,

    chunks_per_region_axis: u32,
    
    cell_size: u32,
    region_size: u32,
}

struct BrushUniform {
    brush_shape: u32,
    brush_radius: f32,
    brush_fade: f32,
    brush_strength: f32,
    brush_modifier: f32,
    brush_limit_enabled: u32,
}

@group(0) @binding(0) var<uniform> screen: ScreenUniform;
@group(0) @binding(1) var<uniform> time: TimeUniform;
@group(0) @binding(2) var<uniform> camera: CameraUniform;
@group(0) @binding(3) var<storage, read_write> hit_result: vec4<f32>;

@group(1) @binding(0) var<uniform> terrain: TerrainParametersUniform;
@group(1) @binding(1) var<uniform> brush: BrushUniform;

@group(2) @binding(0) var<uniform> region_offset: vec4<f32>;
@group(2) @binding(1) var heightmap: texture_2d<f32>;
@group(2) @binding(2) var normals: texture_2d<u32>;

const SENTINEL: f32 = 1e30;
const DEBUG_SHADING: bool = false;

// const GREEN = vec3<f32>(0.171, 0.296, 0.107); // sRGB (0.45, 0.58, 0.36)
// const BROWN = vec3<f32>(0.274, 0.163, 0.078); // sRGB (0.56, 0.44, 0.31)

const GREEN = vec3<f32>(0.20, 0.42, 0.10);
const BROWN = vec3<f32>(0.22, 0.13, 0.055);


const FOG_COLOR = vec3<f32>(0.343, 0.448, 0.554); // sRGB (0.62, 0.70, 0.77)
const FOG_DENSITY: f32 = 0.002;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) height: f32,
    @location(1) world_position: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let local_x = vertex_index % terrain.heights_per_chunk_axis;
    let local_z = vertex_index / terrain.heights_per_chunk_axis;

    let chunk_x = instance_index % terrain.chunks_per_region_axis;
    let chunk_z = instance_index / terrain.chunks_per_region_axis;

    let texel_x = chunk_x * terrain.cells_per_chunk_axis + local_x;
    let texel_z = chunk_z * terrain.cells_per_chunk_axis + local_z;

    let height = textureLoad(heightmap, vec2<i32>(i32(texel_x), i32(texel_z)), 0).r;

    let world_position = vec4<f32>(
        f32(texel_x * terrain.cell_size) + region_offset.x,
        height,
        f32(texel_z * terrain.cell_size) + region_offset.z,
        1.0,
    );

    let packed = textureLoad(normals, vec2<i32>(i32(texel_x), i32(texel_z)), 0).r;
    let enc = unpack2x16unorm(packed);
    let normal = oct_decode_hemi(enc);

    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_position;
    out.height = height;
    out.world_position = world_position.xyz;
    out.world_normal = normal;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color: vec3<f32>;
    let normal = normalize(in.world_normal);
    let light_dir = normalize(vec3<f32>(0.55, 0.75, 0.45));
    let normal_dot_light = max(dot(normal, light_dir), 0.0);
    let hemi = mix(0.18, 0.38, normal.y);
    let diffuse = normal_dot_light * 0.65;
    let intensity = clamp(hemi + diffuse, 0.0, 1.0);

    let base_color = vec3<f32>(0.50, 0.49, 0.43);
    color = base_color * intensity;

    if hit_result.w > 0.0 && hit_result.w < SENTINEL {
    //if camera.xz_intersection.w > 0.0 {
        let smooth_brush = smooth_brush(in, color);
        color = mix(color, smooth_brush, 0.5);
    }

    return vec4<f32>(color, 1.0);
}

fn smooth_brush(in: VertexOutput, base_color: vec3<f32>) -> vec3<f32> {
    let delta = in.world_position.xz - hit_result.xz;
    let dist_sq = dot(delta, delta);

    let radius = brush.brush_radius;
    let radius_sq = radius * radius;

    if dist_sq > radius_sq {
        return base_color;
    }

    let dist = sqrt(dist_sq);

    let falloff = brush_falloff(
        dist,
        radius,
        brush.brush_fade,
        brush.brush_shape
    );

    //let brush_color = vec3<f32>(0.05, 0.20, 1.0);
    let brush_color = vec3<f32>(0.03, 0.20, 1.0);
    let preview_strength = 0.12 + falloff * 0.78;

    var color = mix(
        base_color,
        brush_color,
        preview_strength
    );

    let cell_size = f32(terrain.cell_size);

    // Ring width is a fraction of the radius, with only a small floor for
    // tiny brushes. min() last, so it can never exceed 5% of the radius.
    let ring_width = min(max(radius * 0.015, cell_size * 0.1), radius * 0.05);

    let outer_ring = 1.0 - smoothstep(0.0, ring_width, abs(dist - radius));
    color = mix(color, vec3<f32>(0.26, 0.79, 1.0), outer_ring * 0.9);

    let inner_radius = radius * (1.0 - brush.brush_fade);

    if inner_radius > ring_width {
        let inner_ring = 1.0 - smoothstep(0.0, ring_width, abs(dist - inner_radius));
        color = mix(color, vec3<f32>(1.0, 0.69, 0.17), inner_ring * 0.9);
    }

    return color;
}

// fn smooth_brush(in: VertexOutput, base_color: vec3<f32>) -> vec3<f32> {
//     var color = base_color;
//     let delta = in.world_position.xz - hit_result.xz;
//     //let delta = in.world_position.xz - camera.xz_intersection.xz;
//     let dist_sq = dot(delta, delta);
//     let radius_sq = brush.brush_radius * brush.brush_radius;

//     if dist_sq <= radius_sq {
//         let dist = sqrt(dist_sq);

//         let t = 1.0 - brush_falloff(dist, brush.brush_radius, brush.brush_fade, brush.brush_shape);

//         let inner_color = vec3<f32>(0.0, 0.033, 1.0);
//         let outer_color = vec3<f32>(0.0, 0.033, 1.0);
//         var brush_color = mix(inner_color, outer_color, t);

//         brush_color = mix(brush_color, color, t);
//         color = mix(brush_color, color, 0.3);
//     }

//     return color;
// }

fn oct_decode_hemi(e_in: vec2<f32>) -> vec3<f32> {
    let e = e_in * 2.0 - vec2<f32>(1.0);
    let x = (e.x - e.y) * 0.5;
    let z = (e.x + e.y) * 0.5;
    let y = 1.0 - abs(x) - abs(z);
    return normalize(vec3<f32>(x, y, z));
}

fn brush_falloff(dist: f32, radius: f32, fade: f32, mode: u32) -> f32 {
    let inner = radius * (1.0 - fade);

    // 0 at the start of the falloff, 1 at the brush edge.
    let s = clamp(
        (dist - inner) / max(radius - inner, 1e-5),
        0.0,
        1.0
    );

    switch mode {
        case 1u: { // linear
            return 1.0 - s;
        }

        case 2u: { // spherical: stays strong, then drops sharply
            return sqrt(1.0 - s * s);
        }

        case 3u: { // inverse spherical: drops quickly, then has a long tail
            let r = 1.0 - s;
            return 1.0 - sqrt(1.0 - r * r);
        }

        default: { // 0: smoothstep
            //return 1.0 - smoothstep(0.0, 1.0, s);
            let k = 4.0;
            let e = exp(-k);
            return (exp(-k * s * s) - e) / (1.0 - e); // normalized so it hits exactly 0 at s=1
        }
    }
}

