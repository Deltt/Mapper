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

struct IntersectionData {
    hit_result: vec4<f32>,
    brush_hit_cell_space: vec4<f32>,
    brush_aabb_origin_texel: vec2<i32>,
    brush_aabb_end_texel_exclusive: vec2<i32>,
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

struct ScratchTargetUniform {
    scatter_target: i32,
}

@group(0) @binding(0) var<uniform> screen: ScreenUniform;
@group(0) @binding(1) var<uniform> time: TimeUniform;
@group(0) @binding(2) var<uniform> camera: CameraUniform;
@group(0) @binding(3) var<storage, read_write> intersection_data: IntersectionData;

@group(1) @binding(0) var<uniform> terrain: TerrainParametersUniform;
@group(1) @binding(1) var<uniform> brush: BrushUniform;
@group(1) @binding(2) var scratchmap: texture_storage_2d<r32float, read_write>;
@group(1) @binding(3) var scratchmap2: texture_storage_2d<r32float, read_write>;
@group(1) @binding(4) var<uniform> scratch_targets: ScratchTargetUniform;

// Dispatched once per edit, before the gathers:
// dispatch_workgroups(ceil(heights_per_region_axis / 8), ceil(heights_per_region_axis / 8), 1)
// Clears both scratch textures, which are region-sized.
@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    if any(global_invocation_id.xy >= vec2<u32>(terrain.heights_per_region_axis)) {
        return;
    }

    let scratch_texel = vec2<i32>(global_invocation_id.xy);
    textureStore(scratchmap, scratch_texel, vec4<f32>(0.0));
    textureStore(scratchmap2, scratch_texel, vec4<f32>(0.0));
}