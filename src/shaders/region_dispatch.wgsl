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

struct RegionParametersUniform {
    offset: vec4<f32>,
    index_2d: vec2<i32>,
    aabb_origin_texel: vec2<i32>,
    aabb_end_texel_exclusive: vec2<i32>,
    _pad0: vec2<i32>,
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

@group(2) @binding(0) var<uniform> region: RegionParametersUniform;
@group(2) @binding(1) var<storage, read_write> indirect_args: vec3<u32>;

const SENTINEL: f32 = 1e30;
const WORKGROUP_SIZE_XY: u32 = 8u; // must match the gather/edit @workgroup_size(8, 8, 1)

@compute @workgroup_size(1, 1, 1)
fn cs_main() {
    if intersection_data.hit_result.w <= 0.0 || intersection_data.hit_result.w >= SENTINEL {
        indirect_args = vec3<u32>(0u, 0u, 0u);
        return;
    }

    let overlap_origin_texel = max(intersection_data.brush_aabb_origin_texel, region.aabb_origin_texel);
    let overlap_end_texel_exclusive = min(intersection_data.brush_aabb_end_texel_exclusive, region.aabb_end_texel_exclusive);

    let overlap_size_texel = max(overlap_end_texel_exclusive - overlap_origin_texel, vec2<i32>(0));
    let overlap_size_texel_u32 = vec2<u32>(overlap_size_texel);

    let workgroup_count = (overlap_size_texel_u32 + vec2<u32>(WORKGROUP_SIZE_XY - 1u)) / vec2<u32>(WORKGROUP_SIZE_XY);
    indirect_args = vec3<u32>(workgroup_count.x, workgroup_count.y, 1u);
}