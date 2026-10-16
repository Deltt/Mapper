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
@group(2) @binding(1) var heightmap: texture_2d<f32>;
@group(2) @binding(2) var normals: texture_2d<u32>;

const SENTINEL: f32 = 1e30;
const MARGIN_TEXELS: i32 = 2;
const CIRCLE_EPSILON_CELLS: f32 = 0.001;

fn is_texel_in_footprint(texel: vec2<i32>, hit_cell_space: vec2<f32>, radius_cells: f32) -> bool {
    let nearest_point = clamp(
        vec2<i32>(round(hit_cell_space)),
        texel - vec2<i32>(MARGIN_TEXELS),
        texel + vec2<i32>(MARGIN_TEXELS),
    );
    let delta_cells = vec2<f32>(nearest_point) - hit_cell_space;
    let radius_with_slack = radius_cells + CIRCLE_EPSILON_CELLS;
    return dot(delta_cells, delta_cells) <= radius_with_slack * radius_with_slack;
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    if intersection_data.hit_result.w <= 0.0 || intersection_data.hit_result.w >= SENTINEL {
        return;
    }

    let overlap_origin_texel = max(intersection_data.brush_aabb_origin_texel, region.aabb_origin_texel);
    let overlap_end_texel_exclusive = min(intersection_data.brush_aabb_end_texel_exclusive, region.aabb_end_texel_exclusive);

    let overlap_size_texel = overlap_end_texel_exclusive - overlap_origin_texel;
    if any(vec2<i32>(global_invocation_id.xy) >= overlap_size_texel) {
        return;
    }

    let global_texel = overlap_origin_texel + vec2<i32>(global_invocation_id.xy);
    let radius_cell_space = brush.brush_radius / f32(terrain.cell_size);
    if !is_texel_in_footprint(global_texel, intersection_data.brush_hit_cell_space.xz, radius_cell_space) {
        return;
    }

    let region_local_texel = global_texel - region.aabb_origin_texel;
    let height = textureLoad(heightmap, region_local_texel, 0).r;

    let scratch_center = vec2<i32>(i32(terrain.heights_per_region_axis / 2u));
    let brush_hit_texel = vec2<i32>(floor(intersection_data.brush_hit_cell_space.xz));
    let scratch_texel = scratch_center + (global_texel - brush_hit_texel);
    textureStore(scratchmap, scratch_texel, vec4<f32>(height, 0.0, 0.0, 0.0));
}