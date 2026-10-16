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
@group(2) @binding(1) var heightmap: texture_storage_2d<r32float, write>;
@group(2) @binding(2) var normals: texture_storage_2d<r32uint, write>;

const SENTINEL: f32 = 1e30;
const SCATTER_MARGIN_TEXELS: i32 = 1; // 1 for the changed normals; their stencil is covered by the gather margin of 2
const CIRCLE_EPSILON_CELLS: f32 = 0.001;
const USE_SOBEL: bool = true; // true = Sobel (8 neighbors), false = central differences (4 neighbors)

fn is_texel_in_footprint(texel: vec2<i32>, hit_cell_space: vec2<f32>, radius_cells: f32, margin_texels: i32) -> bool {
    let nearest_point = clamp(
        vec2<i32>(round(hit_cell_space)),
        texel - vec2<i32>(margin_texels),
        texel + vec2<i32>(margin_texels),
    );
    let delta_cells = vec2<f32>(nearest_point) - hit_cell_space;
    let radius_with_slack = radius_cells + CIRCLE_EPSILON_CELLS;
    return dot(delta_cells, delta_cells) <= radius_with_slack * radius_with_slack;
}

fn load_scratch_height(scratch_texel: vec2<i32>) -> f32 {
    if scratch_targets.scatter_target == 0 {
        return textureLoad(scratchmap, scratch_texel).r;
    } else {
        return textureLoad(scratchmap2, scratch_texel).r;
    }
}

fn normal_from_slopes(slope_x: f32, slope_z: f32) -> vec3<f32> {
    return normalize(vec3<f32>(-slope_x, 1.0, -slope_z));
}

fn compute_normal_central_difference(scratch_texel: vec2<i32>, cell_size_f32: f32) -> vec3<f32> {
    let height_left = load_scratch_height(scratch_texel + vec2<i32>(-1, 0));
    let height_right = load_scratch_height(scratch_texel + vec2<i32>(1, 0));
    let height_down = load_scratch_height(scratch_texel + vec2<i32>(0, -1));
    let height_up = load_scratch_height(scratch_texel + vec2<i32>(0, 1));

    let slope_x = (height_right - height_left) * 0.5 / cell_size_f32;
    let slope_z = (height_up - height_down) * 0.5 / cell_size_f32;
    return normal_from_slopes(slope_x, slope_z);
}

fn compute_normal_sobel(scratch_texel: vec2<i32>, cell_size_f32: f32) -> vec3<f32> {
    let height_down_left = load_scratch_height(scratch_texel + vec2<i32>(-1, -1));
    let height_down = load_scratch_height(scratch_texel + vec2<i32>(0, -1));
    let height_down_right = load_scratch_height(scratch_texel + vec2<i32>(1, -1));
    let height_left = load_scratch_height(scratch_texel + vec2<i32>(-1, 0));
    let height_right = load_scratch_height(scratch_texel + vec2<i32>(1, 0));
    let height_up_left = load_scratch_height(scratch_texel + vec2<i32>(-1, 1));
    let height_up = load_scratch_height(scratch_texel + vec2<i32>(0, 1));
    let height_up_right = load_scratch_height(scratch_texel + vec2<i32>(1, 1));

    // Kernel weights sum to 4 per side, so a slope of s per texel gives 8 * s; divide by 8 to normalize.
    let sobel_x = (height_up_right + 2.0 * height_right + height_down_right)
                - (height_up_left + 2.0 * height_left + height_down_left);
    let sobel_z = (height_up_left + 2.0 * height_up + height_up_right)
                - (height_down_left + 2.0 * height_down + height_down_right);

    let slope_x = sobel_x / (8.0 * cell_size_f32);
    let slope_z = sobel_z / (8.0 * cell_size_f32);
    return normal_from_slopes(slope_x, slope_z);
}

fn compute_normal_scharr(scratch_texel: vec2<i32>, cell_size_f32: f32) -> vec3<f32> {
    let height_down_left = load_scratch_height(scratch_texel + vec2<i32>(-1, -1));
    let height_down = load_scratch_height(scratch_texel + vec2<i32>(0, -1));
    let height_down_right = load_scratch_height(scratch_texel + vec2<i32>(1, -1));
    let height_left = load_scratch_height(scratch_texel + vec2<i32>(-1, 0));
    let height_right = load_scratch_height(scratch_texel + vec2<i32>(1, 0));
    let height_up_left = load_scratch_height(scratch_texel + vec2<i32>(-1, 1));
    let height_up = load_scratch_height(scratch_texel + vec2<i32>(0, 1));
    let height_up_right = load_scratch_height(scratch_texel + vec2<i32>(1, 1));

    // Scharr weights 3/10/3: each side sums to 16, so a slope of s per texel gives 32 * s.
    let scharr_x = (3.0 * height_up_right + 10.0 * height_right + 3.0 * height_down_right)
                 - (3.0 * height_up_left + 10.0 * height_left + 3.0 * height_down_left);
    let scharr_z = (3.0 * height_up_left + 10.0 * height_up + 3.0 * height_up_right)
                 - (3.0 * height_down_left + 10.0 * height_down + 3.0 * height_down_right);

    let slope_x = scharr_x / (32.0 * cell_size_f32);
    let slope_z = scharr_z / (32.0 * cell_size_f32);
    return normal_from_slopes(slope_x, slope_z);
}

fn compute_normal(scratch_texel: vec2<i32>, cell_size_f32: f32) -> vec3<f32> {
    if USE_SOBEL {
        //return compute_normal_sobel(scratch_texel, cell_size_f32);
        return compute_normal_scharr(scratch_texel, cell_size_f32);
    }
    return compute_normal_central_difference(scratch_texel, cell_size_f32);
}

fn oct_encode_hemi(normal_in: vec3<f32>) -> vec2<f32> {
    let normal = normal_in / (abs(normal_in.x) + abs(normal_in.y) + abs(normal_in.z));
    return vec2<f32>(normal.x + normal.z, normal.z - normal.x) * 0.5 + vec2<f32>(0.5);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    if intersection_data.hit_result.w <= 0.0 || intersection_data.hit_result.w >= SENTINEL {
        return;
    }

    // Same overlap as gather, so the dispatch grid lines up.
    let overlap_origin_texel = max(intersection_data.brush_aabb_origin_texel, region.aabb_origin_texel);
    let overlap_end_texel_exclusive = min(intersection_data.brush_aabb_end_texel_exclusive, region.aabb_end_texel_exclusive);

    let overlap_size_texel = overlap_end_texel_exclusive - overlap_origin_texel;
    if any(vec2<i32>(global_invocation_id.xy) >= overlap_size_texel) {
        return;
    }

    let global_texel = overlap_origin_texel + vec2<i32>(global_invocation_id.xy);
    let radius_cell_space = brush.brush_radius / f32(terrain.cell_size);
    if !is_texel_in_footprint(global_texel, intersection_data.brush_hit_cell_space.xz, radius_cell_space, SCATTER_MARGIN_TEXELS) {
        return;
    }

    // Same mapping as gather.
    let scratch_center = vec2<i32>(i32(terrain.heights_per_region_axis / 2u));
    let brush_hit_texel = vec2<i32>(floor(intersection_data.brush_hit_cell_space.xz));
    let scratch_texel = scratch_center + (global_texel - brush_hit_texel);

    let cell_size_f32 = f32(terrain.cell_size);
    let height_center = load_scratch_height(scratch_texel);
    let normal = compute_normal(scratch_texel, cell_size_f32);
    let packed_normal = pack2x16unorm(oct_encode_hemi(normal));

    let region_local_texel = global_texel - region.aabb_origin_texel;
    textureStore(heightmap, region_local_texel, vec4<f32>(height_center, 0.0, 0.0, 0.0));
    textureStore(normals, region_local_texel, vec4<u32>(packed_normal, 0u, 0u, 0u));
}