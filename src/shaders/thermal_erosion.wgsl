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
@group(1) @binding(2) var scratchmap: texture_storage_2d<r32float, read_write>;  // A: input, written by gather
@group(1) @binding(3) var scratchmap2: texture_storage_2d<r32float, read_write>; // B: output, read by scatter
@group(1) @binding(4) var<uniform> scratch_targets: ScratchTargetUniform;

const SENTINEL: f32 = 1e30;
const FOOTPRINT_MARGIN_TEXELS: i32 = 2; // must equal the gather margin, the 8 neighbors stay inside it
const CIRCLE_EPSILON_CELLS: f32 = 0.001;
const TALUS_ANGLE_RADIANS: f32 = 0.55; // about 31 degrees, slopes steeper than this get eroded
const EROSION_RATE: f32 = 0.02; // 8 neighbors: keep <= 0.0625 for stability

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

// 0 at and beyond the brush edge.
fn texel_falloff(global_texel: vec2<i32>) -> f32 {
    let delta_cells = vec2<f32>(global_texel) - intersection_data.brush_hit_cell_space.xz;
    let distance_world = length(delta_cells) * f32(terrain.cell_size);
    return brush_falloff(distance_world, brush.brush_radius, brush.brush_fade, brush.brush_shape);
}

// Height flowing INTO the center texel from one neighbor (negative = flowing out).
// Antisymmetric: the neighbor computes the exact opposite for this pair, because the
// pair weight and the allowed height difference are symmetric. No flow crosses the brush edge.
fn erosion_flux(
    center_height: f32,
    center_falloff: f32,
    global_texel: vec2<i32>,
    scratch_texel: vec2<i32>,
    offset_texel: vec2<i32>,
    talus_height_per_cell: f32,
) -> f32 {
    let neighbor_height = textureLoad(scratchmap, scratch_texel + offset_texel).r;
    let pair_weight = min(center_falloff, texel_falloff(global_texel + offset_texel));
    // Diagonal neighbors are sqrt(2) cells away, so they may differ by sqrt(2) times more.
    let talus_height_difference = talus_height_per_cell * length(vec2<f32>(offset_texel));
    let height_difference = neighbor_height - center_height;
    let excess_height = max(abs(height_difference) - talus_height_difference, 0.0);
    return sign(height_difference) * excess_height * pair_weight;
}

// One thermal erosion step: reads scratch A, writes scratch B.
// EVERY texel of the gather footprint is written (copy-through outside the effect),
// because scatter reads its normal stencil from B. Run scatter with scatter_target = 1.
@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    if intersection_data.hit_result.w <= 0.0 || intersection_data.hit_result.w >= SENTINEL {
        return; // no valid hit this frame
    }

    let brush_aabb_size_texel = intersection_data.brush_aabb_end_texel_exclusive - intersection_data.brush_aabb_origin_texel;
    if any(vec2<i32>(global_invocation_id.xy) >= brush_aabb_size_texel) {
        return; // excess thread
    }

    let global_texel = intersection_data.brush_aabb_origin_texel + vec2<i32>(global_invocation_id.xy);
    let radius_cell_space = brush.brush_radius / f32(terrain.cell_size);
    if !is_texel_in_footprint(global_texel, intersection_data.brush_hit_cell_space.xz, radius_cell_space, FOOTPRINT_MARGIN_TEXELS) {
        return; // never read by scatter
    }

    // Same mapping as gather and scatter.
    let scratch_center = vec2<i32>(i32(terrain.heights_per_region_axis / 2u));
    let brush_hit_texel = vec2<i32>(floor(intersection_data.brush_hit_cell_space.xz));
    let scratch_texel = scratch_center + (global_texel - brush_hit_texel);

    let scratch_extent_texel = vec2<i32>(i32(terrain.heights_per_region_axis));
    if any(scratch_texel < vec2<i32>(0)) || any(scratch_texel >= scratch_extent_texel) {
        return; // footprint does not fit into the scratch
    }

    let center_height = textureLoad(scratchmap, scratch_texel).r;
    let center_falloff = texel_falloff(global_texel);

    if center_falloff <= 0.0 {
        textureStore(scratchmap2, scratch_texel, vec4<f32>(center_height, 0.0, 0.0, 0.0));
        return; // outside the brush or the terrain: copy through
    }

    // Allowed height difference per cell of horizontal distance, heights are in world units.
    let talus_height_per_cell = tan(TALUS_ANGLE_RADIANS) * f32(terrain.cell_size);

    var net_height_change = 0.0;
    for (var offset_z = -1; offset_z <= 1; offset_z++) {
        for (var offset_x = -1; offset_x <= 1; offset_x++) {
            if offset_x == 0 && offset_z == 0 {
                continue;
            }
            net_height_change += erosion_flux(
                center_height,
                center_falloff,
                global_texel,
                scratch_texel,
                vec2<i32>(offset_x, offset_z),
                talus_height_per_cell,
            );
        }
    }

    let result_height = center_height + EROSION_RATE * net_height_change;
    textureStore(scratchmap2, scratch_texel, vec4<f32>(result_height, 0.0, 0.0, 0.0));
}

fn brush_falloff(distance_to_hit: f32, radius: f32, fade: f32, mode: u32) -> f32 {
    let inner_radius = radius * (1.0 - fade);

    // 0 at the start of the falloff, 1 at the brush edge.
    let falloff_progress = clamp(
        (distance_to_hit - inner_radius) / max(radius - inner_radius, 1e-5),
        0.0,
        1.0
    );

    switch mode {
        case 1u: { // linear
            return 1.0 - falloff_progress;
        }

        case 2u: { // spherical: stays strong, then drops sharply
            return sqrt(1.0 - falloff_progress * falloff_progress);
        }

        case 3u: { // inverse spherical: drops quickly, then has a long tail
            let inverted_progress = 1.0 - falloff_progress;
            return 1.0 - sqrt(1.0 - inverted_progress * inverted_progress);
        }

        default: { // 0: gaussian-like, normalized so it hits exactly 0 at the edge
            let exponent_strength = 4.0;
            let value_at_edge = exp(-exponent_strength);
            return (exp(-exponent_strength * falloff_progress * falloff_progress) - value_at_edge)
                / (1.0 - value_at_edge);
        }
    }
}