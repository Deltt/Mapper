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

const SENTINEL: f32 = 1e30;
const DIAGONAL_STEP_FACTOR: f32 = 1.4143; // sqrt(2), upper bound for a diagonal crossing
const EXTRA_STEPS: u32 = 2u;              // initial sample + final partial step
const MAX_INTERSECT_DISTANCE: f32 = 2000.0;
const AABB_TEXEL_MARGIN: u32 = 2u;

fn sample_height_bilinear(texel_position: vec2<f32>, max_texel: vec2<i32>) -> f32 {
    let base_texel = vec2<i32>(floor(texel_position));
    let blend = fract(texel_position);
    let min_texel = vec2<i32>(0, 0);

    let texel_x0_z0 = clamp(base_texel, min_texel, max_texel);
    let texel_x1_z0 = clamp(base_texel + vec2<i32>(1, 0), min_texel, max_texel);
    let texel_x0_z1 = clamp(base_texel + vec2<i32>(0, 1), min_texel, max_texel);
    let texel_x1_z1 = clamp(base_texel + vec2<i32>(1, 1), min_texel, max_texel);

    let height_x0_z0 = textureLoad(heightmap, texel_x0_z0, 0).r;
    let height_x1_z0 = textureLoad(heightmap, texel_x1_z0, 0).r;
    let height_x0_z1 = textureLoad(heightmap, texel_x0_z1, 0).r;
    let height_x1_z1 = textureLoad(heightmap, texel_x1_z1, 0).r;

    let height_along_x_at_z0 = mix(height_x0_z0, height_x1_z0, blend.x);
    let height_along_x_at_z1 = mix(height_x0_z1, height_x1_z1, blend.x);
    return mix(height_along_x_at_z0, height_along_x_at_z1, blend.y);
}

fn ray_minus_terrain_height(
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    ray_distance: f32,
    region_min: vec2<f32>,
    cell_size_f32: f32,
    max_texel: vec2<i32>,
) -> f32 {
    let ray_position = ray_origin + ray_direction * ray_distance;
    let terrain_height = sample_height_bilinear((ray_position.xz - region_min) / cell_size_f32, max_texel);
    return ray_position.y - terrain_height;
}

fn write_hit_if_closer(ray_origin: vec3<f32>, ray_direction: vec3<f32>, hit_distance: f32) {
    if hit_distance < intersection_data.hit_result.w {
        let hit_position = ray_origin + ray_direction * hit_distance;

        let cell_size_f32 = f32(terrain.cell_size);
        let brush_hit_cell_space = vec4<f32>(hit_position / cell_size_f32, 0.0);

        let extended_radius = brush.brush_radius + f32(AABB_TEXEL_MARGIN) * cell_size_f32;
        let extended_radius_cell_space = extended_radius / cell_size_f32;
        let brush_aabb_origin_texel = vec2<i32>(ceil(brush_hit_cell_space.xz - vec2<f32>(extended_radius_cell_space)));
        let brush_aabb_end_texel_exclusive = vec2<i32>(floor(brush_hit_cell_space.xz + vec2<f32>(extended_radius_cell_space))) + 1;

        intersection_data.hit_result = vec4<f32>(hit_position.x, hit_position.y, hit_position.z, hit_distance);
        intersection_data.brush_hit_cell_space = brush_hit_cell_space;
        intersection_data.brush_aabb_origin_texel = brush_aabb_origin_texel;
        intersection_data.brush_aabb_end_texel_exclusive = brush_aabb_end_texel_exclusive;
    }
}

@compute @workgroup_size(1, 1, 1)
fn cs_main() {
    // Dispatches are serialized, so no one else writes intersection_data.hit_result during this dispatch.
    let closest_hit_distance = intersection_data.hit_result.w;

    let ray_origin = camera.view_position.xyz;
    let ray_direction = normalize(camera.mouse_ray.xyz);

    let region_min = region.offset.xz;
    let region_max = region.offset.xz + vec2<f32>(f32(terrain.region_size));
    let max_texel = vec2<i32>(i32(terrain.heights_per_region_axis) - 1);
    let cell_size_f32 = f32(terrain.cell_size);

    // Clip the ray to the region's XZ box (slab test).
    var enter_distance = 0.0;
    //var exit_distance = SENTINEL;
    var exit_distance = MAX_INTERSECT_DISTANCE;

    if abs(ray_direction.x) > 1e-6 {
        let distance_to_min_x = (region_min.x - ray_origin.x) / ray_direction.x;
        let distance_to_max_x = (region_max.x - ray_origin.x) / ray_direction.x;
        enter_distance = max(enter_distance, min(distance_to_min_x, distance_to_max_x));
        exit_distance = min(exit_distance, max(distance_to_min_x, distance_to_max_x));
    } else if ray_origin.x < region_min.x || ray_origin.x > region_max.x {
        return;
    }

    if abs(ray_direction.z) > 1e-6 {
        let distance_to_min_z = (region_min.y - ray_origin.z) / ray_direction.z;
        let distance_to_max_z = (region_max.y - ray_origin.z) / ray_direction.z;
        enter_distance = max(enter_distance, min(distance_to_min_z, distance_to_max_z));
        exit_distance = min(exit_distance, max(distance_to_min_z, distance_to_max_z));
    } else if ray_origin.z < region_min.y || ray_origin.z > region_max.y {
        return;
    }

    enter_distance = max(enter_distance, 0.0);
    // Nothing beyond a closer hit found by an earlier region can matter.
    exit_distance = min(exit_distance, closest_hit_distance);

    if enter_distance >= exit_distance {
        return; // misses the region, or can't beat the closest hit
    }

    var previous_height_difference = ray_minus_terrain_height(
        ray_origin, ray_direction, enter_distance, region_min, cell_size_f32, max_texel);

    if previous_height_difference <= 0.0 {
        return; // starts below the surface: no hit (original behavior)
    }

    let step_distance = cell_size_f32 / max(length(ray_direction.xz), 1e-6);
    let max_steps = u32(ceil(DIAGONAL_STEP_FACTOR * f32(terrain.cells_per_region_axis))) + EXTRA_STEPS;
    var current_distance = enter_distance;

    for (var step_index: u32 = 0u; step_index < max_steps; step_index = step_index + 1u) {
        let previous_distance = current_distance;
        current_distance = min(current_distance + step_distance, exit_distance);

        let height_difference = ray_minus_terrain_height(
            ray_origin, ray_direction, current_distance, region_min, cell_size_f32, max_texel);

        if height_difference <= 0.0 {
            // Crossed between the previous and current sample: refine linearly.
            let difference_span = previous_height_difference - height_difference;
            let refine_fraction = select(0.0, previous_height_difference / difference_span, difference_span > 1e-6);
            write_hit_if_closer(ray_origin, ray_direction, mix(previous_distance, current_distance, refine_fraction));
            return;
        }

        previous_height_difference = height_difference;

        if current_distance >= exit_distance {
            return; // reached the region border (or the closest known hit) without a crossing
        }
    }
}