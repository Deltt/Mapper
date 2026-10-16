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
@group(1) @binding(3) var scratchmap2: texture_storage_2d<r32float, read_write>; // B: output, read by scatter (scatter_target = 1)
@group(1) @binding(4) var<uniform> scratch_targets: ScratchTargetUniform;

const SENTINEL: f32 = 1e30;
const FOOTPRINT_MARGIN_TEXELS: i32 = 2; // must equal the gather margin, the Sobel stencil stays inside it
const CIRCLE_EPSILON_CELLS: f32 = 0.001;

const SLOPE_MIN: f32 = 0.15; // below this slope (about 8.5 degrees) nothing is carved
const SLOPE_FULL: f32 = 0.6; // from this slope (about 31 degrees) on, carving is at full strength
const NOISE_FREQUENCY: f32 = 0.15; // noise lattice cells per texel, gully width is about 1 / frequency texels
const STREAK_HALF_SAMPLES: i32 = 6; // samples on each side of the texel along the flow
const STREAK_STEP_TEXELS: f32 = 2.0; // streak length is about 2 * half_samples * step texels
const GROOVE_CONTRAST: f32 = 3.0; // averaging along the flow flattens the noise, this restores contrast
const GULLY_STRENGTH_SCALE: f32 = 0.1; // multiplies brush_strength * frame_time, like the raise brush

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

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

fn lattice_value(lattice_point: vec2<i32>) -> f32 {
    let hashed = pcg_hash(bitcast<u32>(lattice_point.x) * 1597334677u ^ bitcast<u32>(lattice_point.y) * 3812015801u);
    return f32(hashed) * (1.0 / 4294967296.0);
}

// Smooth value noise in [0, 1], a function of the absolute position only.
fn value_noise(position: vec2<f32>) -> f32 {
    let lattice_cell = vec2<i32>(floor(position));
    let fraction = position - vec2<f32>(lattice_cell);
    let smooth_fraction = fraction * fraction * fraction * (fraction * (fraction * 6.0 - 15.0) + 10.0);

    let value_00 = lattice_value(lattice_cell);
    let value_10 = lattice_value(lattice_cell + vec2<i32>(1, 0));
    let value_01 = lattice_value(lattice_cell + vec2<i32>(0, 1));
    let value_11 = lattice_value(lattice_cell + vec2<i32>(1, 1));
    return mix(mix(value_00, value_10, smooth_fraction.x), mix(value_01, value_11, smooth_fraction.x), smooth_fraction.y);
}

// Noise averaged along the downhill line: the features get stretched along the flow, which
// gives channels that follow the slope. The noise itself is isotropic and absolute, so
// there are no phase jumps when the slope direction changes.
fn streak_noise(texel_position: vec2<f32>, downhill_direction: vec2<f32>) -> f32 {
    var weighted_sum = 0.0;
    var weight_sum = 0.0;
    for (var sample_index = -STREAK_HALF_SAMPLES; sample_index <= STREAK_HALF_SAMPLES; sample_index++) {
        let weight = 1.0 - abs(f32(sample_index)) / f32(STREAK_HALF_SAMPLES + 1);
        let sample_position = texel_position + downhill_direction * (f32(sample_index) * STREAK_STEP_TEXELS);
        weighted_sum += weight * value_noise(sample_position * NOISE_FREQUENCY);
        weight_sum += weight;
    }
    return weighted_sum / weight_sum;
}

fn load_height(scratch_texel: vec2<i32>) -> f32 {
    return textureLoad(scratchmap, scratch_texel).r;
}

// Stateless gully carving: reads scratch A, writes scratch B.
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

    let center_height = load_height(scratch_texel);
    let center_falloff = texel_falloff(global_texel);
    if center_falloff <= 0.0 {
        textureStore(scratchmap2, scratch_texel, vec4<f32>(center_height, 0.0, 0.0, 0.0));
        return; // outside the brush: copy through
    }

    // Sobel gradient in world height per cell. "up" is the texel y + 1 neighbor, which is world +z.
    let height_down_left = load_height(scratch_texel + vec2<i32>(-1, -1));
    let height_down = load_height(scratch_texel + vec2<i32>(0, -1));
    let height_down_right = load_height(scratch_texel + vec2<i32>(1, -1));
    let height_left = load_height(scratch_texel + vec2<i32>(-1, 0));
    let height_right = load_height(scratch_texel + vec2<i32>(1, 0));
    let height_up_left = load_height(scratch_texel + vec2<i32>(-1, 1));
    let height_up = load_height(scratch_texel + vec2<i32>(0, 1));
    let height_up_right = load_height(scratch_texel + vec2<i32>(1, 1));

    let sobel_x = (height_up_right + 2.0 * height_right + height_down_right)
                - (height_up_left + 2.0 * height_left + height_down_left);
    let sobel_z = (height_up_left + 2.0 * height_up + height_up_right)
                - (height_down_left + 2.0 * height_down + height_down_right);
    let gradient = vec2<f32>(sobel_x, sobel_z) / 8.0;
    let gradient_length = length(gradient);

    let slope_magnitude = gradient_length / f32(terrain.cell_size);
    let slope_mask = smoothstep(SLOPE_MIN, SLOPE_FULL, slope_magnitude);
    if slope_mask <= 0.0 {
        textureStore(scratchmap2, scratch_texel, vec4<f32>(center_height, 0.0, 0.0, 0.0));
        return; // too flat: untouched
    }

    let downhill_direction = -gradient / gradient_length;
    let streak_value = streak_noise(vec2<f32>(global_texel), downhill_direction);

    // Low noise values become grooves. Squaring gives sharper channels and flatter ridges between them.
    let channel_value = clamp((streak_value - 0.5) * GROOVE_CONTRAST + 0.5, 0.0, 1.0);
    let groove_depth = (1.0 - channel_value) * (1.0 - channel_value);

    let height_change = -brush.brush_strength * time.frame_time * GULLY_STRENGTH_SCALE
        * center_falloff * slope_mask * groove_depth;

    textureStore(scratchmap2, scratch_texel, vec4<f32>(center_height + height_change, 0.0, 0.0, 0.0));
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