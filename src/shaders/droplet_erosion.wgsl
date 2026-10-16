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
@group(1) @binding(2) var scratchmap: texture_storage_2d<r32float, read_write>;  // A: edited IN PLACE, scatter reads it (scatter_target = 0)
@group(1) @binding(3) var scratchmap2: texture_storage_2d<r32float, read_write>; // B: unused here
@group(1) @binding(4) var<uniform> scratch_targets: ScratchTargetUniform;

const SENTINEL: f32 = 1e30;
const TWO_PI: f32 = 6.28318530718;

const DROPLET_STEPS: u32 = 40u; // hard cap per droplet, bounds divergence inside a wave
const DROPLET_EDGE_MARGIN_CELLS: f32 = 1.5; // the 4 corner texels stay inside the radius gather and scatter cover
const MAX_STEP_HEIGHT_CHANGE_CELLS: f32 = 0.02; // times cell_size, caps how much one step may raise or lower the terrain
const FINAL_DEPOSIT_CAP_CELLS: f32 = 0.1; // times cell_size, caps the leftover sediment a droplet leaves at the end
const FLAT_SLOPE_EPSILON: f32 = 0.001; // slope (height per horizontal distance) below which a droplet stops
const MAX_SPEED: f32 = 4.0;
const INERTIA: f32 = 0.05; // how much a droplet keeps its previous direction
const GRAVITY: f32 = 4.0;
const CAPACITY_FACTOR: f32 = 4.0;
const MIN_SLOPE: f32 = 0.01;
const ERODE_SPEED: f32 = 0.3;
const DEPOSIT_SPEED: f32 = 0.3;
const EVAPORATE_SPEED: f32 = 0.02;
const START_SPEED: f32 = 1.0;
const START_WATER: f32 = 1.0;

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Uniform random number in [0, 1].
fn next_random(random_state: ptr<function, u32>) -> f32 {
    *random_state = pcg_hash(*random_state);
    return f32(*random_state) * (1.0 / 4294967296.0);
}

// Spreads a height change over the 4 corners (order: x = (0,0), y = (1,0), z = (0,1), w = (1,1))
// and returns the new corner heights. Every corner is kept inside the height range the 4 corners
// had before the change: a droplet can only level a cell, never dig below its lowest corner
// (no new pits) or raise above its highest corner (no new spikes). A flat cell stays untouched,
// and the stores are skipped when nothing changed.
fn apply_height_change(cell: vec2<i32>, corner_heights: vec4<f32>, corner_weights: vec4<f32>, height_change: f32) -> vec4<f32> {
    let lowest_height = min(min(corner_heights.x, corner_heights.y), min(corner_heights.z, corner_heights.w));
    let highest_height = max(max(corner_heights.x, corner_heights.y), max(corner_heights.z, corner_heights.w));
    let new_heights = clamp(
        corner_heights + corner_weights * height_change,
        vec4<f32>(lowest_height),
        vec4<f32>(highest_height),
    );

    if any(new_heights != corner_heights) {
        textureStore(scratchmap, cell, vec4<f32>(new_heights.x, 0.0, 0.0, 0.0));
        textureStore(scratchmap, cell + vec2<i32>(1, 0), vec4<f32>(new_heights.y, 0.0, 0.0, 0.0));
        textureStore(scratchmap, cell + vec2<i32>(0, 1), vec4<f32>(new_heights.z, 0.0, 0.0, 0.0));
        textureStore(scratchmap, cell + vec2<i32>(1, 1), vec4<f32>(new_heights.w, 0.0, 0.0, 0.0));
    }
    return new_heights;
}

// One thread = one droplet. Dispatch: ceil(droplet_count / 64) workgroups along x.
@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    if intersection_data.hit_result.w <= 0.0 || intersection_data.hit_result.w >= SENTINEL {
        return; // no valid hit this frame
    }

    let cell_size_f32 = f32(terrain.cell_size);
    let radius_cell_space = brush.brush_radius / cell_size_f32;
    let effective_radius = radius_cell_space - DROPLET_EDGE_MARGIN_CELLS;
    if effective_radius <= 0.0 {
        return; // brush too small for droplets
    }

    // Same mapping as gather and scatter, but as a float position in scratch space.
    let scratch_center = vec2<i32>(i32(terrain.heights_per_region_axis / 2u));
    let brush_hit_texel = vec2<i32>(floor(intersection_data.brush_hit_cell_space.xz));
    let hit_scratch_position = vec2<f32>(scratch_center)
        + (intersection_data.brush_hit_cell_space.xz - vec2<f32>(brush_hit_texel));

    // Checked once per droplet instead of once per step: the whole droplet area must fit.
    let scratch_extent = f32(terrain.heights_per_region_axis);
    let droplet_reach = effective_radius + 2.0;
    if any(hit_scratch_position - vec2<f32>(droplet_reach) < vec2<f32>(0.0))
        || any(hit_scratch_position + vec2<f32>(droplet_reach) >= vec2<f32>(scratch_extent)) {
        return;
    }

    let max_step_height_change = MAX_STEP_HEIGHT_CHANGE_CELLS * cell_size_f32;
    let max_final_deposit = FINAL_DEPOSIT_CAP_CELLS * cell_size_f32;

    // frame_time is a frame duration, so mix in more changing values to make sure
    // every frame gets different droplets.
    let frame_seed = pcg_hash(
        bitcast<u32>(time.frame_time) ^ pcg_hash(
            bitcast<u32>(time.delta_time) ^ pcg_hash(
                bitcast<u32>(intersection_data.hit_result.x) ^ bitcast<u32>(intersection_data.hit_result.z)
            )
        )
    );
    var random_state = pcg_hash(global_invocation_id.x ^ frame_seed);

    // Uniform spawn inside the disc (sqrt keeps the density even).
    let spawn_angle = TWO_PI * next_random(&random_state);
    let spawn_distance = effective_radius * sqrt(next_random(&random_state));
    var position = hit_scratch_position + spawn_distance * vec2<f32>(cos(spawn_angle), sin(spawn_angle));

    var direction = vec2<f32>(0.0);
    var speed = START_SPEED;
    var water = START_WATER;
    var sediment = 0.0;

    for (var step_index = 0u; step_index < DROPLET_STEPS; step_index++) {
        let cell = vec2<i32>(floor(position));
        let fraction = position - vec2<f32>(cell);

        // The only loads of the step: 4 corners, fresh because this thread's own stores
        // of the previous step precede them in program order.
        let corner_heights = vec4<f32>(
            textureLoad(scratchmap, cell).r,
            textureLoad(scratchmap, cell + vec2<i32>(1, 0)).r,
            textureLoad(scratchmap, cell + vec2<i32>(0, 1)).r,
            textureLoad(scratchmap, cell + vec2<i32>(1, 1)).r,
        );

        let corner_weights = vec4<f32>(
            (1.0 - fraction.x) * (1.0 - fraction.y),
            fraction.x * (1.0 - fraction.y),
            (1.0 - fraction.x) * fraction.y,
            fraction.x * fraction.y,
        );

        // Bilinear gradient in world height per cell.
        let gradient = vec2<f32>(
            (corner_heights.y - corner_heights.x) * (1.0 - fraction.y) + (corner_heights.w - corner_heights.z) * fraction.y,
            (corner_heights.z - corner_heights.x) * (1.0 - fraction.x) + (corner_heights.w - corner_heights.y) * fraction.x,
        );

        direction = direction * INERTIA - gradient * (1.0 - INERTIA);
        let direction_length = length(direction);
        let slope_magnitude = length(gradient) / cell_size_f32;
        if slope_magnitude < FLAT_SLOPE_EPSILON || direction_length < 1e-6 {
            // Flat ground or a pit: leave the remaining sediment here (limited by the cell's
            // own height range, so flat ground stays flat) and stop.
            let stop_deposit_amount = min(sediment, max_final_deposit);
            if stop_deposit_amount > 0.0 {
                _ = apply_height_change(cell, corner_heights, corner_weights, stop_deposit_amount);
            }
            break;
        }
        direction = direction / direction_length;

        // First-order height change of this one-cell step. No second bilinear sample needed.
        let delta_height_world = dot(gradient, direction);
        let delta_slope = delta_height_world / cell_size_f32;

        let sediment_capacity = max(-delta_slope, MIN_SLOPE) * speed * water * CAPACITY_FACTOR * cell_size_f32;

        var height_change = 0.0; // wanted change: positive = deposit, negative = erode
        if delta_height_world > 0.0 || sediment > sediment_capacity {
            // Going uphill (fill, at most up to the step height) or carrying too much: deposit.
            var deposit_amount = (sediment - sediment_capacity) * DEPOSIT_SPEED;
            if delta_height_world > 0.0 {
                deposit_amount = min(delta_height_world, sediment);
            }
            height_change = min(deposit_amount, max_step_height_change);
        } else {
            // Erode, but never deeper than the height drop of this step or the per-step cap.
            height_change = -min(
                min((sediment_capacity - sediment) * ERODE_SPEED, -delta_height_world),
                max_step_height_change,
            );
        }

        // The range clamp may reduce the change, so the sediment follows what was really applied.
        var heights_after_step = corner_heights;
        if height_change != 0.0 {
            heights_after_step = apply_height_change(cell, corner_heights, corner_weights, height_change);
            sediment = max(sediment - dot(heights_after_step - corner_heights, vec4<f32>(1.0)), 0.0);
        }

        speed = min(sqrt(max(speed * speed - delta_slope * GRAVITY, 0.0)), MAX_SPEED);
        water = water * (1.0 - EVAPORATE_SPEED);

        let next_position = position + direction;
        let next_is_outside = distance(next_position, hit_scratch_position) > effective_radius;
        if next_is_outside || (step_index + 1u == DROPLET_STEPS) {
            // End of life: leave a capped amount of the leftover sediment. It can fill a pit up
            // to the rim of the cell but can no longer form a spike.
            let final_deposit_amount = min(sediment, max_final_deposit);
            if final_deposit_amount > 0.0 {
                _ = apply_height_change(cell, heights_after_step, corner_weights, final_deposit_amount);
            }
            break;
        }
        position = next_position;
    }
}