use glam::{EulerRot, IVec2, Mat4, Quat, Vec3, camera};
use wgpu::util::DeviceExt;
pub use wgpu::*;
use winit::{
    dpi::{LogicalSize, PhysicalSize},
    window::Window,
};
use {glam::Vec2, glam::Vec4};

const DEFAULT_FPS: f32 = 60.0;

pub struct Globals {
    pub globals_bgl: BindGroupLayout,
    pub globals_bg: BindGroup,

    pub screen_uniform: ScreenUniform,
    pub screen_uniform_buffer: Buffer,
    pub screen_uniform_dirty: bool,

    pub time_uniform: TimeUniform,
    pub time_uniform_buffer: Buffer,
    pub time_uniform_dirty: bool,

    pub camera_handle: CameraHandle,
    pub camera_uniform: CameraUniform,
    pub camera_uniform_buffer: Buffer,

    pub intersection_data_buffer: Buffer,
}

impl Globals {
    pub fn new(device: &Device, window: &Window) -> Self {
        let window_size_physical = window.inner_size();
        let scale_factor = window.scale_factor();
        let window_size_logical: LogicalSize<f32> = window_size_physical.to_logical(scale_factor);

        let screen_uniform = ScreenUniform {
            window_size_physical: [
                window_size_physical.width as f32,
                window_size_physical.height as f32,
            ],
            window_size_logical: [
                window_size_logical.width as f32,
                window_size_logical.height as f32,
            ],
            dpi_scale: scale_factor as f32,
            _pad0: 0.0,
        };

        let time_uniform = TimeUniform {
            frame_time: get_monitor_frame_time(window),
            delta_time: 0.0,
        };

        let mut camera_handle = CameraHandle {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            xz_intersection: Vec4::ZERO,
            mouse_ray: Vec3::new(0.0, 0.0, -1.0),
            zoom: 0.0,
            fov_y_radians: 60.0_f32.to_radians(),
            aspect: window_size_physical.width as f32 / window_size_physical.height as f32,
            near: 0.01,
            far: 4000.0,
            camera_speed: 400.0,
            yaw: 0.0,
            pitch: 0.0,
            camera_uniform_dirty: false,
        };
        camera_handle.position = Vec3::new(0.0, 200.0, 0.0);
        camera_handle.rotate_degrees(-135.0, -45.0);

        let camera_uniform = camera_handle.to_uniform();

        let (
            globals_bgl,
            globals_bg,
            screen_uniform_buffer,
            time_uniform_buffer,
            camera_uniform_buffer,
            intersection_data_buffer,
        ) = Self::create_gpu_resources(device, &screen_uniform, &time_uniform, &camera_uniform);

        Self {
            globals_bgl,
            globals_bg,

            screen_uniform,
            screen_uniform_buffer,
            screen_uniform_dirty: false,

            time_uniform,
            time_uniform_buffer,
            time_uniform_dirty: false,

            camera_handle,
            camera_uniform,
            camera_uniform_buffer,

            intersection_data_buffer,
        }
    }

    pub fn rotate_camera_y(&mut self, rotation_input: Vec2) {
        let rotation = rotation_input * 8.0 * self.time_uniform.frame_time;
        self.camera_handle.rotate_degrees(-rotation.x, -rotation.y);

        self.camera_handle.camera_uniform_dirty = true;
    }

    pub fn orbit_camera_y(&mut self, rotation_input_x: f32) {
        let rotation_degrees = -rotation_input_x * 8.0 * self.time_uniform.frame_time;

        let forward = self.camera_handle.rotation * Vec3::NEG_Z;

        if forward.y.abs() < f32::EPSILON {
            return;
        }
        let t = -self.camera_handle.position.y / forward.y;
        if t <= 0.0 {
            return;
        }
        let pivot = self.camera_handle.position + forward * t;

        let rotation = Quat::from_rotation_y(rotation_degrees.to_radians());
        let offset = self.camera_handle.position - pivot;
        self.camera_handle.position = pivot + rotation * offset;

        self.camera_handle.rotate_degrees(rotation_degrees, 0.0);

        self.camera_handle.camera_uniform_dirty = true;
    }

    pub fn move_camera_xz(&mut self, movement_input: Vec3) {
        let mut relative_movement_input = self.camera_handle.rotation * movement_input;
        relative_movement_input.y = 0.0;
        relative_movement_input = relative_movement_input.normalize_or_zero();
        relative_movement_input *= self.camera_handle.camera_speed;
        self.camera_handle.position += relative_movement_input * self.time_uniform.frame_time;
        self.camera_handle.camera_uniform_dirty = true;
    }

    pub fn zoom_camera_xz(&mut self, zoom_input: f32) {
        let relative_movement_input = self.camera_handle.forward() * zoom_input * 700.0;
        self.camera_handle.position += relative_movement_input * self.time_uniform.frame_time;
        self.camera_handle.camera_uniform_dirty = true;
    }

    pub fn update_screen(&mut self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.screen_uniform_buffer,
            0,
            bytemuck::bytes_of(&self.screen_uniform),
        );
        self.screen_uniform_dirty = false;
    }

    pub fn update_time(&mut self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.time_uniform_buffer,
            0,
            bytemuck::bytes_of(&self.time_uniform),
        );
        self.time_uniform_dirty = false;
    }

    pub fn update_camera(
        &mut self,
        queue: &wgpu::Queue,
        surface_config: &SurfaceConfiguration,
        mouse_position: Vec2,
    ) {
        self.camera_handle.mouse_ray = self.mouse_ray_direction(mouse_position, surface_config);

        self.camera_handle.xz_intersection =
            self.mouse_ground_intersection(mouse_position, &surface_config, 3000.0);

        self.camera_uniform = self.camera_handle.to_uniform();

        queue.write_buffer(
            &self.camera_uniform_buffer,
            0,
            bytemuck::bytes_of(&self.camera_uniform),
        );
        self.camera_handle.camera_uniform_dirty = false;
    }

    pub fn mouse_ray_direction(
        &self,
        mouse_position: Vec2,
        surface_config: &SurfaceConfiguration,
    ) -> Vec3 {
        let ndc_x = (mouse_position.x / surface_config.width as f32) * 2.0 - 1.0;
        let ndc_y = 1.0 - (mouse_position.y / surface_config.height as f32) * 2.0;

        let tan_half_fovy = (self.camera_handle.fov_y_radians * 0.5).tan();
        let view_dir = Vec3::new(
            ndc_x * tan_half_fovy * self.camera_handle.aspect,
            ndc_y * tan_half_fovy,
            -1.0,
        );

        let forward = (self.camera_handle.rotation * view_dir).normalize();

        Vec3::new(forward.x, forward.y, forward.z)
    }

    pub fn mouse_ground_intersection(
        &self,
        mouse_position: Vec2,
        surface_config: &SurfaceConfiguration,
        max_distance: f32,
    ) -> Vec4 {
        const EPS: f32 = 1e-5;

        let ndc_x = (mouse_position.x / surface_config.width as f32) * 2.0 - 1.0;
        let ndc_y = 1.0 - (mouse_position.y / surface_config.height as f32) * 2.0;

        let tan_half_fovy = (self.camera_handle.fov_y_radians * 0.5).tan();
        let view_dir = Vec3::new(
            ndc_x * tan_half_fovy * self.camera_handle.aspect,
            ndc_y * tan_half_fovy,
            -1.0,
        );

        let origin = self.camera_handle.position;
        let forward = (self.camera_handle.rotation * view_dir).normalize();

        if forward.y.abs() < EPS {
            return Vec4::ZERO;
        }

        let t = -origin.y / forward.y;

        if t < 0.0 || t > max_distance {
            return Vec4::ZERO;
        }

        let hit = origin + forward * t;
        Vec4::new(hit.x, 0.0, hit.z, 1.0)
    }

    pub fn sync_monitor_frame_time(&mut self, window: &Window) {
        let new_frame_time = get_monitor_frame_time(window);
        if self.time_uniform.frame_time != new_frame_time {
            self.time_uniform.frame_time = new_frame_time;
            self.time_uniform_dirty = true;
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>, new_size_logical: LogicalSize<f32>) {
        self.screen_uniform.window_size_physical = [new_size.width as f32, new_size.height as f32];

        self.screen_uniform.window_size_logical = [
            new_size_logical.width as f32,
            new_size_logical.height as f32,
        ];
        self.screen_uniform_dirty = true;

        self.camera_handle.aspect = new_size.width as f32 / new_size.height as f32;
        self.camera_handle.camera_uniform_dirty = true;
    }

    pub fn change_scale_factor(&mut self, scale_factor: f64) {
        self.screen_uniform.dpi_scale = scale_factor as f32;
        self.screen_uniform_dirty = true;
    }

    fn create_gpu_resources(
        device: &Device,
        screen_uniform: &ScreenUniform,
        time_uniform: &TimeUniform,
        camera_uniform: &CameraUniform,
    ) -> (BindGroupLayout, BindGroup, Buffer, Buffer, Buffer, Buffer) {
        let globals_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Globals Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let screen_uniform_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("Screen Uniform Buffer"),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            contents: bytemuck::bytes_of(screen_uniform),
        });

        let time_uniform_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("Time Uniform Buffer"),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            contents: bytemuck::bytes_of(time_uniform),
        });

        let camera_uniform_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("Camera Uniform Buffer"),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            contents: bytemuck::bytes_of(camera_uniform),
        });

        let intersection_data = IntersectionData {
            hit_result: Vec4::new(0.0, 0.0, 0.0, 1e30),
            brush_hit_cell_space: Vec4::new(0.0, 0.0, 0.0, 0.0),
            brush_aabb_origin_texel: IVec2::new(0, 0),
            brush_aabb_end_texel_exclusive: IVec2::new(0, 0),
        };

        let intersection_data_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("Intersection Data Buffer"),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            contents: bytemuck::bytes_of(&intersection_data),
        });

        let globals_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Globals Bind Group"),
            layout: &globals_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: screen_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: time_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: camera_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: intersection_data_buffer.as_entire_binding(),
                },
            ],
        });

        (
            globals_bgl,
            globals_bg,
            screen_uniform_buffer,
            time_uniform_buffer,
            camera_uniform_buffer,
            intersection_data_buffer,
        )
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScreenUniform {
    pub window_size_physical: [f32; 2],
    pub window_size_logical: [f32; 2],
    pub dpi_scale: f32,
    pub _pad0: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TimeUniform {
    pub frame_time: f32,
    pub delta_time: f32,
}

pub struct CameraHandle {
    pub rotation: Quat,
    pub position: Vec3,
    pub xz_intersection: Vec4,
    pub mouse_ray: Vec3,
    pub zoom: f32,

    pub fov_y_radians: f32,
    pub aspect: f32,
    near: f32,
    far: f32,

    pub camera_speed: f32,
    pub yaw: f32,
    pub pitch: f32,

    pub camera_uniform_dirty: bool,
}

impl CameraHandle {
    pub fn rotate(&mut self, delta_yaw: f32, delta_pitch: f32) {
        let (yaw, pitch, _) = self.rotation.to_euler(EulerRot::YXZ);
        let new_pitch = (pitch + delta_pitch).clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );

        self.yaw = yaw + delta_yaw;
        self.pitch = new_pitch;

        self.rotation = Quat::from_euler(EulerRot::YXZ, self.yaw, new_pitch, 0.0);
    }

    pub fn rotate_degrees(&mut self, delta_yaw_deg: f32, delta_pitch_deg: f32) {
        self.rotate(delta_yaw_deg.to_radians(), delta_pitch_deg.to_radians());
    }

    fn to_uniform(&self) -> CameraUniform {
        let view = self.view_matrix();
        let proj = self.proj_matrix();

        let camera_uniform = CameraUniform {
            view_proj: proj * view,
            view_position: self.position.extend(1.0),
            xz_intersection: self.xz_intersection,
            mouse_ray: self.mouse_ray.extend(1.0),
            zoom: self.zoom,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };

        camera_uniform
    }

    fn view_matrix(&self) -> Mat4 {
        camera::rh::view::look_to_mat4(self.position, self.forward(), Vec3::Y)
    }

    fn proj_matrix(&self) -> Mat4 {
        camera::rh::proj::directx::perspective(self.fov_y_radians, self.aspect, self.near, self.far)
    }

    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }

    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.aspect = new_size.width as f32 / new_size.height as f32;
        self.camera_uniform_dirty = true;
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: Mat4,
    view_position: Vec4,
    xz_intersection: Vec4,
    mouse_ray: Vec4,
    zoom: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct IntersectionData {
    hit_result: Vec4,
    brush_hit_cell_space: Vec4,
    brush_aabb_origin_texel: IVec2,
    brush_aabb_end_texel_exclusive: IVec2,
}

fn get_monitor_frame_time(window: &Window) -> f32 {
    if let Some(monitor) = window.primary_monitor() {
        if let Some(hz) = monitor.refresh_rate_millihertz() {
            return 1000.0 / hz as f32;
        }
    }

    return 1.0 / DEFAULT_FPS;
}
