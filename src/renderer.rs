use std::sync::Arc;

pub use wgpu::*;
use winit::{dpi::PhysicalSize, event_loop::OwnedDisplayHandle, window::Window};

pub struct Renderer {
    pub instance: Instance,
    pub surface: Surface<'static>,
    pub surface_config: SurfaceConfiguration,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub view_format: TextureFormat,

    pub depth_texture: Texture,
    pub depth_view: TextureView,
}

impl Renderer {
    pub async fn new(window: Arc<Window>, display_handle: Option<OwnedDisplayHandle>) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let instance_descriptor =
            InstanceDescriptor::new_with_display_handle(Box::new(display_handle.unwrap()));

        #[cfg(target_arch = "wasm32")]
        let instance_descriptor = InstanceDescriptor::new_without_display_handle();

        let instance = Instance::new(instance_descriptor);

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter_options = RequestAdapterOptions {
            apply_limit_buckets: false,
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        };

        let adapter = instance.request_adapter(&adapter_options).await.unwrap();

        let device_descriptor = DeviceDescriptor::default();

        let (device, queue) = adapter.request_device(&device_descriptor).await.unwrap();

        let size = window.inner_size();

        let surface_capabilities = surface.get_capabilities(&adapter);

        let surface_format = surface_capabilities
            .formats
            .iter()
            .copied()
            .find(|f| {
                matches!(
                    f,
                    TextureFormat::Bgra8UnormSrgb
                        | TextureFormat::Rgba8UnormSrgb
                        | TextureFormat::Bgra8Unorm
                        | TextureFormat::Rgba8Unorm
                )
            })
            .unwrap_or(surface_capabilities.formats[0]);
        log!("Surface format: {:?}", surface_format);

        let view_format = surface_format.add_srgb_suffix();
        log!("View format: {:?}", view_format);

        let view_formats = if view_format != surface_format {
            vec![view_format]
        } else {
            vec![]
        };

        let surface_config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(100),
            height: size.height.max(100),
            present_mode: PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: CompositeAlphaMode::Opaque,
            view_formats,
            color_space: SurfaceColorSpace::Auto,
        };

        surface.configure(&device, &surface_config);

        let depth_texture = device.create_texture(&TextureDescriptor {
            label: Some("Depth Texture"),
            size: Extent3d {
                width: surface_config.width.max(100),
                height: surface_config.height.max(100),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&TextureViewDescriptor::default());

        Self {
            instance,
            surface,
            surface_config,
            adapter,
            device,
            queue,
            view_format,

            depth_texture,
            depth_view,
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_config.width = new_size.width.max(100);
            self.surface_config.height = new_size.height.max(100);
            self.surface.configure(&self.device, &self.surface_config);

            self.depth_texture = self.device.create_texture(&TextureDescriptor {
                label: Some("Depth Texture"),
                size: Extent3d {
                    width: self.surface_config.width.max(100),
                    height: self.surface_config.height.max(100),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Depth32Float,
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });

            self.depth_view = self
                .depth_texture
                .create_view(&TextureViewDescriptor::default());
        }
    }

    pub fn get_render_output(&self) -> Option<(SurfaceTexture, TextureView, &TextureView)> {
        let surface_texture = match self.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(t) => t,
            CurrentSurfaceTexture::Suboptimal(t) => t,
            CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.surface_config);
                return None;
            }
            CurrentSurfaceTexture::Timeout
            | CurrentSurfaceTexture::Occluded
            | CurrentSurfaceTexture::Validation => {
                return None;
            }
            CurrentSurfaceTexture::Lost => {
                return None;
            }
        };

        let surface_view = surface_texture.texture.create_view(&TextureViewDescriptor {
            format: Some(self.view_format),
            ..Default::default()
        });

        Some((surface_texture, surface_view, &self.depth_view))
    }
}
