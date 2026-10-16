use wgpu::util::DeviceExt;
pub use wgpu::*;

pub struct Pipelines {
    region_render_pipeline_layout: PipelineLayout,
    pub region_render_pipeline: RenderPipeline,

    scratch_clear_pipeline_layout: PipelineLayout,
    pub scratch_clear_pipeline: ComputePipeline,

    region_gather_pipeline_layout: PipelineLayout,
    pub region_gather_pipeline: ComputePipeline,

    region_scatter_pipeline_layout: PipelineLayout,
    pub region_scatter_pipeline: ComputePipeline,

    edit_single_pipeline_layout: PipelineLayout,
    pub edit_single_pipeline: ComputePipeline,

    smooth_pipeline_layout: PipelineLayout,
    pub smooth_pipeline: ComputePipeline,

    thermal_erosion_pipeline_layout: PipelineLayout,
    pub thermal_erosion_pipeline: ComputePipeline,

    droplet_erosion_pipeline_layout: PipelineLayout,
    pub droplet_erosion_pipeline: ComputePipeline,

    gully_erosion_pipeline_layout: PipelineLayout,
    pub gully_erosion_pipeline: ComputePipeline,

    region_intersect_pipeline_layout: PipelineLayout,
    pub region_intersect_pipeline: ComputePipeline,

    region_dispatch_pipeline_layout: PipelineLayout,
    pub region_dispatch_pipeline: ComputePipeline,
}

impl Pipelines {
    pub fn new(
        device: &Device,
        view_format: TextureFormat,
        globals_bgl: &BindGroupLayout,
        terrain_bgl: &BindGroupLayout,
        region_render_bgl: &BindGroupLayout,
        region_gather_bgl: &BindGroupLayout,
        region_scatter_bgl: &BindGroupLayout,
        region_intersect_bgl: &BindGroupLayout,
        region_dispatch_bgl: &BindGroupLayout,
    ) -> Self {
        let (
            region_render_pipeline_layout,
            region_render_pipeline,
            scratch_clear_pipeline_layout,
            scratch_clear_pipeline,
            region_gather_pipeline_layout,
            region_gather_pipeline,
            region_scatter_pipeline_layout,
            region_scatter_pipeline,
            edit_single_pipeline_layout,
            edit_single_pipeline,
            smooth_pipeline_layout,
            smooth_pipeline,
            thermal_erosion_pipeline_layout,
            thermal_erosion_pipeline,
            gully_erosion_pipeline_layout,
            gully_erosion_pipeline,
            droplet_erosion_pipeline_layout,
            droplet_erosion_pipeline,
            region_intersect_pipeline_layout,
            region_intersect_pipeline,
            region_dispatch_pipeline_layout,
            region_dispatch_pipeline,
        ) = Self::create_gpu_resources(
            device,
            view_format,
            globals_bgl,
            terrain_bgl,
            region_render_bgl,
            region_gather_bgl,
            region_scatter_bgl,
            region_intersect_bgl,
            region_dispatch_bgl,
        );

        Self {
            region_render_pipeline_layout,
            region_render_pipeline,

            scratch_clear_pipeline_layout,
            scratch_clear_pipeline,

            region_gather_pipeline_layout,
            region_gather_pipeline,

            region_scatter_pipeline_layout,
            region_scatter_pipeline,

            edit_single_pipeline_layout,
            edit_single_pipeline,

            smooth_pipeline_layout,
            smooth_pipeline,

            thermal_erosion_pipeline_layout,
            thermal_erosion_pipeline,

            gully_erosion_pipeline_layout,
            gully_erosion_pipeline,

            droplet_erosion_pipeline_layout,
            droplet_erosion_pipeline,

            region_intersect_pipeline_layout,
            region_intersect_pipeline,

            region_dispatch_pipeline_layout,
            region_dispatch_pipeline,
        }
    }

    fn create_gpu_resources(
        device: &Device,
        view_format: TextureFormat,
        globals_bgl: &BindGroupLayout,
        terrain_bgl: &BindGroupLayout,
        region_render_bgl: &BindGroupLayout,
        region_gather_bgl: &BindGroupLayout,
        region_scatter_bgl: &BindGroupLayout,
        region_intersect_bgl: &BindGroupLayout,
        region_dispatch_bgl: &BindGroupLayout,
    ) -> (
        PipelineLayout,
        RenderPipeline,
        PipelineLayout,
        ComputePipeline,
        PipelineLayout,
        ComputePipeline,
        PipelineLayout,
        ComputePipeline,
        PipelineLayout,
        ComputePipeline,
        PipelineLayout,
        ComputePipeline,
        PipelineLayout,
        ComputePipeline,
        PipelineLayout,
        ComputePipeline,
        PipelineLayout,
        ComputePipeline,
        PipelineLayout,
        ComputePipeline,
        PipelineLayout,
        ComputePipeline,
    ) {
        let region_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Region Render Pipeline Layout"),
                bind_group_layouts: &[
                    Some(globals_bgl),
                    Some(terrain_bgl),
                    Some(region_render_bgl),
                ],
                immediate_size: 0,
            });

        let region_render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Region Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/region_render.wgsl").into()),
        });

        let region_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Region Render Pipeline"),
                layout: Some(&region_render_pipeline_layout),
                multiview_mask: None,
                vertex: wgpu::VertexState {
                    module: &region_render_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &region_render_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: view_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                cache: None,
            });

        let scratch_clear_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Scratch Clear Pipeline Layout"),
                bind_group_layouts: &[Some(globals_bgl), Some(terrain_bgl)],
                immediate_size: 0,
            });

        let scratch_clear_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Scratch Clear Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scratch_clear.wgsl").into()),
        });

        let scratch_clear_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Scratch Clear Pipeline"),
                layout: Some(&scratch_clear_pipeline_layout),
                module: &scratch_clear_shader,
                entry_point: Some("cs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let region_gather_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Region Gather Pipeline Layout"),
                bind_group_layouts: &[
                    Some(globals_bgl),
                    Some(terrain_bgl),
                    Some(region_gather_bgl),
                ],
                immediate_size: 0,
            });

        let gather_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Region Gather Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/region_gather.wgsl").into()),
        });

        let region_gather_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Region Gather Pipeline"),
                layout: Some(&region_gather_pipeline_layout),
                module: &gather_shader,
                entry_point: Some("cs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let region_scatter_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Region Scatter Pipeline Layout"),
                bind_group_layouts: &[
                    Some(globals_bgl),
                    Some(terrain_bgl),
                    Some(region_scatter_bgl),
                ],
                immediate_size: 0,
            });

        let scatter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Region Scatter Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/region_scatter.wgsl").into()),
        });

        let region_scatter_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Region Scatter Pipeline"),
                layout: Some(&region_scatter_pipeline_layout),
                module: &scatter_shader,
                entry_point: Some("cs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let edit_single_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Edit Single Pipeline Layout"),
                bind_group_layouts: &[Some(globals_bgl), Some(terrain_bgl)],
                immediate_size: 0,
            });

        let edit_single_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Edit Single Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/edit_single.wgsl").into()),
        });

        let edit_single_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Edit Single Pipeline"),
                layout: Some(&edit_single_pipeline_layout),
                module: &edit_single_shader,
                entry_point: Some("cs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let smooth_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Smooth Pipeline Layout"),
                bind_group_layouts: &[Some(globals_bgl), Some(terrain_bgl)],
                immediate_size: 0,
            });

        let smooth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Smooth Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/smooth.wgsl").into()),
        });

        let smooth_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Smooth Pipeline"),
            layout: Some(&smooth_pipeline_layout),
            module: &smooth_shader,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let thermal_erosion_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Thermal Erosion Pipeline Layout"),
                bind_group_layouts: &[Some(globals_bgl), Some(terrain_bgl)],
                immediate_size: 0,
            });

        let thermal_erosion_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Thermal Erosion Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/thermal_erosion.wgsl").into()),
        });

        let thermal_erosion_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Thermal Erosion Pipeline"),
                layout: Some(&thermal_erosion_pipeline_layout),
                module: &thermal_erosion_shader,
                entry_point: Some("cs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let droplet_erosion_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Droplet Erosion Pipeline Layout"),
                bind_group_layouts: &[Some(globals_bgl), Some(terrain_bgl)],
                immediate_size: 0,
            });

        let droplet_erosion_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Droplet Erosion Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/droplet_erosion.wgsl").into()),
        });

        let droplet_erosion_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Droplet Erosion Pipeline"),
                layout: Some(&droplet_erosion_pipeline_layout),
                module: &droplet_erosion_shader,
                entry_point: Some("cs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let gully_erosion_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Gully Erosion Pipeline Layout"),
                bind_group_layouts: &[Some(globals_bgl), Some(terrain_bgl)],
                immediate_size: 0,
            });

        let gully_erosion_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gully Erosion Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gully_erosion.wgsl").into()),
        });

        let gully_erosion_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Gully Erosion Pipeline"),
                layout: Some(&gully_erosion_pipeline_layout),
                module: &gully_erosion_shader,
                entry_point: Some("cs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let region_intersect_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Region Intersect Pipeline Layout"),
                bind_group_layouts: &[
                    Some(globals_bgl),
                    Some(terrain_bgl),
                    Some(region_intersect_bgl),
                ],
                immediate_size: 0,
            });

        let intersect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Region Intersect Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/region_intersect.wgsl").into()),
        });

        let region_intersect_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Region Intersect Pipeline"),
                layout: Some(&region_intersect_pipeline_layout),
                module: &intersect_shader,
                entry_point: Some("cs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let region_dispatch_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Region Dispatch Pipeline Layout"),
                bind_group_layouts: &[
                    Some(globals_bgl),
                    Some(terrain_bgl),
                    Some(region_dispatch_bgl),
                ],
                immediate_size: 0,
            });

        let dispatch_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Region Dispatch Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/region_dispatch.wgsl").into()),
        });

        let region_dispatch_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Region Dispatch Pipeline"),
                layout: Some(&region_dispatch_pipeline_layout),
                module: &dispatch_shader,
                entry_point: Some("cs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        (
            region_render_pipeline_layout,
            region_render_pipeline,
            scratch_clear_pipeline_layout,
            scratch_clear_pipeline,
            region_gather_pipeline_layout,
            region_gather_pipeline,
            region_scatter_pipeline_layout,
            region_scatter_pipeline,
            edit_single_pipeline_layout,
            edit_single_pipeline,
            smooth_pipeline_layout,
            smooth_pipeline,
            thermal_erosion_pipeline_layout,
            thermal_erosion_pipeline,
            gully_erosion_pipeline_layout,
            gully_erosion_pipeline,
            droplet_erosion_pipeline_layout,
            droplet_erosion_pipeline,
            region_intersect_pipeline_layout,
            region_intersect_pipeline,
            region_dispatch_pipeline_layout,
            region_dispatch_pipeline,
        )
    }
}
