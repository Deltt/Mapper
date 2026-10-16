use glam::{IVec2, Vec2, Vec4};
use wgpu::util::DeviceExt;
pub use wgpu::*;

use crate::{renderer::Renderer, terrain::TerrainParametersUniform};

pub struct Region {
    pub region_parameters_uniform: RegionParametersUniform,
    region_parameters_uniform_buffer: Buffer,
    pub dispatch_arguments_buffer: Buffer,

    heightmap_texture: Texture,
    normal_texture: Texture,

    pub render_bind_group: BindGroup,
    pub gather_bind_group: BindGroup,
    pub scatter_bind_group: BindGroup,
    pub intersect_bind_group: BindGroup,
    pub dispatch_bind_group: BindGroup,
}

impl Region {
    pub fn new(
        device: &Device,
        queue: &Queue,
        terrain_parameters_uniform: &TerrainParametersUniform,
        render_bind_group_layout: &BindGroupLayout,
        gather_bind_group_layout: &BindGroupLayout,
        scatter_bind_group_layout: &BindGroupLayout,
        intersect_bind_group_layout: &BindGroupLayout,
        dispatch_bind_group_layout: &BindGroupLayout,
        x: i32,
        z: i32,
    ) -> Self {
        let index_2d = IVec2::new(x, z);

        let offset = glam::Vec4::new(
            x as f32 * terrain_parameters_uniform.region_size as f32,
            0.0,
            z as f32 * terrain_parameters_uniform.region_size as f32,
            0.0,
        );

        let cell_size_f32 = terrain_parameters_uniform.cell_size as f32;
        let region_origin_cell_space = offset / cell_size_f32; // Vec3

        let size_cell_space =
            (terrain_parameters_uniform.region_size / terrain_parameters_uniform.cell_size) as i32;
        let aabb_origin_texel = IVec2::new(
            region_origin_cell_space.x.floor() as i32,
            region_origin_cell_space.z.floor() as i32,
        );
        let aabb_end_texel_exclusive =
            aabb_origin_texel + IVec2::splat(size_cell_space) + IVec2::ONE;

        let region_parameters_uniform = RegionParametersUniform {
            offset,
            index_2d,
            aabb_origin_texel,
            aabb_end_texel_exclusive,
            _pad0: IVec2::new(0, 0),
        };

        let region_parameters_uniform_buffer =
            device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("Region Parameters Uniform Buffer"),
                contents: bytemuck::bytes_of(&region_parameters_uniform),
                usage: BufferUsages::UNIFORM,
            });

        let dispatch_arguments_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("Dispatch Argugments Buffer"),
            contents: bytemuck::bytes_of(&[0_u32, 0_u32, 0_u32]),
            usage: BufferUsages::INDIRECT | BufferUsages::STORAGE,
        });

        let heightmap_data: Box<[f32]> = vec![
            0.0;
            (terrain_parameters_uniform.heights_per_region_axis
                * terrain_parameters_uniform.heights_per_region_axis)
                as usize
        ]
        .into_boxed_slice();

        let heightmap_texture = device.create_texture_with_data(
            queue,
            &TextureDescriptor {
                label: Some("Region Heightmap Texture"),
                size: Extent3d {
                    width: terrain_parameters_uniform.heights_per_region_axis,
                    height: terrain_parameters_uniform.heights_per_region_axis,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R32Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&heightmap_data),
        );

        let heightmap_view = heightmap_texture.create_view(&TextureViewDescriptor {
            label: Some("Region Heightmap View"),
            format: Some(TextureFormat::R32Float),
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
            usage: None,
        });

        let normal_data: Box<[u32]> = vec![
            2147516416_u32;
            (terrain_parameters_uniform.heights_per_region_axis
                * terrain_parameters_uniform.heights_per_region_axis)
                as usize
        ]
        .into_boxed_slice();

        let normal_texture = device.create_texture_with_data(
            queue,
            &TextureDescriptor {
                label: Some("Region Normal Texture"),
                size: Extent3d {
                    width: terrain_parameters_uniform.heights_per_region_axis,
                    height: terrain_parameters_uniform.heights_per_region_axis,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R32Uint,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&normal_data),
        );

        let normal_view = normal_texture.create_view(&TextureViewDescriptor {
            label: Some("Region Normal View"),
            format: Some(TextureFormat::R32Uint),
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
            usage: None,
        });

        let render_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Region Render Bind Group"),
            layout: render_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: region_parameters_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&heightmap_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&normal_view),
                },
            ],
        });

        let gather_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Region Gather Bind Group"),
            layout: gather_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: region_parameters_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&heightmap_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&normal_view),
                },
            ],
        });

        let scatter_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Region Scatter Bind Group"),
            layout: scatter_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: region_parameters_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&heightmap_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&normal_view),
                },
            ],
        });

        let intersect_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Region Intersect Bind Group"),
            layout: intersect_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: region_parameters_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&heightmap_view),
                },
            ],
        });

        let dispatch_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Region Dispatch Bind Group"),
            layout: dispatch_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: region_parameters_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: dispatch_arguments_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            region_parameters_uniform,
            region_parameters_uniform_buffer,
            dispatch_arguments_buffer,

            heightmap_texture,
            normal_texture,

            render_bind_group,
            gather_bind_group,
            scatter_bind_group,
            intersect_bind_group,
            dispatch_bind_group,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RegionParametersUniform {
    pub offset: Vec4,
    pub index_2d: IVec2,
    pub aabb_origin_texel: IVec2,
    pub aabb_end_texel_exclusive: IVec2,
    pub _pad0: IVec2,
}
