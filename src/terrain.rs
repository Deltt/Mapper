use std::{collections::HashMap, mem::MaybeUninit};

use glam::Vec2;
use wgpu::util::DeviceExt;
pub use wgpu::*;

use crate::regions::Region;

pub struct Terrain {
    pub regions: HashMap<(i32, i32), Region>,

    pub terrain_bgl: BindGroupLayout,
    pub terrain_bg: BindGroup,

    pub chunk_mesh_index_format: ChunkMeshIndexFormat,
    pub chunk_mesh_index_buffer: Buffer,

    pub terrain_parameters_uniform: TerrainParametersUniform,
    terrain_parameters_uniform_buffer: Buffer,

    pub brush_uniform: BrushUniform,
    brush_uniform_buffer: Buffer,
    pub brush_uniform_dirty: bool,

    pub scratchmap_texture: Texture,
    pub scratchmap2_texture: Texture,
    scratch_targets_buffer: Buffer,

    pub region_render_bgl: BindGroupLayout,
    pub region_gather_bgl: BindGroupLayout,
    pub region_scatter_bgl: BindGroupLayout,
    pub region_intersect_bgl: BindGroupLayout,
    pub region_dispatch_bgl: BindGroupLayout,
}

impl Terrain {
    pub fn new(
        device: &Device,
        queue: &Queue,
        cells_per_chunk_axis: u32,
        chunks_per_region_axis: u32,
        cell_size: u32,
    ) -> Self {
        let regions: HashMap<(i32, i32), Region> = HashMap::new();

        let cells_per_region_axis: u32 = cells_per_chunk_axis * chunks_per_region_axis;
        let heights_per_chunk_axis: u32 = cells_per_chunk_axis + 1;
        let heights_per_chunk_total: u32 = heights_per_chunk_axis * heights_per_chunk_axis;
        let max_height_index = heights_per_chunk_total - 1;
        let index_count = cells_per_chunk_axis * cells_per_chunk_axis * 6;
        let region_size: u32 = cells_per_region_axis * cell_size;
        let heights_per_region_axis: u32 = cells_per_region_axis + 1;

        log!("INFO - Terrain created:");
        log!(
            "{}x{} resolution",
            heights_per_region_axis,
            heights_per_region_axis
        );
        log!(
            "{}x{} chunks ({} total)",
            chunks_per_region_axis,
            chunks_per_region_axis,
            chunks_per_region_axis * chunks_per_region_axis
        );
        log!(
            "{}x{} heights per chunk ({} total)",
            heights_per_chunk_axis,
            heights_per_chunk_axis,
            heights_per_chunk_axis * heights_per_chunk_axis
        );
        log!(
            "{}x{} heights per region ({} total)",
            heights_per_region_axis,
            heights_per_region_axis,
            heights_per_region_axis * heights_per_region_axis
        );
        log!(
            "{}x{} region size (cell size {})",
            region_size,
            region_size,
            cell_size
        );

        let chunk_mesh_index_format = if max_height_index <= u16::MAX as u32 {
            let mut indices: Box<[MaybeUninit<u16>]> = Box::new_uninit_slice(index_count as usize);
            Self::fill_heightmap_indices(
                &mut indices,
                cells_per_chunk_axis,
                heights_per_chunk_axis,
            );
            ChunkMeshIndexFormat::U16(unsafe { indices.assume_init() })
        } else {
            let mut indices: Box<[MaybeUninit<u32>]> = Box::new_uninit_slice(index_count as usize);
            Self::fill_heightmap_indices(
                &mut indices,
                cells_per_chunk_axis,
                heights_per_chunk_axis,
            );
            ChunkMeshIndexFormat::U32(unsafe { indices.assume_init() })
        };

        let terrain_parameters_uniform = TerrainParametersUniform {
            cells_per_chunk_axis,
            cells_per_region_axis,

            heights_per_chunk_axis,
            heights_per_region_axis,

            chunks_per_region_axis,

            cell_size,
            region_size,
        };

        let brush_uniform = BrushUniform {
            brush_shape: 0,
            brush_fade: 0.5,
            brush_radius: 25.0,
            brush_strength: 50.0,
            brush_modifier: 1.0,
            brush_limit_enabled: 0,
        };

        let (
            terrain_bgl,
            terrain_bg,
            chunk_mesh_index_buffer,
            terrain_parameters_uniform_buffer,
            brush_uniform_buffer,
            scratchmap_texture,
            scratchmap2_texture,
            scratch_targets_buffer,
            region_render_bgl,
            region_gather_bgl,
            region_scatter_bgl,
            region_intersect_bgl,
            region_dispatch_bgl,
        ) = Self::create_gpu_resources(
            device,
            queue,
            &chunk_mesh_index_format,
            &terrain_parameters_uniform,
            &brush_uniform,
        );

        let mut terrain = Terrain {
            regions,
            terrain_bgl,
            terrain_bg,

            chunk_mesh_index_format,
            chunk_mesh_index_buffer,
            terrain_parameters_uniform,
            terrain_parameters_uniform_buffer,

            brush_uniform,
            brush_uniform_buffer,
            brush_uniform_dirty: false,

            scratchmap_texture,
            scratchmap2_texture,
            scratch_targets_buffer,

            region_render_bgl,
            region_gather_bgl,
            region_scatter_bgl,
            region_intersect_bgl,
            region_dispatch_bgl,
        };

        terrain.add_region(device, queue, 0, 0);

        terrain
    }

    pub fn default(device: &Device, queue: &Queue) -> Self {
        let terrain = Self::new(device, queue, 64, 8, 1);
        terrain
    }

    pub fn add_region(&mut self, device: &Device, queue: &Queue, x: i32, z: i32) {
        if x > i16::MAX.into() || x < i16::MIN.into() {
            log!("INVALID - x {} exceeds the limit ({})", x, i16::MAX);
            return;
        }

        if z > i16::MAX.into() || z < i16::MIN.into() {
            log!("INVALID - z {} exceeds the limit ({})", z, i16::MAX);
            return;
        }

        if self.regions.contains_key(&(x, z)) {
            log!("INVALID - region {}, {} already exists", x, z);
            return;
        }

        let region = Region::new(
            device,
            queue,
            &self.terrain_parameters_uniform,
            &self.region_render_bgl,
            &self.region_gather_bgl,
            &self.region_scatter_bgl,
            &self.region_intersect_bgl,
            &self.region_dispatch_bgl,
            x,
            z,
        );

        self.regions.insert((x, z), region);
        log!("INFO - Region {}, {} added", x, z);
    }

    pub fn delete_region(&mut self, x: i32, z: i32) {
        if !self.regions.contains_key(&(x, z)) {
            log!("INVALID - region {}, {} doesn't exists", x, z);
            return;
        }

        self.regions.remove(&(x, z));
        log!("INFO - Region {}, {} removed", x, z);
    }

    pub fn update_brush(&mut self, queue: &Queue) {
        queue.write_buffer(
            &self.brush_uniform_buffer,
            0,
            bytemuck::bytes_of(&self.brush_uniform),
        );
        self.brush_uniform_dirty = false;
    }

    pub fn update_scratch_targets(&self, queue: &Queue, mut scatter: i32) {
        if scatter != 1 {
            scatter = 0;
        }

        queue.write_buffer(
            &self.scratch_targets_buffer,
            0,
            bytemuck::bytes_of(&[scatter]),
        );
    }

    fn create_gpu_resources(
        device: &Device,
        queue: &Queue,
        chunk_mesh_index_format: &ChunkMeshIndexFormat,
        terrain_parameters_uniform: &TerrainParametersUniform,
        brush_uniform: &BrushUniform,
    ) -> (
        BindGroupLayout,
        BindGroup,
        Buffer,
        Buffer,
        Buffer,
        Texture,
        Texture,
        Buffer,
        BindGroupLayout,
        BindGroupLayout,
        BindGroupLayout,
        BindGroupLayout,
        BindGroupLayout,
    ) {
        let scratch_targets_buffer: Buffer =
            device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("Scratch Target Buffer"),
                contents: bytemuck::bytes_of(&[0i32]),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

        let scratchmap_data: Box<[f32]> = vec![
            0.0;
            (terrain_parameters_uniform.heights_per_region_axis
                * terrain_parameters_uniform.heights_per_region_axis)
                as usize
        ]
        .into_boxed_slice();

        let scratchmap_texture = device.create_texture_with_data(
            queue,
            &TextureDescriptor {
                label: Some("Scratchmap Texture"),
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
            bytemuck::cast_slice(&scratchmap_data),
        );

        let scratchmap_view = scratchmap_texture.create_view(&TextureViewDescriptor {
            label: Some("Scratchmap View"),
            format: Some(TextureFormat::R32Float),
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
            usage: None,
        });

        let scratchmap2_texture = device.create_texture_with_data(
            queue,
            &TextureDescriptor {
                label: Some("Scratchmap2 Texture"),
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
            bytemuck::cast_slice(&scratchmap_data),
        );

        let scratchmap2_view = scratchmap2_texture.create_view(&TextureViewDescriptor {
            label: Some("Scratchmap2 View"),
            format: Some(TextureFormat::R32Float),
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
            usage: None,
        });

        let terrain_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Terrain Bind Group Layout"),
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
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadWrite,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadWrite,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let chunk_mesh_index_buffer: Buffer =
            device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("Chunk Mesh Index Buffer"),
                contents: chunk_mesh_index_format.as_bytes(),
                usage: BufferUsages::INDEX,
            });

        let terrain_parameters_uniform_buffer =
            device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("Terrain Uniform Buffer"),
                contents: bytemuck::bytes_of(terrain_parameters_uniform),
                usage: BufferUsages::UNIFORM,
            });

        let brush_uniform_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("Brush Uniform Buffer"),
            contents: bytemuck::bytes_of(brush_uniform),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let terrain_bg = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Terrain Bind Group"),
            layout: &terrain_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: terrain_parameters_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: brush_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&scratchmap_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&scratchmap2_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: scratch_targets_buffer.as_entire_binding(),
                },
            ],
        });

        let region_render_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Region Render Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Uint,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let region_gather_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Region Gather Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Uint,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let region_scatter_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Region Scatter Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let region_intersect_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Region Intersect Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let region_dispatch_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Region Dispatch Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        (
            terrain_bgl,
            terrain_bg,
            chunk_mesh_index_buffer,
            terrain_parameters_uniform_buffer,
            brush_uniform_buffer,
            scratchmap_texture,
            scratchmap2_texture,
            scratch_targets_buffer,
            region_render_bgl,
            region_gather_bgl,
            region_scatter_bgl,
            region_intersect_bgl,
            region_dispatch_bgl,
        )
    }

    fn fill_heightmap_indices<T: IndexElem>(
        indices: &mut [MaybeUninit<T>],
        cells_per_chunk_axis: u32,
        heights_per_chunk_axis: u32,
    ) {
        let mut iter = indices.chunks_exact_mut(6);
        let mut row_start_flip = false;

        for y in 0..cells_per_chunk_axis {
            let mut flip = row_start_flip;
            for x in 0..cells_per_chunk_axis {
                let tl = y * heights_per_chunk_axis + x;
                let tr = tl + 1;
                let bl = tl + heights_per_chunk_axis;
                let br = bl + 1;

                let quad = iter.next().unwrap();
                if !flip {
                    quad[0].write(T::from_u32(tl));
                    quad[1].write(T::from_u32(bl));
                    quad[2].write(T::from_u32(br));
                    quad[3].write(T::from_u32(tl));
                    quad[4].write(T::from_u32(br));
                    quad[5].write(T::from_u32(tr));
                } else {
                    quad[0].write(T::from_u32(tl));
                    quad[1].write(T::from_u32(bl));
                    quad[2].write(T::from_u32(tr));
                    quad[3].write(T::from_u32(tr));
                    quad[4].write(T::from_u32(bl));
                    quad[5].write(T::from_u32(br));
                }

                flip = !flip;
            }
            row_start_flip = !row_start_flip;
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainParametersUniform {
    pub cells_per_chunk_axis: u32,
    pub cells_per_region_axis: u32,

    pub heights_per_chunk_axis: u32,
    pub heights_per_region_axis: u32,

    pub chunks_per_region_axis: u32,

    pub cell_size: u32,
    pub region_size: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BrushUniform {
    pub brush_shape: u32,
    pub brush_radius: f32,
    pub brush_fade: f32,
    pub brush_strength: f32,
    pub brush_modifier: f32,
    pub brush_limit_enabled: u32,
}

trait IndexElem: Copy {
    fn from_u32(v: u32) -> Self;
    fn max_value_as_u64() -> u64;
}

impl IndexElem for u16 {
    fn from_u32(v: u32) -> Self {
        v as u16
    }
    fn max_value_as_u64() -> u64 {
        u16::MAX as u64
    }
}

impl IndexElem for u32 {
    fn from_u32(v: u32) -> Self {
        v
    }
    fn max_value_as_u64() -> u64 {
        u32::MAX as u64
    }
}

pub enum ChunkMeshIndexFormat {
    U16(Box<[u16]>),
    U32(Box<[u32]>),
}

impl ChunkMeshIndexFormat {
    pub fn wgpu_format(&self) -> IndexFormat {
        match self {
            ChunkMeshIndexFormat::U16(_) => IndexFormat::Uint16,
            ChunkMeshIndexFormat::U32(_) => IndexFormat::Uint32,
        }
    }
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            ChunkMeshIndexFormat::U16(b) => bytemuck::cast_slice(b),
            ChunkMeshIndexFormat::U32(b) => bytemuck::cast_slice(b),
        }
    }

    pub fn len(&self) -> u32 {
        match self {
            ChunkMeshIndexFormat::U16(b) => b.len() as u32,
            ChunkMeshIndexFormat::U32(b) => b.len() as u32,
        }
    }
}
