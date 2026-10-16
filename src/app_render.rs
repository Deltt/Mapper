use glam::Vec4Swizzles;
pub use wgpu::*;

use crate::{
    input::InputState, pipelines::Pipelines, regions::Region, running_app::RunningApp,
    terrain::Terrain,
};

impl RunningApp {
    pub fn render(&mut self) {
        let (surface_texture, surface_view, depth_view) = match self.renderer.get_render_output() {
            Some(output) => output,
            None => {
                return;
            }
        };

        let mut encoder = self
            .renderer
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Main Encoder"),
            });

        self.intersect_regions(
            &mut encoder,
            &self.renderer.queue,
            &self.pipelines.region_intersect_pipeline,
            &self.globals.globals_bg,
            &self.globals.intersection_data_buffer,
            &self.terrain,
            self.globals.camera_handle.position,
            self.globals.camera_handle.mouse_ray,
        );

        if self.input.is_down(InputState::MouseLeft) {
            self.dispatch_regions(
                &mut encoder,
                &self.pipelines.region_dispatch_pipeline,
                &self.globals.globals_bg,
                &self.terrain,
            );

            self.scratch_clear(
                &mut encoder,
                &self.pipelines,
                &self.globals.globals_bg,
                &self.terrain,
            );

            if !self.input.is_down(InputState::KeyShift) {
                self.edit_regions(
                    &self.renderer.queue,
                    &mut encoder,
                    &self.pipelines,
                    &self.globals.globals_bg,
                    &self.terrain,
                );
            } else {
                self.smooth_regions(
                    &self.renderer.queue,
                    &mut encoder,
                    &self.pipelines,
                    &self.globals.globals_bg,
                    &self.terrain,
                );
            }
        } else if self.input.is_down(InputState::KeyT) {
            self.dispatch_regions(
                &mut encoder,
                &self.pipelines.region_dispatch_pipeline,
                &self.globals.globals_bg,
                &self.terrain,
            );

            self.scratch_clear(
                &mut encoder,
                &self.pipelines,
                &self.globals.globals_bg,
                &self.terrain,
            );

            self.thermal_erosion(
                &self.renderer.queue,
                &mut encoder,
                &self.pipelines,
                &self.globals.globals_bg,
                &self.terrain,
            );
        } else if self.input.is_down(InputState::KeyG) {
            self.dispatch_regions(
                &mut encoder,
                &self.pipelines.region_dispatch_pipeline,
                &self.globals.globals_bg,
                &self.terrain,
            );

            self.scratch_clear(
                &mut encoder,
                &self.pipelines,
                &self.globals.globals_bg,
                &self.terrain,
            );

            self.gully_erosion(
                &self.renderer.queue,
                &mut encoder,
                &self.pipelines,
                &self.globals.globals_bg,
                &self.terrain,
            );
        } else if self.input.is_down(InputState::KeyV) {
            self.dispatch_regions(
                &mut encoder,
                &self.pipelines.region_dispatch_pipeline,
                &self.globals.globals_bg,
                &self.terrain,
            );

            self.scratch_clear(
                &mut encoder,
                &self.pipelines,
                &self.globals.globals_bg,
                &self.terrain,
            );

            self.droplet_erosion(
                &self.renderer.queue,
                &mut encoder,
                &self.pipelines,
                &self.globals.globals_bg,
                &self.terrain,
            );
        }

        self.render_regions(
            &mut encoder,
            &surface_view,
            depth_view,
            &self.pipelines.region_render_pipeline,
            &self.globals.globals_bg,
            &self.terrain,
        );

        self.renderer.queue.submit([encoder.finish()]);
        self.renderer.queue.present(surface_texture);
    }

    pub fn render_regions(
        &self,
        encoder: &mut CommandEncoder,
        view: &TextureView,
        depth_view: &TextureView,
        region_render_pipeline: &RenderPipeline,
        globals_bg: &BindGroup,
        terrain: &Terrain,
    ) {
        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Region Render Pass"),
            multiview_mask: None,
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color {
                        r: 0.01,
                        g: 0.01,
                        b: 0.01,
                        a: 1.0,
                    }),
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(region_render_pipeline);

        render_pass.set_bind_group(0, globals_bg, &[]);
        render_pass.set_bind_group(1, &terrain.terrain_bg, &[]);

        render_pass.set_index_buffer(
            terrain.chunk_mesh_index_buffer.slice(..),
            terrain.chunk_mesh_index_format.wgpu_format(),
        );

        let instance_count = terrain.terrain_parameters_uniform.chunks_per_region_axis
            * terrain.terrain_parameters_uniform.chunks_per_region_axis;

        for region in terrain.regions.values() {
            render_pass.set_bind_group(2, &region.render_bind_group, &[]);

            render_pass.draw_indexed(
                0..terrain.chunk_mesh_index_format.len(),
                0,
                0..instance_count,
            );
        }
    }

    pub fn scratch_clear(
        &self,
        encoder: &mut CommandEncoder,
        pipelines: &Pipelines,
        globals_bg: &BindGroup,
        terrain: &Terrain,
    ) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Scratch Clear Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipelines.scratch_clear_pipeline);
        compute_pass.set_bind_group(0, globals_bg, &[]);
        compute_pass.set_bind_group(1, &terrain.terrain_bg, &[]);
        let workgroup_size = terrain
            .terrain_parameters_uniform
            .heights_per_region_axis
            .div_ceil(8);
        compute_pass.dispatch_workgroups(workgroup_size, workgroup_size, 1);
    }

    pub fn edit_regions(
        &self,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        pipelines: &Pipelines,
        globals_bg: &BindGroup,
        terrain: &Terrain,
    ) {
        terrain.update_scratch_targets(queue, 0i32);

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Region Edit Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipelines.region_gather_pipeline);
        compute_pass.set_bind_group(0, globals_bg, &[]);
        compute_pass.set_bind_group(1, &terrain.terrain_bg, &[]);

        for region in terrain.regions.values() {
            compute_pass.set_bind_group(2, &region.gather_bind_group, &[]);
            compute_pass.dispatch_workgroups_indirect(&region.dispatch_arguments_buffer, 0);
        }

        compute_pass.set_pipeline(&pipelines.edit_single_pipeline);
        const EDIT_WORKGROUP_SIZE: u32 = 8; // match shader
        let workgroups_per_axis = terrain
            .terrain_parameters_uniform
            .heights_per_region_axis
            .div_ceil(EDIT_WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(workgroups_per_axis, workgroups_per_axis, 1);

        compute_pass.set_pipeline(&pipelines.region_scatter_pipeline);
        for region in terrain.regions.values() {
            compute_pass.set_bind_group(2, &region.scatter_bind_group, &[]);
            compute_pass.dispatch_workgroups_indirect(&region.dispatch_arguments_buffer, 0);
        }
    }

    pub fn smooth_regions(
        &self,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        pipelines: &Pipelines,
        globals_bg: &BindGroup,
        terrain: &Terrain,
    ) {
        // encoder.clear_texture(
        //     &terrain.scratchmap_texture,
        //     &ImageSubresourceRange::default(),
        // );

        terrain.update_scratch_targets(queue, 1i32);

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Region Smooth Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipelines.region_gather_pipeline);
        compute_pass.set_bind_group(0, globals_bg, &[]);
        compute_pass.set_bind_group(1, &terrain.terrain_bg, &[]);

        for region in terrain.regions.values() {
            compute_pass.set_bind_group(2, &region.gather_bind_group, &[]);
            compute_pass.dispatch_workgroups_indirect(&region.dispatch_arguments_buffer, 0);
        }

        compute_pass.set_pipeline(&pipelines.smooth_pipeline);
        const EDIT_WORKGROUP_SIZE: u32 = 8; // match shader
        let workgroups_per_axis = terrain
            .terrain_parameters_uniform
            .heights_per_region_axis
            .div_ceil(EDIT_WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(workgroups_per_axis, workgroups_per_axis, 1);

        compute_pass.set_pipeline(&pipelines.region_scatter_pipeline);
        for region in terrain.regions.values() {
            compute_pass.set_bind_group(2, &region.scatter_bind_group, &[]);
            compute_pass.dispatch_workgroups_indirect(&region.dispatch_arguments_buffer, 0);
        }
    }

    pub fn thermal_erosion(
        &self,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        pipelines: &Pipelines,
        globals_bg: &BindGroup,
        terrain: &Terrain,
    ) {
        terrain.update_scratch_targets(queue, 1i32);

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Thermal Erosion Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipelines.region_gather_pipeline);
        compute_pass.set_bind_group(0, globals_bg, &[]);
        compute_pass.set_bind_group(1, &terrain.terrain_bg, &[]);

        for region in terrain.regions.values() {
            compute_pass.set_bind_group(2, &region.gather_bind_group, &[]);
            compute_pass.dispatch_workgroups_indirect(&region.dispatch_arguments_buffer, 0);
        }

        let workgroup_count = terrain
            .terrain_parameters_uniform
            .heights_per_region_axis
            .div_ceil(8);

        compute_pass.set_pipeline(&pipelines.thermal_erosion_pipeline);
        compute_pass.set_bind_group(0, globals_bg, &[]);
        compute_pass.set_bind_group(1, &terrain.terrain_bg, &[]);
        compute_pass.dispatch_workgroups(workgroup_count, workgroup_count, 1);

        compute_pass.set_pipeline(&pipelines.region_scatter_pipeline);
        for region in terrain.regions.values() {
            compute_pass.set_bind_group(2, &region.scatter_bind_group, &[]);
            compute_pass.dispatch_workgroups_indirect(&region.dispatch_arguments_buffer, 0);
        }
    }

    pub fn droplet_erosion(
        &self,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        pipelines: &Pipelines,
        globals_bg: &BindGroup,
        terrain: &Terrain,
    ) {
        terrain.update_scratch_targets(queue, 0i32);

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Droplet Erosion Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipelines.region_gather_pipeline);
        compute_pass.set_bind_group(0, globals_bg, &[]);
        compute_pass.set_bind_group(1, &terrain.terrain_bg, &[]);

        for region in terrain.regions.values() {
            compute_pass.set_bind_group(2, &region.gather_bind_group, &[]);
            compute_pass.dispatch_workgroups_indirect(&region.dispatch_arguments_buffer, 0);
        }

        const DROPLETS_PER_CELL: f32 = 0.01; // start value, tune by eye
        const DROPLET_WORKGROUP_SIZE: u32 = 64;

        let radius_cells = terrain.brush_uniform.brush_radius
            / terrain.terrain_parameters_uniform.cell_size as f32;
        let droplet_count =
            (std::f32::consts::PI * radius_cells * radius_cells * DROPLETS_PER_CELL).ceil() as u32;
        let workgroup_count = droplet_count.div_ceil(DROPLET_WORKGROUP_SIZE).max(1);

        compute_pass.set_pipeline(&pipelines.droplet_erosion_pipeline);
        compute_pass.set_bind_group(0, globals_bg, &[]);
        compute_pass.set_bind_group(1, &terrain.terrain_bg, &[]);
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);

        compute_pass.set_pipeline(&pipelines.region_scatter_pipeline);
        for region in terrain.regions.values() {
            compute_pass.set_bind_group(2, &region.scatter_bind_group, &[]);
            compute_pass.dispatch_workgroups_indirect(&region.dispatch_arguments_buffer, 0);
        }
    }

    pub fn gully_erosion(
        &self,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        pipelines: &Pipelines,
        globals_bg: &BindGroup,
        terrain: &Terrain,
    ) {
        terrain.update_scratch_targets(queue, 1i32);

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Gully Erosion Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipelines.region_gather_pipeline);
        compute_pass.set_bind_group(0, globals_bg, &[]);
        compute_pass.set_bind_group(1, &terrain.terrain_bg, &[]);

        for region in terrain.regions.values() {
            compute_pass.set_bind_group(2, &region.gather_bind_group, &[]);
            compute_pass.dispatch_workgroups_indirect(&region.dispatch_arguments_buffer, 0);
        }

        let workgroup_count = terrain
            .terrain_parameters_uniform
            .heights_per_region_axis
            .div_ceil(8);

        compute_pass.set_pipeline(&pipelines.gully_erosion_pipeline);
        compute_pass.set_bind_group(0, globals_bg, &[]);
        compute_pass.set_bind_group(1, &terrain.terrain_bg, &[]);
        compute_pass.dispatch_workgroups(workgroup_count, workgroup_count, 1);

        compute_pass.set_pipeline(&pipelines.region_scatter_pipeline);
        for region in terrain.regions.values() {
            compute_pass.set_bind_group(2, &region.scatter_bind_group, &[]);
            compute_pass.dispatch_workgroups_indirect(&region.dispatch_arguments_buffer, 0);
        }
    }

    // pub fn edit_regions(
    //     &self,
    //     encoder: &mut CommandEncoder,
    //     region_edit_pipeline: &ComputePipeline,
    //     globals_bg: &BindGroup,
    //     terrain: &Terrain,
    // ) {
    //     let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
    //         label: Some("Region Edit Pass"),
    //         timestamp_writes: None,
    //     });
    //     compute_pass.set_pipeline(region_edit_pipeline);
    //     compute_pass.set_bind_group(0, globals_bg, &[]);
    //     compute_pass.set_bind_group(1, &terrain.terrain_bg, &[]);

    //     for region in terrain.regions.values() {
    //         compute_pass.set_bind_group(2, &region.edit_bind_group, &[]);
    //         compute_pass.dispatch_workgroups_indirect(&region.dispatch_arguments_buffer, 0);
    //     }
    // }

    // pub fn calculate_normals_regions(
    //     &self,
    //     encoder: &mut CommandEncoder,
    //     region_normal_pipeline: &ComputePipeline,
    //     globals_bg: &BindGroup,
    //     terrain: &Terrain,
    // ) {
    //     let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
    //         label: Some("Region Normal Pass"),
    //         timestamp_writes: None,
    //     });
    //     compute_pass.set_pipeline(region_normal_pipeline);
    //     compute_pass.set_bind_group(0, globals_bg, &[]);
    //     compute_pass.set_bind_group(1, &terrain.terrain_bg, &[]);

    //     for region in terrain.regions.values() {
    //         compute_pass.set_bind_group(2, &region.normal_bind_group, &[]);
    //         compute_pass.dispatch_workgroups_indirect(&region.dispatch_arguments_buffer, 0);
    //     }
    // }

    pub fn intersect_regions(
        &self,
        encoder: &mut CommandEncoder,
        queue: &Queue,
        region_intersect_pipeline: &ComputePipeline,
        globals_bg: &BindGroup,
        intersection_data_buffer: &Buffer,
        terrain: &Terrain,
        ray_origin: glam::Vec3,
        ray_dir: glam::Vec3,
    ) {
        let reset = glam::Vec4::new(0.0, 0.0, 0.0, 1e30);
        queue.write_buffer(intersection_data_buffer, 0, bytemuck::bytes_of(&reset));

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Region Intersect Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(region_intersect_pipeline);

        compute_pass.set_bind_group(0, globals_bg, &[]);
        compute_pass.set_bind_group(1, &terrain.terrain_bg, &[]);

        let ray_direction = ray_dir.normalize();
        let mut candidate_regions: Vec<(f32, &Region)> = terrain
            .regions
            .values()
            .filter_map(|region| {
                ray_region_entry_distance(
                    region.region_parameters_uniform.offset.xz(),
                    terrain.terrain_parameters_uniform.region_size,
                    ray_origin,
                    ray_direction,
                )
                .map(|entry_distance| (entry_distance, region))
            })
            .collect();
        candidate_regions.sort_by(|a, b| a.0.total_cmp(&b.0)); // front to back

        for (_, region) in candidate_regions {
            compute_pass.set_bind_group(2, &region.intersect_bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }
    }

    pub fn dispatch_regions(
        &self,
        encoder: &mut CommandEncoder,
        dispatch_size_pipeline: &ComputePipeline,
        globals_bg: &BindGroup,
        terrain: &Terrain,
    ) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Region Dispatch Size Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(dispatch_size_pipeline);
        compute_pass.set_bind_group(0, globals_bg, &[]);
        compute_pass.set_bind_group(1, &terrain.terrain_bg, &[]);

        for region in terrain.regions.values() {
            compute_pass.set_bind_group(2, &region.dispatch_bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }
    }

    pub fn render_empty_pass(&self, encoder: &mut CommandEncoder, view: &TextureView) {
        let _render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Empty Render Pass"),
            multiview_mask: None,
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color {
                        r: 0.1,
                        g: 0.1,
                        b: 0.1,
                        a: 1.0,
                    }),
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
    }

    pub fn debug_pass(&self, encoder: &mut CommandEncoder, view: &TextureView) {
        let _render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Debug Clear Pass"),
            multiview_mask: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 1.0,
                        g: 0.0,
                        b: 1.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
    }
}

const MAX_INTERSECT_DISTANCE: f32 = 2000.0;

fn ray_intersects_region_aabb(
    region_offset_xz: glam::Vec2,
    region_size: u32,
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    max_ray_length: f32,
) -> bool {
    const EPS: f32 = 1e-6;

    let region_min = region_offset_xz;
    let region_max = region_offset_xz + glam::Vec2::splat(region_size as f32);

    let mut t_min = 0.0f32;
    let mut t_max = max_ray_length;

    if ray_dir.x.abs() > EPS {
        let tx1 = (region_min.x - ray_origin.x) / ray_dir.x;
        let tx2 = (region_max.x - ray_origin.x) / ray_dir.x;
        t_min = t_min.max(tx1.min(tx2));
        t_max = t_max.min(tx1.max(tx2));
    } else if ray_origin.x < region_min.x || ray_origin.x > region_max.x {
        return false;
    }

    if ray_dir.z.abs() > EPS {
        let tz1 = (region_min.y - ray_origin.z) / ray_dir.z;
        let tz2 = (region_max.y - ray_origin.z) / ray_dir.z;
        t_min = t_min.max(tz1.min(tz2));
        t_max = t_max.min(tz1.max(tz2));
    } else if ray_origin.z < region_min.y || ray_origin.z > region_max.y {
        return false;
    }

    t_min <= t_max && t_max >= 0.0
}

fn ray_region_entry_distance(
    region_offset_xz: glam::Vec2,
    region_size: u32,
    ray_origin: glam::Vec3,
    ray_direction: glam::Vec3,
) -> Option<f32> {
    const PARALLEL_EPSILON: f32 = 1e-6;

    let region_min = region_offset_xz;
    let region_max = region_offset_xz + glam::Vec2::splat(region_size as f32);

    let mut enter_distance = 0.0f32;
    //let mut exit_distance = f32::MAX;
    let mut exit_distance = MAX_INTERSECT_DISTANCE;

    if ray_direction.x.abs() > PARALLEL_EPSILON {
        let distance_to_min_x = (region_min.x - ray_origin.x) / ray_direction.x;
        let distance_to_max_x = (region_max.x - ray_origin.x) / ray_direction.x;
        enter_distance = enter_distance.max(distance_to_min_x.min(distance_to_max_x));
        exit_distance = exit_distance.min(distance_to_min_x.max(distance_to_max_x));
    } else if ray_origin.x < region_min.x || ray_origin.x > region_max.x {
        return None;
    }

    if ray_direction.z.abs() > PARALLEL_EPSILON {
        let distance_to_min_z = (region_min.y - ray_origin.z) / ray_direction.z;
        let distance_to_max_z = (region_max.y - ray_origin.z) / ray_direction.z;
        enter_distance = enter_distance.max(distance_to_min_z.min(distance_to_max_z));
        exit_distance = exit_distance.min(distance_to_min_z.max(distance_to_max_z));
    } else if ray_origin.z < region_min.y || ray_origin.z > region_max.y {
        return None;
    }

    (enter_distance <= exit_distance && exit_distance >= 0.0).then_some(enter_distance)
}
