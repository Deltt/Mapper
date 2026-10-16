use std::sync::Arc;

use winit::{
    event_loop::{ActiveEventLoop, EventLoopProxy, OwnedDisplayHandle},
    window::{CursorGrabMode, Window},
};

pub use wgpu::*;

use crate::{
    UserEvent, command_queue::CommandSlot, globals::Globals, input::Input, pipelines::Pipelines,
    renderer::Renderer, terrain::Terrain,
};

pub struct RunningApp {
    pub window: Arc<Window>,
    pub renderer: Renderer,
    pub input: Input,
    pub globals: Globals,
    pub command_slot: CommandSlot,

    pub terrain: Terrain,
    pub pipelines: Pipelines,

    pub redraw_pending: bool,
    pub frame_count: u32,
}

impl RunningApp {
    pub async fn new(
        window: Arc<Window>,
        display_handle: Option<OwnedDisplayHandle>,
        proxy: EventLoopProxy<UserEvent>,
    ) -> Self {
        let renderer = Renderer::new(window.clone(), display_handle).await;

        let input = Input::default();

        let globals = Globals::new(&renderer.device, &window);

        let command_slot = CommandSlot::new(proxy.clone());

        let terrain = Terrain::default(&renderer.device, &renderer.queue);

        let pipelines = Pipelines::new(
            &renderer.device,
            renderer.view_format,
            &globals.globals_bgl,
            &terrain.terrain_bgl,
            &terrain.region_render_bgl,
            &terrain.region_gather_bgl,
            &terrain.region_scatter_bgl,
            &terrain.region_intersect_bgl,
            &terrain.region_dispatch_bgl,
        );

        let running_app = RunningApp {
            window,
            renderer,
            input,
            globals,
            command_slot,

            terrain,
            pipelines,

            redraw_pending: false,
            frame_count: 0,
        };

        running_app.window.set_visible(true);
        //running_app.window.set_cursor_visible(false);

        running_app
    }

    pub fn create_terrain(
        &mut self,
        cells_per_chunk_axis: u32,
        chunks_per_region_axis: u32,
        cell_size: u32,
    ) {
        if !is_pow2(cells_per_chunk_axis) {
            log!(
                "INVALID - cells per chunk axis {} are not power of two",
                cells_per_chunk_axis
            );
            return;
        }

        if !is_pow2(chunks_per_region_axis) {
            log!(
                "INVALID - chunks per axis {} are not power of two",
                chunks_per_region_axis
            );
            return;
        }

        if cell_size < 1 {
            log!("INVALID - cell size can't be zero");
            return;
        }

        self.terrain = Terrain::new(
            &self.renderer.device,
            &self.renderer.queue,
            cells_per_chunk_axis,
            chunks_per_region_axis,
            cell_size,
        );
    }

    pub fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {}

    pub fn request_frame(&mut self) {
        if !self.redraw_pending {
            self.redraw_pending = true;
            self.window.request_redraw();
        }
    }
}

fn is_pow2(n: u32) -> bool {
    n != 0 && (n & (n - 1)) == 0
}
