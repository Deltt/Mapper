use glam::Vec3;
use winit::window::CursorGrabMode;

use crate::{input::InputState, running_app::RunningApp, terrain::Terrain};

impl RunningApp {
    pub fn logic(&mut self) {
        self.globals.sync_monitor_frame_time(&self.window);

        if self.globals.screen_uniform_dirty {
            self.globals.update_screen(&self.renderer.queue);
        }

        if self.globals.time_uniform_dirty {
            self.globals.update_time(&self.renderer.queue);
        }

        self.interpret_command();

        let mut new_brush_shape = self.terrain.brush_uniform.brush_shape;
        if self.input.is_down(InputState::Key0) {
            new_brush_shape = 0;
        } else if self.input.is_down(InputState::Key1) {
            new_brush_shape = 1;
        } else if self.input.is_down(InputState::Key2) {
            new_brush_shape = 2;
        } else if self.input.is_down(InputState::Key3) {
            new_brush_shape = 3;
        }

        if new_brush_shape != self.terrain.brush_uniform.brush_shape {
            self.terrain.brush_uniform.brush_shape = new_brush_shape;
            self.terrain.brush_uniform_dirty = true;
        }

        if self.input.is_down(InputState::MouseMiddle) {
            self.globals.orbit_camera_y(self.input.mouse_delta.x);
            let _ = self.window.set_cursor_grab(CursorGrabMode::Locked);
        } else if self.input.is_down(InputState::KeyCTRL) {
            if self.input.mouse_delta.x.abs() > self.input.mouse_delta.y.abs() {
                self.terrain.brush_uniform.brush_radius +=
                    self.input.mouse_delta.x * self.globals.time_uniform.frame_time * 5.0;

                let max_radius =
                    (self.terrain.terrain_parameters_uniform.region_size - 7) as f32 * 0.5;
                self.terrain.brush_uniform.brush_radius = self
                    .terrain
                    .brush_uniform
                    .brush_radius
                    .clamp(0.5, max_radius);
            } else {
                self.terrain.brush_uniform.brush_fade +=
                    self.input.mouse_delta.y * self.globals.time_uniform.frame_time * 0.4;
                self.terrain.brush_uniform.brush_fade =
                    self.terrain.brush_uniform.brush_fade.clamp(0.0, 1.0);
            }
            let _ = self.window.set_cursor_grab(CursorGrabMode::Locked);
            self.terrain.brush_uniform_dirty = true;
        } else {
            let _ = self.window.set_cursor_grab(CursorGrabMode::None);
        }

        if self.input.is_down(InputState::KeyALT) {
            if self.terrain.brush_uniform.brush_modifier != -1.0 {
                self.terrain.brush_uniform.brush_modifier = -1.0;
                self.terrain.brush_uniform_dirty = true;
            }
        } else {
            if self.terrain.brush_uniform.brush_modifier != 1.0 {
                self.terrain.brush_uniform.brush_modifier = 1.0;
                self.terrain.brush_uniform_dirty = true;
            }
        }

        if self.terrain.brush_uniform_dirty {
            self.terrain.update_brush(&self.renderer.queue);
        }

        let camera_movement_input = self.input.get_movement_input();
        if camera_movement_input != Vec3::ZERO {
            self.globals.move_camera_xz(self.input.get_movement_input());
        }

        if self.input.mouse_scroll != 0.0 {
            self.globals.zoom_camera_xz(self.input.mouse_scroll);
        }

        if self.globals.camera_handle.camera_uniform_dirty {
            self.globals.update_camera(
                &self.renderer.queue,
                &self.renderer.surface_config,
                self.input.mouse_position,
            );
        }
    }

    pub fn interpret_command(&mut self) {
        let Some(cmd) = self.command_slot.take() else {
            return;
        };
        let tokens: Vec<&str> = cmd.split_whitespace().collect();

        let valid = match tokens.as_slice() {
            ["test"] => {
                log!("Test command");
                false
            }

            ["create", "default"] => {
                self.terrain = Terrain::default(&self.renderer.device, &self.renderer.queue);
                for (x, z) in [(0, 0), (1, 0), (1, 1)] {
                    self.terrain
                        .add_region(&self.renderer.device, &self.renderer.queue, x, z);
                }
                true
            }

            ["create", "terrain", rest @ ..] => {
                let args: Vec<u32> = rest.iter().filter_map(|s| s.parse().ok()).collect();
                match args.as_slice() {
                    &[_, _, 0] => {
                        log!("INVALID - cell size cannot be 0");
                        false
                    }
                    &[cells_per_chunk_axis, chunks_per_region_axis, cell_size] => {
                        self.create_terrain(
                            cells_per_chunk_axis,
                            chunks_per_region_axis,
                            cell_size,
                        );
                        true
                    }
                    _ => {
                        log!(
                            "INVALID - create terrain expects 3 u32 args, got: {:?}",
                            args
                        );
                        false
                    }
                }
            }

            ["add", "region", rest @ ..] => {
                let args: Vec<i32> = rest.iter().filter_map(|s| s.parse().ok()).collect();
                match args.as_slice() {
                    &[x, z] => {
                        self.terrain
                            .add_region(&self.renderer.device, &self.renderer.queue, x, z);
                        true
                    }
                    _ => {
                        log!("INVALID - add region expects 2 i32 args, got: {:?}", args);
                        false
                    }
                }
            }

            ["delete", "region", rest @ ..] => {
                let args: Vec<i32> = rest.iter().filter_map(|s| s.parse().ok()).collect();
                match args.as_slice() {
                    &[x, z] => {
                        self.terrain.delete_region(x, z);
                        true
                    }
                    _ => {
                        log!(
                            "INVALID - delete region expects 2 i32 args, got: {:?}",
                            args
                        );
                        false
                    }
                }
            }

            _ => {
                log!("unknown command: {:?}", cmd);
                false
            }
        };

        if valid {
            //
        }
    }
}
