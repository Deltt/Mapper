use glam::Vec2;
use winit::{
    dpi::LogicalSize,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::WindowId,
};

use crate::{UserEvent, running_app::RunningApp};

#[repr(u8)]
#[derive(Clone, Copy)]
pub enum InputState {
    MouseLeft,
    MouseRight,
    MouseMiddle,
    KeyW,
    KeyA,
    KeyS,
    KeyD,
    KeyQ,
    KeyE,
    KeyT,
    KeyG,
    KeyV,
    KeyEsc,
    KeySpace,
    KeyCTRL,
    KeyShift,
    KeyALT,
    Key0,
    Key1,
    Key2,
    Key3,
}

impl InputState {
    #[inline]
    const fn bit(self) -> u32 {
        1 << (self as u8)
    }

    fn from_key(code: KeyCode) -> Option<Self> {
        Some(match code {
            KeyCode::KeyW => Self::KeyW,
            KeyCode::KeyA => Self::KeyA,
            KeyCode::KeyS => Self::KeyS,
            KeyCode::KeyD => Self::KeyD,
            KeyCode::KeyQ => Self::KeyQ,
            KeyCode::KeyE => Self::KeyE,
            KeyCode::KeyT => Self::KeyT,
            KeyCode::KeyG => Self::KeyG,
            KeyCode::KeyV => Self::KeyV,
            KeyCode::Escape => Self::KeyEsc,
            KeyCode::Space => Self::KeySpace,
            KeyCode::ControlLeft => Self::KeyCTRL,
            KeyCode::ShiftLeft => Self::KeyShift,
            KeyCode::AltLeft => Self::KeyALT,
            KeyCode::Digit0 => Self::Key0,
            KeyCode::Digit1 => Self::Key1,
            KeyCode::Digit2 => Self::Key2,
            KeyCode::Digit3 => Self::Key3,
            _ => return None,
        })
    }

    fn from_mouse(button: MouseButton) -> Option<Self> {
        Some(match button {
            MouseButton::Left => Self::MouseLeft,
            MouseButton::Right => Self::MouseRight,
            MouseButton::Middle => Self::MouseMiddle,
            _ => return None,
        })
    }
}

pub struct Input {
    down: u32,
    pressed: u32,
    released: u32,

    pub mouse_delta: Vec2,
    pub mouse_position: Vec2,
    pub mouse_scroll: f32,

    pub window_focused: bool,
}

impl Input {
    pub fn default() -> Self {
        Self {
            down: 0,
            pressed: 0,
            released: 0,

            mouse_delta: Vec2::new(0.0, 0.0),
            mouse_position: Vec2::new(0.0, 0.0),
            mouse_scroll: 0.0,

            window_focused: false,
        }
    }

    #[inline]
    fn set_down(&mut self, s: InputState) {
        let b = s.bit();
        if self.down & b == 0 {
            self.pressed |= b;
        }
        self.down |= b;
    }

    #[inline]
    fn set_up(&mut self, s: InputState) {
        let b = s.bit();
        self.down &= !b;
        self.released |= b;
    }

    #[inline]
    pub fn is_down(&self, s: InputState) -> bool {
        self.down & s.bit() != 0
    }

    #[inline]
    pub fn is_pressed(&self, s: InputState) -> bool {
        self.pressed & s.bit() != 0
    }

    #[inline]
    pub fn is_released(&self, s: InputState) -> bool {
        self.released & s.bit() != 0
    }

    #[inline]
    pub fn any_down(&self) -> bool {
        self.down != 0
    }

    pub fn end_frame(&mut self) {
        self.pressed = 0;
        self.released = 0;
        self.mouse_delta = Vec2::ZERO;
        self.mouse_scroll = 0.0;
    }

    pub fn user_event(app: &mut RunningApp, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::CommandReceived => {
                app.request_frame();
            }
            UserEvent::Wake => {
                app.request_frame();
            }
            _ => (),
        }
    }

    pub fn window_event(
        app: &mut RunningApp,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let _ = window_id;

        match event {
            WindowEvent::DroppedFile(fat_arrow) => {}
            WindowEvent::RedrawRequested => {
                app.redraw_pending = false;

                log!("Frame Count: {}", app.frame_count);
                app.frame_count += 1;

                app.logic();
                app.render();

                if app.input.any_down() {
                    app.request_frame();
                }

                app.input.end_frame();
            }

            WindowEvent::CloseRequested => {
                log!("App closed by click");
                event_loop.exit();
            }

            WindowEvent::ScaleFactorChanged {
                scale_factor,
                inner_size_writer: _,
            } => {
                app.globals.change_scale_factor(scale_factor);
            }

            WindowEvent::Focused(focused) => {
                app.input.window_focused = focused;
            }

            WindowEvent::Resized(new_size) => {
                if new_size.height == 0 || new_size.width == 0 {
                    return;
                }

                let scale_factor = app.window.scale_factor();
                let new_size_logical: LogicalSize<f32> = new_size.to_logical(scale_factor);

                app.globals.resize(new_size, new_size_logical);
                app.renderer.resize(new_size);
            }

            WindowEvent::CursorMoved {
                device_id: _,
                position,
            } => {
                app.input.mouse_position = Vec2::new(position.x as f32, position.y as f32);
                app.globals.camera_handle.camera_uniform_dirty = true;
                app.request_frame();
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(btn) = InputState::from_mouse(button) {
                    match state {
                        ElementState::Pressed => app.input.set_down(btn),
                        ElementState::Released => app.input.set_up(btn),
                    }
                }

                app.request_frame();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if let Some(key) = InputState::from_key(code) {
                        match event.state {
                            ElementState::Pressed => app.input.set_down(key),
                            ElementState::Released => app.input.set_up(key),
                        }
                    }
                }

                if app.input.any_down() {
                    app.request_frame();
                }
            }

            _ => (),
        }
    }

    pub fn device_event(
        app: &mut RunningApp,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                app.input.mouse_delta += Vec2::new(delta.0 as f32, delta.1 as f32);
            }

            DeviceEvent::MouseWheel { delta } => {
                app.input.mouse_scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_x, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
                };
                app.input.mouse_scroll = app.input.mouse_scroll.clamp(-1.0, 1.0);

                app.request_frame();
            }

            DeviceEvent::Key(_) => {
                if app.input.any_down() {
                    app.request_frame();
                }
            }

            _ => (),
        }
    }

    pub fn get_movement_input(&self) -> glam::Vec3 {
        let w = self.is_down(InputState::KeyW) as i32;
        let a = self.is_down(InputState::KeyA) as i32;
        let s = self.is_down(InputState::KeyS) as i32;
        let d = self.is_down(InputState::KeyD) as i32;

        let movement_input =
            glam::Vec3::new((d - a) as f32, 0.0, (s - w) as f32).normalize_or_zero();

        movement_input
    }
}
