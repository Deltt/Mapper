use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::window::WindowId;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::EventLoopExtWebSys;

use crate::input::Input;
use crate::running_app::RunningApp;

#[macro_use]
mod macros;
mod app_logic;
mod app_render;
mod command_queue;
mod globals;
mod input;
mod renderer;
mod running_app;

mod pipelines;
mod regions;
mod terrain;

const APP_NAME: &str = "Dummy App";

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn start() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    let event_loop = EventLoop::<UserEvent>::with_user_event().build().unwrap();
    let proxy = event_loop.create_proxy();

    let mut app = App::Starting { proxy };

    #[cfg(not(target_arch = "wasm32"))]
    event_loop
        .run_app(&mut app)
        .expect("Failed to expect run_app in start");

    #[cfg(target_arch = "wasm32")]
    event_loop.spawn_app(app);
}

enum UserEvent {
    AppReady(RunningApp),
    CommandReceived,
    Wake,
}

enum App {
    Starting { proxy: EventLoopProxy<UserEvent> },
    Ready(RunningApp, EventLoopProxy<UserEvent>),
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let proxy = match self {
            App::Starting { proxy } => proxy.clone(),
            App::Ready(..) => return,
        };

        let min_size = LogicalSize::new(640, 360);

        #[cfg(not(target_arch = "wasm32"))]
        let attrs = winit::window::Window::default_attributes()
            .with_visible(false)
            .with_min_inner_size(min_size)
            .with_title(APP_NAME);

        #[cfg(target_arch = "wasm32")]
        let attrs = {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;
            let canvas = web_sys::window()
                .unwrap()
                .document()
                .unwrap()
                .get_element_by_id("canvas")
                .unwrap()
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .unwrap();
            winit::window::Window::default_attributes()
                .with_visible(false)
                .with_maximized(true)
                .with_title(APP_NAME)
                .with_canvas(Some(canvas))
        };

        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        #[cfg(not(target_arch = "wasm32"))]
        let display_handle = Some(event_loop.owned_display_handle());

        #[cfg(target_arch = "wasm32")]
        let display_handle = None;

        let fut = async move {
            let app = RunningApp::new(window, display_handle, proxy.clone()).await;
            let _ = proxy.send_event(UserEvent::AppReady(app));
            #[cfg(target_arch = "wasm32")]
            let _ = proxy.send_event(UserEvent::Wake);
        };

        event_loop.set_control_flow(ControlFlow::Wait);

        #[cfg(not(target_arch = "wasm32"))]
        pollster::block_on(fut);

        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(fut);
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::AppReady(app) => {
                let proxy = match self {
                    App::Starting { proxy } => proxy.clone(),
                    App::Ready(_, proxy) => proxy.clone(),
                };
                *self = App::Ready(app, proxy);
            }
            event => {
                if let App::Ready(app, _) = self {
                    Input::user_event(app, event_loop, event);
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if let App::Ready(app, _) = self {
            Input::window_event(app, event_loop, id, event);
        }
    }

    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let App::Ready(app, _) = self {
            Input::device_event(app, event_loop, device_id, event);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if event_loop.exiting() {
            return;
        }

        if let App::Ready(app, _) = self {
            app.about_to_wait(event_loop);
        }
    }
}
