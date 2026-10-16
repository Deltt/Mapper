use std::sync::{Arc, Mutex};

use winit::event_loop::EventLoopProxy;

use crate::UserEvent;

#[derive(Clone)]
pub struct CommandSlot {
    inner: Arc<Mutex<Option<String>>>,
}

impl CommandSlot {
    pub fn new(proxy: EventLoopProxy<UserEvent>) -> Self {
        let slot = CommandSlot {
            inner: Arc::new(Mutex::new(None)),
        };

        #[cfg(not(target_arch = "wasm32"))]
        slot.spawn_stdin_reader(proxy);

        #[cfg(target_arch = "wasm32")]
        wasm_bridge::register_global(slot.clone(), proxy);

        slot
    }

    pub fn set(&self, cmd: impl Into<String>) {
        *self.inner.lock().unwrap() = Some(cmd.into());
    }

    pub fn take(&self) -> Option<String> {
        self.inner.lock().unwrap().take()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn spawn_stdin_reader(&self, proxy: EventLoopProxy<UserEvent>) {
        let slot = self.clone();
        std::thread::spawn(move || {
            use std::io::BufRead;
            for line in std::io::stdin().lock().lines() {
                let Ok(line) = line else { break };
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }
                slot.set(line);
                let _ = proxy.send_event(UserEvent::CommandReceived);
            }
        });
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm_bridge {
    use super::CommandSlot;
    use crate::UserEvent;
    use std::cell::RefCell;
    use wasm_bindgen::prelude::*;
    use winit::event_loop::EventLoopProxy;

    thread_local! {
        static GLOBAL: RefCell<Option<(CommandSlot, EventLoopProxy<UserEvent>)>> = RefCell::new(None);
    }

    pub fn register_global(slot: CommandSlot, proxy: EventLoopProxy<UserEvent>) {
        GLOBAL.with(|g| *g.borrow_mut() = Some((slot, proxy)));
    }

    #[wasm_bindgen]
    pub fn cmd(command: &str) {
        GLOBAL.with(|g| {
            if let Some((slot, proxy)) = g.borrow().as_ref() {
                slot.set(command);
                let _ = proxy.send_event(UserEvent::CommandReceived);
            }
        });
    }
}
