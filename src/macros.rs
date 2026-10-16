#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => {{
        #[cfg(not(target_arch = "wasm32"))]
        println!($($arg)*);
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&format!($($arg)*).into());
    }};
}

#[macro_export]
macro_rules! size_of_field {
    ($type:ty, $field:ident) => {
        $crate::macros::size_of_return_value(&|s: $type| s.$field)
    };
}

#[macro_export]
macro_rules! define_uniform_range {
    ($struct:ty, $method_name:ident, $first:ident ..= $last:ident) => {
        impl $struct {
            const UNIFORM_START: usize = ::core::mem::offset_of!($struct, $first);
            const UNIFORM_END: usize =
                ::core::mem::offset_of!($struct, $last) + $crate::size_of_field!($struct, $last);

            pub fn $method_name(&self) -> &[u8] {
                &bytemuck::bytes_of(self)[Self::UNIFORM_START..Self::UNIFORM_END]
            }
        }
    };
}
