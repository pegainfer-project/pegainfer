//! Declarative typed GPU buffer structs.

/// Declares a GPU buffer struct with generated `new()` and `set_batch_size()`.
#[macro_export]
macro_rules! gpu_buffers {
    (
        $(#[$struct_meta:meta])*
        $vis:vis struct $name:ident {
            $(
                $(#[$field_meta:meta])*
                $field_vis:vis $field:ident : $kind:ident < { $dim:expr } >
            ),* $(,)?
        }
    ) => {
        $(#[$struct_meta])*
        $vis struct $name {
            $(
                $(#[$field_meta])*
                $field_vis $field : $crate::tensor:: $kind < { $dim } >,
            )*
        }

        impl $name {
            $vis fn new(
                ctx: &$crate::tensor::DeviceContext,
                batch_size: usize,
            ) -> ::anyhow::Result<Self> {
                Ok(Self {
                    $(
                        $field: $crate::gpu_buffers!(@alloc ctx, batch_size, $kind, $dim),
                    )*
                })
            }

            $vis fn set_batch_size(&mut self, bs: usize) {
                $(
                    $crate::gpu_buffers!(@set_bs self, bs, $field, $kind);
                )*
            }
        }
    };

    (@alloc $ctx:ident, $bs:ident, GpuTensor, $dim:expr) => {
        $crate::tensor::GpuTensor::<{ $dim }>::zeros($ctx, $bs)?
    };
    (@alloc $ctx:ident, $bs:ident, GpuRawSlice, $dim:expr) => {
        $crate::tensor::GpuRawSlice::<{ $dim }>::zeros($ctx, $bs)?
    };
    (@alloc $ctx:ident, $bs:ident, GpuRawSliceI32, $dim:expr) => {
        $crate::tensor::GpuRawSliceI32::<{ $dim }>::zeros($ctx, $bs)?
    };

    (@set_bs $self:ident, $bs:ident, $field:ident, GpuTensor) => {
        $self.$field.seq_len = $bs;
    };
    (@set_bs $self:ident, $bs:ident, $field:ident, GpuRawSlice) => {
        $self.$field.batch_size = $bs;
    };
    (@set_bs $self:ident, $bs:ident, $field:ident, GpuRawSliceI32) => {
        $self.$field.batch_size = $bs;
    };
}
