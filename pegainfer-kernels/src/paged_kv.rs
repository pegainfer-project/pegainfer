#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvStorage {
    Bf16,
    E4m3,
}

impl KvStorage {
    pub const fn elem_bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::E4m3 => 1,
        }
    }
}

/// Page-first geometry used by paged-KV kernels.
///
/// This is kernel-facing shape metadata only. Pool allocation, page ownership,
/// and request state live in the root runtime crate.
#[derive(Clone, Copy, Debug)]
pub struct PagedKvLayout {
    pub page_size: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Elements in one K (or V) block: page_size x num_kv_heads x head_dim.
    pub kv_block_len: usize,
    /// Elements between layers within a page: 2 x kv_block_len (K then V).
    pub layer_stride: usize,
    /// Elements per page (all layers): num_layers x layer_stride.
    pub page_stride: usize,
    pub storage: KvStorage,
}

impl PagedKvLayout {
    pub fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize, page_size: usize) -> Self {
        Self::with_storage(
            num_layers,
            num_kv_heads,
            head_dim,
            page_size,
            KvStorage::Bf16,
        )
    }

    pub fn with_storage(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        page_size: usize,
        storage: KvStorage,
    ) -> Self {
        let kv_block_len = page_size * num_kv_heads * head_dim;
        let layer_stride = 2 * kv_block_len;
        let page_stride = num_layers * layer_stride;
        Self {
            page_size,
            num_layers,
            num_kv_heads,
            head_dim,
            kv_block_len,
            layer_stride,
            page_stride,
            storage,
        }
    }
}
