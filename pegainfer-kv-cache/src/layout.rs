/// Page-first geometry: dimensions and derived strides for one page.
///
/// Pure value type — no GPU, no allocation.
#[derive(Clone, Copy, Debug)]
pub struct KvLayout {
    pub page_size: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Elements in one K (or V) block: page_size × num_kv_heads × head_dim.
    pub kv_block_len: usize,
    /// Elements between layers within a page: 2 × kv_block_len (K then V).
    pub layer_stride: usize,
    /// Elements per page (all layers): num_layers × layer_stride.
    pub page_stride: usize,
}

impl KvLayout {
    pub fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize, page_size: usize) -> Self {
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
        }
    }
}
