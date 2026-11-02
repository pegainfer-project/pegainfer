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

/// How one token's keys and values are laid out within a page's layer block.
///
/// This is the shape axis of the layout, separate from `KvStorage`, which is
/// the element-width axis. The strides every consumer derives come from here
/// and nowhere else; a reader that can only walk one of these has to say so
/// at its geometry rather than find out at the wrong rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvFormat {
    /// K then V, each `head_dim` wide: two blocks per layer per page.
    Split,
    /// One block per layer of `head_dim + rotary` values per token, for a
    /// family whose K and V fork from one projection: `K = RoPE(V * w)`, so a
    /// proportional RoPE over `rotary` columns leaves the rest of K an exact
    /// `V * w` that folds into the query. The row is
    /// `[K_rot | V_identity | V_rot]`: score operand the first `head_dim`
    /// columns, value operand the last. `rotate_half` pairs
    /// `(d, d + head_dim/2)`, so the rotated set is `[0, rotary/2)` and
    /// `[head_dim/2, head_dim/2 + rotary/2)`. See [`KvFormat::permute`].
    Folded { rotary: usize },
}

impl KvFormat {
    /// Values stored per token per kv head across the layer block.
    pub const fn values_per_token(self, head_dim: usize) -> usize {
        match self {
            Self::Split => 2 * head_dim,
            Self::Folded { rotary } => head_dim + rotary,
        }
    }

    /// The width of one row when the pool is read as rows of
    /// `[num_kv_heads, row_width]`: the unit the generated kernels index in.
    pub const fn row_width(self, head_dim: usize) -> usize {
        match self {
            Self::Split => head_dim,
            Self::Folded { rotary } => head_dim + rotary,
        }
    }

    /// Blocks per layer: K and V apart, or one row holding both.
    pub const fn blocks_per_layer(self) -> usize {
        match self {
            Self::Split => 2,
            Self::Folded { .. } => 1,
        }
    }

    /// The rotated columns that `Folded` keeps of K, as the count the writer
    /// and the readers derive the permutation from; zero for `Split`, whose
    /// rows are the head's own order.
    pub const fn fold_rotary(self) -> usize {
        match self {
            Self::Split => 0,
            Self::Folded { rotary } => rotary,
        }
    }

    /// Where head column `d` lands in a folded row's first `head_dim`
    /// columns: the rotated set first, in order, then the identity set. The
    /// writer places K and V by this, the readers undo it on the store, and
    /// the format parity gate holds the two implementations to this one.
    pub const fn permute(self, head_dim: usize, d: usize) -> usize {
        match self {
            Self::Split => d,
            Self::Folded { rotary } => {
                let rh = rotary / 2;
                let h = head_dim / 2;
                if d < rh {
                    d
                } else if d >= h && d < h + rh {
                    d - h + rh
                } else if d < h {
                    d + rh
                } else {
                    d
                }
            }
        }
    }

    /// Whether the format's invariants hold for this head: an even rotary
    /// that fits the half-head the rotation pairs across.
    pub const fn fits(self, head_dim: usize) -> bool {
        match self {
            Self::Split => head_dim > 0,
            Self::Folded { rotary } => {
                rotary > 0
                    && rotary.is_multiple_of(2)
                    && rotary <= head_dim
                    && head_dim.is_multiple_of(2)
            }
        }
    }
}

/// The one derivation of a page's strides, in elements: the K (or V) block,
/// the layer block, and the page. Both the kernel-facing layout and the
/// runtime's own copy call this, so the arithmetic exists once. `None` on
/// overflow; the caller decides whether that is a panic or an error.
pub fn derive_strides(
    page_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
    format: KvFormat,
) -> Option<(usize, usize, usize)> {
    if !format.fits(head_dim) {
        return None;
    }
    let layer_stride = page_size
        .checked_mul(num_kv_heads)?
        .checked_mul(format.values_per_token(head_dim))?;
    // A block is a K (or V) block for the split format and the whole layer
    // row for the folded one, so `kv_block_len` is the block either way.
    let kv_block_len = layer_stride / format.blocks_per_layer();
    let page_stride = num_layers.checked_mul(layer_stride)?;
    Some((kv_block_len, layer_stride, page_stride))
}

/// The pool read as rows of `[num_kv_heads, row_width]`, which is how the
/// generated kernels address it: a layer's block starts `layer_row` rows into
/// the page and the pool holds `pool_rows` rows in all. Format-generic, so a
/// reader that walks rows can take any format the row width describes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowGeometry {
    pub row_elems: usize,
    pub rows_per_page: usize,
    pub layer_row: usize,
    pub pool_rows: usize,
}

/// One layer's block in the pool, format-generic: where it starts and how
/// the page is strided. The prep that writes the pool takes this for either
/// format; the split-only readers take the K/V offset pair instead.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockGeometry {
    pub block_offset_elems: usize,
    pub page_size: usize,
    pub num_pages: usize,
    pub stride_page: usize,
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
    /// Elements between layers within a page: the format's values per token
    /// over the page's tokens and heads (K then V for `KvFormat::Split`).
    pub layer_stride: usize,
    /// Elements per page (all layers): num_layers x layer_stride.
    pub page_stride: usize,
    pub storage: KvStorage,
    pub format: KvFormat,
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
        Self::with_storage_and_format(
            num_layers,
            num_kv_heads,
            head_dim,
            page_size,
            storage,
            KvFormat::Split,
        )
    }

    pub fn with_storage_and_format(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        page_size: usize,
        storage: KvStorage,
        format: KvFormat,
    ) -> Self {
        let (kv_block_len, layer_stride, page_stride) =
            derive_strides(page_size, num_kv_heads, head_dim, num_layers, format)
                .expect("paged KV layout strides overflow usize or the format does not fit");
        Self {
            page_size,
            num_layers,
            num_kv_heads,
            head_dim,
            kv_block_len,
            layer_stride,
            page_stride,
            storage,
            format,
        }
    }

    /// A layer's block in a pool of `pool_len` elements, for either format.
    /// The strides are re-derived from the primitives and held to what the
    /// layout carries, since its fields are public.
    pub fn block_geometry(
        &self,
        pool_len: usize,
        layer: usize,
        what: &str,
    ) -> anyhow::Result<BlockGeometry> {
        let (kv_block_len, layer_stride, page_stride) = derive_strides(
            self.page_size,
            self.num_kv_heads,
            self.head_dim,
            self.num_layers,
            self.format,
        )
        .ok_or_else(|| anyhow::anyhow!("{what} strides overflow or the format does not fit"))?;
        anyhow::ensure!(
            self.kv_block_len == kv_block_len
                && self.layer_stride == layer_stride
                && self.page_stride == page_stride,
            "{what} layout strides ({}, {}, {}) are not the format's ({kv_block_len}, \
             {layer_stride}, {page_stride})",
            self.kv_block_len,
            self.layer_stride,
            self.page_stride
        );
        anyhow::ensure!(
            layer < self.num_layers,
            "{what} layer {layer} >= layout.num_layers {}",
            self.num_layers
        );
        anyhow::ensure!(
            page_stride > 0 && pool_len.is_multiple_of(page_stride),
            "{what} pool of {pool_len} elements is not whole pages of {page_stride}"
        );
        let num_pages = pool_len / page_stride;
        anyhow::ensure!(
            num_pages >= 1,
            "{what} pool of {pool_len} elements holds no whole page of {page_stride}"
        );
        Ok(BlockGeometry {
            block_offset_elems: layer * layer_stride,
            page_size: self.page_size,
            num_pages,
            stride_page: page_stride,
        })
    }

    /// The row view of a pool of `pool_len` elements at `layer`, refused
    /// rather than truncated when the pool or the page does not divide into
    /// rows, or the layer is past the layout.
    pub fn row_geometry(
        &self,
        pool_len: usize,
        layer: usize,
        what: &str,
    ) -> anyhow::Result<RowGeometry> {
        let row_elems = self.num_kv_heads * self.format.row_width(self.head_dim);
        anyhow::ensure!(row_elems > 0, "{what} layout has an empty row");
        anyhow::ensure!(
            layer < self.num_layers,
            "{what} layer {layer} >= layout.num_layers {}",
            self.num_layers
        );
        anyhow::ensure!(
            self.page_stride.is_multiple_of(row_elems)
                && self.layer_stride.is_multiple_of(row_elems)
                && pool_len.is_multiple_of(row_elems),
            "{what} pool does not divide into rows of {row_elems}"
        );
        Ok(RowGeometry {
            row_elems,
            rows_per_page: self.page_stride / row_elems,
            layer_row: layer * (self.layer_stride / row_elems),
            pool_rows: pool_len / row_elems,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_folded_permutation_puts_the_rotated_set_first_and_is_a_bijection() {
        let format = KvFormat::Folded { rotary: 128 };
        let cols: Vec<usize> = (0..512).map(|d| format.permute(512, d)).collect();
        let mut seen = vec![false; 512];
        for &col in &cols {
            assert!(!seen[col], "column {col} taken twice");
            seen[col] = true;
        }
        assert!(seen.iter().all(|&taken| taken));
        let rotated: Vec<usize> = (0..64).chain(256..320).collect();
        let first: Vec<usize> = rotated.iter().map(|&d| cols[d]).collect();
        assert_eq!(first, (0..128).collect::<Vec<_>>());
        assert!(
            (0..512)
                .filter(|d| !rotated.contains(d))
                .all(|d| cols[d] >= 128)
        );
        assert!((0..512).all(|d| KvFormat::Split.permute(512, d) == d));
    }

    #[test]
    fn a_rotary_the_head_cannot_fold_is_refused_at_the_strides() {
        for rotary in [0, 127, 1024] {
            assert!(
                derive_strides(64, 4, 512, 10, KvFormat::Folded { rotary }).is_none(),
                "rotary {rotary} must not derive strides"
            );
        }
        assert!(derive_strides(64, 4, 512, 10, KvFormat::Folded { rotary: 128 }).is_some());
    }
}
