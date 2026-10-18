use cudarc::driver::CudaSlice;
use half::bf16;

use crate::buffer::KvBuffer;
use crate::layout::KvLayout;

/// Lightweight, non-owning view of a request's KV state.
///
/// Built from a `SchedulableSequence`'s assigned block IDs before each
/// forward pass. Block lifecycle is managed externally by `BlockManager`.
#[derive(Clone)]
pub struct KvView {
    page_indices: Vec<i32>,
    seq_len: usize,
    page_size: usize,
}

impl KvView {
    /// `page_indices` must cover `seq_len` exactly: attention kernels derive
    /// the sequence length as `(num_pages - 1) * page_size + last_page_len`,
    /// so a surplus page makes them read garbage K/V past the sequence and a
    /// missing page makes them read out of bounds (#291).
    pub fn new(page_indices: Vec<i32>, seq_len: usize, page_size: usize) -> Self {
        assert_eq!(
            page_indices.len(),
            seq_len.div_ceil(page_size),
            "KvView pages must exactly cover seq_len={seq_len} (page_size={page_size})"
        );
        Self {
            page_indices,
            seq_len,
            page_size,
        }
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn num_pages(&self) -> usize {
        self.page_indices.len()
    }

    pub fn last_page_len(&self) -> usize {
        if self.seq_len == 0 {
            0
        } else {
            let rem = self.seq_len % self.page_size;
            if rem == 0 { self.page_size } else { rem }
        }
    }

    pub fn page_indices(&self) -> &[i32] {
        &self.page_indices
    }

    pub fn desc<'a>(&'a self, buffer: &'a KvBuffer) -> KvViewDesc<'a> {
        KvViewDesc {
            layout: *buffer.layout(),
            buffer: buffer.buffer(),
            pages: &self.page_indices,
            seq_len: self.seq_len,
            last_page_len: self.last_page_len(),
        }
    }
}

/// Kernel-facing metadata bundle.
pub struct KvViewDesc<'a> {
    layout: KvLayout,
    buffer: &'a CudaSlice<bf16>,
    pages: &'a [i32],
    seq_len: usize,
    last_page_len: usize,
}

impl KvViewDesc<'_> {
    pub fn layout(&self) -> &KvLayout {
        &self.layout
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn last_page_len(&self) -> usize {
        self.last_page_len
    }

    pub fn num_pages(&self) -> usize {
        self.pages.len()
    }

    pub fn page_indices(&self) -> &[i32] {
        self.pages
    }

    pub fn buffer(&self) -> &CudaSlice<bf16> {
        self.buffer
    }
}
