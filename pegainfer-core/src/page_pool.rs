use std::sync::Arc;

use parking_lot::Mutex;

/// Shared fixed-page allocator with RAII-owned permits.
#[derive(Clone, Debug)]
pub struct PagePool {
    inner: Arc<PagePoolInner>,
}

#[derive(Debug)]
struct PagePoolInner {
    capacity_pages: usize,
    free_list: Mutex<Vec<PageId>>,
}

/// A set of pages leased from a [`PagePool`]. Pages return automatically on drop.
#[derive(Debug)]
pub struct OwnedPagePermit {
    inner: Arc<PagePoolInner>,
    pages: Vec<PageId>,
}

/// Opaque page identifier within a pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PageId(usize);

impl PageId {
    pub(crate) fn index(self) -> usize {
        self.0
    }
}

impl PagePool {
    pub(crate) fn new(capacity_pages: usize) -> Self {
        let free_list = (0..capacity_pages).rev().map(PageId).collect();
        Self {
            inner: Arc::new(PagePoolInner {
                capacity_pages,
                free_list: Mutex::new(free_list),
            }),
        }
    }

    pub(crate) fn try_acquire_many(&self, n: usize) -> Option<OwnedPagePermit> {
        if n == 0 {
            return Some(OwnedPagePermit {
                inner: Arc::clone(&self.inner),
                pages: Vec::new(),
            });
        }

        let mut free_list = self.inner.free_list.lock();
        if free_list.len() < n {
            return None;
        }

        let mut pages = Vec::with_capacity(n);
        for _ in 0..n {
            pages.push(free_list.pop().expect("free_list length checked above"));
        }

        Some(OwnedPagePermit {
            inner: Arc::clone(&self.inner),
            pages,
        })
    }

    pub(crate) fn capacity_pages(&self) -> usize {
        self.inner.capacity_pages
    }

    pub(crate) fn available_pages(&self) -> usize {
        self.inner.free_list.lock().len()
    }
}

impl OwnedPagePermit {
    pub(crate) fn pages(&self) -> &[PageId] {
        &self.pages
    }

    // `is_empty` was dead code (hawk sweep, #743); `len` stands alone.
    #[allow(clippy::len_without_is_empty)]
    pub(crate) fn len(&self) -> usize {
        self.pages.len()
    }

    /// Both permits must lease from the same pool; merging across pools would
    /// return pages to the wrong free list.
    pub(crate) fn absorb(&mut self, mut other: OwnedPagePermit) {
        assert!(
            Arc::ptr_eq(&self.inner, &other.inner),
            "absorb requires permits from the same pool"
        );
        self.pages.append(&mut other.pages);
    }

    /// Try to acquire `n` more pages, appending them to this permit.
    /// Returns `true` on success. On failure the permit is unchanged.
    pub(crate) fn try_grow(&mut self, n: usize) -> bool {
        if n == 0 {
            return true;
        }

        let mut free_list = self.inner.free_list.lock();
        if free_list.len() < n {
            return false;
        }

        self.pages.reserve(n);
        for _ in 0..n {
            self.pages
                .push(free_list.pop().expect("free_list length checked above"));
        }
        true
    }
}

impl Drop for OwnedPagePermit {
    fn drop(&mut self) {
        if self.pages.is_empty() {
            return;
        }

        let pages = std::mem::take(&mut self.pages);
        let mut free_list = self.inner.free_list.lock();
        free_list.extend(pages.into_iter().rev());
    }
}

#[cfg(test)]
mod tests {
    use super::PageId;
    use super::PagePool;

    #[test]
    fn acquires_pages_and_restores_on_drop() {
        let pool = PagePool::new(4);
        assert_eq!(pool.capacity_pages(), 4);

        {
            let permit = pool.try_acquire_many(2).expect("acquire should succeed");
            assert_eq!(permit.len(), 2);
            assert_eq!(permit.pages(), &[PageId(0), PageId(1)]);
            assert_eq!(pool.available_pages(), 2);

            let permit_b = pool
                .try_acquire_many(1)
                .expect("second acquire should succeed");
            assert_eq!(permit_b.pages(), &[PageId(2)]);
            assert_eq!(pool.available_pages(), 1);
        }

        assert_eq!(pool.available_pages(), 4);
        let permit = pool
            .try_acquire_many(4)
            .expect("released pages should be reusable");
        assert_eq!(
            permit.pages(),
            &[PageId(0), PageId(1), PageId(2), PageId(3)]
        );
    }

    #[test]
    fn reports_insufficient_capacity_with_none() {
        let pool = PagePool::new(2);
        let _permit = pool.try_acquire_many(1).expect("acquire should succeed");

        assert!(pool.try_acquire_many(2).is_none());
    }

    #[test]
    fn try_grow_extends_permit_and_returns_all_on_drop() {
        let pool = PagePool::new(4);

        {
            let mut permit = pool.try_acquire_many(1).expect("initial acquire");
            assert_eq!(permit.pages(), &[PageId(0)]);

            assert!(permit.try_grow(2));
            assert_eq!(permit.pages(), &[PageId(0), PageId(1), PageId(2)]);
            assert_eq!(pool.available_pages(), 1);

            // grow beyond remaining capacity fails, permit unchanged
            assert!(!permit.try_grow(2));
            assert_eq!(permit.len(), 3);
            assert_eq!(pool.available_pages(), 1);
        }

        // all 4 pages back after drop
        assert_eq!(pool.available_pages(), 4);
    }
}
