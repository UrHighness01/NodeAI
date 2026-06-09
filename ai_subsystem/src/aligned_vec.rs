use alloc::alloc::{alloc_zeroed, dealloc, realloc, Layout};
use core::ptr::NonNull;
use core::ops::{Deref, DerefMut};

/// A dynamically sized array that guarantees a specific memory alignment.
/// Crucial for AVX2 instructions which trap on unaligned access.
pub struct AlignedVec<T, const ALIGN: usize> {
    ptr: NonNull<T>,
    cap: usize,
    len: usize,
}

unsafe impl<T: Send, const ALIGN: usize> Send for AlignedVec<T, ALIGN> {}
unsafe impl<T: Sync, const ALIGN: usize> Sync for AlignedVec<T, ALIGN> {}

impl<T, const ALIGN: usize> AlignedVec<T, ALIGN> {
    pub fn new() -> Self {
        assert!(ALIGN.is_power_of_two(), "Alignment must be a power of two");
        Self {
            ptr: NonNull::dangling(),
            cap: 0,
            len: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut vec = Self::new();
        vec.reserve(capacity);
        vec
    }

    pub fn push(&mut self, value: T) {
        if self.len == self.cap {
            let new_cap = if self.cap == 0 { 4 } else { self.cap * 2 };
            self.reserve(new_cap);
        }
        unsafe {
            self.ptr.as_ptr().add(self.len).write(value);
        }
        self.len += 1;
    }

    pub fn resize(&mut self, new_len: usize, value: T) where T: Clone {
        if new_len > self.len {
            self.reserve(new_len);
            for _ in self.len..new_len {
                self.push(value.clone());
            }
        } else {
            self.truncate(new_len);
        }
    }

    pub fn clear(&mut self) {
        self.truncate(0);
    }

    pub fn truncate(&mut self, len: usize) {
        if len > self.len {
            return;
        }
        unsafe {
            let mut current_len = self.len;
            while current_len > len {
                current_len -= 1;
                core::ptr::drop_in_place(self.ptr.as_ptr().add(current_len));
            }
        }
        self.len = len;
    }

    pub fn reserve(&mut self, additional: usize) {
        let new_cap = self.len.checked_add(additional).expect("Capacity overflow");
        if new_cap <= self.cap {
            return;
        }

        let new_layout = Layout::from_size_align(new_cap * core::mem::size_of::<T>(), ALIGN)
            .expect("Invalid layout");
        
        unsafe {
            let new_ptr = if self.cap == 0 {
                alloc_zeroed(new_layout)
            } else {
                let old_layout = Layout::from_size_align(self.cap * core::mem::size_of::<T>(), ALIGN)
                    .unwrap();
                realloc(self.ptr.as_ptr() as *mut u8, old_layout, new_layout.size())
            };
            
            if new_ptr.is_null() {
                alloc::alloc::handle_alloc_error(new_layout);
            }
            
            self.ptr = NonNull::new_unchecked(new_ptr as *mut T);
            self.cap = new_cap;
        }
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T, const ALIGN: usize> Drop for AlignedVec<T, ALIGN> {
    fn drop(&mut self) {
        if self.cap != 0 {
            self.clear();
            let layout = Layout::from_size_align(self.cap * core::mem::size_of::<T>(), ALIGN)
                .unwrap();
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl<T, const ALIGN: usize> Deref for AlignedVec<T, ALIGN> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const ALIGN: usize> DerefMut for AlignedVec<T, ALIGN> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

// Convert from slice for convenience
impl<T: Clone, const ALIGN: usize> From<&[T]> for AlignedVec<T, ALIGN> {
    fn from(slice: &[T]) -> Self {
        let mut vec = Self::with_capacity(slice.len());
        for item in slice {
            vec.push(item.clone());
        }
        vec
    }
}
