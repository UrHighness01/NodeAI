//! AI-Managed Hot-Lock Splitting
//!
//! A self-optimizing lock that starts as a single global Mutex but dynamically
//! shards into an array of locks when high contention is detected by the
//! AI scheduler.

use alloc::{boxed::Box, collections::BTreeMap};
use spin::Mutex;
use core::sync::atomic::{AtomicU64, AtomicBool, Ordering};

const SHARD_COUNT: usize = 8;

/// A self-optimizing map that shards its internal BTreeMap under high contention.
pub struct HotMap<K, V> {
    global_lock: Mutex<BTreeMap<K, V>>,
    sharded:     AtomicBool,
    shards:      [Mutex<BTreeMap<K, V>>; SHARD_COUNT],
    contention:  AtomicU64,
}

impl<K: Ord + Clone + AsRef<[u8]>, V: Clone> HotMap<K, V> {
    pub fn new() -> Self {
        Self {
            global_lock: Mutex::new(BTreeMap::new()),
            sharded:     AtomicBool::new(false),
            // Initialize empty shards
            shards: [
                Mutex::new(BTreeMap::new()), Mutex::new(BTreeMap::new()),
                Mutex::new(BTreeMap::new()), Mutex::new(BTreeMap::new()),
                Mutex::new(BTreeMap::new()), Mutex::new(BTreeMap::new()),
                Mutex::new(BTreeMap::new()), Mutex::new(BTreeMap::new()),
            ],
            contention:  AtomicU64::new(0),
        }
    }

    /// Hash the key to determine the shard index using a simple FNV-1a hash
    fn shard_idx(key: &K) -> usize {
        let mut hash: u64 = 0xcbf29ce484222325;
        for &b in key.as_ref() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        (hash as usize) % SHARD_COUNT
    }

    /// Insert a value into the HotMap.
    pub fn insert(&self, key: K, value: V) {
        if self.sharded.load(Ordering::Relaxed) {
            let idx = Self::shard_idx(&key);
            self.shards[idx].lock().insert(key, value);
        } else {
            // Attempt to acquire global lock. If it fails to acquire immediately, record contention.
            match self.global_lock.try_lock() {
                Some(mut guard) => {
                    guard.insert(key, value);
                }
                None => {
                    self.contention.fetch_add(1, Ordering::Relaxed);
                    // Fallback to blocking acquire
                    self.global_lock.lock().insert(key, value);
                    self.check_split();
                }
            }
        }
    }

    /// Get a value from the HotMap.
    pub fn get(&self, key: &K) -> Option<V> {
        if self.sharded.load(Ordering::Relaxed) {
            let idx = Self::shard_idx(key);
            self.shards[idx].lock().get(key).cloned()
        } else {
            match self.global_lock.try_lock() {
                Some(guard) => guard.get(key).cloned(),
                None => {
                    self.contention.fetch_add(1, Ordering::Relaxed);
                    let val = self.global_lock.lock().get(key).cloned();
                    self.check_split();
                    val
                }
            }
        }
    }

    /// Remove a value from the HotMap.
    pub fn remove(&self, key: &K) -> Option<V> {
        if self.sharded.load(Ordering::Relaxed) {
            let idx = Self::shard_idx(key);
            self.shards[idx].lock().remove(key)
        } else {
            match self.global_lock.try_lock() {
                Some(mut guard) => guard.remove(key),
                None => {
                    self.contention.fetch_add(1, Ordering::Relaxed);
                    let val = self.global_lock.lock().remove(key);
                    self.check_split();
                    val
                }
            }
        }
    }
    
    /// Collect all items. This is expensive but necessary for full directory listings.
    pub fn snapshot(&self) -> BTreeMap<K, V> {
        if self.sharded.load(Ordering::Relaxed) {
            let mut result = BTreeMap::new();
            for shard in &self.shards {
                for (k, v) in shard.lock().iter() {
                    result.insert(k.clone(), v.clone());
                }
            }
            result
        } else {
            self.global_lock.lock().clone()
        }
    }

    /// Check if contention threshold is met, and if so, dynamically split.
    fn check_split(&self) {
        if self.contention.load(Ordering::Relaxed) > 1000 {
            // Threshold met. Only one thread should split it.
            let mut guard = self.global_lock.lock();
            if self.sharded.load(Ordering::Relaxed) { return; } // already split by another thread

            crate::klog!(WARN, "hot_lock: AI detected >1000 lock contention events. Dynamically sharding HotMap into {} granular locks.", SHARD_COUNT);

            // Move all items from the global map to the shards.
            let mut keys_to_move = alloc::vec::Vec::new();
            for (k, _) in guard.iter() {
                keys_to_move.push(k.clone());
            }

            for k in keys_to_move {
                if let Some(v) = guard.remove(&k) {
                    let idx = Self::shard_idx(&k);
                    self.shards[idx].lock().insert(k, v);
                }
            }

            // Engage EL scriptable policy to permanently route to shards.
            self.sharded.store(true, Ordering::SeqCst);
        }
    }
}
