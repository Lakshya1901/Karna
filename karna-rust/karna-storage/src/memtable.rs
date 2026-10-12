use std::collections::BTreeMap;
use std::sync::RwLock;

/// MemTable is the in-memory store for key-value pairs before they are flushed to disk.
///
/// Ownership and Concurrency:
/// - We use `std::sync::RwLock` to allow multiple concurrent readers (get operations) 
///   but only exclusive access for writers (put/delete operations).
/// - The `BTreeMap` owns the keys and values (Strings) to ensure that the data 
///   persists in memory as long as the MemTable exists. 
/// - When flushing to an SSTable, we will lock the memtable, extract the data (transferring ownership),
///   and clear the tree.
pub struct MemTable {
    data: RwLock<BTreeMap<String, String>>,
}

impl MemTable {
    /// Creates a new, empty MemTable.
    pub fn new() -> Self {
        MemTable {
            data: RwLock::new(BTreeMap::new()),
        }
    }

    /// Inserts a key-value pair into the MemTable.
    pub fn put(&self, key: String, value: String) {
        // We acquire a write lock here. Only one thread can write at a time.
        // The lock is automatically dropped when `_guard` goes out of scope.
        let mut guard = self.data.write().unwrap();
        guard.insert(key, value);
    }

    /// Retrieves a value by its key.
    pub fn get(&self, key: &str) -> Option<String> {
        // We acquire a read lock. Multiple threads can read concurrently.
        let guard = self.data.read().unwrap();
        guard.get(key).cloned()
    }

    /// Clears the memtable, taking ownership of the current data and returning it.
    /// This is typically used during a flush to an SSTable.
    pub fn extract_data(&self) -> BTreeMap<String, String> {
        let mut guard = self.data.write().unwrap();
        // Replace the current map with an empty one, taking ownership of the old data.
        std::mem::take(&mut *guard)
    }
}

