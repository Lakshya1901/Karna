use crate::memtable::MemTable;
use crate::sstable::{SSTableReader, SSTableWriter};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// LsmEngine coordinates between the in-memory MemTable and on-disk SSTables.
///
/// Ownership and Concurrency:
/// - `LsmEngine` owns the `MemTable` and the state for on-disk files.
/// - We use `Arc` to allow multiple threads to share ownership of the engine.
/// - The `MemTable` internally handles concurrent reads/writes using an `RwLock`.
/// - The list of SSTable files is protected by a `Mutex` to serialize updates to the file list 
///   when flushes occur. 
pub struct LsmEngine {
    memtable: MemTable,
    storage_dir: PathBuf,
    // The list of SSTable file paths. Protected by a Mutex because it changes when we flush.
    sstable_files: Mutex<Vec<PathBuf>>,
}

impl LsmEngine {
    /// Initializes a new LSM engine with a given storage directory.
    pub fn new(storage_dir: PathBuf) -> Arc<Self> {
        // Ensure storage directory exists
        std::fs::create_dir_all(&storage_dir).unwrap_or_default();
        
        Arc::new(LsmEngine {
            memtable: MemTable::new(),
            storage_dir,
            sstable_files: Mutex::new(Vec::new()),
        })
    }

    /// Puts a key-value pair into the database.
    pub fn put(&self, key: String, value: String) {
        // Writes directly to the in-memory memtable.
        self.memtable.put(key, value);
    }

    /// Retrieves a value for a given key.
    pub fn get(&self, key: &str) -> Option<String> {
        // 1. First, check the memtable
        if let Some(val) = self.memtable.get(key) {
            return Some(val);
        }

        // 2. If not found, check the SSTables (from newest to oldest ideally, but we just scan here)
        let files_guard = self.sstable_files.lock().unwrap();
        for file_path in files_guard.iter().rev() {
            if let Ok(mut reader) = SSTableReader::new(file_path) {
                if let Ok(Some(val)) = reader.get(key) {
                    return Some(val);
                }
            }
        }
        
        None
    }

    /// Flushes the current MemTable to an SSTable on disk.
    pub fn flush(&self) {
        let data = self.memtable.extract_data();
        if data.is_empty() {
            return;
        }

        let mut files_guard = self.sstable_files.lock().unwrap();
        let file_id = files_guard.len();
        let file_path = self.storage_dir.join(format!("sstable_{}.sst", file_id));
        
        if SSTableWriter::flush_to_disk(&file_path, data).is_ok() {
            files_guard.push(file_path);
        }
    }
}

