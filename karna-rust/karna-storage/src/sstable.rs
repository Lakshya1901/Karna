use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// SSTable (Sorted String Table) provides persistent, immutable storage for key-value pairs.
///
/// Ownership and Concurrency:
/// - Once created, an SSTable on disk is immutable.
/// - The `SSTableReader` owns a `File` handle (wrapped in a `BufReader`).
/// - Since the file is read-only, multiple threads could technically read from it,
///   but in a concurrent environment, it's often better for each thread to have its own file handle,
///   or to use memory-mapped files (mmap) to let the OS handle concurrency.
/// - Here, we provide simple synchronous I/O operations for demonstration.
pub struct SSTableWriter;

impl SSTableWriter {
    /// Flushes a BTreeMap of data to an SSTable file.
    /// Ownership of the `data` is taken to consume it completely during the write.
    pub fn flush_to_disk<P: AsRef<Path>>(path: P, data: BTreeMap<String, String>) -> io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Simple encoding: 
        // [key length (8 bytes)] [key bytes] [value length (8 bytes)] [value bytes]
        for (key, value) in data {
            let key_bytes = key.as_bytes();
            let val_bytes = value.as_bytes();
            
            writer.write_all(&(key_bytes.len() as u64).to_le_bytes())?;
            writer.write_all(key_bytes)?;
            
            writer.write_all(&(val_bytes.len() as u64).to_le_bytes())?;
            writer.write_all(val_bytes)?;
        }
        
        writer.flush()?;
        Ok(())
    }
}

pub struct SSTableReader {
    reader: BufReader<File>,
}

impl SSTableReader {
    /// Opens an SSTable file for reading.
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        Ok(SSTableReader {
            reader: BufReader::new(file),
        })
    }

    /// Scans the SSTable to find a specific key.
    /// This is a naive linear scan for demonstration.
    pub fn get(&mut self, target_key: &str) -> io::Result<Option<String>> {
        use std::io::Seek;
        
        // Rewind to the beginning for each search
        self.reader.seek(io::SeekFrom::Start(0))?;
        
        loop {
            let mut len_buf = [0u8; 8];
            match self.reader.read_exact(&mut len_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
                Err(e) => return Err(e),
            }
            
            let key_len = u64::from_le_bytes(len_buf) as usize;
            let mut key_buf = vec![0u8; key_len];
            self.reader.read_exact(&mut key_buf)?;
            
            self.reader.read_exact(&mut len_buf)?;
            let val_len = u64::from_le_bytes(len_buf) as usize;
            let mut val_buf = vec![0u8; val_len];
            self.reader.read_exact(&mut val_buf)?;
            
            if let Ok(key_str) = String::from_utf8(key_buf) {
                if key_str == target_key {
                    if let Ok(val_str) = String::from_utf8(val_buf) {
                        return Ok(Some(val_str));
                    }
                }
            }
        }
    }
}

