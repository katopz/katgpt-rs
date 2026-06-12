//! SenseModule serialization — binary format with BLAKE3 verification.

use crate::types::SenseModule;
use std::io::{self, Read, Write};

const MAGIC: &[u8; 4] = b"SNSE";
const VERSION: u8 = 1;

/// Save a SenseModule to a writer.
pub fn save_module(module: &SenseModule, mut w: impl Write) -> io::Result<()> {
    w.write_all(MAGIC)?;
    w.write_all(&[VERSION])?;
    w.write_all(&[module.kind as u8])?;
    // Stack buffer to zero padding — avoids cloning the full SenseModule.
    let mut copy: std::mem::MaybeUninit<SenseModule> = std::mem::MaybeUninit::uninit();
    unsafe {
        std::ptr::copy_nonoverlapping(module, copy.as_mut_ptr(), 1);
        // SAFETY: copy is now a valid SenseModule. Zero padding in directions.
        let copy_ref = copy.assume_init_mut();
        for dir in &mut copy_ref.directions {
            dir.zero_padding();
        }
        let bytes: &[u8] = std::slice::from_raw_parts(
            copy_ref as *const SenseModule as *const u8,
            std::mem::size_of::<SenseModule>(),
        );
        w.write_all(bytes)?;
        // SenseModule has no Drop impl — MaybeUninit drops as unit, no cleanup needed.
    }
    Ok(())
}

/// Load a SenseModule from a reader. Verifies BLAKE3.
pub fn load_module(mut r: impl Read) -> io::Result<SenseModule> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid SNSE magic",
        ));
    }
    let mut ver = [0u8; 1];
    r.read_exact(&mut ver)?;
    if ver[0] != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported SNSE version",
        ));
    }
    let mut kind_buf = [0u8; 1];
    r.read_exact(&mut kind_buf)?;

    let mut module_bytes = [0u8; std::mem::size_of::<SenseModule>()];
    r.read_exact(&mut module_bytes)?;

    // SAFETY: buffer is zero-initialized (covers padding), SenseModule is
    // repr(C) so field layout matches the byte offsets. All fields are
    // valid for any bit pattern (integers, f32, arrays).
    let module: SenseModule =
        unsafe { std::ptr::read(module_bytes.as_ptr() as *const SenseModule) };

    if !module.verify() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "BLAKE3 checksum mismatch",
        ));
    }
    Ok(module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sense::octree::{KgEmbedding, SenseOctreeBuilder};
    use crate::types::SenseKind;

    #[test]
    fn test_roundtrip() {
        let builder = SenseOctreeBuilder::new(3);
        let emb = KgEmbedding {
            entity_hash: 42,
            relation_hash: 7,
            embedding: [0.5, -0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            sign: true,
            confidence: 1.0,
        };
        let original = builder.build(SenseKind::FighterSense, &[emb]);

        let mut buf = Vec::new();
        save_module(&original, &mut buf).unwrap();

        let loaded = load_module(&buf[..]).unwrap();
        assert_eq!(loaded.kind, original.kind);
        assert_eq!(loaded.octree_bits, original.octree_bits);
        assert_eq!(loaded.n_directions, original.n_directions);
    }

    #[test]
    fn test_invalid_magic_rejected() {
        let buf = b"XXXX\x01\x00\x00";
        assert!(load_module(&buf[..]).is_err());
    }
}
