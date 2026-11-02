//! Interleaved Q4_K weight layout for the batched prefill GEMM.
//!
//! Eight consecutive output rows are packed into one block so that a GEMM can hold eight *different*
//! output rows in the eight i32 lanes of a YMM register. Today's
//! `compute_q4_k_batched_row_range_avx2` instead puts partial sums of a *single* output element in
//! those lanes, which forces a horizontal reduction per (output row, batch row) and makes each
//! weight load serve one output instead of eight. Removing that is the point of this layout; see
//! `qk_batched_tile_lanes` in `lib.rs`, where the lane decision lives.
//!
//! The layout is ggml's `block_q4_Kx8` and the packing is a port of its `make_block_q4_Kx8` and
//! `repack_q4_K_to_q4_K_8_bl` (MIT, ggerganov/ggml). Byte-for-byte compatibility with ggml is
//! deliberate: it lets their kernel be used as an oracle against ours on identical bytes.
//!
//! This module only rearranges bytes. It decodes nothing and loses nothing: `interleave_rows`
//! followed by `deinterleave_row` returns the original block, which `round_trips_every_byte`
//! asserts over pseudo-random blocks.

/// Elements per Q4_K super-block.
pub(crate) const QK_K: usize = 256;
/// Bytes per source `block_q4_K`: d, dmin, twelve packed six-bit scale bytes, 128 nibble bytes.
pub(crate) const Q4_K_BLOCK_BYTES: usize = 144;
/// Output rows packed per interleaved block.
pub(crate) const INTERLEAVE_ROWS: usize = 8;
/// Bytes copied per interleaving step. ggml calls this `blck_size_interleave`.
pub(crate) const INTERLEAVE_BLOCK: usize = 8;
/// Bytes per `block_q4_Kx8`: eight d, eight dmin, 96 scale bytes, 1024 nibble bytes.
pub(crate) const Q4_KX8_BLOCK_BYTES: usize = 16 + 16 + 96 + 1024;

const SCALES_OFFSET: usize = 4;
const QUANTS_OFFSET: usize = 16;

/// Packs eight source blocks, one per output row, into a single interleaved block.
///
/// `sources[i]` is row `i`'s `Q4_K_BLOCK_BYTES` for one super-block. `out` receives
/// `Q4_KX8_BLOCK_BYTES`.
pub(crate) fn make_block(sources: &[&[u8]; INTERLEAVE_ROWS], out: &mut [u8]) {
    debug_assert_eq!(out.len(), Q4_KX8_BLOCK_BYTES);
    for (row, source) in sources.iter().enumerate() {
        debug_assert_eq!(source.len(), Q4_K_BLOCK_BYTES);
        // d and dmin keep their little-endian fp16 bytes; this layout never decodes them.
        out[row * 2] = source[0];
        out[row * 2 + 1] = source[1];
        out[16 + row * 2] = source[2];
        out[16 + row * 2 + 1] = source[3];
    }

    // Nibbles, eight bytes at a time, round-robin across the eight rows.
    let steps = QK_K * 4 / INTERLEAVE_BLOCK;
    for step in 0..steps {
        let row = step % INTERLEAVE_ROWS;
        let src_offset = (step / INTERLEAVE_ROWS) * INTERLEAVE_BLOCK;
        let dst_offset = 128 + step * INTERLEAVE_BLOCK;
        let from = QUANTS_OFFSET + src_offset;
        out[dst_offset..dst_offset + INTERLEAVE_BLOCK]
            .copy_from_slice(&sources[row][from..from + INTERLEAVE_BLOCK]);
    }

    // Scales and mins. Each twelve output bytes carry one sub-block's eight scales and eight mins,
    // one per row, with the high two bits folded in exactly as ggml packs them.
    let mut s = [0_u8; INTERLEAVE_ROWS];
    let mut m = [0_u8; INTERLEAVE_ROWS];
    for i in 0..4 {
        for j in 0..INTERLEAVE_ROWS {
            s[j] = sources[j][SCALES_OFFSET + i] & 63;
            m[j] = sources[j][SCALES_OFFSET + i + 4] & 63;
        }
        write_twelve(&mut out[32 + i * 12..], &s, &m);
    }
    for i in 0..4 {
        for j in 0..INTERLEAVE_ROWS {
            let sc = sources[j][SCALES_OFFSET + i];
            let mn = sources[j][SCALES_OFFSET + i + 4];
            let hi = sources[j][SCALES_OFFSET + i + 8];
            s[j] = ((sc & 192) >> 2) | (hi & 15);
            m[j] = ((mn & 192) >> 2) | ((hi & 240) >> 4);
        }
        write_twelve(&mut out[32 + 48 + i * 12..], &s, &m);
    }
}

#[inline]
fn write_twelve(out: &mut [u8], s: &[u8; INTERLEAVE_ROWS], m: &[u8; INTERLEAVE_ROWS]) {
    for k in 0..4 {
        out[k] = (s[k] & 63) + ((s[k + 4] & 48) << 2);
        out[k + 4] = (m[k] & 63) + ((m[k + 4] & 48) << 2);
        out[k + 8] = (s[k + 4] & 15) + ((m[k + 4] & 15) << 4);
    }
}

/// Interleaves a whole weight matrix. `rows` must be a multiple of eight.
///
/// Returns `None` rather than guessing when the shape cannot be interleaved, because a silently
/// mis-shaped repack would read as a correctness bug in the kernel that consumes it.
pub(crate) fn interleave_rows(weights: &[u8], rows: usize, cols: usize) -> Option<Vec<u8>> {
    if rows % INTERLEAVE_ROWS != 0 || cols % QK_K != 0 {
        return None;
    }
    let blocks = cols / QK_K;
    let row_bytes = blocks * Q4_K_BLOCK_BYTES;
    if weights.len() < rows * row_bytes {
        return None;
    }
    let mut out = vec![0_u8; rows / INTERLEAVE_ROWS * blocks * Q4_KX8_BLOCK_BYTES];
    let mut cursor = 0;
    for group in (0..rows).step_by(INTERLEAVE_ROWS) {
        for block in 0..blocks {
            let mut slots: [&[u8]; INTERLEAVE_ROWS] = [&[]; INTERLEAVE_ROWS];
            for row in 0..INTERLEAVE_ROWS {
                let at = (group + row) * row_bytes + block * Q4_K_BLOCK_BYTES;
                slots[row] = &weights[at..at + Q4_K_BLOCK_BYTES];
            }
            make_block(&slots, &mut out[cursor..cursor + Q4_KX8_BLOCK_BYTES]);
            cursor += Q4_KX8_BLOCK_BYTES;
        }
    }
    Some(out)
}

/// Recovers one source row's block from an interleaved block. Inverse of `make_block`.
///
/// Exists so the layout can be proven lossless rather than assumed to be, and so a kernel reading
/// it can be checked against the scalar path on the same bytes.
pub(crate) fn deinterleave_row(block: &[u8], row: usize, out: &mut [u8]) {
    debug_assert_eq!(block.len(), Q4_KX8_BLOCK_BYTES);
    debug_assert_eq!(out.len(), Q4_K_BLOCK_BYTES);
    out[0] = block[row * 2];
    out[1] = block[row * 2 + 1];
    out[2] = block[16 + row * 2];
    out[3] = block[16 + row * 2 + 1];

    let steps = QK_K * 4 / INTERLEAVE_BLOCK;
    for step in 0..steps {
        if step % INTERLEAVE_ROWS != row {
            continue;
        }
        let src_offset = (step / INTERLEAVE_ROWS) * INTERLEAVE_BLOCK;
        let dst_offset = 128 + step * INTERLEAVE_BLOCK;
        out[QUANTS_OFFSET + src_offset..QUANTS_OFFSET + src_offset + INTERLEAVE_BLOCK]
            .copy_from_slice(&block[dst_offset..dst_offset + INTERLEAVE_BLOCK]);
    }

    for i in 0..4 {
        let low = &block[32 + i * 12..];
        let high = &block[32 + 48 + i * 12..];
        let (s_low, m_low) = read_pair(low, row);
        let (s_high, m_high) = read_pair(high, row);
        out[SCALES_OFFSET + i] = (s_low & 63) | ((s_high & 48) << 2);
        out[SCALES_OFFSET + i + 4] = (m_low & 63) | ((m_high & 48) << 2);
        out[SCALES_OFFSET + i + 8] = (s_high & 15) | ((m_high & 15) << 4);
    }
}

#[inline]
fn read_pair(twelve: &[u8], row: usize) -> (u8, u8) {
    if row < 4 {
        (twelve[row] & 63, twelve[row + 4] & 63)
    } else {
        let k = row - 4;
        let s = (twelve[k + 8] & 15) | ((twelve[k] >> 2) & 48);
        let m = ((twelve[k + 8] >> 4) & 15) | ((twelve[k + 4] >> 2) & 48);
        (s, m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_block(seed: &mut u32) -> Vec<u8> {
        let mut block = vec![0_u8; Q4_K_BLOCK_BYTES];
        for byte in block.iter_mut() {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 17;
            *seed ^= *seed << 5;
            *byte = (*seed & 0xFF) as u8;
        }
        block
    }

    #[test]
    fn round_trips_every_byte() {
        // The layout is only a rearrangement, so de-interleaving has to return the exact source
        // bytes for all eight rows. Anything less means the GEMM reading it would be scoring
        // weights that are not the model's.
        let mut seed = 0x9E37_79B9_u32;
        for _ in 0..64 {
            let sources: Vec<Vec<u8>> = (0..INTERLEAVE_ROWS).map(|_| pseudo_block(&mut seed)).collect();
            let refs: [&[u8]; INTERLEAVE_ROWS] = std::array::from_fn(|i| sources[i].as_slice());
            let mut packed = vec![0_u8; Q4_KX8_BLOCK_BYTES];
            make_block(&refs, &mut packed);
            let mut recovered = vec![0_u8; Q4_K_BLOCK_BYTES];
            for row in 0..INTERLEAVE_ROWS {
                deinterleave_row(&packed, row, &mut recovered);
                assert_eq!(recovered, sources[row], "row {row} did not round-trip");
            }
        }
    }

    #[test]
    fn interleaves_a_matrix_and_recovers_each_row() {
        let rows = 16;
        let cols = 512;
        let blocks = cols / QK_K;
        let row_bytes = blocks * Q4_K_BLOCK_BYTES;
        let mut seed = 0x1234_5678_u32;
        let mut weights = vec![0_u8; rows * row_bytes];
        for byte in weights.iter_mut() {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *byte = (seed & 0xFF) as u8;
        }
        let packed = interleave_rows(&weights, rows, cols).expect("shape is interleavable");
        assert_eq!(packed.len(), rows / INTERLEAVE_ROWS * blocks * Q4_KX8_BLOCK_BYTES);
        let mut recovered = vec![0_u8; Q4_K_BLOCK_BYTES];
        for row in 0..rows {
            for block in 0..blocks {
                let group = row / INTERLEAVE_ROWS;
                let at = (group * blocks + block) * Q4_KX8_BLOCK_BYTES;
                deinterleave_row(&packed[at..at + Q4_KX8_BLOCK_BYTES], row % INTERLEAVE_ROWS, &mut recovered);
                let from = row * row_bytes + block * Q4_K_BLOCK_BYTES;
                assert_eq!(&recovered[..], &weights[from..from + Q4_K_BLOCK_BYTES], "row {row} block {block}");
            }
        }
    }

    #[test]
    fn refuses_shapes_it_cannot_interleave() {
        let cols = 256;
        let weights = vec![0_u8; 8 * Q4_K_BLOCK_BYTES];
        assert!(interleave_rows(&weights, 7, cols).is_none(), "rows not a multiple of eight");
        assert!(interleave_rows(&weights, 8, 300).is_none(), "cols not a multiple of QK_K");
        assert!(interleave_rows(&weights[..16], 8, cols).is_none(), "truncated input");
    }
}

/// Exposes the interleaver to an out-of-crate oracle check.
///
/// The round-trip tests prove the layout is lossless; they cannot prove it is the *same* layout
/// ggml defines, because ggml's packer is `static` and cannot be linked. This entry lets a harness
/// feed our bytes to ggml's own `ggml_gemv_q4_K_8x8_q8_K` and compare against its row-major
/// `vec_dot_q4_K_q8_K` on the source weights, which settles compatibility using their kernel as the
/// oracle rather than our reading of their source.
///
/// Returns the bytes written, or 0 if the shape cannot be interleaved or the output is too small.
///
/// # Safety
///
/// `weights` must address `rows * (cols / 256) * 144` bytes and `out` at least `out_len`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jmodels_q4k_interleave_rows(
    weights: *const u8,
    rows: u32,
    cols: u32,
    out: *mut u8,
    out_len: u64,
) -> u64 {
    if weights.is_null() || out.is_null() {
        return 0;
    }
    let rows = rows as usize;
    let cols = cols as usize;
    if cols % QK_K != 0 {
        return 0;
    }
    let needed = rows * (cols / QK_K) * Q4_K_BLOCK_BYTES;
    // SAFETY: the caller guarantees the source extent.
    let source = unsafe { std::slice::from_raw_parts(weights, needed) };
    let Some(packed) = interleave_rows(source, rows, cols) else {
        return 0;
    };
    if (out_len as usize) < packed.len() {
        return 0;
    }
    // SAFETY: the caller guarantees out_len bytes and we just checked it is large enough.
    unsafe { std::ptr::copy_nonoverlapping(packed.as_ptr(), out, packed.len()) };
    packed.len() as u64
}
