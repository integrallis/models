// Copyright 2025-2026 Integrallis Software, LLC
// SPDX-License-Identifier: Apache-2.0
//! Owned HF tokenizer handles. Callers own buffers; no Rust allocation crosses the ABI.
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use tokenizers::Tokenizer;

static NEXT: AtomicU64 = AtomicU64::new(1);
static TOKENIZERS: LazyLock<Mutex<HashMap<u64, Arc<Tokenizer>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[unsafe(no_mangle)]
pub extern "C" fn jmodels_tokenizer_abi() -> i32 {
    1
}

/// Returns zero on invalid JSON or a panic. Input must address `length` readable bytes.
///
/// # Safety
/// `data` must address `length` readable bytes for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jmodels_tokenizer_create(data: *const u8, length: usize) -> u64 {
    catch_unwind(AssertUnwindSafe(|| {
        if data.is_null() || length == 0 || length > isize::MAX as usize {
            return 0;
        }
        let bytes = unsafe { std::slice::from_raw_parts(data, length) };
        let Ok(mut tokenizer) = Tokenizer::from_bytes(bytes) else {
            return 0;
        };
        // Schema compilation owns boundaries; never silently pad or truncate raw pieces.
        if tokenizer.with_truncation(None).is_err() {
            return 0;
        }
        tokenizer.with_padding(None);
        let handle = NEXT.fetch_add(1, Ordering::Relaxed);
        if handle == 0 {
            return 0;
        }
        let Ok(mut registry) = TOKENIZERS.lock() else {
            return 0;
        };
        registry.insert(handle, Arc::new(tokenizer));
        handle
    }))
    .unwrap_or(0)
}

/// Returns token count, or -1 for an invalid handle/input. If capacity is too small, returns
/// required count without writing. Pointers must address the stated readable/writable lengths.
///
/// # Safety
/// `data` must address `length` readable bytes. A non-null `output` must be aligned for u32,
/// writable for `capacity` elements, and must not alias input or tokenizer-owned memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jmodels_tokenizer_encode(
    handle: u64,
    data: *const u8,
    length: usize,
    output: *mut u32,
    capacity: usize,
) -> i64 {
    catch_unwind(AssertUnwindSafe(|| {
        if data.is_null() || length > isize::MAX as usize {
            return -1;
        }
        let tokenizer = match TOKENIZERS.lock() {
            Ok(registry) => registry.get(&handle).cloned(),
            Err(_) => None,
        };
        let Some(tokenizer) = tokenizer else {
            return -1;
        };
        let bytes = unsafe { std::slice::from_raw_parts(data, length) };
        let Ok(text) = std::str::from_utf8(bytes) else {
            return -1;
        };
        let Ok(encoded) = tokenizer.encode(text, false) else {
            return -1;
        };
        let ids = encoded.get_ids();
        if ids.len() > capacity {
            return ids.len() as i64;
        }
        if !ids.is_empty() {
            if output.is_null() {
                return -1;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(ids.as_ptr(), output, ids.len());
            }
        }
        ids.len() as i64
    }))
    .unwrap_or(-1)
}

#[unsafe(no_mangle)]
pub extern "C" fn jmodels_tokenizer_destroy(handle: u64) {
    let _ = catch_unwind(|| {
        if let Ok(mut registry) = TOKENIZERS.lock() {
            registry.remove(&handle);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_inputs_and_closed_handles_fail_without_unwinding() {
        assert_eq!(unsafe { jmodels_tokenizer_create(b"{}".as_ptr(), 2) }, 0);
        assert_eq!(
            unsafe { jmodels_tokenizer_encode(0, b"x".as_ptr(), 1, std::ptr::null_mut(), 0) },
            -1
        );
        jmodels_tokenizer_destroy(0);
    }
    #[test]
    fn caller_buffers_and_handle_lifetime() {
        let tokenizer = Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default());
        let json = tokenizer.to_string(false).unwrap();
        let handle = unsafe { jmodels_tokenizer_create(json.as_ptr(), json.len()) };
        assert_ne!(handle, 0);
        assert_eq!(
            unsafe { jmodels_tokenizer_encode(handle, b"".as_ptr(), 0, std::ptr::null_mut(), 0) },
            0
        );
        jmodels_tokenizer_destroy(handle);
        assert_eq!(
            unsafe { jmodels_tokenizer_encode(handle, b"".as_ptr(), 0, std::ptr::null_mut(), 0) },
            -1
        );
        jmodels_tokenizer_destroy(handle);
    }

    #[test]
    fn insufficient_buffers_utf8_and_concurrent_calls() {
        let json = br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1},"unk_token":"[UNK]"}}"#;
        let handle = unsafe { jmodels_tokenizer_create(json.as_ptr(), json.len()) };
        assert_ne!(handle, 0);
        let mut output = [99u32; 2];
        assert_eq!(
            unsafe {
                jmodels_tokenizer_encode(
                    handle,
                    b"hello hello".as_ptr(),
                    11,
                    output.as_mut_ptr(),
                    1,
                )
            },
            2
        );
        assert_eq!(output, [99, 99]);
        assert_eq!(
            unsafe {
                jmodels_tokenizer_encode(handle, [255u8].as_ptr(), 1, output.as_mut_ptr(), 2)
            },
            -1
        );
        let workers: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(move || {
                    let mut output = [99u32; 2];
                    for _ in 0..20 {
                        assert_eq!(
                            unsafe {
                                jmodels_tokenizer_encode(
                                    handle,
                                    b"hello hello".as_ptr(),
                                    11,
                                    output.as_mut_ptr(),
                                    2,
                                )
                            },
                            2
                        );
                        assert_eq!(output, [1, 1]);
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        jmodels_tokenizer_destroy(handle);
    }
}
