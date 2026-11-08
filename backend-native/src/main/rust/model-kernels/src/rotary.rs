//! Float rotary factors for the reference image-decoder arithmetic contract.
use super::{STATUS_INVALID_SHAPE, STATUS_NULL_POINTER, STATUS_OK};

#[unsafe(no_mangle)]
/// # Safety
/// Non-null buffers must contain `pairs` floats. The writable cosine and sine
/// buffers must be disjoint and must not alias the optional frequency divisors.
pub unsafe extern "C" fn jmodels_vision_rotary_factors(
    position: i32,
    pairs: i32,
    base: f32,
    scale: f32,
    magnitude: f32,
    divisors: *const f32,
    cosine: *mut f32,
    sine: *mut f32,
) -> i32 {
    if cosine.is_null() || sine.is_null() {
        return STATUS_NULL_POINTER;
    }
    if position < 0
        || !(1..=8192).contains(&pairs)
        || cosine == sine
        || !base.is_finite()
        || base <= 0.0
        || !scale.is_finite()
        || scale <= 0.0
        || !magnitude.is_finite()
        || magnitude <= 0.0
    {
        return STATUS_INVALID_SHAPE;
    }
    let pairs = pairs as usize;
    // SAFETY: the caller guarantees the advertised lengths and non-aliasing outputs.
    let (cs, sn) = unsafe {
        (
            std::slice::from_raw_parts_mut(cosine, pairs),
            std::slice::from_raw_parts_mut(sine, pairs),
        )
    };
    let ff = if divisors.is_null() {
        None
    } else {
        // SAFETY: a non-null divisor buffer has the same advertised length.
        Some(unsafe { std::slice::from_raw_parts(divisors, pairs) })
    };
    if ff.is_some_and(|values| values.iter().any(|x| !x.is_finite() || *x <= 0.0)) {
        return STATUS_INVALID_SHAPE;
    }
    let step = base.powf(-2.0 / (2 * pairs) as f32);
    let mut angle = position as f32;
    for pair in 0..pairs {
        let divisor = ff.map_or(1.0, |values| values[pair]);
        let scaled = scale * (angle / divisor);
        cs[pair] = scaled.cos() * magnitude;
        sn[pair] = scaled.sin() * magnitude;
        angle *= step;
    }
    STATUS_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_shapes_do_not_write_outputs() {
        let mut cs = [7_f32; 64];
        let mut sn = [8_f32; 64];
        // SAFETY: valid exclusive buffers; invalid shape must be rejected before writes.
        let result = unsafe {
            jmodels_vision_rotary_factors(
                -1,
                64,
                10000.0,
                0.25,
                1.0,
                std::ptr::null(),
                cs.as_mut_ptr(),
                sn.as_mut_ptr(),
            )
        };
        assert_eq!(result, STATUS_INVALID_SHAPE);
        assert_eq!(cs, [7.0; 64]);
        assert_eq!(sn, [8.0; 64]);
    }
    #[test]
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    fn float_factors_match_independent_linux_reference_bits() {
        let expected = include_bytes!("../testdata/rotary-factors/factors.f32");
        for (row, position) in [0, 1, 31, 639, 4095].into_iter().enumerate() {
            let mut cs = [0_f32; 64];
            let mut sn = [0_f32; 64];
            // SAFETY: complete and disjoint factor buffers.
            assert_eq!(
                unsafe {
                    jmodels_vision_rotary_factors(
                        position,
                        64,
                        10000.0,
                        0.25,
                        1.0,
                        std::ptr::null(),
                        cs.as_mut_ptr(),
                        sn.as_mut_ptr(),
                    )
                },
                STATUS_OK
            );
            for pair in 0..64 {
                for (lane, value) in [cs[pair], sn[pair]].into_iter().enumerate() {
                    let off = (row * 128 + pair * 2 + lane) * 4;
                    assert_eq!(
                        value.to_bits(),
                        u32::from_le_bytes(expected[off..off + 4].try_into().unwrap()),
                        "position={position}, pair={pair}, lane={lane}"
                    );
                }
            }
        }
    }
}
