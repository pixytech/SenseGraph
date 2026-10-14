pub fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) & 1) as u32;
    let exponent = ((half >> 10) & 0x1F) as u32;
    let mantissa = (half & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal: normalize
        let mut e = exponent;
        let mut m = mantissa;
        while (m & 0x400) == 0 {
            m <<= 1;
            e = e.wrapping_sub(1);
        }
        e = e.wrapping_add(1);
        m &= !0x400;
        let f32_exp = (e as i32 + 127 - 15) as u32;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }

    if exponent == 31 {
        let f32_mantissa = mantissa << 13;
        return f32::from_bits((sign << 31) | (0xFF << 23) | f32_mantissa);
    }

    let f32_exp = (exponent as i32 + 127 - 15) as u32;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
}

pub fn bf16_to_f32(half: u16) -> f32 {
    f32::from_bits((half as u32) << 16)
}

pub fn decode_f16_buffer(src: &[u8], dst: &mut [f32]) {
    assert!(src.len() >= dst.len() * 2);
    for (i, chunk) in src.chunks_exact(2).enumerate().take(dst.len()) {
        let half = u16::from_le_bytes([chunk[0], chunk[1]]);
        dst[i] = f16_to_f32(half);
    }
}

pub fn decode_bf16_buffer(src: &[u8], dst: &mut [f32]) {
    assert!(src.len() >= dst.len() * 2);
    for (i, chunk) in src.chunks_exact(2).enumerate().take(dst.len()) {
        let half = u16::from_le_bytes([chunk[0], chunk[1]]);
        dst[i] = bf16_to_f32(half);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
    }

    #[test]
    fn test_f16_one() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xBC00), -1.0);
    }

    #[test]
    fn test_f16_half() {
        assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-6);
    }
}
