/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.integrallis.models.backend.purejava.quant;

/** Utility for IEEE 754 half-precision (float16) conversions. */
public final class Float16 {

  private Float16() {}

  /** Converts a 16-bit half-precision float to a 32-bit float. */
  public static float toFloat(short f16Bits) {
    int bits = Short.toUnsignedInt(f16Bits);
    int sign = (bits >>> 15) & 0x1;
    int exponent = (bits >>> 10) & 0x1F;
    int mantissa = bits & 0x3FF;

    if (exponent == 0) {
      if (mantissa == 0) {
        // Zero (positive or negative)
        return Float.intBitsToFloat(sign << 31);
      }
      // Subnormal: convert to normalized float
      float value = mantissa / 1024.0f;
      value *= (1.0f / 16384.0f); // 2^-14
      return sign == 0 ? value : -value;
    } else if (exponent == 31) {
      if (mantissa == 0) {
        return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
      }
      return Float.NaN;
    }

    // Normalized value
    int f32Exponent = exponent - 15 + 127;
    int f32Mantissa = mantissa << 13;
    int f32Bits = (sign << 31) | (f32Exponent << 23) | f32Mantissa;
    return Float.intBitsToFloat(f32Bits);
  }

  /** Converts a 32-bit float to a 16-bit half-precision representation. */
  public static short fromFloat(float value) {
    int bits = Float.floatToRawIntBits(value);
    int sign = (bits >>> 31) & 0x1;
    int exponent = (bits >>> 23) & 0xFF;
    int mantissa = bits & 0x7FFFFF;

    if (exponent == 0xFF) {
      // Infinity or NaN
      if (mantissa == 0) {
        return (short) ((sign << 15) | 0x7C00);
      }
      return (short) ((sign << 15) | 0x7C00 | (mantissa >>> 13));
    }

    int f16Exponent = exponent - 127 + 15;
    if (f16Exponent <= 0) {
      if (f16Exponent < -10) {
        return (short) (sign << 15);
      }
      // Subnormal
      int f16Mantissa = (mantissa | 0x800000) >>> (1 - f16Exponent + 13);
      return (short) ((sign << 15) | f16Mantissa);
    }

    if (f16Exponent >= 31) {
      return (short) ((sign << 15) | 0x7C00);
    }

    int f16Mantissa = mantissa >>> 13;
    return (short) ((sign << 15) | (f16Exponent << 10) | f16Mantissa);
  }
}
