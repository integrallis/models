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
package com.integrallis.models.backend.purejava.lfm2;

import java.util.Objects;

/**
 * LFM2's short convolution: the operator that stands where attention stands on most of its layers.
 *
 * <p>LFM2.5-1.2B has 16 layers of which only six are attention. The other ten replace it with a
 * gated depthwise convolution over a three-token window, which needs a <b>recurrent state</b>
 * rather than a key-value cache: two past values per channel, shifted forward on every token. That
 * is why this architecture cannot simply be added to the Llama decoder -- the state has a different
 * shape and a different update rule.
 *
 * <p>The block, transcribed from llama.cpp's {@code build_shortconv_block}:
 *
 * <pre>{@code
 * bcx = in_proj . x            // 3 * dim, three equal chunks in the order b, c, x
 * bx  = b * x                  // elementwise gate
 * window = [state..., bx]      // lCache values per channel, oldest first
 * conv   = sum(kernel[tap] * window[tap])
 * y      = c * conv            // the second gate
 * out    = out_proj . y
 * }</pre>
 *
 * <p>Two orderings here are read from the reference rather than assumed, because both are invisible
 * when wrong -- they produce plausible numbers, not errors:
 *
 * <ul>
 *   <li><b>Tap 0 multiplies the oldest element.</b> ggml's {@code
 *       ggml_compute_forward_ssm_conv_f32} accumulates {@code s[i0 + i1*ncs] * c[i0 + i1*nc]} with
 *       the window starting at the current token's position, so index 0 of the kernel lines up with
 *       the furthest-back value.
 *   <li><b>The chunks are b, c, x in that order.</b> The reference takes chunk 0 as {@code b},
 *       chunk 1 as {@code c} and chunk 2 as {@code x}, then gates with {@code b * x} before the
 *       convolution and with {@code c} after it. Swapping {@code b} and {@code c} still type-checks
 *       and still produces fluent text.
 * </ul>
 *
 * <p>State layout is channel-major with the taps contiguous, matching the reference's reshape to
 * {@code [d_conv, n_embd]} and the kernel's own {@code [lCache, n_embd]} GGUF shape.
 */
public final class Lfm2ShortConv {

  private Lfm2ShortConv() {}

  /**
   * Applies one token of the short convolution and advances the state.
   *
   * @param projected the {@code in_proj} output, {@code 3 * dim} values as b, c, x
   * @param dim the model width
   * @param kernel the depthwise taps, {@code lCache * dim}, tap-contiguous per channel
   * @param lCache the convolution width, at least two
   * @param state the recurrent state, {@code (lCache - 1) * dim}, tap-contiguous per channel and
   *     oldest first; advanced in place
   * @param gated receives {@code dim} values, the convolution output gated by c
   */
  public static void applyToken(
      float[] projected, int dim, float[] kernel, int lCache, float[] state, float[] gated) {
    Objects.requireNonNull(projected, "projected");
    Objects.requireNonNull(kernel, "kernel");
    Objects.requireNonNull(state, "state");
    Objects.requireNonNull(gated, "gated");
    if (lCache < 2) {
      throw new IllegalArgumentException("lCache must be at least 2, was " + lCache);
    }
    int taps = lCache - 1;
    if (projected.length < 3 * dim) {
      throw new IllegalArgumentException(
          "projected must hold 3 * " + dim + " values, was " + projected.length);
    }
    if (kernel.length < (long) lCache * dim) {
      throw new IllegalArgumentException(
          "kernel must hold " + lCache + " * " + dim + " values, was " + kernel.length);
    }
    if (state.length < (long) taps * dim) {
      throw new IllegalArgumentException(
          "state must hold " + taps + " * " + dim + " values, was " + state.length);
    }
    if (gated.length < dim) {
      throw new IllegalArgumentException("gated must hold " + dim + " values");
    }

    for (int channel = 0; channel < dim; channel++) {
      float b = projected[channel];
      float c = projected[dim + channel];
      float x = projected[2 * dim + channel];
      float bx = b * x;

      int stateBase = channel * taps;
      int kernelBase = channel * lCache;
      // Tap 0 is the oldest value, which is state slot 0; the newest tap multiplies this token.
      float sum = 0.0f;
      for (int tap = 0; tap < taps; tap++) {
        sum += kernel[kernelBase + tap] * state[stateBase + tap];
      }
      sum += kernel[kernelBase + taps] * bx;

      gated[channel] = c * sum;

      // Shift the window forward: the oldest value falls out and this token's gate enters.
      for (int tap = 0; tap < taps - 1; tap++) {
        state[stateBase + tap] = state[stateBase + tap + 1];
      }
      state[stateBase + taps - 1] = bx;
    }
  }
}
