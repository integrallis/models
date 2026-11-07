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
package com.integrallis.models.api;

/** Stateful image-and-text inference. Image positions share the text sequence and context limit. */
public interface ImageInferenceBackend extends InferenceBackend {
  /** Number of decoder positions occupied by one image. */
  int imageTokenCount();

  /**
   * Decodes and consumes one PNG/JPEG image at the next sequence position. Returns stable,
   * caller-owned logits after its final projected patch. The image must precede the following text
   * prefill at position + imageTokenCount(). Call reset before an independent request.
   * Implementations are not thread safe.
   */
  float[] prefillImage(byte[] encodedImage, int position);
}
