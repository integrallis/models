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

/**
 * Immutable description of one exported portable-prefix block (a contiguous token range).
 *
 * <p>A complete prefix may be assembled from independently verified blocks. Each block carries its
 * own start/end positions and payload identity so the receiver can validate coverage before
 * allocating mutable tensors.
 */
public record PortablePrefixBlock(int startToken, int endToken, int payloadBytes) {

  public PortablePrefixBlock {
    if (startToken < 0)
      throw new IllegalArgumentException("startToken must be >= 0: " + startToken);
    if (endToken <= startToken)
      throw new IllegalArgumentException(
          "endToken must be > startToken: " + endToken + " <= " + startToken);
    if (payloadBytes < 0)
      throw new IllegalArgumentException("payloadBytes must be >= 0: " + payloadBytes);
  }

  /** Number of tokens covered by this block. */
  public int tokenCount() {
    return endToken - startToken;
  }
}
