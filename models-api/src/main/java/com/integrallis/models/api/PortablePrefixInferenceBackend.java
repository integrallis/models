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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Optional lossless transfer of immutable base-model KV prefixes between compatible backends.
 *
 * <p>Callers provide authenticated, bounded transport and isolation. Implementations must validate
 * compatibility before allocating tensors and must not modify an existing session during import.
 * This transfers computed KV state, not sampler state, next-token logits or a whole conversation.
 * Identity, byte-size queries and immutable-prefix export must be safe alongside inference; they
 * must not acquire the model's generation lock or read mutable sequence buffers.
 */
public interface PortablePrefixInferenceBackend extends SharedPrefixInferenceBackend {
  boolean supportsPortablePrefixes();

  /**
   * Describes portability without hashing model files, acquiring the generation lock, or accessing
   * mutable session state. The default preserves legacy providers' boolean result and leaves their
   * truncation semantics explicitly unspecified. Implementations should supply a specific reason
   * when unavailable and keep this result consistent with {@link #supportsPortablePrefixes()}.
   */
  default PortablePrefixCapabilities portablePrefixCapabilities() {
    return supportsPortablePrefixes()
        ? PortablePrefixCapabilities.supported(PortablePrefixCapabilities.Truncation.UNSPECIFIED)
        : PortablePrefixCapabilities.unsupported(
            PortablePrefixCapabilities.Status.BACKEND_UNSUPPORTED);
  }

  /** Stable digest binding exact weights/tokenizer, cache ABI and effective execution settings. */
  String portablePrefixFingerprint() throws IOException;

  /** Exact serialized backend payload bytes, including identity and dimensions. */
  long portablePrefixBytes(int tokenCount);

  /**
   * Returns a shorter immutable view, retaining its original token positions and storage. Supported
   * boundaries follow {@link #portablePrefixCapabilities()}; recurrent state at the end of a prefix
   * cannot be reused as state at an earlier boundary. Unavailable boundaries must be rejected.
   */
  SharedInferencePrefix truncatePrefix(SharedInferencePrefix prefix, int tokenCount);

  /** Writes a bounded-length prefix in the backend's versioned wire format. */
  void writePrefix(SharedInferencePrefix prefix, int tokenCount, DataOutput output)
      throws IOException;

  /** Reads a new detached prefix after validating this backend's identity and shape. */
  SharedInferencePrefix readPrefix(DataInput input, int tokenCount, long maxBytes)
      throws IOException;

  /**
   * Writes a single contiguous token range of an existing prefix as a self-describing block.
   *
   * <p>The block includes the backend identity, range metadata ({@link PortablePrefixBlock}) and the
   * payload bytes for that range only. Callers stream multiple blocks independently; assembly is
   * atomic on the receiving side.
   *
   * @param prefix        the shared prefix to export from
   * @param startToken    first token included in the block (inclusive)
   * @param endToken      first token after the block (exclusive)
   * @param output        destination stream; must remain open for the caller
   * @throws IOException  if the range is unavailable or streaming fails
   */
  default void writePrefixBlock(
      SharedInferencePrefix prefix, int startToken, int endToken, DataOutput output)
      throws IOException {
    throw new UnsupportedOperationException("block-range export is not implemented");
  }

  /**
   * Reads a sequence of independently verified blocks and assembles a complete prefix.
   *
   * <p>The input format is the same identity used by {@link #writePrefix}, followed by a count and a
   * sequence of blocks. The receiver validates every block's range, identity and shape before
   * allocating tensors. Overlapping or missing ranges are rejected.
   *
   * @param input       stream containing the identity, block descriptors and payloads
   * @param tokenCount  total tokens the assembled prefix must cover (sum of block ranges)
   * @param maxBytes    maximum bytes to read from the stream
   * @return an assembled detached prefix covering exactly {@code tokenCount} tokens
   * @throws IOException if validation fails, ranges overlap or are missing, or the budget is exceeded
   */
  default SharedInferencePrefix readPrefixBlocks(DataInput input, int tokenCount, long maxBytes)
      throws IOException {
    throw new UnsupportedOperationException("block-range assembly is not implemented");
  }
}
