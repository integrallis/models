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

  /** Stable digest binding exact weights/tokenizer, cache ABI and effective execution settings. */
  String portablePrefixFingerprint() throws IOException;

  /** Exact serialized backend payload bytes, including identity and dimensions. */
  long portablePrefixBytes(int tokenCount);

  /** Returns a shorter immutable view, retaining its original token positions and storage. */
  SharedInferencePrefix truncatePrefix(SharedInferencePrefix prefix, int tokenCount);

  /** Writes a bounded-length prefix in the backend's versioned wire format. */
  void writePrefix(SharedInferencePrefix prefix, int tokenCount, DataOutput output)
      throws IOException;

  /** Reads a new detached prefix after validating this backend's identity and shape. */
  SharedInferencePrefix readPrefix(DataInput input, int tokenCount, long maxBytes)
      throws IOException;
}
