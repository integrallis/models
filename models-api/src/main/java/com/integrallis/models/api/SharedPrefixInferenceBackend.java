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

import java.util.Optional;

/**
 * Optional backend capability for physically sharing an immutable KV-cache prefix between divergent
 * inference branches of the same loaded model.
 *
 * <p>This capability is deliberately narrower than model routing. Prefixes cannot cross backend
 * instances or incompatible weights, and every fork begins at the prefix checkpoint. The source
 * session is consumed by {@link #freezePrefix}; callers must use a fork for subsequent inference.
 */
public interface SharedPrefixInferenceBackend extends BatchInferenceBackend {

  /** Returns whether the currently loaded model graph supports immutable physical prefixes. */
  default boolean supportsSharedPrefixes() {
    return true;
  }

  /** Selects the computation applied after the shared prefix. */
  enum Branch {
    /** Continue with the exact base model. */
    BASE,

    /** Continue with the backend's pinned activated adapter. */
    ACTIVATED_ADAPTER
  }

  /** Returns whether this loaded backend has a qualified activated adapter. */
  boolean supportsActivatedBranch();

  /** Returns the exact adapter provenance and invocation contract when one is loaded. */
  default Optional<ActivatedAdapterMetadata> activatedAdapter() {
    return Optional.empty();
  }

  /**
   * Arms a base-prefilled independent session so its next tokens must begin the pinned adapter
   * invocation sequence and execute on the activated branch.
   *
   * <p>This supports an honest recomputation control for prompts below a measured prefix-sharing
   * crossover. The already-evaluated base prefix is never retroactively adapter-modified.
   */
  default void activateAdapter(InferenceSession session) {
    throw new UnsupportedOperationException("backend cannot activate an adapter branch");
  }

  /**
   * Freezes a nonempty session as immutable shared storage and consumes the source session.
   *
   * @throws IllegalArgumentException if the source is empty or belongs to another backend
   */
  SharedInferencePrefix freezePrefix(InferenceSession source);

  /** Opens a base-model branch without copying or recomputing the prefix. */
  default InferenceSession fork(SharedInferencePrefix prefix) {
    return fork(prefix, Branch.BASE);
  }

  /** Opens the requested branch without copying or recomputing the prefix. */
  InferenceSession fork(SharedInferencePrefix prefix, Branch branch);

  /**
   * Returns whether two open sessions reference the same physical immutable KV-cache prefix.
   *
   * <p>This diagnostic exists so qualification and runtime telemetry can prove physical sharing
   * rather than infer it from timing alone.
   */
  boolean sharesPrefixStorage(InferenceSession first, InferenceSession second);
}
