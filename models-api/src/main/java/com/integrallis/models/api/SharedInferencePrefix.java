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
 * Opaque immutable key/value-cache prefix owned by one loaded backend.
 *
 * <p>The prefix remains valid until its backend is closed. Forking it creates independent mutable
 * suffixes that reference the same immutable key/value storage; implementations must not satisfy
 * this contract by copying or recomputing the prefix.
 */
public interface SharedInferencePrefix {

  /** Returns the next token position after the shared prefix. */
  int checkpoint();

  /** Returns the key/value and bookkeeping bytes physically shared by all forks. */
  long sharedBytes();
}
