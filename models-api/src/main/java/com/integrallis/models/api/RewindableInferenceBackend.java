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

/** Optional state capability for retaining and restoring an exact sequence prefix. */
public interface RewindableInferenceBackend extends InferenceBackend {

  /** Returns the next sequence position, suitable for a later {@link #rewind(int)} call. */
  int checkpoint();

  /** Discards cached sequence state at and after {@code checkpoint}. */
  void rewind(int checkpoint);
}
