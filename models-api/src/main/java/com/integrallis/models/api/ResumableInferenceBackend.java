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
 * A backend that can return to a position it has already read, without reading those tokens again.
 *
 * <p>This is narrower than {@link SharedPrefixInferenceBackend}, deliberately. A shared prefix
 * hands out branches that live at the same time; a resumption is strictly sequential -- capture a
 * point, run past it, come back, run past it again. Sequential reuse is what a repeated decision
 * over one piece of evidence needs, and it is the only shape a recurrent state can offer cheaply: a
 * Gated DeltaNet state is a running summary, so it can be copied back but not sliced.
 *
 * <p>Without it, every question about the same document pays for the whole document again.
 */
public interface ResumableInferenceBackend extends InferenceBackend {

  /** Whether the loaded model graph can resume, which not every architecture can. */
  default boolean supportsResumption() {
    return false;
  }

  /**
   * Captures the current position so it can be returned to.
   *
   * @return an opaque handle to this backend's state at the current position
   */
  Resumption capture();

  /**
   * Returns to a captured position, discarding everything read after it.
   *
   * @param point a handle previously returned by {@link #capture()} on this backend
   */
  void resume(Resumption point);

  /** An opaque position handle. Only the backend that issued it can resume it. */
  interface Resumption {
    /** The token position this handle returns to. */
    int position();
  }
}
