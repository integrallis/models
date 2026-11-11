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

import java.util.Objects;

/**
 * Immutable description of portable-prefix support for a loaded backend configuration.
 *
 * <p>This describes capability, not peer compatibility or model qualification. A supported backend
 * must still validate the exact execution identity, token sequence and state layout on import.
 * Unsupported configurations report the first blocking condition, not an exhaustive diagnosis.
 */
public record PortablePrefixCapabilities(Status status, Truncation truncation) {
  /** Machine-readable support status. */
  public enum Status {
    SUPPORTED,
    /** The backend does not implement portability, or supplies no more specific reason. */
    BACKEND_UNSUPPORTED,
    /** Complete decoder state has no portable representation. */
    DECODER_STATE_UNSUPPORTED,
    /** The model file needed to establish exact identity is unavailable. */
    MODEL_IDENTITY_UNAVAILABLE,
    /** Activated-adapter state has no portable representation. */
    ACTIVATED_STATE_UNSUPPORTED,
    /** The attention kernel's state has no portable representation. */
    ATTENTION_STATE_UNSUPPORTED,
    /** The matrix kernel has not opted in for its current state and execution identity. */
    MATRIX_KERNEL_UNSUPPORTED,
    /** The native library needed to establish exact identity is unavailable. */
    NATIVE_IDENTITY_UNAVAILABLE
  }

  /** Which shorter token boundaries can be represented without recomputation. */
  public enum Truncation {
    /** Every nonempty prefix up to the captured token count. */
    ANY_PREFIX,
    /** Only boundaries for which complete state was explicitly retained. */
    RETAINED_CHECKPOINTS,
    /** A legacy provider supports portability but has not declared truncation semantics. */
    UNSPECIFIED,
    /** Portable prefixes are unavailable. */
    UNSUPPORTED
  }

  public PortablePrefixCapabilities {
    Objects.requireNonNull(status, "status");
    Objects.requireNonNull(truncation, "truncation");
    if ((status == Status.SUPPORTED) == (truncation == Truncation.UNSUPPORTED)) {
      throw new IllegalArgumentException("support status and truncation must agree");
    }
  }

  public boolean supported() {
    return status == Status.SUPPORTED;
  }

  public static PortablePrefixCapabilities supported(Truncation truncation) {
    return new PortablePrefixCapabilities(Status.SUPPORTED, truncation);
  }

  public static PortablePrefixCapabilities unsupported(Status reason) {
    return new PortablePrefixCapabilities(reason, Truncation.UNSUPPORTED);
  }
}
