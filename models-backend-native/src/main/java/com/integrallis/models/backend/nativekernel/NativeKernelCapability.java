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
package com.integrallis.models.backend.nativekernel;

/** Compute operations implemented by a compatible Models native-kernel library. */
public enum NativeKernelCapability {
  Q4_0_F32_BATCHED_MATMUL(1L),
  Q4_0_F32_GROUPED_BATCHED_MATMUL(1L << 1),
  PERSISTENT_WORKER_CONTEXT(1L << 2);

  private final long mask;

  NativeKernelCapability(long mask) {
    this.mask = mask;
  }

  long mask() {
    return mask;
  }
}
