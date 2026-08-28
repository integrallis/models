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
package com.integrallis.models.backend.purejava.internal;

import java.lang.foreign.Arena;
import java.util.Objects;

/** Owns an arena used across inference worker threads. Internal backend infrastructure only. */
public final class ModelMemoryArena implements AutoCloseable {

  private static final String NATIVE_IMAGE_CODE_PROPERTY = "org.graalvm.nativeimage.imagecode";

  private final Arena arena;
  private final boolean closeable;

  private ModelMemoryArena(Arena arena, boolean closeable) {
    this.arena = Objects.requireNonNull(arena, "arena");
    this.closeable = closeable;
  }

  public static ModelMemoryArena create() {
    return create(System.getProperty(NATIVE_IMAGE_CODE_PROPERTY) != null);
  }

  public static ModelMemoryArena create(boolean nativeImage) {
    return nativeImage
        ? new ModelMemoryArena(Arena.ofAuto(), false)
        : new ModelMemoryArena(Arena.ofShared(), true);
  }

  public static ModelMemoryArena owning(Arena arena) {
    return new ModelMemoryArena(arena, true);
  }

  public Arena arena() {
    return arena;
  }

  @Override
  public void close() {
    if (closeable) {
      arena.close();
    }
  }
}
