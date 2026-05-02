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
package com.integrallis.models.backend.purejava.gguf;

/** Static-only utility class for GGUF binary format constants. */
public final class GgufConstants {

  private GgufConstants() {}

  /** GGUF magic number: "GGUF" in little-endian (bytes 0x47, 0x47, 0x55, 0x46). */
  public static final int MAGIC = 0x46554747;

  /** Default tensor data alignment in bytes. */
  public static final int DEFAULT_ALIGNMENT = 32;

  private static final int MIN_SUPPORTED_VERSION = 2;
  private static final int MAX_SUPPORTED_VERSION = 3;

  /** Returns {@code true} if the given GGUF version is supported by this parser. */
  public static boolean isVersionSupported(int version) {
    return version >= MIN_SUPPORTED_VERSION && version <= MAX_SUPPORTED_VERSION;
  }
}
