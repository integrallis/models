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
package com.integrallis.models.backend.purejava.cache;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.*;
import java.util.HexFormat;

/** Exact artifact and implementation identity; computed once when portability is requested. */
public final class PrefixFingerprint {
  private PrefixFingerprint() {}

  /** Stable value representation, including array-valued record fields (never JVM object IDs). */
  public static String configuration(Object value) {
    if (value == null) return "null";
    if (value instanceof boolean[] booleans) return java.util.Arrays.toString(booleans);
    if (value instanceof float[] floats) return java.util.Arrays.toString(floats);
    if (value instanceof Record record) {
      var text = new StringBuilder(record.getClass().getName()).append('[');
      for (var field : record.getClass().getRecordComponents()) {
        try {
          text.append(field.getName())
              .append('=')
              .append(configuration(field.getAccessor().invoke(record)))
              .append(';');
        } catch (ReflectiveOperationException failed) {
          throw new IllegalStateException("Cannot identify cache configuration", failed);
        }
      }
      return text.append(']').toString();
    }
    if (value instanceof String
        || value instanceof Number
        || value instanceof Boolean
        || value instanceof Enum<?>) return value.toString();
    throw new IllegalArgumentException(
        "Unsupported cache configuration type: " + value.getClass().getName());
  }

  public static String of(Path model, String configuration, Class<?>... implementations)
      throws IOException {
    MessageDigest digest = sha256();
    digest.update("models-portable-kv-f32-v1\n".getBytes(StandardCharsets.UTF_8));
    digest.update(fileDigest(model));
    digest.update(configuration.getBytes(StandardCharsets.UTF_8));
    for (Class<?> implementation : implementations) {
      digest.update(implementation.getName().getBytes(StandardCharsets.UTF_8));
      try {
        var location = implementation.getProtectionDomain().getCodeSource();
        Path path = location == null ? null : Path.of(location.getLocation().toURI());
        if (path != null && Files.isRegularFile(path)) {
          digest.update(fileDigest(path)); // Published JAR, including bundled kernels when present.
        } else {
          String resource = "/" + implementation.getName().replace('.', '/') + ".class";
          try (InputStream in = implementation.getResourceAsStream(resource)) {
            if (in == null) throw new IOException("missing implementation identity");
            digest.update(streamDigest(in));
          }
        }
      } catch (java.net.URISyntaxException invalid) {
        throw new IOException("invalid implementation location", invalid);
      }
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  private static byte[] fileDigest(Path path) throws IOException {
    try (InputStream in = Files.newInputStream(path)) {
      return streamDigest(in);
    }
  }

  private static byte[] streamDigest(InputStream in) throws IOException {
    var digest = sha256();
    byte[] buffer = new byte[1024 * 1024];
    for (int count; (count = in.read(buffer)) != -1; ) digest.update(buffer, 0, count);
    return digest.digest();
  }

  private static MessageDigest sha256() {
    try {
      return MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException impossible) {
      throw new AssertionError(impossible);
    }
  }
}
