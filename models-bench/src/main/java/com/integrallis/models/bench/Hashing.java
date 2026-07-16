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
package com.integrallis.models.bench;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

/** SHA-256 helpers used for model, prompt, and output identity. */
final class Hashing {

  private Hashing() {}

  static String sha256(Path path) throws IOException {
    MessageDigest digest = digest();
    try (InputStream input = Files.newInputStream(path)) {
      byte[] buffer = new byte[1024 * 1024];
      int read;
      while ((read = input.read(buffer)) >= 0) {
        digest.update(buffer, 0, read);
      }
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  static String sha256(String value) {
    return HexFormat.of().formatHex(digest().digest(value.getBytes(StandardCharsets.UTF_8)));
  }

  static String sha256(float[] values) {
    MessageDigest digest = digest();
    byte[] bytes = new byte[Float.BYTES];
    for (float value : values) {
      int bits = Float.floatToRawIntBits(value);
      bytes[0] = (byte) (bits >>> 24);
      bytes[1] = (byte) (bits >>> 16);
      bytes[2] = (byte) (bits >>> 8);
      bytes[3] = (byte) bits;
      digest.update(bytes);
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  private static MessageDigest digest() {
    try {
      return MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException impossible) {
      throw new IllegalStateException("SHA-256 is unavailable", impossible);
    }
  }
}
