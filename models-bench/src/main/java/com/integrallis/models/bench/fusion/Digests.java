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
package com.integrallis.models.bench.fusion;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** SHA-256 helpers; file digests are memoized per path, size and modification time. */
final class Digests {

  private static final Map<String, String> FILE_CACHE = new ConcurrentHashMap<>();

  private Digests() {}

  static String sha256(String value) {
    return sha256(value.getBytes(StandardCharsets.UTF_8));
  }

  static String sha256(byte[] bytes) {
    return HexFormat.of().formatHex(digest().digest(bytes));
  }

  static String sha256(Path path) throws IOException {
    Path real = path.toRealPath();
    String key = real + "|" + Files.size(real) + "|" + Files.getLastModifiedTime(real).toMillis();
    String cached = FILE_CACHE.get(key);
    if (cached != null) {
      return cached;
    }
    MessageDigest digest = digest();
    try (InputStream input = Files.newInputStream(real)) {
      byte[] buffer = new byte[4 * 1024 * 1024];
      int read;
      while ((read = input.read(buffer)) >= 0) {
        digest.update(buffer, 0, read);
      }
    }
    String value = HexFormat.of().formatHex(digest.digest());
    FILE_CACHE.put(key, value);
    return value;
  }

  static MessageDigest digest() {
    try {
      return MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException impossible) {
      throw new IllegalStateException("SHA-256 is unavailable", impossible);
    }
  }
}
