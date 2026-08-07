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

import com.integrallis.models.router.TaskIndexBuilder;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Remembers prompt embeddings between runs so comparisons do not re-embed the corpus.
 *
 * <p>Embedding dominates an index build — around 25 minutes for the router corpus against a few
 * seconds for everything downstream. Comparing storage choices such as quantization means building
 * the same vectors repeatedly, and paying for them each time makes the comparison expensive enough
 * to discourage running it.
 *
 * <p>Keyed by prompt text and scoped by a caller-supplied identity covering the model, its width
 * and anything else that changes what a vector means. A cache reused across models would silently
 * answer with another model's vectors, which is worse than the cost it saves.
 */
final class EmbeddingCache {

  private static final int MAGIC = 0x54_49_45_43; // TIEC
  private static final int VERSION = 1;

  private final Path file;
  private final String identity;
  private final Map<String, float[]> vectors = new HashMap<>();
  private int loaded;

  private EmbeddingCache(Path file, String identity) {
    this.file = file;
    this.identity = identity;
  }

  /**
   * Opens the cache at {@code file}, reading it when it exists and matches {@code identity}.
   *
   * @param file where the cache lives
   * @param identity what the cached vectors mean; a mismatch discards them rather than mixing
   * @return the cache
   */
  static EmbeddingCache open(Path file, String identity) {
    Objects.requireNonNull(file, "file");
    Objects.requireNonNull(identity, "identity");
    EmbeddingCache cache = new EmbeddingCache(file, identity);
    if (Files.isReadable(file)) {
      cache.read();
    }
    return cache;
  }

  /** How many vectors were read from disk. */
  int loaded() {
    return loaded;
  }

  /** How many vectors the cache holds now, including any computed this run. */
  int size() {
    return vectors.size();
  }

  /**
   * Wraps an embedder so its results are remembered.
   *
   * @param delegate the real embedder
   * @return an embedder that consults the cache first
   */
  TaskIndexBuilder.TaskEmbedder wrap(TaskIndexBuilder.TaskEmbedder delegate) {
    Objects.requireNonNull(delegate, "delegate");
    return text -> vectors.computeIfAbsent(text, delegate::embed);
  }

  /** Writes the cache back, replacing whatever was there. */
  void save() {
    try {
      Path parent = file.toAbsolutePath().getParent();
      if (parent != null) {
        Files.createDirectories(parent);
      }
      try (DataOutputStream out =
          new DataOutputStream(
              new java.io.BufferedOutputStream(Files.newOutputStream(file), 1 << 16))) {
        out.writeInt(MAGIC);
        out.writeInt(VERSION);
        byte[] identityBytes = identity.getBytes(StandardCharsets.UTF_8);
        out.writeInt(identityBytes.length);
        out.write(identityBytes);
        out.writeInt(vectors.size());
        for (Map.Entry<String, float[]> entry : vectors.entrySet()) {
          // Length-prefixed rather than writeUTF: a corpus prompt can exceed the 64KB that
          // modified UTF-8 encoding allows.
          byte[] key = entry.getKey().getBytes(StandardCharsets.UTF_8);
          out.writeInt(key.length);
          out.write(key);
          float[] vector = entry.getValue();
          out.writeInt(vector.length);
          for (float value : vector) {
            out.writeFloat(value);
          }
        }
      }
    } catch (IOException e) {
      throw new UncheckedIOException("cannot write embedding cache " + file, e);
    }
  }

  private void read() {
    try (DataInputStream in =
        new DataInputStream(new java.io.BufferedInputStream(Files.newInputStream(file), 1 << 16))) {
      if (in.readInt() != MAGIC || in.readInt() != VERSION) {
        return;
      }
      byte[] identityBytes = in.readNBytes(in.readInt());
      if (!identity.equals(new String(identityBytes, StandardCharsets.UTF_8))) {
        // A different model or width wrote this. Its vectors are not comparable with ours.
        return;
      }
      int count = in.readInt();
      for (int index = 0; index < count; index++) {
        String key = new String(in.readNBytes(in.readInt()), StandardCharsets.UTF_8);
        float[] vector = new float[in.readInt()];
        for (int component = 0; component < vector.length; component++) {
          vector[component] = in.readFloat();
        }
        vectors.put(key, vector);
      }
      loaded = vectors.size();
    } catch (IOException e) {
      // A truncated cache is a performance problem, not a correctness one: drop it and re-embed.
      vectors.clear();
      loaded = 0;
    }
  }
}
