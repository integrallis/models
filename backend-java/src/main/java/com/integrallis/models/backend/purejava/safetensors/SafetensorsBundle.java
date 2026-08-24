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
package com.integrallis.models.backend.purejava.safetensors;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** One standard Safetensors file or an indexed set of root-level shards. */
public final class SafetensorsBundle {

  private static final String INDEX_FILENAME = "model.safetensors.index.json";
  private static final String SINGLE_FILENAME = "model.safetensors";

  private final Map<String, Path> weightMap;
  private final Map<Path, SafetensorsFile> shards;

  private SafetensorsBundle(Map<String, Path> weightMap, Map<Path, SafetensorsFile> shards) {
    this.weightMap = java.util.Collections.unmodifiableMap(new LinkedHashMap<>(weightMap));
    this.shards = Map.copyOf(shards);
  }

  /** Opens either a Safetensors file or a directory using standard Hugging Face names. */
  public static SafetensorsBundle open(Path source, Arena arena) throws IOException {
    Objects.requireNonNull(source, "source");
    Objects.requireNonNull(arena, "arena");
    Path normalized = source.toAbsolutePath().normalize();
    if (Files.isRegularFile(normalized)) {
      return single(normalized, arena);
    }
    if (!Files.isDirectory(normalized)) {
      throw new IOException("Safetensors source does not exist: " + normalized);
    }
    Path index = normalized.resolve(INDEX_FILENAME);
    if (Files.isRegularFile(index)) {
      return indexed(normalized, index, arena);
    }
    Path model = normalized.resolve(SINGLE_FILENAME);
    if (Files.isRegularFile(model)) {
      return single(model, arena);
    }
    throw new IOException(
        "Safetensors directory contains neither " + INDEX_FILENAME + " nor " + SINGLE_FILENAME);
  }

  /** Tensor names in index order, or header order for a single file. */
  public List<String> tensorNames() {
    return List.copyOf(weightMap.keySet());
  }

  /** Resolves a tensor through the index rather than assuming related tensors share a shard. */
  public SafetensorsTensor tensor(String name) {
    Path shard = weightMap.get(name);
    if (shard == null) {
      throw new IllegalArgumentException("Safetensors tensor not found: " + name);
    }
    return shards.get(shard).tensor(name);
  }

  /** Returns the mapped shard that the index assigns to a tensor. */
  public Path shardPath(String name) {
    Path shard = weightMap.get(name);
    if (shard == null) {
      throw new IllegalArgumentException("Safetensors tensor not found: " + name);
    }
    return shard;
  }

  private static SafetensorsBundle single(Path model, Arena arena) throws IOException {
    SafetensorsFile file = SafetensorsParser.parse(model, arena);
    Map<String, Path> weightMap = new LinkedHashMap<>();
    for (String name : file.tensorNames()) {
      weightMap.put(name, model);
    }
    return new SafetensorsBundle(weightMap, Map.of(model, file));
  }

  private static SafetensorsBundle indexed(Path directory, Path index, Arena arena)
      throws IOException {
    Map<String, String> encodedMap = readWeightMap(index);
    Map<String, Path> weightMap = new LinkedHashMap<>();
    Map<Path, SafetensorsFile> shards = new LinkedHashMap<>();
    for (Map.Entry<String, String> entry : encodedMap.entrySet()) {
      Path shard = resolveRootShard(directory, entry.getValue());
      SafetensorsFile file = shards.get(shard);
      if (file == null) {
        if (!Files.isRegularFile(shard)) {
          throw new MalformedSafetensorsException("indexed shard does not exist: " + shard);
        }
        file = SafetensorsParser.parse(shard, arena);
        shards.put(shard, file);
      }
      if (!file.contains(entry.getKey())) {
        throw new MalformedSafetensorsException(
            "index maps tensor "
                + entry.getKey()
                + " to "
                + shard.getFileName()
                + ", but that shard does not contain it");
      }
      weightMap.put(entry.getKey(), shard);
    }
    for (Map.Entry<Path, SafetensorsFile> shard : shards.entrySet()) {
      for (String name : shard.getValue().tensorNames()) {
        if (!shard.getKey().equals(weightMap.get(name))) {
          throw new MalformedSafetensorsException(
              "tensor "
                  + name
                  + " in shard "
                  + shard.getKey().getFileName()
                  + " is absent from the index");
        }
      }
    }
    return new SafetensorsBundle(weightMap, shards);
  }

  private static Map<String, String> readWeightMap(Path index) throws IOException {
    Map<String, String> weightMap = new LinkedHashMap<>();
    try (JsonParser parser = SafetensorsParser.JSON.createParser(index.toFile())) {
      require(parser.nextToken(), JsonToken.START_OBJECT, "index root");
      boolean found = false;
      while (parser.nextToken() != JsonToken.END_OBJECT) {
        require(parser.currentToken(), JsonToken.FIELD_NAME, "index field");
        String field = parser.currentName();
        JsonToken value = parser.nextToken();
        if ("weight_map".equals(field)) {
          require(value, JsonToken.START_OBJECT, "weight_map");
          found = true;
          while (parser.nextToken() != JsonToken.END_OBJECT) {
            require(parser.currentToken(), JsonToken.FIELD_NAME, "weight_map tensor name");
            String name = parser.currentName();
            require(parser.nextToken(), JsonToken.VALUE_STRING, "shard filename for " + name);
            weightMap.put(name, parser.getText());
          }
        } else {
          parser.skipChildren();
        }
      }
      if (parser.nextToken() != null) {
        throw new MalformedSafetensorsException("index has content after its root object");
      }
      if (!found || weightMap.isEmpty()) {
        throw new MalformedSafetensorsException("index must contain a non-empty weight_map");
      }
      return weightMap;
    } catch (MalformedSafetensorsException malformed) {
      throw malformed;
    } catch (IOException malformed) {
      throw new MalformedSafetensorsException(
          "invalid or duplicate Safetensors index JSON: " + malformed.getMessage(), malformed);
    }
  }

  private static Path resolveRootShard(Path directory, String filename) {
    if (filename.isBlank()
        || filename.contains("/")
        || filename.contains("\\")
        || !filename.endsWith(".safetensors")) {
      throw new MalformedSafetensorsException(
          "index shard must be a root shard filename ending in .safetensors: " + filename);
    }
    Path relative = Path.of(filename);
    if (relative.isAbsolute() || relative.getNameCount() != 1) {
      throw new MalformedSafetensorsException(
          "index shard must be a root shard filename: " + filename);
    }
    Path resolved = directory.resolve(relative).normalize();
    if (!directory.equals(resolved.getParent())) {
      throw new MalformedSafetensorsException(
          "index shard must be a root shard filename: " + filename);
    }
    return resolved;
  }

  private static void require(JsonToken actual, JsonToken expected, String description) {
    if (actual != expected) {
      throw new MalformedSafetensorsException(
          description + " must be " + expected + "; got " + actual);
    }
  }
}
