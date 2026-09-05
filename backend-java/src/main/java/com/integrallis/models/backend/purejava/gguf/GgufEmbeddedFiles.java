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

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Embedded configuration and tokenizer sidecars carried by an audio.cpp GGUF artifact. */
public final class GgufEmbeddedFiles {

  private static final String NAMES_KEY = "audiocpp.embedded_files.names";
  private static final String OFFSETS_KEY = "audiocpp.embedded_files.offsets";
  private static final String DATA_KEY = "audiocpp.embedded_files.data";

  private final Map<String, byte[]> files;

  private GgufEmbeddedFiles(Map<String, byte[]> files) {
    this.files = files;
  }

  /** Parses the binary sidecar layout used by standalone audio.cpp GGUF packages. */
  public static GgufEmbeddedFiles from(GgufMetadata metadata) {
    Objects.requireNonNull(metadata, "metadata");
    GgufMetadataValue namesValue = metadata.entries().get(NAMES_KEY);
    GgufMetadataValue offsetsValue = metadata.entries().get(OFFSETS_KEY);
    GgufMetadataValue dataValue = metadata.entries().get(DATA_KEY);
    if (namesValue == null && offsetsValue == null && dataValue == null) {
      return new GgufEmbeddedFiles(Map.of());
    }

    List<String> names = stringArray(namesValue, NAMES_KEY);
    long[] offsets = uint64Array(offsetsValue, OFFSETS_KEY);
    byte[] data = uint8Array(dataValue, DATA_KEY);
    if (offsets.length != names.size() + 1) {
      throw new IllegalArgumentException(
          "GGUF embedded file offsets must contain one more entry than names");
    }

    Map<String, byte[]> parsed = new LinkedHashMap<>();
    for (int index = 0; index < names.size(); index++) {
      String name = safeRelativeName(names.get(index));
      long start = offsets[index];
      long end = offsets[index + 1];
      if (start < 0 || end < start || end > data.length) {
        throw new IllegalArgumentException(
            "GGUF embedded file byte range is invalid for " + name + ": " + start + ".." + end);
      }
      byte[] previous = parsed.put(name, Arrays.copyOfRange(data, (int) start, (int) end));
      if (previous != null) {
        throw new IllegalArgumentException("GGUF contains duplicate embedded file name: " + name);
      }
    }
    return new GgufEmbeddedFiles(Collections.unmodifiableMap(new LinkedHashMap<>(parsed)));
  }

  /** Returns embedded file names in artifact order. */
  public List<String> names() {
    return List.copyOf(files.keySet());
  }

  /** Returns a caller-owned copy of one embedded file. */
  public byte[] read(String name) {
    Objects.requireNonNull(name, "name");
    byte[] value = files.get(name);
    if (value == null) {
      throw new IllegalArgumentException("Embedded GGUF file not found: " + name);
    }
    return value.clone();
  }

  /** Decodes one embedded file as UTF-8 text. */
  public String readUtf8(String name) {
    return new String(read(name), StandardCharsets.UTF_8);
  }

  private static List<String> stringArray(GgufMetadataValue value, String key) {
    GgufMetadataValue.ArrayValue array = requireArray(value, key, GgufValueType.STRING);
    List<String> result = new ArrayList<>(array.elements().size());
    for (GgufMetadataValue element : array.elements()) {
      result.add(((GgufMetadataValue.StringValue) element).value());
    }
    return result;
  }

  private static long[] uint64Array(GgufMetadataValue value, String key) {
    GgufMetadataValue.ArrayValue array = requireArray(value, key, GgufValueType.UINT64);
    long[] result = new long[array.elements().size()];
    for (int index = 0; index < result.length; index++) {
      result[index] = ((GgufMetadataValue.Uint64Value) array.elements().get(index)).value();
    }
    return result;
  }

  private static byte[] uint8Array(GgufMetadataValue value, String key) {
    GgufMetadataValue.ArrayValue array = requireArray(value, key, GgufValueType.UINT8);
    byte[] result = new byte[array.elements().size()];
    for (int index = 0; index < result.length; index++) {
      result[index] = (byte) ((GgufMetadataValue.Uint8Value) array.elements().get(index)).value();
    }
    return result;
  }

  private static GgufMetadataValue.ArrayValue requireArray(
      GgufMetadataValue value, String key, GgufValueType elementType) {
    if (!(value instanceof GgufMetadataValue.ArrayValue array)
        || array.elementType() != elementType) {
      throw new IllegalArgumentException(
          "GGUF embedded file metadata " + key + " must be a " + elementType + " array");
    }
    return array;
  }

  private static String safeRelativeName(String name) {
    if (name == null || name.isBlank()) {
      throw new IllegalArgumentException("GGUF contains an unsafe empty embedded file name");
    }
    String portable = name.replace('\\', '/');
    Path path = Path.of(portable).normalize();
    if (portable.startsWith("/")
        || portable.matches("^[A-Za-z]:.*")
        || path.isAbsolute()
        || path.startsWith("..")) {
      throw new IllegalArgumentException("GGUF contains an unsafe embedded file name: " + name);
    }
    return path.toString().replace('\\', '/');
  }
}
