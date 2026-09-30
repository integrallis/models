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

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Strict {@code --name value} options: unknown or repeated names are errors. */
final class CliOptions {
  private final Map<String, String> values;

  private CliOptions(Map<String, String> values) {
    this.values = values;
  }

  static CliOptions parse(String[] args, int from, Set<String> allowed) {
    if (((args.length - from) & 1) != 0) {
      throw new IllegalArgumentException("options must be provided as --name value pairs");
    }
    Map<String, String> values = new HashMap<>();
    for (int i = from; i < args.length; i += 2) {
      String option = args[i];
      if (!option.startsWith("--")) {
        throw new IllegalArgumentException("expected an option, got: " + option);
      }
      String name = option.substring(2);
      if (!allowed.contains(name)) {
        throw new IllegalArgumentException(
            "unknown option: " + option + " (allowed: " + allowed + ")");
      }
      if (values.put(name, args[i + 1]) != null) {
        throw new IllegalArgumentException("duplicate option: " + option);
      }
    }
    return new CliOptions(values);
  }

  boolean has(String name) {
    return values.containsKey(name);
  }

  String get(String name, String fallback) {
    return values.getOrDefault(name, fallback);
  }

  String required(String name) {
    String value = values.get(name);
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("--" + name + " is required");
    }
    return value;
  }

  int integer(String name, int fallback) {
    String value = values.get(name);
    if (value == null) {
      return fallback;
    }
    try {
      return Integer.parseInt(value);
    } catch (NumberFormatException failure) {
      throw new IllegalArgumentException("--" + name + " must be an integer: " + value, failure);
    }
  }

  long longValue(String name, long fallback) {
    String value = values.get(name);
    if (value == null) {
      return fallback;
    }
    try {
      return Long.parseLong(value);
    } catch (NumberFormatException failure) {
      throw new IllegalArgumentException("--" + name + " must be an integer: " + value, failure);
    }
  }

  float decimal(String name, float fallback) {
    String value = values.get(name);
    if (value == null) {
      return fallback;
    }
    try {
      return Float.parseFloat(value);
    } catch (NumberFormatException failure) {
      throw new IllegalArgumentException("--" + name + " must be a number: " + value, failure);
    }
  }

  boolean bool(String name, boolean fallback) {
    String value = values.get(name);
    if (value == null) {
      return fallback;
    }
    return switch (value) {
      case "true", "on" -> true;
      case "false", "off" -> false;
      default -> throw new IllegalArgumentException("--" + name + " must be true/false or on/off");
    };
  }

  Path existingFile(String name) {
    Path path = Path.of(required(name));
    if (!Files.isRegularFile(path)) {
      throw new IllegalArgumentException("--" + name + " is not a regular file: " + path);
    }
    return path;
  }

  List<Path> fileList(String name) {
    List<Path> paths = new ArrayList<>();
    if (!has(name)) {
      return paths;
    }
    for (String part : required(name).split(",")) {
      Path path = Path.of(part.strip());
      if (!Files.isRegularFile(path)) {
        throw new IllegalArgumentException("--" + name + " entry is not a regular file: " + path);
      }
      paths.add(path);
    }
    return paths;
  }

  /** Parses {@code A=/p/a.gguf,B=/p/b.gguf} preserving order. */
  Map<String, Path> members() {
    Map<String, Path> members = new LinkedHashMap<>();
    for (String entry : required("members").split(",")) {
      int equals = entry.indexOf('=');
      if (equals <= 0 || equals == entry.length() - 1) {
        throw new IllegalArgumentException("--members entries must be name=path: " + entry);
      }
      String name = entry.substring(0, equals).strip();
      Path path = Path.of(entry.substring(equals + 1).strip());
      if (!Files.isRegularFile(path)) {
        throw new IllegalArgumentException("member " + name + " is not a regular file: " + path);
      }
      if (members.put(name, path) != null) {
        throw new IllegalArgumentException("duplicate member name: " + name);
      }
    }
    return members;
  }
}
