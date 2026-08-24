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
package com.integrallis.models.backend.purejava.fixture;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;
import java.util.Properties;
import java.util.Set;

/** Classpath registry for Models-owned, pinned integration-test artifacts. */
public final class ModelFixtureRegistry {
  private static final String RESOURCE = "model-fixtures.properties";
  private static final int LEGACY_FIELD_COUNT = 15;
  private static final int FORMAT_FIELD_COUNT = 16;

  private final List<ModelFixtureDescriptor> descriptors;

  private ModelFixtureRegistry(List<ModelFixtureDescriptor> descriptors) {
    this.descriptors = List.copyOf(descriptors);
  }

  public static ModelFixtureRegistry fromClasspath() {
    ClassLoader loader = Thread.currentThread().getContextClassLoader();
    if (loader == null) {
      loader = ModelFixtureRegistry.class.getClassLoader();
    }
    try (InputStream input = loader.getResourceAsStream(RESOURCE)) {
      if (input == null) {
        throw new IllegalStateException("Missing test fixture manifest: " + RESOURCE);
      }
      Properties properties = new Properties();
      properties.load(input);
      Path directory =
          Path.of(
              System.getProperty(
                  "models.fixtures.directory",
                  Path.of(System.getProperty("user.home"), ".jvllm", "models").toString()));
      List<ModelFixtureDescriptor> entries =
          properties.stringPropertyNames().stream()
              .sorted()
              .map(id -> parse(id, properties.getProperty(id), directory))
              .toList();
      return new ModelFixtureRegistry(entries);
    } catch (IOException failure) {
      throw new IllegalStateException("Unable to read " + RESOURCE, failure);
    }
  }

  public Optional<ModelFixtureDescriptor> resolve(ModelFixtureRequirement requirement) {
    List<ModelFixtureDescriptor> matches =
        descriptors.stream().filter(requirement::matches).toList();
    if (matches.size() > 1) {
      throw new IllegalArgumentException("Fixture requirement is ambiguous");
    }
    return matches.stream().findFirst();
  }

  public List<ModelFixtureDescriptor> descriptors() {
    return descriptors;
  }

  private static ModelFixtureDescriptor parse(String id, String encoded, Path directory) {
    String[] fields = encoded.split("\\|", -1);
    if (fields.length != LEGACY_FIELD_COUNT && fields.length != FORMAT_FIELD_COUNT) {
      throw new IllegalStateException(
          "Fixture "
              + id
              + " has "
              + fields.length
              + " fields; expected "
              + LEGACY_FIELD_COUNT
              + " or "
              + FORMAT_FIELD_COUNT);
    }
    return new ModelFixtureDescriptor(
        id,
        fields[0],
        fields[1],
        fields[2],
        fields[3],
        fields[4],
        URI.create(fields[6]),
        directory.resolve(fields[7]),
        fields[8],
        Long.parseLong(fields[9]),
        fields[10],
        fields[11],
        commaSeparated(fields[12]),
        commaSeparated(fields[13]),
        Boolean.parseBoolean(fields[14]),
        fields.length == FORMAT_FIELD_COUNT ? fields[15] : "gguf");
  }

  private static Set<String> commaSeparated(String value) {
    if (value.isBlank()) {
      return Set.of();
    }
    return Arrays.stream(value.split(","))
        .map(String::trim)
        .filter(entry -> !entry.isEmpty())
        .collect(java.util.stream.Collectors.toCollection(LinkedHashSet::new));
  }
}
