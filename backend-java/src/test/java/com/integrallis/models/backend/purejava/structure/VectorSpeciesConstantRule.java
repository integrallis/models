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
package com.integrallis.models.backend.purejava.structure;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.reflect.Executable;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import jdk.incubator.vector.VectorSpecies;

/**
 * Structural rule: a Vector API {@link VectorSpecies} must be a {@code static final} constant. It
 * must not be a method or constructor parameter (including a lambda's synthetic parameter) and must
 * not live in an instance field or a mutable static field.
 */
final class VectorSpeciesConstantRule {

  private static final byte[] SPECIES_DESCRIPTOR =
      "jdk/incubator/vector/VectorSpecies".getBytes(StandardCharsets.US_ASCII);

  private VectorSpeciesConstantRule() {}

  /** Classes whose bytecode references {@code VectorSpecies}, and the violations found in them. */
  record Scan(List<String> scannedClasses, List<String> violations) {}

  /** Returns every violation declared directly by {@code type}. */
  static List<String> violations(Class<?> type) {
    List<String> violations = new ArrayList<>();
    for (Field field : type.getDeclaredFields()) {
      if (VectorSpecies.class.isAssignableFrom(field.getType())) {
        int modifiers = field.getModifiers();
        if (!Modifier.isStatic(modifiers) || !Modifier.isFinal(modifiers)) {
          violations.add(type.getName() + "." + field.getName() + " is not static final");
        }
      }
    }
    Stream.concat(
            Arrays.stream(type.getDeclaredMethods()), Arrays.stream(type.getDeclaredConstructors()))
        .forEach(executable -> addParameterViolations(type, executable, violations));
    return violations;
  }

  /**
   * Loads, without initializing, every class under {@code root} whose bytecode mentions {@code
   * VectorSpecies} and applies {@link #violations(Class)}. A referencing class that cannot be
   * loaded is itself reported, so the scan can never pass by silently skipping a class.
   */
  static Scan scanClassDirectory(Path root, ClassLoader loader) {
    if (Files.isRegularFile(root)) {
      try (FileSystem jar = FileSystems.newFileSystem(root)) {
        return scanClassDirectory(jar.getPath("/"), loader);
      } catch (IOException failure) {
        throw new UncheckedIOException(failure);
      }
    }
    List<String> scanned = new ArrayList<>();
    List<String> violations = new ArrayList<>();
    try (Stream<Path> files = Files.walk(root)) {
      for (Path file : files.filter(path -> path.toString().endsWith(".class")).sorted().toList()) {
        if (!mentionsSpecies(Files.readAllBytes(file))) {
          continue;
        }
        String relative = root.relativize(file).toString();
        if (relative.startsWith("META-INF")) {
          continue;
        }
        String name =
            relative
                .substring(0, relative.length() - ".class".length())
                .replace(file.getFileSystem().getSeparator(), ".");
        scanned.add(name);
        try {
          violations.addAll(violations(Class.forName(name, false, loader)));
        } catch (ClassNotFoundException | LinkageError unloadable) {
          violations.add(name + " could not be loaded for inspection: " + unloadable);
        }
      }
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
    return new Scan(List.copyOf(scanned), List.copyOf(violations));
  }

  private static void addParameterViolations(
      Class<?> owner, Executable executable, List<String> violations) {
    Class<?>[] parameters = executable.getParameterTypes();
    for (int index = 0; index < parameters.length; index++) {
      if (VectorSpecies.class.isAssignableFrom(parameters[index])) {
        String name =
            executable instanceof java.lang.reflect.Constructor<?>
                ? "<init>"
                : executable.getName();
        String signature =
            Arrays.stream(parameters).map(Class::getSimpleName).collect(Collectors.joining(", "));
        violations.add(owner.getName() + "#" + name + "(" + signature + ") parameter " + index);
      }
    }
  }

  private static boolean mentionsSpecies(byte[] bytes) {
    outer:
    for (int start = 0; start <= bytes.length - SPECIES_DESCRIPTOR.length; start++) {
      for (int offset = 0; offset < SPECIES_DESCRIPTOR.length; offset++) {
        if (bytes[start + offset] != SPECIES_DESCRIPTOR[offset]) {
          continue outer;
        }
      }
      return true;
    }
    return false;
  }
}
