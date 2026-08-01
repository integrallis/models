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

import java.lang.management.ManagementFactory;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.management.JMException;
import javax.management.MBeanServer;
import javax.management.ObjectName;

/** Best-effort in-process access to the JDK native-memory summary. */
final class NativeMemoryTracking {

  private static final long KIBIBYTE = 1_024;
  private static final Pattern TOTAL =
      Pattern.compile(
          "^Total:\\s+reserved=(\\d+)KB,\\s+committed=(\\d+)KB\\s*$", Pattern.MULTILINE);
  private static final Pattern CATEGORY =
      Pattern.compile(
          "^-\\s+(.+?)\\s+\\(reserved=(\\d+)KB,\\s+committed=(\\d+)KB\\)\\s*$", Pattern.MULTILINE);

  private NativeMemoryTracking() {}

  static Summary capture() {
    try {
      MBeanServer server = ManagementFactory.getPlatformMBeanServer();
      ObjectName diagnosticCommand = new ObjectName("com.sun.management:type=DiagnosticCommand");
      String output =
          (String)
              server.invoke(
                  diagnosticCommand,
                  "vmNativeMemory",
                  new Object[] {new String[] {"summary", "scale=KB"}},
                  new String[] {String[].class.getName()});
      return parse(output);
    } catch (JMException | RuntimeException unavailable) {
      return Summary.unavailable();
    }
  }

  static Summary parse(String output) {
    if (output == null) {
      return Summary.unavailable();
    }
    Matcher total = TOTAL.matcher(output);
    if (!total.find()) {
      return Summary.unavailable();
    }
    Map<String, Category> categories = new LinkedHashMap<>();
    Matcher category = CATEGORY.matcher(output);
    while (category.find()) {
      categories.put(
          category.group(1).trim(),
          new Category(bytes(category.group(2)), bytes(category.group(3))));
    }
    return new Summary(true, bytes(total.group(1)), bytes(total.group(2)), categories);
  }

  private static long bytes(String kibibytes) {
    return Math.multiplyExact(Long.parseLong(kibibytes), KIBIBYTE);
  }

  record Summary(
      boolean available,
      long totalReservedBytes,
      long totalCommittedBytes,
      Map<String, Category> categories) {

    Summary {
      categories = Map.copyOf(categories);
    }

    static Summary unavailable() {
      return new Summary(false, 0, 0, Map.of());
    }
  }

  record Category(long reservedBytes, long committedBytes) {}
}
