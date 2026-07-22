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

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/** Shared strict parsing for benchmark subcommands that accept name/value options. */
final class BenchmarkCliArguments {

  private BenchmarkCliArguments() {}

  static Map<String, String> parse(String[] args, Set<String> allowedOptions) {
    if ((args.length & 1) != 0) {
      throw new IllegalArgumentException("options must be provided as --name value pairs");
    }
    Map<String, String> values = new HashMap<>();
    for (int index = 0; index < args.length; index += 2) {
      String option = args[index];
      if (!option.startsWith("--")) {
        throw new IllegalArgumentException("expected option, got: " + option);
      }
      String name = option.substring(2);
      if (!allowedOptions.contains(name)) {
        throw new IllegalArgumentException("unknown option: " + option);
      }
      if (values.put(name, args[index + 1]) != null) {
        throw new IllegalArgumentException("duplicate option: " + option);
      }
    }
    return values;
  }

  static int integer(Map<String, String> values, String name, int defaultValue) {
    String value = values.get(name);
    if (value == null) {
      return defaultValue;
    }
    try {
      return Integer.parseInt(value);
    } catch (NumberFormatException exception) {
      throw new IllegalArgumentException("--" + name + " must be an integer: " + value, exception);
    }
  }
}
