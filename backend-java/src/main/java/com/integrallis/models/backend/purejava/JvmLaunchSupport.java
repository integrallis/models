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
package com.integrallis.models.backend.purejava;

import java.lang.management.ManagementFactory;
import java.util.List;
import java.util.Objects;

/** Validates process-wide JVM settings required by the pure-Java inference kernels. */
final class JvmLaunchSupport {

  private static final String TIERED_STOP_PREFIX = "-XX:TieredStopAtLevel=";

  private JvmLaunchSupport() {}

  static void requireOptimizingCompiler() {
    requireOptimizingCompiler(ManagementFactory.getRuntimeMXBean().getInputArguments());
  }

  static void requireOptimizingCompiler(List<String> inputArguments) {
    Objects.requireNonNull(inputArguments, "inputArguments");
    for (String argument : inputArguments) {
      if ("-Xint".equals(argument)) {
        throw unsupported(argument);
      }
      if (argument.startsWith(TIERED_STOP_PREFIX)) {
        int level = parseTieredStopLevel(argument);
        if (level > 0 && level < 4) {
          throw unsupported(argument);
        }
      }
    }
  }

  private static int parseTieredStopLevel(String argument) {
    try {
      return Integer.parseInt(argument.substring(TIERED_STOP_PREFIX.length()));
    } catch (NumberFormatException ignored) {
      return -1;
    }
  }

  private static IllegalStateException unsupported(String argument) {
    return new IllegalStateException(
        "Pure-Java model inference cannot run with "
            + argument
            + " because it prevents HotSpot from using the optimizing C2 compiler and makes "
            + "Vector API kernels impractically slow. For Spring Boot Gradle, configure "
            + "tasks.named('bootRun') { optimizedLaunch = false }. For Spring Boot Maven, set "
            + "<optimizedLaunch>false</optimizedLaunch>. In an IntelliJ Spring Boot run "
            + "configuration, select 'Disable launch optimization'.");
  }
}
