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
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Peak resident set size of this process.
 *
 * <p>Linux reads the kernel high-water mark ({@code VmHWM}); elsewhere a daemon thread samples
 * {@code ps -o rss=} every two seconds, so short spikes can be missed. The method is reported.
 */
final class PeakRss implements AutoCloseable {
  private static final Path STATUS = Path.of("/proc/self/status");
  private final AtomicLong sampledPeak = new AtomicLong();
  private final Thread sampler;

  PeakRss() {
    if (Files.isRegularFile(STATUS)) {
      sampler = null;
    } else {
      sampler =
          Thread.ofPlatform()
              .daemon()
              .name("peak-rss-sampler")
              .start(
                  () -> {
                    while (!Thread.currentThread().isInterrupted()) {
                      sampledPeak.accumulateAndGet(currentRss(), Math::max);
                      try {
                        Thread.sleep(2000);
                      } catch (InterruptedException interrupted) {
                        return;
                      }
                    }
                  });
    }
  }

  String method() {
    return sampler == null ? "linux-VmHWM" : "sampled-ps-rss-2s";
  }

  long peakBytes() {
    if (sampler == null) {
      try {
        for (String line : Files.readAllLines(STATUS, StandardCharsets.UTF_8)) {
          if (line.startsWith("VmHWM:")) {
            return Long.parseLong(line.replaceAll("[^0-9]", "")) * 1024;
          }
        }
      } catch (IOException | NumberFormatException ignored) {
        return -1;
      }
      return -1;
    }
    return sampledPeak.accumulateAndGet(currentRss(), Math::max);
  }

  private static long currentRss() {
    try {
      Process process =
          new ProcessBuilder("ps", "-o", "rss=", "-p", Long.toString(ProcessHandle.current().pid()))
              .start();
      String value =
          new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8).strip();
      return process.waitFor() == 0 && !value.isEmpty() ? Long.parseLong(value) * 1024 : 0;
    } catch (IOException | NumberFormatException failure) {
      return 0;
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      return 0;
    }
  }

  @Override
  public void close() {
    if (sampler != null) {
      sampler.interrupt();
    }
  }
}
