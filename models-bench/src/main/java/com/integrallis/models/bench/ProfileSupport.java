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

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.ParseException;
import jdk.jfr.Recording;

/** Shared JFR and GC measurement infrastructure for isolated inference profiles. */
final class ProfileSupport {

  private ProfileSupport() {}

  static ProfileRecording jfr(String name) throws IOException, ParseException {
    return new JfrProfileRecording(name);
  }

  static GcMetrics gcMetrics() {
    long collections = 0;
    long pauseMillis = 0;
    for (var collector : ManagementFactory.getGarbageCollectorMXBeans()) {
      if (collector.getCollectionCount() >= 0) {
        collections += collector.getCollectionCount();
      }
      if (collector.getCollectionTime() >= 0) {
        pauseMillis += collector.getCollectionTime();
      }
    }
    return new GcMetrics(collections, pauseMillis);
  }

  record GcMetrics(long collections, long pauseMillis) {}

  @FunctionalInterface
  interface GcMetricsSource {
    GcMetrics snapshot();
  }

  interface ProfileRecording extends AutoCloseable {
    void start();

    void stop();

    void dump(Path output) throws IOException;

    @Override
    void close();
  }

  private static final class JfrProfileRecording implements ProfileRecording {
    private final Recording recording;

    private JfrProfileRecording(String name) throws IOException, ParseException {
      recording = new Recording(jdk.jfr.Configuration.getConfiguration("profile"));
      recording.setName(name);
    }

    @Override
    public void start() {
      recording.start();
    }

    @Override
    public void stop() {
      recording.stop();
    }

    @Override
    public void dump(Path output) throws IOException {
      Path absolute = output.toAbsolutePath();
      Path parent = absolute.getParent();
      if (parent != null) {
        Files.createDirectories(parent);
      }
      recording.dump(absolute);
    }

    @Override
    public void close() {
      recording.close();
    }
  }
}
