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
package com.integrallis.models.backend.purejava.diagnostics;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;

/**
 * Test support: forgets earlier once-per-process reports, records {@link PerformanceCliffEvent}s in
 * process, and returns what was committed.
 */
public final class PerformanceCliffRecording implements AutoCloseable {

  private final Recording recording;
  private List<RecordedEvent> events;

  private PerformanceCliffRecording() {
    PerformanceCliffs.reset();
    recording = new Recording();
    recording.enable(PerformanceCliffEvent.NAME).withStackTrace();
    recording.start();
  }

  /** Resets the registry and starts recording cliff events. */
  public static PerformanceCliffRecording start() {
    return new PerformanceCliffRecording();
  }

  /** Stops the recording (once) and returns every committed cliff event. */
  public List<RecordedEvent> events() {
    if (events == null) {
      recording.stop();
      try {
        Path dump = Files.createTempFile("performance-cliffs", ".jfr");
        try {
          recording.dump(dump);
          events =
              RecordingFile.readAllEvents(dump).stream()
                  .filter(
                      event -> event.getEventType().getName().equals(PerformanceCliffEvent.NAME))
                  .toList();
        } finally {
          Files.deleteIfExists(dump);
        }
      } catch (IOException failure) {
        throw new UncheckedIOException(failure);
      }
    }
    return events;
  }

  /** Number of committed events for {@code cliff}. */
  public long count(PerformanceCliff cliff) {
    return events().stream().filter(event -> cliff.id().equals(event.getString("reason"))).count();
  }

  /** Committed reason ids, in commit order. */
  public List<String> reasons() {
    return events().stream().map(event -> event.getString("reason")).toList();
  }

  @Override
  public void close() {
    events();
    recording.close();
  }
}
