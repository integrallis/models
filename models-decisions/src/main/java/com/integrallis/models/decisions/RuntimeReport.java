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
package com.integrallis.models.decisions;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * What a decision run actually resolved: the model, the kernel, the device, the adapter.
 *
 * <p>Every invalidated measurement in this campaign has the same shape. A kernel was expected and a
 * fallback ran, or an adapter was expected and the base ran, and the result was a plausible number
 * attached to the wrong claim: a pure-Java fallback read as a latency measurement, a GPU arm that
 * silently used the CPU because the ABI did not match, a benchmark arm that never loaded the device
 * at all. None of those failed. They all produced numbers.
 *
 * <p>So the resolved configuration is carried as data rather than printed ad hoc by whichever tool
 * remembered to. A caller can print it beside a result, assert on it before trusting one, or write
 * it into an evidence file, and a report that says {@code pure-java} cannot be mistaken for one
 * that says {@code cuda}.
 *
 * <p>This is deliberately not a logger. A log line is discarded by whoever is not reading stderr; a
 * value has to be either used or visibly ignored.
 */
public final class RuntimeReport {

  private final Map<String, String> facts = new LinkedHashMap<>();

  /** Records one resolved fact. The last value for a key wins. */
  public RuntimeReport with(String key, String value) {
    facts.put(Objects.requireNonNull(key, "key"), Objects.requireNonNull(value, "value"));
    return this;
  }

  /** Records the kernel that resolved, and the device when there is one. */
  public RuntimeReport kernel(String kernel, String device, String detail) {
    with("kernel", kernel);
    if (!device.isBlank()) {
      with("device", device);
    }
    if (!detail.isBlank()) {
      with("kernelDetail", detail);
    }
    return this;
  }

  /** The value recorded for a key, or empty. */
  public String get(String key) {
    return facts.getOrDefault(key, "");
  }

  /**
   * Fails unless the recorded value matches.
   *
   * <p>For the case where a run is only meaningful on one configuration: a GPU comparison that
   * quietly ran on the CPU is not a slower result, it is a different experiment.
   */
  public void require(String key, String expected) {
    String actual = get(key);
    if (!expected.equals(actual)) {
      throw new IllegalStateException(
          "expected "
              + key
              + "="
              + expected
              + " but the run resolved "
              + key
              + "="
              + (actual.isEmpty() ? "<unset>" : actual)
              + "; "
              + this);
    }
  }

  /** The facts, in the order they were recorded, as one line fit for a results file. */
  @Override
  public String toString() {
    StringBuilder line = new StringBuilder("runtime[");
    boolean first = true;
    for (Map.Entry<String, String> fact : facts.entrySet()) {
      if (!first) {
        line.append(", ");
      }
      line.append(fact.getKey()).append('=').append(fact.getValue());
      first = false;
    }
    return line.append(']').toString();
  }

  /** The facts as JSON, for writing beside a measurement. */
  public String toJson() {
    StringBuilder json = new StringBuilder("{");
    boolean first = true;
    for (Map.Entry<String, String> fact : facts.entrySet()) {
      if (!first) {
        json.append(',');
      }
      json.append('"')
          .append(fact.getKey())
          .append("\":\"")
          .append(fact.getValue().replace("\\", "\\\\").replace("\"", "\\\""))
          .append('"');
      first = false;
    }
    return json.append('}').toString();
  }
}
