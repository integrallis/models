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
package com.integrallis.models.backend.apple;

import com.integrallis.models.api.catalog.DiscoveredModel;
import com.integrallis.models.api.catalog.ModelCatalogProvider;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Reports Apple's on-device model when the machine actually has one.
 *
 * <p>Opt-in, unlike a catalog of models someone installed. Apple Intelligence is present because of
 * the hardware, so discovering it by default would make identical code route differently on a
 * developer's Mac than in production — a difference that shows up as unexplained behaviour rather
 * than as a decision anyone made. Callers ask for it with {@code discoverLocal(true)}.
 *
 * <p>Availability is checked rather than assumed. macOS on Apple silicon is necessary but not
 * sufficient: Apple Intelligence can be switched off, unsupported in a region, or still
 * downloading, and the framework reports that at runtime.
 */
public final class AppleFoundationModelsCatalog implements ModelCatalogProvider {

  /** Identifier this catalog reports the on-device model under. */
  public static final String MODEL_ID = "apple-foundation-model";

  /**
   * Published context window of Apple's on-device model.
   *
   * <p>A documented property of the framework rather than something measurable from here.
   */
  private static final int CONTEXT_WINDOW = 4096;

  @Override
  public String name() {
    return "apple-foundation-models";
  }

  @Override
  public boolean requiresOptIn() {
    return true;
  }

  /** Short, deterministic prompt used only to time the model once. */
  private static final String PROBE = "Reply with the single word: ready";

  @Override
  public List<DiscoveredModel> discover() {
    try (AppleFoundationModelsClient client = AppleFoundationModels.create()) {
      if (!client.availability().available()) {
        return List.of();
      }
      DiscoveredModel.Performance performance = measure(client);
      if (performance == null) {
        return List.of();
      }
      return List.of(
          new DiscoveredModel(
              MODEL_ID,
              true,
              Set.of("chat", "summarization", "extraction", "creative"),
              CONTEXT_WINDOW,
              0.0,
              0.0,
              performance,
              Map.of(),
              1.0));
    } catch (RuntimeException e) {
      // Absent is a normal answer on anything that is not a recent Mac, and discovery runs on every
      // platform. Reporting nothing beats propagating a platform detail to a caller who has other
      // models to route between.
      return List.of();
    }
  }

  /**
   * Times one short generation.
   *
   * <p>Apple publishes no throughput figures and the framework exposes no counters, so the choice
   * is between measuring, inventing, or reporting nothing and being skipped. Measuring is the only
   * one that leaves the router scoring on something true.
   *
   * <p>Runs once per discovery, behind an explicit opt-in, on a fixed short prompt. It is a single
   * datapoint on a device whose rate moves with thermal state — good enough to rank against a 7B
   * running locally, not good enough to publish.
   */
  private static DiscoveredModel.Performance measure(AppleFoundationModelsClient client) {
    try {
      long started = System.nanoTime();
      AppleFoundationModelsResponse response =
          client.generate(AppleFoundationModelsRequest.builder(PROBE).build());
      long elapsedNanos = System.nanoTime() - started;
      if (response == null || response.text() == null || response.text().isBlank()) {
        return null;
      }
      double elapsedSeconds = elapsedNanos / 1_000_000_000.0;
      if (!(elapsedSeconds > 0)) {
        return null;
      }
      // Whole-call latency stands in for time to first token: this path is not streaming, so the
      // first token is not separately observable. It overstates TTFT, which biases the router away
      // from this model rather than towards it — the safe direction for an estimate.
      long latencyMillis = Math.max(1L, elapsedNanos / 1_000_000L);
      double tokens = Math.max(1.0, response.text().trim().split("\\s+").length);
      return new DiscoveredModel.Performance(latencyMillis, tokens / elapsedSeconds);
    } catch (RuntimeException e) {
      return null;
    }
  }
}
