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

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.api.TokenStream;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Prompt/response client backed by Apple Foundation Models when available. */
public final class AppleFoundationModelsClient implements TextGenerationModel {

  private static final String MODEL_NAME = "apple-system-language-model";
  private static final BackendDiagnostics DIAGNOSTICS =
      new BackendDiagnostics(
          "apple-foundation-models",
          "system-language-model-v1",
          Map.of(
              "execution", "on-device",
              "model", "SystemLanguageModel.default",
              "weights", "managed-by-macOS"),
          List.of());

  private final AppleFoundationModelsBridge bridge;

  private AppleFoundationModelsClient(AppleFoundationModelsBridge bridge) {
    this.bridge = Objects.requireNonNull(bridge, "bridge");
  }

  /** Creates a client for an internal bridge implementation. */
  static AppleFoundationModelsClient of(AppleFoundationModelsBridge bridge) {
    return new AppleFoundationModelsClient(bridge);
  }

  /** Creates an unavailable client with the supplied reason. */
  public static AppleFoundationModelsClient unavailable(String reason) {
    return unavailable(AppleFoundationModelsAvailability.unavailable(reason));
  }

  static AppleFoundationModelsClient unavailable(AppleFoundationModelsAvailability availability) {
    return new AppleFoundationModelsClient(
        new UnavailableAppleFoundationModelsBridge(availability));
  }

  /** Returns current availability for Apple Foundation Models. */
  public AppleFoundationModelsAvailability availability() {
    return bridge.availability();
  }

  /** Generates a response for a prompt using default request options. */
  public AppleFoundationModelsResponse generate(String prompt) {
    return generate(AppleFoundationModelsRequest.builder(prompt).build());
  }

  /** Generates a response for the supplied request. */
  public AppleFoundationModelsResponse generate(AppleFoundationModelsRequest request) {
    Objects.requireNonNull(request, "request");
    AppleFoundationModelsAvailability availability = availability();
    if (!availability.available()) {
      throw new IllegalStateException(availability.reason());
    }
    return bridge.generate(request);
  }

  /**
   * Returns the stable Models identifier for Apple's OS-managed default language model.
   *
   * <p>Apple does not expose a downloadable weight revision through Foundation Models.
   */
  @Override
  public String modelName() {
    return MODEL_NAME;
  }

  /** Returns the execution identity exposed to Models framework adapters. */
  @Override
  public BackendDiagnostics diagnostics() {
    return DIAGNOSTICS;
  }

  /**
   * Generates through the shared Models contract.
   *
   * <p>Apple Foundation Models currently returns a completed response through this bridge, so the
   * response is delivered as one stream event. {@link SamplingOptions#maxTokens()} maps to Apple's
   * maximum response-token option; Apple owns the remaining sampling behavior.
   */
  @Override
  public void generate(String prompt, SamplingOptions options, TokenStream stream) {
    Objects.requireNonNull(prompt, "prompt");
    Objects.requireNonNull(options, "options");
    Objects.requireNonNull(stream, "stream");
    try {
      AppleFoundationModelsResponse response =
          generate(
              AppleFoundationModelsRequest.builder(prompt)
                  .maxOutputTokens(options.maxTokens())
                  .build());
      stream.onToken(response.text());
      stream.onComplete();
    } catch (RuntimeException failure) {
      stream.onError(failure);
    }
  }

  @Override
  public void close() {
    bridge.close();
  }
}
