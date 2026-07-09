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

import java.util.Objects;

/** Prompt/response client backed by Apple Foundation Models when available. */
public final class AppleFoundationModelsClient implements AutoCloseable {

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

  @Override
  public void close() {
    bridge.close();
  }
}
