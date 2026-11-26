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
package com.integrallis.models.spring.boot.decision;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * Configuration of the auto-configured Spring AI {@code DecisionModel}.
 *
 * @param enabled whether to contribute a {@code DecisionModel} bean; off unless set
 * @param backend name of the {@code InferenceBackend} bean to answer with; empty to use the one the
 *     chat model uses (the application's single backend, or the {@code
 *     integrallis.models.catalog-model} one)
 * @param modelName the name reported in response metadata and observations; empty for the backend's
 *     own name
 * @param temperature positive, finite temperature applied across the answer letters
 */
@ConfigurationProperties("integrallis.models.decision")
public record ModelsDecisionProperties(
    @DefaultValue("false") boolean enabled,
    String backend,
    String modelName,
    @DefaultValue("1.0") double temperature) {

  /** Normalises blank names to absent and rejects a temperature the adapter cannot use. */
  public ModelsDecisionProperties {
    backend = backend == null || backend.isBlank() ? null : backend.strip();
    modelName = modelName == null || modelName.isBlank() ? null : modelName.strip();
    if (!Double.isFinite(temperature) || temperature <= 0) {
      throw new IllegalArgumentException(
          "integrallis.models.decision.temperature must be positive and finite: " + temperature);
    }
  }
}
