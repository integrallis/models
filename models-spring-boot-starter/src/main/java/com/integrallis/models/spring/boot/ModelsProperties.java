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
package com.integrallis.models.spring.boot;

import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.util.List;
import java.util.Objects;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;
import org.springframework.boot.context.properties.bind.DefaultValue;

/** Configuration for the auto-configured Models Spring AI chat model. */
@ConfigurationProperties("integrallis.models")
public record ModelsProperties(
    @DefaultValue("raw") String chatTemplate,
    @DefaultValue @NestedConfigurationProperty Sampling sampling) {

  public ModelsProperties {
    Objects.requireNonNull(chatTemplate, "chatTemplate");
    Objects.requireNonNull(sampling, "sampling");
  }

  ChatTemplate parsedChatTemplate() {
    return ChatTemplate.parse(chatTemplate);
  }

  SamplingOptions samplingOptions() {
    return sampling.toOptions();
  }

  /** Sampling properties for local generation. */
  public record Sampling(
      @DefaultValue("1.0") float temperature,
      @DefaultValue("0.9") float topP,
      @DefaultValue("40") int topK,
      @DefaultValue("256") int maxTokens,
      Long seed,
      @DefaultValue("1.0") float repetitionPenalty,
      List<String> stopSequences) {

    public Sampling {
      stopSequences = stopSequences == null ? List.of() : List.copyOf(stopSequences);
    }

    SamplingOptions toOptions() {
      SamplingOptions.Builder builder =
          SamplingOptions.builder()
              .temperature(temperature)
              .topP(topP)
              .topK(topK)
              .maxTokens(maxTokens)
              .repetitionPenalty(repetitionPenalty)
              .stopSequences(stopSequences);
      if (seed != null) {
        builder.seed(seed);
      }
      return builder.build();
    }
  }
}
