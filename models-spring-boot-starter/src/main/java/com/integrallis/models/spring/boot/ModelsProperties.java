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
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;

/** Configuration for the auto-configured Models Spring AI chat model. */
@ConfigurationProperties("integrallis.models")
public class ModelsProperties {

  /** Stable Models chat-template identifier, such as {@code chatml} or {@code llama3}. */
  private String chatTemplate = ChatTemplate.RAW.id();

  /** Default sampling values used when a Spring AI prompt does not override them. */
  @NestedConfigurationProperty private Sampling sampling = new Sampling();

  public String getChatTemplate() {
    return chatTemplate;
  }

  public void setChatTemplate(String chatTemplate) {
    this.chatTemplate = Objects.requireNonNull(chatTemplate, "chatTemplate");
  }

  public Sampling getSampling() {
    return sampling;
  }

  public void setSampling(Sampling sampling) {
    this.sampling = Objects.requireNonNull(sampling, "sampling");
  }

  ChatTemplate chatTemplate() {
    return ChatTemplate.parse(chatTemplate);
  }

  SamplingOptions samplingOptions() {
    return sampling.toOptions();
  }

  /** Sampling properties for local generation. */
  public static class Sampling {
    private float temperature = 1.0f;
    private float topP = 0.9f;
    private int topK = 40;
    private int maxTokens = 256;
    private Long seed;
    private float repetitionPenalty = 1.0f;
    private List<String> stopSequences = new ArrayList<>();

    public float getTemperature() {
      return temperature;
    }

    public void setTemperature(float temperature) {
      this.temperature = temperature;
    }

    public float getTopP() {
      return topP;
    }

    public void setTopP(float topP) {
      this.topP = topP;
    }

    public int getTopK() {
      return topK;
    }

    public void setTopK(int topK) {
      this.topK = topK;
    }

    public int getMaxTokens() {
      return maxTokens;
    }

    public void setMaxTokens(int maxTokens) {
      this.maxTokens = maxTokens;
    }

    public Long getSeed() {
      return seed;
    }

    public void setSeed(Long seed) {
      this.seed = seed;
    }

    public float getRepetitionPenalty() {
      return repetitionPenalty;
    }

    public void setRepetitionPenalty(float repetitionPenalty) {
      this.repetitionPenalty = repetitionPenalty;
    }

    public List<String> getStopSequences() {
      return List.copyOf(stopSequences);
    }

    public void setStopSequences(List<String> stopSequences) {
      this.stopSequences = new ArrayList<>(Objects.requireNonNull(stopSequences, "stopSequences"));
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
