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

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.SharedPrefixInferenceBackend;
import com.integrallis.models.spring.ai.decision.ModelsSpringAiDecisionModel;
import com.integrallis.models.spring.boot.ModelsAutoConfiguration;
import io.micrometer.observation.ObservationRegistry;
import java.util.Arrays;
import org.springframework.ai.decision.DecisionModel;
import org.springframework.beans.factory.ListableBeanFactory;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

/**
 * Contributes a Spring AI {@link DecisionModel} answered by the same Models backend as the chat
 * model, when {@code integrallis.models.decision.enabled=true}.
 *
 * <p>The backend is the one named by {@code integrallis.models.decision.backend}, or else the
 * application's single {@link InferenceBackend} bean, which includes the one contributed for {@code
 * integrallis.models.catalog-model}. An application that declares its own {@code DecisionModel}
 * keeps it. With the property set and no usable backend the context fails to start and says why:
 * the property is a request, and a request that cannot be met is not silently dropped.
 *
 * <p>Requires Spring AI 2.1 or later. The class-level guard names the type as a string, so on an
 * older line Spring Boot skips this configuration from its class-file metadata and never loads it.
 *
 * <p>The decision model and the chat model serialise their own calls but not each other's: an
 * application that uses both concurrently over one backend must serialise them itself, as {@link
 * ModelsSpringAiDecisionModel} documents.
 */
@AutoConfiguration(after = ModelsAutoConfiguration.class)
@ConditionalOnClass(name = "org.springframework.ai.decision.DecisionModel")
@ConditionalOnProperty(
    prefix = "integrallis.models.decision",
    name = "enabled",
    havingValue = "true")
@EnableConfigurationProperties(ModelsDecisionProperties.class)
public class ModelsDecisionAutoConfiguration {

  /** Bean name of the auto-configured decision model. */
  public static final String MODELS_DECISION_MODEL_BEAN_NAME = "modelsDecisionModel";

  @Bean(name = MODELS_DECISION_MODEL_BEAN_NAME)
  @ConditionalOnMissingBean(DecisionModel.class)
  ModelsSpringAiDecisionModel modelsDecisionModel(
      ModelsDecisionProperties properties,
      ListableBeanFactory beanFactory,
      ObjectProvider<ObservationRegistry> observations) {
    InferenceBackend backend = backend(properties, beanFactory);
    if (!(backend instanceof SharedPrefixInferenceBackend shared)) {
      throw new IllegalStateException(
          "integrallis.models.decision.enabled=true, but the backend "
              + backend.name()
              + " ("
              + backend.getClass().getName()
              + ") is not a SharedPrefixInferenceBackend, which the DecisionModel adapter reads"
              + " questions through");
    }
    String name = properties.modelName() != null ? properties.modelName() : shared.name();
    return new ModelsSpringAiDecisionModel(
        shared,
        name,
        properties.temperature(),
        observations.getIfAvailable(() -> ObservationRegistry.NOOP));
  }

  private static InferenceBackend backend(
      ModelsDecisionProperties properties, ListableBeanFactory beanFactory) {
    if (properties.backend() != null) {
      if (!beanFactory.containsBean(properties.backend())) {
        throw new IllegalStateException(
            "integrallis.models.decision.backend names "
                + properties.backend()
                + ", but there is no such bean");
      }
      return beanFactory.getBean(properties.backend(), InferenceBackend.class);
    }
    String[] names = beanFactory.getBeanNamesForType(InferenceBackend.class, true, false);
    if (names.length == 0) {
      throw new IllegalStateException(
          "integrallis.models.decision.enabled=true, but there is no InferenceBackend bean: declare"
              + " one, or set integrallis.models.catalog-model");
    }
    InferenceBackend unique = beanFactory.getBeanProvider(InferenceBackend.class).getIfUnique();
    if (unique == null) {
      throw new IllegalStateException(
          "integrallis.models.decision.enabled=true and there are several InferenceBackend beans "
              + Arrays.toString(names)
              + ": mark one @Primary or name it in integrallis.models.decision.backend");
    }
    return unique;
  }
}
