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

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

import com.integrallis.models.api.InferenceBackend;
import java.util.List;
import java.util.function.Predicate;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.context.annotation.ImportCandidates;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Import;
import org.springframework.context.annotation.ImportSelector;
import org.springframework.core.type.AnnotationMetadata;
import org.springframework.util.ClassUtils;

/**
 * The DecisionModel auto-configuration on whatever Spring AI line this build runs against.
 *
 * <p>It is imported exactly as Spring Boot imports it: by name from the starter's
 * AutoConfiguration.imports, so its class-level guard is evaluated from class-file metadata. On a
 * line without {@code DecisionModel} (2.0, 1.1) the property must change nothing and the context
 * must start; on 2.1 it must contribute the bean. The CI matrix runs this on every line.
 */
@Tag("unit")
class ModelsDecisionAutoConfigurationOnEveryLineTest {

  static final String DECISION_CONFIGURATION =
      "com.integrallis.models.spring.boot.decision.ModelsDecisionAutoConfiguration";

  /** Imports this starter's auto-configurations by name, the way Spring Boot does. */
  static final class StarterAutoConfigurations implements ImportSelector {
    @Override
    public String[] selectImports(AnnotationMetadata metadata) {
      return ImportCandidates.load(AutoConfiguration.class, getClass().getClassLoader())
          .getCandidates()
          .stream()
          .filter(name -> name.startsWith("com.integrallis.models.spring.boot."))
          .toArray(String[]::new);
    }

    @Override
    public Predicate<String> getExclusionFilter() {
      return null;
    }
  }

  @Configuration(proxyBeanMethods = false)
  @Import(StarterAutoConfigurations.class)
  static class Application {}

  @Test
  void theStarterListsTheDecisionConfiguration() {
    List<String> listed =
        ImportCandidates.load(AutoConfiguration.class, getClass().getClassLoader()).getCandidates();
    assertThat(listed).contains(DECISION_CONFIGURATION);
  }

  @Test
  void enablingDecisionsContributesTheBeanOnlyWhereSpringAiHasDecisionModel() {
    boolean decisionApi =
        ClassUtils.isPresent(
            "org.springframework.ai.decision.DecisionModel", getClass().getClassLoader());
    new ApplicationContextRunner()
        .withUserConfiguration(Application.class)
        .withBean(InferenceBackend.class, () -> mock(InferenceBackend.class))
        .withPropertyValues("integrallis.models.decision.enabled=true")
        .run(
            context -> {
              if (decisionApi) {
                // A mock is not a SharedPrefixInferenceBackend: on 2.1 the request is honoured
                // and refused with the reason, which proves the configuration was evaluated.
                assertThat(context).hasFailed();
                assertThat(context.getStartupFailure())
                    .rootCause()
                    .hasMessageContaining("is not a SharedPrefixInferenceBackend");
              } else {
                assertThat(context).hasNotFailed();
                assertThat(context).doesNotHaveBean("modelsDecisionModel");
                assertThat(context).hasBean(ModelsAutoConfiguration.MODELS_CHAT_MODEL_BEAN_NAME);
              }
            });
  }
}
