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

import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarException;
import org.modeljars.ModelJarLocator;
import org.modeljars.ModelJarRegistry;
import org.modeljars.ModelJarRequirement;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

/** Spring Boot auto-configuration for ModelJars registry access. */
@AutoConfiguration
@EnableConfigurationProperties(ModelJarsProperties.class)
public class ModelJarsAutoConfiguration {

  @Bean
  @ConditionalOnMissingBean
  ModelJarRegistry modelJarRegistry() {
    return ModelJarRegistry.fromClasspath();
  }

  @Bean
  @ConditionalOnMissingBean
  ModelJarLocator modelJarLocator(ModelJarRegistry registry) {
    return new ModelJarLocator(registry);
  }

  @Bean
  @ConditionalOnMissingBean
  @ConditionalOnProperty(prefix = "models.modeljars", name = "source")
  ModelJarRequirement modelJarRequirement(ModelJarsProperties properties) {
    return properties.toRequirement();
  }

  @Bean
  @ConditionalOnMissingBean
  @ConditionalOnProperty(prefix = "models.modeljars", name = "source")
  ModelJarDescriptor modelJarDescriptor(
      ModelJarRegistry registry, ModelJarRequirement requirement) {
    return registry
        .resolve(requirement)
        .orElseThrow(() -> new ModelJarException("No ModelJars descriptor matched " + requirement));
  }
}
