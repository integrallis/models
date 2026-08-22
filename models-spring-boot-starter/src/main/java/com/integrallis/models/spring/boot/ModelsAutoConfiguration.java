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

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.spring.ai.ModelsSpringAiChatModel;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.observation.ObservationRegistry;
import org.springframework.ai.chat.observation.ChatModelMeterObservationHandler;
import org.springframework.ai.embedding.observation.EmbeddingModelMeterObservationHandler;
import org.springframework.beans.BeansException;
import org.springframework.beans.factory.config.BeanDefinition;
import org.springframework.beans.factory.config.ConfigurableListableBeanFactory;
import org.springframework.beans.factory.support.BeanDefinitionRegistry;
import org.springframework.beans.factory.support.BeanDefinitionRegistryPostProcessor;
import org.springframework.beans.factory.support.RootBeanDefinition;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.AutoConfigureAfter;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

/**
 * Spring Boot auto-configuration for the Models Spring AI adapter.
 *
 * <p>The application owns model resolution and contributes either a {@link TextGenerationModel} or
 * an {@link InferenceBackend}. This configuration adapts that runtime to Spring AI without taking a
 * dependency on a model catalog. A high-level {@code TextGenerationModel} takes precedence when
 * both contracts are present.
 *
 * <p>The adapter definition is registered after configuration classes have contributed their bean
 * definitions but before application beans are instantiated. This lets model beans from later
 * auto-configurations participate without leaving a phantom {@code ChatModel} definition when no
 * local runtime exists.
 */
@AutoConfiguration
@AutoConfigureAfter(
    name = {
      "org.springframework.ai.model.chat.observation.autoconfigure.ChatObservationAutoConfiguration",
      "org.springframework.ai.model.embedding.observation.autoconfigure.EmbeddingObservationAutoConfiguration"
    })
@ConditionalOnClass(name = "org.springframework.ai.chat.model.ChatModel")
@EnableConfigurationProperties(ModelsProperties.class)
public class ModelsAutoConfiguration {

  /** Stable bean name for the auto-configured local Spring AI chat model. */
  public static final String MODELS_CHAT_MODEL_BEAN_NAME = "modelsChatModel";

  /**
   * Registers the local Spring AI adapter when a Models runtime definition is present.
   *
   * @return the definition registrar
   */
  @Bean
  static BeanDefinitionRegistryPostProcessor modelsChatModelBeanDefinitionRegistrar() {
    return new ModelsChatModelBeanDefinitionRegistrar();
  }

  @Bean
  @ConditionalOnBean(MeterRegistry.class)
  @ConditionalOnMissingBean(ChatModelMeterObservationHandler.class)
  ChatModelMeterObservationHandler modelsChatModelMeterObservationHandler(
      MeterRegistry meterRegistry) {
    return new ChatModelMeterObservationHandler(meterRegistry);
  }

  @Bean
  @ConditionalOnBean(MeterRegistry.class)
  @ConditionalOnMissingBean(EmbeddingModelMeterObservationHandler.class)
  EmbeddingModelMeterObservationHandler modelsEmbeddingModelMeterObservationHandler(
      MeterRegistry meterRegistry) {
    return new EmbeddingModelMeterObservationHandler(meterRegistry);
  }

  private static final class ModelsChatModelBeanDefinitionRegistrar
      implements BeanDefinitionRegistryPostProcessor {

    @Override
    public void postProcessBeanDefinitionRegistry(BeanDefinitionRegistry registry)
        throws BeansException {
      if (!(registry instanceof ConfigurableListableBeanFactory beanFactory)) {
        throw new IllegalStateException(
            "Models requires a ConfigurableListableBeanFactory, got " + registry.getClass());
      }
      if (registry.containsBeanDefinition(MODELS_CHAT_MODEL_BEAN_NAME)
          || beanFactory.containsSingleton(MODELS_CHAT_MODEL_BEAN_NAME)
          || beanFactory.getBeanNamesForType(ModelsSpringAiChatModel.class, true, false).length
              > 0) {
        return;
      }

      String[] modelNames = beanFactory.getBeanNamesForType(TextGenerationModel.class, true, false);
      String[] backendNames = beanFactory.getBeanNamesForType(InferenceBackend.class, true, false);
      if (modelNames.length == 0 && backendNames.length == 0) {
        return;
      }

      RootBeanDefinition definition = new RootBeanDefinition(ModelsSpringAiChatModel.class);
      definition.setRole(BeanDefinition.ROLE_APPLICATION);
      definition.setInstanceSupplier(
          () -> {
            ModelsProperties properties = beanFactory.getBean(ModelsProperties.class);
            ObservationRegistry observations =
                beanFactory
                    .getBeanProvider(ObservationRegistry.class)
                    .getIfAvailable(() -> ObservationRegistry.NOOP);
            TextGenerationModel model =
                beanFactory.getBeanProvider(TextGenerationModel.class).getIfAvailable();
            if (model != null) {
              return new ModelsSpringAiChatModel(
                  model,
                  properties.parsedChatTemplate(),
                  properties.samplingOptions(),
                  observations);
            }
            InferenceBackend backend =
                beanFactory.getBeanProvider(InferenceBackend.class).getIfAvailable();
            if (backend == null) {
              throw new IllegalStateException(
                  "A Models runtime bean disappeared while creating "
                      + MODELS_CHAT_MODEL_BEAN_NAME);
            }
            return new ModelsSpringAiChatModel(
                backend,
                properties.parsedChatTemplate(),
                properties.samplingOptions(),
                observations);
          });
      registry.registerBeanDefinition(MODELS_CHAT_MODEL_BEAN_NAME, definition);
    }

    @Override
    public void postProcessBeanFactory(ConfigurableListableBeanFactory beanFactory)
        throws BeansException {}
  }
}
