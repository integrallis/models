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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarLocator;
import org.modeljars.ModelJarRegistry;
import org.modeljars.ModelJarRequirement;
import org.modeljars.ModelPerformanceProfile;
import org.modeljars.ModelPerformanceProfileRegistry;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ConfigurableApplicationContext;

@Tag("unit")
class ModelJarsSpringBootSmokeTest {

  @Test
  void springApplicationResolvesModelJarsMarkerCatalog() {
    try (ConfigurableApplicationContext context =
        new SpringApplicationBuilder(TestApplication.class)
            .web(WebApplicationType.NONE)
            .properties(
                "models.modeljars.source=hf://ggml-org/Qwen3-0.6B-GGUF",
                "models.modeljars.version-range=[3.0.0,4.0.0)",
                "models.modeljars.variant=q4_0",
                "models.modeljars.backend=pure-java",
                "models.modeljars.capability=text-generation")
            .run()) {
      assertThat(context.getBean(ModelJarRegistry.class).descriptors()).isNotEmpty();
      assertThat(context.getBean(ModelPerformanceProfileRegistry.class).profiles())
          .extracting(ModelPerformanceProfile::id)
          .contains("qwen3_0_6b_q4_0_epyc_milan_jdk25");
      assertThat(context.getBean(ModelJarLocator.class)).isNotNull();
      assertThat(context.getBean(ModelJarRequirement.class).source())
          .isEqualTo("hf://ggml-org/Qwen3-0.6B-GGUF");

      ModelJarDescriptor descriptor = context.getBean(ModelJarDescriptor.class);
      assertThat(descriptor.markerCoordinate().groupId()).isEqualTo("org.modeljars.huggingface");
      assertThat(descriptor.localPath().orElseThrow().toString())
          .endsWith(".jvllm/models/Qwen3-0.6B-Q4_0.gguf");
    }
  }

  @SpringBootApplication
  static class TestApplication {}
}
