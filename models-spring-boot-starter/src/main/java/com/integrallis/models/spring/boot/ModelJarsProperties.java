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

import org.modeljars.ModelJarRequirement;
import org.springframework.boot.context.properties.ConfigurationProperties;

/** Configuration properties used to select a ModelJars descriptor. */
@ConfigurationProperties(prefix = "models.modeljars")
public class ModelJarsProperties {
  private String source;
  private String versionRange;
  private String variant;
  private String backend = "pure-java";
  private String capability;

  public String getSource() {
    return source;
  }

  public void setSource(String source) {
    this.source = source;
  }

  public String getVersionRange() {
    return versionRange;
  }

  public void setVersionRange(String versionRange) {
    this.versionRange = versionRange;
  }

  public String getVariant() {
    return variant;
  }

  public void setVariant(String variant) {
    this.variant = variant;
  }

  public String getBackend() {
    return backend;
  }

  public void setBackend(String backend) {
    this.backend = backend;
  }

  public String getCapability() {
    return capability;
  }

  public void setCapability(String capability) {
    this.capability = capability;
  }

  ModelJarRequirement toRequirement() {
    ModelJarRequirement.Builder builder = ModelJarRequirement.forSource(source);
    if (versionRange != null && !versionRange.isBlank()) {
      builder.versionRange(versionRange);
    }
    if (variant != null && !variant.isBlank()) {
      builder.variant(variant);
    }
    if (backend != null && !backend.isBlank()) {
      builder.backend(backend);
    }
    if (capability != null && !capability.isBlank()) {
      builder.capability(capability);
    }
    return builder.build();
  }
}
