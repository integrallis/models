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
package com.integrallis.models.bench;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.nativekernel.NativeKernelPlatform;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.List;
import java.util.Properties;
import org.junit.jupiter.api.Test;

class NativeBenchmarkRuntimeTest {

  @Test
  void suppliesExactlyOneHostNativeKernelArtifact() throws Exception {
    NativeKernelPlatform platform = NativeKernelPlatform.current();
    ClassLoader classLoader = getClass().getClassLoader();
    String metadataResource = platform.resourceDirectory() + "native.properties";
    List<URL> metadataResources = Collections.list(classLoader.getResources(metadataResource));

    assertThat(metadataResources)
        .as("runtime classpath must contain one native artifact for %s", platform.id())
        .hasSize(1);

    Properties metadata = new Properties();
    try (InputStream input = metadataResources.getFirst().openStream()) {
      metadata.load(new java.io.InputStreamReader(input, StandardCharsets.UTF_8));
    }
    assertThat(metadata.getProperty("platform")).isEqualTo(platform.id());
    assertThat(metadata.getProperty("library")).isEqualTo(platform.libraryFileName());
    assertThat(
            Collections.list(
                classLoader.getResources(
                    platform.resourceDirectory() + platform.libraryFileName())))
        .as("runtime classpath must contain one native library for %s", platform.id())
        .hasSize(1);
  }
}
