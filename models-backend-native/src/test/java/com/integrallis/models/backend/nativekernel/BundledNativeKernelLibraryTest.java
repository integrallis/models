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
package com.integrallis.models.backend.nativekernel;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class BundledNativeKernelLibraryTest {
  private static final byte[] LIBRARY_BYTES =
      "models-native-test-library".getBytes(StandardCharsets.UTF_8);

  @TempDir Path temporaryDirectory;

  @Test
  void verifiesAndExtractsTheCurrentPlatformLibraryOnce() throws IOException {
    NativeKernelPlatform platform = NativeKernelPlatform.detect("Linux", "amd64");
    Path artifact = createArtifact(platform, sha256(LIBRARY_BYTES), LIBRARY_BYTES);
    Path cache = temporaryDirectory.resolve("cache");

    try (URLClassLoader loader = classLoader(artifact)) {
      Path first = BundledNativeKernelLibrary.resolve(loader, platform, cache);
      Path second = BundledNativeKernelLibrary.resolve(loader, platform, cache);

      assertThat(first).isEqualTo(second).isRegularFile();
      assertThat(Files.readAllBytes(first)).containsExactly(LIBRARY_BYTES);
      assertThat(first)
          .startsWith(
              cache.resolve("abi-" + NativeKernelLibrary.ABI_VERSION).resolve(platform.id()));
    }
  }

  @Test
  void rejectsAnArtifactWhoseLibraryDoesNotMatchItsDigest() throws IOException {
    NativeKernelPlatform platform = NativeKernelPlatform.detect("Mac OS X", "arm64");
    Path artifact = createArtifact(platform, "0".repeat(64), LIBRARY_BYTES);

    try (URLClassLoader loader = classLoader(artifact)) {
      assertThatThrownBy(
              () ->
                  BundledNativeKernelLibrary.resolve(
                      loader, platform, temporaryDirectory.resolve("cache")))
          .isInstanceOf(SecurityException.class)
          .hasMessageContaining("SHA-256");
    }
  }

  @Test
  void rejectsMetadataForAnotherPlatform() throws IOException {
    NativeKernelPlatform packagedPlatform = NativeKernelPlatform.detect("Linux", "aarch64");
    NativeKernelPlatform runtimePlatform = NativeKernelPlatform.detect("Linux", "amd64");
    Path artifact = createArtifact(packagedPlatform, sha256(LIBRARY_BYTES), LIBRARY_BYTES);

    try (URLClassLoader loader = classLoader(artifact)) {
      assertThatThrownBy(
              () ->
                  BundledNativeKernelLibrary.resolve(
                      loader, runtimePlatform, temporaryDirectory.resolve("cache")))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining(runtimePlatform.id());
    }
  }

  @Test
  void packagedHostArtifactCanBeExtractedAndOpened() throws IOException {
    String configuredArtifact = System.getProperty("models.native.test.artifact");
    assertThat(configuredArtifact)
        .as("Gradle must supply the packaged host native artifact")
        .isNotBlank();
    Path artifact = Path.of(configuredArtifact);
    NativeKernelPlatform platform = NativeKernelPlatform.current();

    try (URLClassLoader loader = classLoader(artifact)) {
      Path library =
          BundledNativeKernelLibrary.resolve(
              loader, platform, temporaryDirectory.resolve("packaged-cache"));

      assertThat(library.getFileName().toString()).isEqualTo(platform.libraryFileName());
      try (NativeKernelLibrary kernels = NativeKernelLibrary.open(library)) {
        assertThat(kernels.abiVersion()).isEqualTo(NativeKernelLibrary.ABI_VERSION);
        assertThat(kernels.supports(NativeKernelCapability.PERSISTENT_WORKER_CONTEXT)).isTrue();
      }
    }
  }

  private Path createArtifact(NativeKernelPlatform platform, String digest, byte[] libraryBytes)
      throws IOException {
    Path artifact = temporaryDirectory.resolve(platform.id() + ".jar");
    String metadata =
        """
        abi=%d
        platform=%s
        library=%s
        sha256=%s
        """
            .formatted(
                NativeKernelLibrary.ABI_VERSION, platform.id(), platform.libraryFileName(), digest);
    try (JarOutputStream jar = new JarOutputStream(Files.newOutputStream(artifact))) {
      addEntry(
          jar,
          platform.resourceDirectory() + BundledNativeKernelLibrary.METADATA_FILE_NAME,
          metadata.getBytes(StandardCharsets.UTF_8));
      addEntry(jar, platform.resourceDirectory() + platform.libraryFileName(), libraryBytes);
    }
    return artifact;
  }

  private static URLClassLoader classLoader(Path artifact) throws IOException {
    return new URLClassLoader(new URL[] {artifact.toUri().toURL()}, null);
  }

  private static void addEntry(JarOutputStream jar, String name, byte[] bytes) throws IOException {
    jar.putNextEntry(new JarEntry(name));
    jar.write(bytes);
    jar.closeEntry();
  }

  private static String sha256(byte[] bytes) {
    try {
      return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    } catch (NoSuchAlgorithmException impossible) {
      throw new IllegalStateException("SHA-256 is required by the Java platform", impossible);
    }
  }
}
