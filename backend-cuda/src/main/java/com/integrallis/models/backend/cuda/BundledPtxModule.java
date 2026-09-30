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
package com.integrallis.models.backend.cuda;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
import java.net.URLConnection;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HexFormat;
import java.util.List;
import java.util.Properties;

/**
 * Resolves the integrity-checked PTX module packaged inside this jar.
 *
 * <p>This mirrors {@code BundledNativeKernelLibrary} in {@code backend-native}: a resource
 * directory holding the artifact and a {@code .properties} descriptor, a declared SHA-256 that is
 * recomputed at load, and an ABI field that must match the Java binding. The mechanism is
 * deliberately the same one; a second packaging scheme would be a second thing to get wrong.
 *
 * <p>It is simpler in one respect, and that is a real advantage of shipping PTX rather than a
 * shared library. PTX is <em>device</em> code: the same bytes serve every host platform, because
 * the CUDA driver assembles them for whatever GPU is present. {@code backend-native} needs six
 * per-platform artifacts and a host-architecture check; this needs one artifact and none. Nothing
 * is extracted to disk either — the bytes go straight to {@code cuModuleLoadData}, so there is no
 * cache directory, no executable bit, and no temporary file.
 */
final class BundledPtxModule {

  /** Classpath directory holding the PTX module and its descriptor. */
  static final String RESOURCE_DIRECTORY = "META-INF/models/cuda/";

  /** Descriptor filename inside {@link #RESOURCE_DIRECTORY}. */
  static final String METADATA_FILE_NAME = "cuda.properties";

  /** PTX filename inside {@link #RESOURCE_DIRECTORY}. */
  static final String MODULE_FILE_NAME = "models-cuda-kernels.ptx";

  private static final String ABI_PROPERTY = "abi";
  private static final String MODULE_PROPERTY = "module";
  private static final String SHA_256_PROPERTY = "sha256";
  private static final String TARGET_PROPERTY = "target";
  private static final String KERNELS_PROPERTY = "kernels";
  private static final String TOOLCHAIN_PROPERTY = "toolchain";

  private final byte[] ptx;
  private final String sha256;
  private final String target;
  private final String toolchain;
  private final List<String> kernelNames;

  private BundledPtxModule(
      byte[] ptx, String sha256, String target, String toolchain, List<String> kernelNames) {
    this.ptx = ptx;
    this.sha256 = sha256;
    this.target = target;
    this.toolchain = toolchain;
    this.kernelNames = List.copyOf(kernelNames);
  }

  /** Loads and verifies the PTX module packaged with this class. */
  static BundledPtxModule resolve() {
    return resolve(BundledPtxModule.class.getClassLoader());
  }

  /** Loads and verifies the PTX module visible to {@code classLoader}. */
  static BundledPtxModule resolve(ClassLoader classLoader) {
    URL metadataUrl = uniqueResource(classLoader, RESOURCE_DIRECTORY + METADATA_FILE_NAME);
    Properties metadata = loadMetadata(metadataUrl);

    String abi = requireMetadata(metadata, ABI_PROPERTY);
    if (!Integer.toString(CudaKernelAbi.VERSION).equals(abi)) {
      throw new IllegalStateException(
          "PTX artifact ABI " + abi + " does not match Java ABI " + CudaKernelAbi.VERSION);
    }
    String moduleName = requireMetadata(metadata, MODULE_PROPERTY);
    if (!MODULE_FILE_NAME.equals(moduleName)) {
      throw new IllegalStateException(
          "PTX artifact declares module " + moduleName + " but expected " + MODULE_FILE_NAME);
    }
    String expectedDigest = requireMetadata(metadata, SHA_256_PROPERTY);
    if (!expectedDigest.matches("[0-9a-f]{64}")) {
      throw new IllegalStateException("PTX artifact has an invalid SHA-256: " + expectedDigest);
    }

    URL moduleUrl = uniqueResource(classLoader, RESOURCE_DIRECTORY + moduleName);
    byte[] ptx = readBytes(moduleUrl);
    String actualDigest = sha256(ptx);
    if (!MessageDigest.isEqual(
        expectedDigest.getBytes(StandardCharsets.US_ASCII),
        actualDigest.getBytes(StandardCharsets.US_ASCII))) {
      throw new SecurityException(
          "PTX module SHA-256 mismatch: expected " + expectedDigest + " but found " + actualDigest);
    }

    List<String> kernels = List.of(requireMetadata(metadata, KERNELS_PROPERTY).split(","));
    if (!kernels.containsAll(CudaKernelAbi.KERNEL_NAMES)) {
      throw new IllegalStateException(
          "PTX artifact declares kernels "
              + kernels
              + " but Java requires "
              + CudaKernelAbi.KERNEL_NAMES);
    }
    return new BundledPtxModule(
        ptx,
        expectedDigest,
        requireMetadata(metadata, TARGET_PROPERTY),
        requireMetadata(metadata, TOOLCHAIN_PROPERTY),
        kernels);
  }

  /** The verified PTX bytes, ready for {@code cuModuleLoadData}. */
  byte[] ptx() {
    return ptx.clone();
  }

  /**
   * The module's SHA-256, reported with every benchmark result as the kernel identity.
   *
   * <p>A performance number without this is not reproducible: it does not say which kernels ran.
   */
  String sha256() {
    return sha256;
  }

  /** The PTX virtual architecture the module was compiled for, such as {@code sm_80}. */
  String target() {
    return target;
  }

  /** The exact Rust toolchain that produced the module, so a regression is attributable. */
  String toolchain() {
    return toolchain;
  }

  /** Entry-point names the module exports. */
  List<String> kernelNames() {
    return kernelNames;
  }

  /** The minimum compute capability implied by {@link #target()}, as {@code major * 10 + minor}. */
  int minimumComputeCapability() {
    if (!target.startsWith("sm_")) {
      throw new IllegalStateException("unrecognised PTX target: " + target);
    }
    return Integer.parseInt(target.substring(3));
  }

  private static URL uniqueResource(ClassLoader classLoader, String resourceName) {
    try {
      Enumeration<URL> resources = classLoader.getResources(resourceName);
      List<URL> matches = new ArrayList<>(2);
      while (resources.hasMoreElements()) {
        matches.add(resources.nextElement());
      }
      if (matches.isEmpty()) {
        throw new IllegalStateException(
            "no Models CUDA kernel artifact; expected classpath resource " + resourceName);
      }
      if (matches.size() != 1) {
        throw new IllegalStateException(
            "multiple Models CUDA kernel artifacts provide " + resourceName + ": " + matches);
      }
      return matches.getFirst();
    } catch (IOException failure) {
      throw new IllegalStateException(
          "failed to locate classpath resource " + resourceName, failure);
    }
  }

  private static Properties loadMetadata(URL metadataUrl) {
    Properties metadata = new Properties();
    try (InputStream input = openUncachedStream(metadataUrl);
        Reader reader = new InputStreamReader(input, StandardCharsets.UTF_8)) {
      metadata.load(reader);
      return metadata;
    } catch (IOException failure) {
      throw new IllegalStateException(
          "failed to read CUDA kernel metadata " + metadataUrl, failure);
    }
  }

  private static String requireMetadata(Properties metadata, String name) {
    String value = metadata.getProperty(name);
    if (value == null || value.isBlank()) {
      throw new IllegalStateException("CUDA kernel metadata is missing " + name);
    }
    return value.strip();
  }

  private static byte[] readBytes(URL resource) {
    try (InputStream input = openUncachedStream(resource)) {
      return input.readAllBytes();
    } catch (IOException failure) {
      throw new IllegalStateException("failed to read PTX module " + resource, failure);
    }
  }

  private static InputStream openUncachedStream(URL resource) throws IOException {
    URLConnection connection = resource.openConnection();
    connection.setUseCaches(false);
    return connection.getInputStream();
  }

  private static String sha256(byte[] bytes) {
    try {
      return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    } catch (NoSuchAlgorithmException impossible) {
      throw new IllegalStateException("SHA-256 is required by the Java platform", impossible);
    }
  }
}
