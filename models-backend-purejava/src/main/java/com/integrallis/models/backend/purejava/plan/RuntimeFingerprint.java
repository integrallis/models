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
package com.integrallis.models.backend.purejava.plan;

import com.integrallis.vectors.core.VectorRuntimeCapabilities;
import com.integrallis.vectors.core.VectorUtil;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/** Stable runtime properties used to select a pure-Java execution plan. */
public record RuntimeFingerprint(
    String javaVersion,
    String vmName,
    String vmVendor,
    String vmVersion,
    String compiler,
    String osName,
    String architecture,
    String vectorProvider,
    boolean vectorApi,
    int preferredVectorBits,
    int vectorBits,
    boolean fastVectorFma,
    boolean fastScalarFma,
    boolean sve,
    String ggufExecutor,
    int ggufThreads,
    int ggufChunksPerThread,
    int processors) {

  public RuntimeFingerprint {
    javaVersion = requireText(javaVersion, "javaVersion");
    vmName = requireText(vmName, "vmName");
    vmVendor = requireText(vmVendor, "vmVendor");
    vmVersion = requireText(vmVersion, "vmVersion");
    compiler = requireText(compiler, "compiler");
    osName = requireText(osName, "osName");
    architecture = requireText(architecture, "architecture");
    vectorProvider = requireText(vectorProvider, "vectorProvider");
    if (preferredVectorBits <= 0) {
      throw new IllegalArgumentException("preferredVectorBits must be > 0");
    }
    if (vectorBits < 0) {
      throw new IllegalArgumentException("vectorBits must be >= 0");
    }
    if (vectorApi != (vectorBits > 0)) {
      throw new IllegalArgumentException("vectorApi and vectorBits disagree");
    }
    ggufExecutor = requireText(ggufExecutor, "ggufExecutor");
    if (ggufThreads <= 0 || ggufChunksPerThread <= 0) {
      throw new IllegalArgumentException("GGUF thread and chunk counts must be > 0");
    }
    if (processors <= 0) {
      throw new IllegalArgumentException("processors must be > 0");
    }
  }

  RuntimeFingerprint(
      String javaVersion,
      String vmName,
      String vmVendor,
      String vmVersion,
      String compiler,
      String osName,
      String architecture,
      int vectorBits,
      int processors) {
    this(
        javaVersion,
        vmName,
        vmVendor,
        vmVersion,
        compiler,
        osName,
        architecture,
        vectorBits > 0 ? "test-vector" : "test-scalar",
        vectorBits > 0,
        Math.max(64, vectorBits),
        vectorBits,
        false,
        false,
        false,
        "persistent",
        processors,
        2,
        processors);
  }

  /** Captures properties without running a startup performance probe. */
  public static RuntimeFingerprint capture() {
    Map<String, String> properties = new LinkedHashMap<>();
    for (String name : System.getProperties().stringPropertyNames()) {
      properties.put(name, System.getProperty(name));
    }
    VectorRuntimeCapabilities vectors = VectorUtil.runtimeCapabilities();
    return from(properties, vectors, Runtime.getRuntime().availableProcessors());
  }

  static RuntimeFingerprint from(Map<String, String> properties, int vectorBits, int processors) {
    Objects.requireNonNull(properties, "properties");
    String vmName = property(properties, "java.vm.name");
    String vmVendor = property(properties, "java.vm.vendor");
    String vmVersion = property(properties, "java.vm.version");
    String runtimeName = properties.getOrDefault("java.runtime.name", "");
    String compiler = compiler(vmName, vmVendor, vmVersion, runtimeName);
    return new RuntimeFingerprint(
        property(properties, "java.version"),
        vmName,
        vmVendor,
        vmVersion,
        compiler,
        property(properties, "os.name"),
        property(properties, "os.arch"),
        vectorBits,
        processors);
  }

  private static RuntimeFingerprint from(
      Map<String, String> properties, VectorRuntimeCapabilities vectors, int processors) {
    Objects.requireNonNull(properties, "properties");
    Objects.requireNonNull(vectors, "vectors");
    String vmName = property(properties, "java.vm.name");
    String vmVendor = property(properties, "java.vm.vendor");
    String vmVersion = property(properties, "java.vm.version");
    String runtimeName = properties.getOrDefault("java.runtime.name", "");
    return new RuntimeFingerprint(
        property(properties, "java.version"),
        vmName,
        vmVendor,
        vmVersion,
        compiler(vmName, vmVendor, vmVersion, runtimeName),
        property(properties, "os.name"),
        property(properties, "os.arch"),
        vectors.providerName(),
        vectors.vectorApi(),
        vectors.preferredVectorBits(),
        vectors.activeVectorBits(),
        vectors.fastVectorFma(),
        vectors.fastScalarFma(),
        vectors.sve(),
        vectors.ggufExecutor(),
        vectors.ggufThreads(),
        vectors.ggufChunksPerThread(),
        processors);
  }

  /** Returns the backend-neutral diagnostic representation. */
  public Map<String, String> asEnvironment() {
    Map<String, String> environment = new LinkedHashMap<>();
    environment.put("java-version", javaVersion);
    environment.put("vm-name", vmName);
    environment.put("vm-vendor", vmVendor);
    environment.put("vm-version", vmVersion);
    environment.put("compiler", compiler);
    environment.put("os", osName);
    environment.put("architecture", architecture);
    environment.put("vector-provider", vectorProvider);
    environment.put("vector-api", Boolean.toString(vectorApi));
    environment.put("preferred-vector-bits", Integer.toString(preferredVectorBits));
    environment.put("active-vector-bits", Integer.toString(vectorBits));
    environment.put("fast-vector-fma", Boolean.toString(fastVectorFma));
    environment.put("fast-scalar-fma", Boolean.toString(fastScalarFma));
    environment.put("sve", Boolean.toString(sve));
    environment.put("gguf-executor", ggufExecutor);
    environment.put("gguf-threads", Integer.toString(ggufThreads));
    environment.put("gguf-chunks-per-thread", Integer.toString(ggufChunksPerThread));
    environment.put("processors", Integer.toString(processors));
    return Map.copyOf(environment);
  }

  private static String compiler(
      String vmName, String vmVendor, String vmVersion, String runtimeName) {
    String identity =
        String.join(" ", vmName, vmVendor, vmVersion, runtimeName).toLowerCase(Locale.ROOT);
    if (identity.contains("graal")) {
      return "graal-jvmci";
    }
    if (identity.contains("openj9")) {
      return "openj9";
    }
    return "hotspot-c2";
  }

  private static String property(Map<String, String> properties, String name) {
    return requireText(properties.get(name), name);
  }

  private static String requireText(String value, String name) {
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException(name + " must not be blank");
    }
    return value.trim();
  }
}
