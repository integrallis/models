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
package com.integrallis.models.bench.fusion;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.bench.fusion.ReportModel.BackendConfig;
import com.integrallis.models.bench.fusion.ReportModel.MemberConfig;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Loads GGUF members on the pure-Java or Rust/FFM backend and records their identity. */
final class MemberLoader implements AutoCloseable {
  static final String NATIVE_LIBRARY_PROPERTY = "models.native.kernels.library";
  static final String NATIVE_LIBRARY_ENV = "MODELS_NATIVE_KERNELS_LIBRARY";

  private final Map<String, FusionMember> members = new LinkedHashMap<>();
  private final List<MemberConfig> configs = new ArrayList<>();
  private final BackendConfig backend;
  private long loadMillis;

  private MemberLoader(BackendConfig backend) {
    this.backend = backend;
  }

  static MemberLoader load(
      Map<String, Path> paths,
      List<String> names,
      String backendName,
      int contextLength,
      ObjectMapper mapper)
      throws IOException {
    if (!"pure-java".equals(backendName) && !"rust-ffm".equals(backendName)) {
      throw new IllegalArgumentException("--backend must be pure-java or rust-ffm");
    }
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(contextLength));
    MemberLoader loader = new MemberLoader(backendConfig(backendName));
    try {
      for (String name : names) {
        Path path = paths.get(name);
        if (path == null) {
          throw new IllegalArgumentException("arm member " + name + " is not named in --members");
        }
        TokenizerIdentity identity = TokenizerIdentity.read(path);
        long started = System.nanoTime();
        InferenceBackend loaded =
            "rust-ffm".equals(backendName) ? loadNative(path) : PureJavaBackend.load(path);
        PipelineMember member = new PipelineMember(name, loaded);
        long millis = (System.nanoTime() - started) / 1_000_000L;
        loader.loadMillis += millis;
        loader.members.put(name, member);
        loader.configs.add(
            new MemberConfig(
                name,
                path.toAbsolutePath().toString(),
                Digests.sha256(path),
                Files.size(path),
                identity.tokenizerSha256(),
                identity.fields(),
                identity.chatTemplateSha256(),
                identity.architecture(),
                identity.fileType(),
                identity.vocabularySize(),
                mapper.valueToTree(member.pipeline().diagnostics()),
                millis));
      }
      return loader;
    } catch (IOException | RuntimeException | Error failure) {
      loader.close();
      throw failure;
    }
  }

  /** Identity only, without loading weights (gate G0). */
  static MemberConfig describe(String name, Path path) throws IOException {
    TokenizerIdentity identity = TokenizerIdentity.read(path);
    return new MemberConfig(
        name,
        path.toAbsolutePath().toString(),
        Digests.sha256(path),
        Files.size(path),
        identity.tokenizerSha256(),
        identity.fields(),
        identity.chatTemplateSha256(),
        identity.architecture(),
        identity.fileType(),
        identity.vocabularySize(),
        null,
        0);
  }

  private static InferenceBackend loadNative(Path model) {
    try {
      Class<?> type = Class.forName("com.integrallis.models.backend.nativekernel.RustFfmBackend");
      return (InferenceBackend) type.getMethod("load", Path.class).invoke(null, model);
    } catch (ClassNotFoundException missing) {
      throw new IllegalStateException(
          "rust-ffm requires the optional backend-native runtime; build with -PmodelsBenchNative=true",
          missing);
    } catch (ReflectiveOperationException failure) {
      throw new IllegalStateException("could not load the Rust/FFM backend", failure);
    }
  }

  static BackendConfig backendConfig(String backendName) {
    if (!"rust-ffm".equals(backendName)) {
      return new BackendConfig(backendName, null, null);
    }
    String configured = System.getProperty(NATIVE_LIBRARY_PROPERTY);
    if (configured == null || configured.isBlank()) {
      configured = System.getenv(NATIVE_LIBRARY_ENV);
    }
    Path library = null;
    if (configured != null && !configured.isBlank()) {
      library = Path.of(configured);
    } else {
      try {
        Class<?> bundled =
            Class.forName("com.integrallis.models.backend.nativekernel.BundledNativeKernelLibrary");
        java.lang.reflect.Method resolve = bundled.getDeclaredMethod("resolve");
        resolve.setAccessible(true);
        library = (Path) resolve.invoke(null);
      } catch (ReflectiveOperationException | RuntimeException unavailable) {
        library = null;
      }
    }
    String sha = null;
    if (library != null && Files.isRegularFile(library)) {
      try {
        sha = Digests.sha256(library);
      } catch (IOException unreadable) {
        sha = null;
      }
    }
    return new BackendConfig(backendName, library == null ? null : library.toString(), sha);
  }

  Map<String, FusionMember> members() {
    return Map.copyOf(members);
  }

  List<MemberConfig> configs() {
    return List.copyOf(configs);
  }

  BackendConfig backend() {
    return backend;
  }

  long loadMillis() {
    return loadMillis;
  }

  @Override
  public void close() {
    RuntimeException failure = null;
    for (FusionMember member : members.values()) {
      try {
        member.close();
      } catch (RuntimeException closeFailure) {
        if (failure == null) {
          failure = closeFailure;
        } else {
          failure.addSuppressed(closeFailure);
        }
      }
    }
    members.clear();
    if (failure != null) {
      throw failure;
    }
  }
}
