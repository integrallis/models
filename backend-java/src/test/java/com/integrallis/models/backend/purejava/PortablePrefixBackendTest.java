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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.*;

import com.integrallis.models.api.PortablePrefixCapabilities;
import com.integrallis.models.api.PortablePrefixCapabilities.Status;
import com.integrallis.models.api.PortablePrefixCapabilities.Truncation;
import java.io.*;
import java.nio.file.*;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class PortablePrefixBackendTest {
  @Test
  void aNamedInjectedKernelMustExplicitlySupplyItsPortableIdentity(@TempDir Path dir)
      throws Exception {
    var type = com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel.class;
    var kernel =
        (com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel)
            java.lang.reflect.Proxy.newProxyInstance(
                type.getClassLoader(),
                new Class<?>[] {type},
                (proxy, method, args) -> {
                  if (method.getName().equals("implementation")) return "cuda-ptx";
                  if (method.getName().equals("supports")) return false;
                  if (method.isDefault())
                    return java.lang.reflect.InvocationHandler.invokeDefault(proxy, method, args);
                  throw new UnsupportedOperationException(method.getName());
                });
    var file = PureJavaBackendTest.buildNanoModelFile(dir, new Random(42));
    try (var backend = PureJavaBackend.load(file, kernel)) {
      assertThat(backend.supportsSharedPrefixes()).isTrue();
      assertThat(backend.supportsPortablePrefixes()).isFalse();
      assertThat(backend.portablePrefixCapabilities())
          .isEqualTo(PortablePrefixCapabilities.unsupported(Status.MATRIX_KERNEL_UNSUPPORTED));
      assertThatThrownBy(backend::portablePrefixFingerprint)
          .isInstanceOf(UnsupportedOperationException.class);
    }
  }

  @Test
  void injectedAttentionStateCannotBeAdvertisedAsPortable(@TempDir Path dir) throws Exception {
    var attention =
        org.mockito.Mockito.mock(
            com.integrallis.models.backend.purejava.spi.BatchedCausalAttentionKernel.class);
    var file = PureJavaBackendTest.buildNanoModelFile(dir, new Random(42));
    try (var backend =
        PureJavaBackend.load(
            file,
            com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel.none(),
            attention)) {
      assertThat(backend.supportsPortablePrefixes()).isFalse();
      assertThat(backend.portablePrefixCapabilities())
          .isEqualTo(PortablePrefixCapabilities.unsupported(Status.ATTENTION_STATE_UNSUPPORTED));
    }
  }

  @Test
  void aMissingModelIdentityIsReportedBeforeFingerprinting(@TempDir Path dir) throws Exception {
    var file = PureJavaBackendTest.buildNanoModelFile(dir, new Random(42));
    try (var backend = PureJavaBackend.load(file)) {
      Files.move(file, dir.resolve("moved.gguf"));
      assertThat(backend.supportsPortablePrefixes()).isFalse();
      assertThat(backend.portablePrefixCapabilities())
          .isEqualTo(PortablePrefixCapabilities.unsupported(Status.MODEL_IDENTITY_UNAVAILABLE));
    }
  }

  @Test
  void capabilityDiscoveryDoesNotWaitForInferenceOrComputeAnIdentity(@TempDir Path dir)
      throws Exception {
    var file = PureJavaBackendTest.buildNanoModelFile(dir, new Random(42));
    var kernel =
        org.mockito.Mockito.mock(
            com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel.class,
            org.mockito.Mockito.CALLS_REAL_METHODS);
    org.mockito.Mockito.when(kernel.supportsPortablePrefixes()).thenReturn(true);
    var backend = PureJavaBackend.load(file, kernel);
    try (var pipeline = new com.integrallis.models.runtime.InferencePipeline(backend);
        var workers = java.util.concurrent.Executors.newSingleThreadExecutor()) {
      synchronized (backend) {
        var report = workers.submit(pipeline::portablePromptPrefixCapabilities);
        assertThat(report.get(2, java.util.concurrent.TimeUnit.SECONDS))
            .isEqualTo(PortablePrefixCapabilities.supported(Truncation.ANY_PREFIX));
      }
      assertThat(pipeline.portablePromptPrefixCapabilities().supported())
          .isEqualTo(pipeline.supportsPortablePromptPrefixes());
      org.mockito.Mockito.verify(kernel, org.mockito.Mockito.never()).portablePrefixFingerprint();
    }
  }

  @Test
  void immutableExportDoesNotWaitForTheInferenceExecutionLock(@TempDir Path dir) throws Exception {
    var file = PureJavaBackendTest.buildNanoModelFile(dir, new Random(42));
    var backend = PureJavaBackend.load(file);
    try (var pipeline = new com.integrallis.models.runtime.InferencePipeline(backend);
        var workers = java.util.concurrent.Executors.newFixedThreadPool(2)) {
      var prefix = pipeline.prepareSharedTokenPrefix(new int[] {2, 3, 4});
      pipeline.portablePrefixFingerprint();
      var locked = new java.util.concurrent.CountDownLatch(1);
      var release = new java.util.concurrent.CountDownLatch(1);
      var inference =
          workers.submit(
              () -> {
                synchronized (backend) {
                  locked.countDown();
                  try {
                    release.await();
                  } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                  }
                }
              });
      try {
        assertThat(locked.await(2, java.util.concurrent.TimeUnit.SECONDS)).isTrue();
        var export =
            workers.submit(
                () -> {
                  var output = new ByteArrayOutputStream();
                  pipeline.writeSharedPromptPrefix(prefix, 3, output);
                  return output.size();
                });
        assertThat(export.get(2, java.util.concurrent.TimeUnit.SECONDS)).isPositive();
      } finally {
        release.countDown();
        inference.get();
      }
    }
  }

  @Test
  void runtimeChecksIntegrityTokensAndContinuesWithoutPrefillingImportedTokens(@TempDir Path dir)
      throws Exception {
    var file = PureJavaBackendTest.buildNanoModelFile(dir, new Random(42));
    try (var donor =
            new com.integrallis.models.runtime.InferencePipeline(PureJavaBackend.load(file));
        var receiver =
            new com.integrallis.models.runtime.InferencePipeline(PureJavaBackend.load(file))) {
      assertThat(donor.portablePromptPrefixCapabilities())
          .isEqualTo(PortablePrefixCapabilities.supported(Truncation.ANY_PREFIX));
      assertThat(donor.supportsPortablePromptPrefixes()).isTrue();
      var prefix = donor.prepareSharedTokenPrefix(new int[] {2, 3, 4, 5});
      var bytes = new ByteArrayOutputStream();
      donor.writeSharedPromptPrefix(prefix, 3, bytes);
      var imported =
          receiver.readSharedPromptPrefix(
              new ByteArrayInputStream(bytes.toByteArray()), bytes.size());
      assertThat(imported.tokens()).containsExactly(2, 3, 4);
      assertThat(imported.commonPrefixLength(new int[] {2, 3, 9})).isEqualTo(2);
      try (var branch = receiver.openGenerationSession(imported)) {
        assertThat(branch.isClosed()).isFalse();
      }
      byte[] corrupt = bytes.toByteArray();
      corrupt[corrupt.length - 33] ^= 1;
      assertThatThrownBy(
              () ->
                  receiver.readSharedPromptPrefix(
                      new ByteArrayInputStream(corrupt), corrupt.length))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("checksum");
      assertThatThrownBy(
              () ->
                  receiver.readSharedPromptPrefix(
                      new ByteArrayInputStream(bytes.toByteArray()), 10))
          .isInstanceOf(IOException.class);
      assertThat(receiver.truncateSharedPromptPrefix(imported, 2).tokens()).containsExactly(2, 3);
      assertThatThrownBy(() -> receiver.truncateSharedPromptPrefix(prefix, 2))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }

  @Test
  void continuesOnAnotherBackendWithExactLogitsAndRejectsDifferentWeights(@TempDir Path dir)
      throws Exception {
    Path file = PureJavaBackendTest.buildNanoModelFile(dir, new Random(42));
    Path other =
        PureJavaBackendTest.buildNanoModelFile(
            Files.createDirectory(dir.resolve("other")), new Random(43));
    try (var donor = PureJavaBackend.load(file);
        var receiver = PureJavaBackend.load(file);
        var wrong = PureJavaBackend.load(other)) {
      var source = donor.openSession();
      donor.prefill(source, new int[] {2, 3, 4, 5}, 0);
      var prefix = donor.freezePrefix(source);
      var bytes = new ByteArrayOutputStream();
      donor.writePrefix(prefix, 3, new DataOutputStream(bytes));
      var imported = receiver.readPrefix(input(bytes), 3, 1_000_000);
      try (var actual = receiver.fork(imported);
          var control = receiver.openSession()) {
        receiver.prefill(control, new int[] {2, 3, 4}, 0);
        assertThat(receiver.forward(actual, 9, 3)).containsExactly(receiver.forward(control, 9, 3));
        assertThat(receiver.forward(actual, 8, 4)).containsExactly(receiver.forward(control, 8, 4));
      }
      assertThat(source.isClosed()).isTrue();
      assertThatThrownBy(() -> wrong.readPrefix(input(bytes), 3, 1_000_000))
          .isInstanceOf(IOException.class);
      assertThatThrownBy(() -> receiver.readPrefix(input(bytes), 3, 20))
          .isInstanceOf(IOException.class);
      assertThatThrownBy(() -> receiver.fork(prefix)).isInstanceOf(IllegalArgumentException.class);
    }
  }

  private static DataInputStream input(ByteArrayOutputStream bytes) {
    return new DataInputStream(new ByteArrayInputStream(bytes.toByteArray()));
  }
}
