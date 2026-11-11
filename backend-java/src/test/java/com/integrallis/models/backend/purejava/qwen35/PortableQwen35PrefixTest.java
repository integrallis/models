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
package com.integrallis.models.backend.purejava.qwen35;

import static org.assertj.core.api.Assertions.*;

import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.runtime.InferencePipeline;
import java.io.*;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.concurrent.*;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class PortableQwen35PrefixTest {
  @Test
  void forkedStateGrowsWithTheSuffixRatherThanTheDeclaredContext(@TempDir Path dir)
      throws Exception {
    var file = Qwen35ForwardPassTest.writeToyModel(dir, false, 262_144);
    try (var arena = Arena.ofConfined()) {
      var graph = Qwen35ForwardPass.fromGgufFile(GgufParser.parse(file, arena), 4);
      var session = graph.openSession(262_144);
      assertThat(session.allocatedCacheBytes()).isZero();
      graph.prefill(session, tokens(17), 0);
      assertThat(session.allocatedCacheBytes())
          .isEqualTo(32L * 8 * graph.config().attentionKeyDim());
      var prefix = graph.freezePrefix(session);
      var branch = graph.fork(prefix);
      long shared = 17L * 8 * graph.config().attentionKeyDim();
      assertThat(branch.allocatedCacheBytes()).isEqualTo(shared);
      graph.forward(branch, 4, 17);
      assertThat(branch.allocatedCacheBytes())
          .isEqualTo(shared + 16L * 8 * graph.config().attentionKeyDim());
      graph.reset(branch);
      assertThat(branch.checkpoint()).isEqualTo(17);
      assertThat(branch.allocatedCacheBytes())
          .isEqualTo(shared + 16L * 8 * graph.config().attentionKeyDim());
    }
  }

  @Test
  void completeCheckpointsContinueExactlyAcrossIndependentGraphs(@TempDir Path dir)
      throws Exception {
    var file = Qwen35ForwardPassTest.writeToyModel(dir, false, 64);
    for (int batch : new int[] {1, 2, 4}) {
      try (var firstArena = Arena.ofConfined();
          var secondArena = Arena.ofConfined()) {
        var donor = Qwen35ForwardPass.fromGgufFile(GgufParser.parse(file, firstArena), batch);
        var receiver = Qwen35ForwardPass.fromGgufFile(GgufParser.parse(file, secondArena), batch);
        for (int length : new int[] {1, 2, 3, 7, 17}) {
          int[] tokens = tokens(length);
          var source = donor.openSession(64);
          donor.prefill(source, tokens, 0);
          var prefix = donor.freezePrefix(source);
          assertThatThrownBy(() -> donor.forward(source, 2, length))
              .isInstanceOf(IllegalStateException.class);
          byte[] bytes = encode(prefix);
          assertThat(bytes.length).isEqualTo(donor.portablePrefixBytes(length));
          var imported = receiver.readPrefix(input(bytes), length, bytes.length, 64);
          imported.validateTokens(tokens);
          assertThat(encode(imported)).containsExactly(bytes);
          var a = receiver.fork(imported);
          var b = receiver.fork(imported);
          var controlA = receiver.openSession(64);
          var controlB = receiver.openSession(64);
          receiver.prefill(controlA, tokens, 0);
          receiver.prefill(controlB, tokens, 0);
          assertThat(receiver.sharesPrefixStorage(a, b)).isTrue();
          assertThat(a.allocatedCacheBytes())
              .isEqualTo(8L * length * receiver.config().attentionKeyDim());
          for (int index = 0; index < 4; index++) {
            float[] actual = receiver.forward(a, 2 + index, length + index);
            float[] expected = receiver.forward(controlA, 2 + index, length + index);
            assertThat(actual).containsExactly(expected);
            assertThat(argmax(actual)).isEqualTo(argmax(expected));
          }
          assertThat(receiver.prefill(b, new int[] {6, 5, 4}, length))
              .containsExactly(receiver.prefill(controlB, new int[] {6, 5, 4}, length));
          assertThat(receiver.sharesPrefixStorage(a, b)).isTrue();
          assertThat(encode(imported)).containsExactly(bytes);
          receiver.reset(b);
          assertThat(b.checkpoint()).isEqualTo(length);
          var fresh = receiver.fork(imported);
          assertThat(receiver.forward(b, 7, length))
              .containsExactly(receiver.forward(fresh, 7, length));
          assertThatThrownBy(() -> receiver.rewind(b, length - 1))
              .isInstanceOf(IllegalArgumentException.class);
          assertThatThrownBy(() -> imported.truncate(length - 1))
              .isInstanceOf(IllegalArgumentException.class);
          assertThat(imported.truncate(length)).isSameAs(imported);
        }
      }
    }
  }

  @Test
  void extendedCheckpointsKeepEarlierStorageAndReplayOnlyTheirPrivateSuffix(@TempDir Path dir)
      throws Exception {
    var file = Qwen35ForwardPassTest.writeToyModel(dir, false, 64);
    try (var arena = Arena.ofConfined()) {
      var graph = Qwen35ForwardPass.fromGgufFile(GgufParser.parse(file, arena), 4);
      var source = graph.openSession(64);
      graph.prefill(source, new int[] {2, 3, 4}, 0);
      var parent = graph.freezePrefix(source);
      var oldBranch = graph.fork(parent);
      var extension = graph.fork(parent);
      graph.prefill(extension, new int[] {5, 6}, 3);
      var child = graph.freezePrefix(extension);
      var branch = graph.fork(child);
      assertThat(graph.sharesPrefixStorage(oldBranch, branch)).isTrue();
      var control = graph.openSession(64);
      graph.prefill(control, new int[] {2, 3, 4}, 0);
      graph.prefill(control, new int[] {5, 6}, 3);
      assertThat(graph.prefill(branch, new int[] {7, 2}, 5))
          .containsExactly(graph.prefill(control, new int[] {7, 2}, 5));
      graph.rewind(branch, 6);
      var replay = graph.fork(child);
      graph.prefill(replay, new int[] {7}, 5);
      assertThat(graph.forward(branch, 4, 6)).containsExactly(graph.forward(replay, 4, 6));
      var imported = graph.readPrefix(input(encode(child)), 5, encode(child).length, 64);
      var importedBranch = graph.fork(imported);
      var childBranch = graph.fork(child);
      assertThat(graph.forward(importedBranch, 3, 5))
          .containsExactly(graph.forward(childBranch, 3, 5));
      assertThat(graph.decideGrouped(graph.fork(child), new int[][] {{2, 3}, {4, 5}}))
          .isDeepEqualTo(graph.decideGrouped(graph.fork(imported), new int[][] {{2, 3}, {4, 5}}));
    }
  }

  @Test
  void runtimeValidatesHistoryIntegrityAndIdentityWithoutTouchingLiveSessions(@TempDir Path dir)
      throws Exception {
    var file = Qwen35ForwardPassTest.writeToyModel(dir, false, 64);
    var donorBackend = PureJavaBackend.load(file);
    var receiverBackend = PureJavaBackend.load(file);
    try (var donor = new InferencePipeline(donorBackend);
        var receiver = new InferencePipeline(receiverBackend);
        var live = receiverBackend.openSession()) {
      receiverBackend.prefill(live, new int[] {4, 5}, 0);
      var prefix = donor.prepareSharedTokenPrefix(new int[] {2, 3, 4});
      var output = new ByteArrayOutputStream();
      donor.writeSharedPromptPrefix(prefix, 3, output);
      byte[] bytes = output.toByteArray();
      assertThat(bytes.length).isEqualTo(donor.portablePrefixBytes(3));
      var imported = receiver.readSharedPromptPrefix(new ByteArrayInputStream(bytes), bytes.length);
      assertThat(imported.tokens()).containsExactly(2, 3, 4);
      try (var session = receiver.openGenerationSession(imported)) {
        session.resetContext();
        assertThat(session.isClosed()).isFalse();
      }
      assertThat(live.checkpoint()).isEqualTo(2);
      byte[] historyMismatch = bytes.clone();
      ByteBuffer.wrap(historyMismatch)
          .putInt(8, 5); // Valid token, inconsistent with retained history.
      resign(historyMismatch);
      assertThatThrownBy(
              () ->
                  receiver.readSharedPromptPrefix(
                      new ByteArrayInputStream(historyMismatch), historyMismatch.length))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("history");
      byte[] corruption = bytes.clone();
      corruption[corruption.length - 33] ^= 1;
      assertThatThrownBy(
              () ->
                  receiver.readSharedPromptPrefix(
                      new ByteArrayInputStream(corruption), corruption.length))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("checksum");
      assertThatThrownBy(
              () ->
                  receiver.readSharedPromptPrefix(
                      new ByteArrayInputStream(bytes), bytes.length - 1))
          .isInstanceOf(IOException.class);
      assertThatThrownBy(() -> donor.truncateSharedPromptPrefix(prefix, 2))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("boundary");
      assertThat(live.checkpoint()).isEqualTo(2);
      // Multiple detached imports do not change an already-open session.
      var again = receiver.readSharedPromptPrefix(new ByteArrayInputStream(bytes), bytes.length);
      try (var session = receiver.openGenerationSession(again)) {
        assertThat(session.isClosed()).isFalse();
      }
    }
  }

  @Test
  void importedStateSurvivesDonorClosureAndRejectsDifferentWeights(@TempDir Path dir)
      throws Exception {
    var file = Qwen35ForwardPassTest.writeToyModel(dir);
    byte[] payload;
    try (var donor = new InferencePipeline(PureJavaBackend.load(file))) {
      var prefix = donor.prepareSharedTokenPrefix(new int[] {2, 3});
      var bytes = new ByteArrayOutputStream();
      donor.writeSharedPromptPrefix(prefix, 2, bytes);
      payload = bytes.toByteArray();
    }
    try (var receiver = new InferencePipeline(PureJavaBackend.load(file))) {
      var prefix =
          receiver.readSharedPromptPrefix(new ByteArrayInputStream(payload), payload.length);
      try (var session = receiver.openGenerationSession(prefix)) {
        assertThat(session.isClosed()).isFalse();
      }
    }
    byte[] different = Files.readAllBytes(file);
    different[different.length - 1] ^= 1;
    var wrong = Files.write(dir.resolve("different.gguf"), different);
    try (var receiver = new InferencePipeline(PureJavaBackend.load(wrong))) {
      assertThatThrownBy(
              () ->
                  receiver.readSharedPromptPrefix(
                      new ByteArrayInputStream(payload), payload.length))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("identity");
    }
  }

  @Test
  void importRejectsMalformedLayoutAndBudgetBeforeReadingTensorPayload(@TempDir Path dir)
      throws Exception {
    var file = Qwen35ForwardPassTest.writeToyModel(dir);
    try (var arena = Arena.ofConfined()) {
      var graph = Qwen35ForwardPass.fromGgufFile(GgufParser.parse(file, arena));
      var session = graph.openSession(8);
      graph.prefill(session, new int[] {2, 3}, 0);
      byte[] bytes = encode(graph.freezePrefix(session));
      for (int offset : new int[] {0, 4, 8, 12, 16, 20, 24, 25}) {
        byte[] invalid = Arrays.copyOf(bytes, 26); // No token/tensor payload available.
        invalid[offset] ^= 2;
        assertThatThrownBy(() -> graph.readPrefix(input(invalid), 2, bytes.length, 8))
            .isInstanceOf(IOException.class)
            .isNotInstanceOf(EOFException.class);
      }
      assertThatThrownBy(() -> graph.readPrefix(input(new byte[0]), 2, bytes.length - 1, 8))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("budget");
      assertThatThrownBy(() -> graph.readPrefix(input(bytes), 2, bytes.length, 1))
          .isInstanceOf(IOException.class)
          .hasMessageContaining("capacity");
      assertThatThrownBy(
              () ->
                  graph.readPrefix(
                      input(Arrays.copyOf(bytes, bytes.length - 1)), 2, bytes.length, 8))
          .isInstanceOf(EOFException.class);
    }
  }

  @Test
  void immutableHybridExportDoesNotWaitForTheDonorsGenerationLock(@TempDir Path dir)
      throws Exception {
    var backend = PureJavaBackend.load(Qwen35ForwardPassTest.writeToyModel(dir));
    try (var pipeline = new InferencePipeline(backend);
        var workers = Executors.newSingleThreadExecutor()) {
      var prefix = pipeline.prepareSharedTokenPrefix(new int[] {2, 3});
      synchronized (backend) {
        var export =
            workers.submit(
                () -> {
                  var bytes = new ByteArrayOutputStream();
                  pipeline.writeSharedPromptPrefix(prefix, 2, bytes);
                  return bytes.size();
                });
        assertThat(export.get(2, TimeUnit.SECONDS)).isEqualTo(pipeline.portablePrefixBytes(2));
      }
    }
  }

  private static int[] tokens(int count) {
    int[] tokens = new int[count];
    for (int index = 0; index < count; index++) tokens[index] = 2 + index % 6;
    return tokens;
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int index = 1; index < values.length; index++)
      if (values[index] > values[best]) best = index;
    return best;
  }

  private static byte[] encode(Qwen35ForwardPass.SessionPrefix prefix) throws IOException {
    var bytes = new ByteArrayOutputStream();
    prefix.writeTo(new DataOutputStream(bytes), prefix.checkpoint());
    return bytes.toByteArray();
  }

  private static DataInputStream input(byte[] bytes) {
    return new DataInputStream(new ByteArrayInputStream(bytes));
  }

  private static void resign(byte[] bytes) throws Exception {
    var digest = MessageDigest.getInstance("SHA-256");
    digest.update(bytes, 0, bytes.length - 32);
    System.arraycopy(digest.digest(), 0, bytes, bytes.length - 32, 32);
  }
}
