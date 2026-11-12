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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * #428: Block-range assembly of portable prefixes.
 *
 * <p>A complete prefix may be transferred as multiple independently verified blocks. The receiver
 * assembles them atomically after validating identity, coverage and non-overlap.
 */
@Tag("unit")
class PortablePrefixBlockAssemblyTest {

  @Test
  void assemblyOfTwoBlocksMatchesWholePrefixExport(@TempDir Path dir) throws Exception {
    var file = PureJavaBackendTest.buildNanoModelFile(dir, new Random(42));
    try (var donor = PureJavaBackend.load(file);
        var receiver = PureJavaBackend.load(file)) {
      var session = donor.openSession();
      donor.prefill(session, new int[] {2, 3, 4, 5, 6}, 0);
      var prefix = donor.freezePrefix(session);
      assertThat(prefix.checkpoint()).isEqualTo(5);

      // Export the whole prefix as a baseline.
      var whole = new ByteArrayOutputStream();
      donor.writePrefix(prefix, 5, new DataOutputStream(whole));
      var importedWhole =
          receiver.readPrefix(
              new DataInputStream(new ByteArrayInputStream(whole.toByteArray())), 5, whole.size());

      // Export the same prefix in two blocks: [0,2) and [2,5).
      var block1 = new ByteArrayOutputStream();
      donor.writePrefixBlock(prefix, 0, 2, new DataOutputStream(block1));
      var block2 = new ByteArrayOutputStream();
      donor.writePrefixBlock(prefix, 2, 5, new DataOutputStream(block2));

      // Assemble the two blocks on the receiver.
      var assembled = new ByteArrayOutputStream();
      assembled.write(block1.toByteArray());
      assembled.write(block2.toByteArray());
      var importedBlocks =
          receiver.readPrefixBlocks(
              new DataInputStream(new ByteArrayInputStream(assembled.toByteArray())),
              5,
              assembled.size());

      // Both imports must yield the same checkpoint and continuation logits.
      assertThat(importedBlocks.checkpoint()).isEqualTo(importedWhole.checkpoint()).isEqualTo(5);
      try (var forkWhole = receiver.fork(importedWhole);
          var forkBlocks = receiver.fork(importedBlocks)) {
        float[] wholeLogits = receiver.forward(forkWhole, 9, 5);
        float[] blockLogits = receiver.forward(forkBlocks, 9, 5);
        assertThat(blockLogits).containsExactly(wholeLogits);
      }
    }
  }

  @Test
  void assemblyRejectsOverlappingBlockRanges(@TempDir Path dir) throws Exception {
    var file = PureJavaBackendTest.buildNanoModelFile(dir, new Random(42));
    try (var donor = PureJavaBackend.load(file);
        var receiver = PureJavaBackend.load(file)) {
      var session = donor.openSession();
      donor.prefill(session, new int[] {2, 3, 4, 5, 6}, 0);
      var prefix = donor.freezePrefix(session);

      // Two overlapping blocks: [0,3) and [2,5).
      var block1 = new ByteArrayOutputStream();
      donor.writePrefixBlock(prefix, 0, 3, new DataOutputStream(block1));
      var block2 = new ByteArrayOutputStream();
      donor.writePrefixBlock(prefix, 2, 5, new DataOutputStream(block2));

      var assembled = new ByteArrayOutputStream();
      assembled.write(block1.toByteArray());
      assembled.write(block2.toByteArray());

      assertThatThrownBy(
              () ->
                  receiver.readPrefixBlocks(
                      new DataInputStream(new ByteArrayInputStream(assembled.toByteArray())),
                      5,
                      assembled.size()))
          .isInstanceOf(java.io.IOException.class)
          .hasMessageContaining("overlap");
    }
  }

  @Test
  void assemblyRejectsMissingRangeAtHead(@TempDir Path dir) throws Exception {
    var file = PureJavaBackendTest.buildNanoModelFile(dir, new Random(42));
    try (var donor = PureJavaBackend.load(file);
        var receiver = PureJavaBackend.load(file)) {
      var session = donor.openSession();
      donor.prefill(session, new int[] {2, 3, 4, 5, 6}, 0);
      var prefix = donor.freezePrefix(session);

      // Only export [2,5), omitting [0,2).
      var block = new ByteArrayOutputStream();
      donor.writePrefixBlock(prefix, 2, 5, new DataOutputStream(block));

      assertThatThrownBy(
              () ->
                  receiver.readPrefixBlocks(
                      new DataInputStream(new ByteArrayInputStream(block.toByteArray())),
                      5,
                      block.size()))
          .isInstanceOf(java.io.IOException.class)
          .hasMessageContaining("missing");
    }
  }

  @Test
  void assemblyAcceptsReorderedBlocks(@TempDir Path dir) throws Exception {
    var file = PureJavaBackendTest.buildNanoModelFile(dir, new Random(42));
    try (var donor = PureJavaBackend.load(file);
        var receiver = PureJavaBackend.load(file)) {
      var session = donor.openSession();
      donor.prefill(session, new int[] {2, 3, 4, 5, 6}, 0);
      var prefix = donor.freezePrefix(session);

      // Export blocks out of order: [2,5) then [0,2).
      var blockTail = new ByteArrayOutputStream();
      donor.writePrefixBlock(prefix, 2, 5, new DataOutputStream(blockTail));
      var blockHead = new ByteArrayOutputStream();
      donor.writePrefixBlock(prefix, 0, 2, new DataOutputStream(blockHead));

      var assembled = new ByteArrayOutputStream();
      assembled.write(blockTail.toByteArray());
      assembled.write(blockHead.toByteArray());

      // Receiver must assemble correctly regardless of order.
      var imported =
          receiver.readPrefixBlocks(
              new DataInputStream(new ByteArrayInputStream(assembled.toByteArray())),
              5,
              assembled.size());
      assertThat(imported.checkpoint()).isEqualTo(5);
      try (var fork = receiver.fork(imported)) {
        assertThat(receiver.forward(fork, 9, 5)).isNotEmpty();
      }
    }
  }

  @Test
  void assemblyRejectsDuplicateBlocks(@TempDir Path dir) throws Exception {
    var file = PureJavaBackendTest.buildNanoModelFile(dir, new Random(42));
    try (var donor = PureJavaBackend.load(file);
        var receiver = PureJavaBackend.load(file)) {
      var session = donor.openSession();
      donor.prefill(session, new int[] {2, 3, 4, 5, 6}, 0);
      var prefix = donor.freezePrefix(session);

      // Duplicate the same block twice.
      var block = new ByteArrayOutputStream();
      donor.writePrefixBlock(prefix, 0, 3, new DataOutputStream(block));

      var assembled = new ByteArrayOutputStream();
      assembled.write(block.toByteArray());
      assembled.write(block.toByteArray());

      assertThatThrownBy(
              () ->
                  receiver.readPrefixBlocks(
                      new DataInputStream(new ByteArrayInputStream(assembled.toByteArray())),
                      5,
                      assembled.size()))
          .isInstanceOf(java.io.IOException.class)
          .hasMessageContaining("overlap");
    }
  }

  @Test
  void assemblyRejectsWrongModelIdentity(@TempDir Path dir) throws Exception {
    var file = PureJavaBackendTest.buildNanoModelFile(dir, new Random(42));
    var otherFile =
        PureJavaBackendTest.buildNanoModelFile(
            Files.createDirectory(dir.resolve("other")), new Random(43));
    try (var donor = PureJavaBackend.load(file);
        var receiver = PureJavaBackend.load(otherFile)) {
      var session = donor.openSession();
      donor.prefill(session, new int[] {2, 3, 4, 5, 6}, 0);
      var prefix = donor.freezePrefix(session);

      var block = new ByteArrayOutputStream();
      donor.writePrefixBlock(prefix, 0, 5, new DataOutputStream(block));

      assertThatThrownBy(
              () ->
                  receiver.readPrefixBlocks(
                      new DataInputStream(new ByteArrayInputStream(block.toByteArray())),
                      5,
                      block.size()))
          .isInstanceOf(java.io.IOException.class)
          .hasMessageContaining("identity");
    }
  }

  @Test
  void assemblyRejectsBudgetTooSmall(@TempDir Path dir) throws Exception {
    var file = PureJavaBackendTest.buildNanoModelFile(dir, new Random(42));
    try (var donor = PureJavaBackend.load(file);
        var receiver = PureJavaBackend.load(file)) {
      var session = donor.openSession();
      donor.prefill(session, new int[] {2, 3, 4, 5, 6}, 0);
      var prefix = donor.freezePrefix(session);

      var block = new ByteArrayOutputStream();
      donor.writePrefixBlock(prefix, 0, 5, new DataOutputStream(block));

      assertThatThrownBy(
              () ->
                  receiver.readPrefixBlocks(
                      new DataInputStream(new ByteArrayInputStream(block.toByteArray())),
                      5,
                      10)) // way below the minimum 44
          .isInstanceOf(java.io.IOException.class)
          .hasMessageContaining("budget");
    }
  }

  @Test
  void assemblyRejectsNegativeBlockStart(@TempDir Path dir) throws Exception {
    var file = PureJavaBackendTest.buildNanoModelFile(dir, new Random(42));
    try (var donor = PureJavaBackend.load(file);
        var receiver = PureJavaBackend.load(file)) {
      var session = donor.openSession();
      donor.prefill(session, new int[] {2, 3, 4, 5, 6}, 0);
      var prefix = donor.freezePrefix(session);

      // Manually craft a block unit with negative start after a valid identity.
      var crafted = new ByteArrayOutputStream();
      var craftedOut = new DataOutputStream(crafted);
      craftedOut.write(java.util.HexFormat.of().parseHex(donor.portablePrefixFingerprint()));
      craftedOut.writeInt(1); // block count
      craftedOut.writeInt(-1); // negative start
      craftedOut.writeInt(3); // end

      assertThatThrownBy(
              () ->
                  receiver.readPrefixBlocks(
                      new DataInputStream(new ByteArrayInputStream(crafted.toByteArray())),
                      5,
                      crafted.size()))
          .isInstanceOf(java.io.IOException.class)
          .hasMessageContaining("invalid");
    }
  }

  @Test
  void assemblyRejectsEndBeyondTokenCount(@TempDir Path dir) throws Exception {
    var file = PureJavaBackendTest.buildNanoModelFile(dir, new Random(42));
    try (var donor = PureJavaBackend.load(file);
        var receiver = PureJavaBackend.load(file)) {
      var session = donor.openSession();
      donor.prefill(session, new int[] {2, 3, 4, 5, 6}, 0);
      var prefix = donor.freezePrefix(session);

      var crafted = new ByteArrayOutputStream();
      var craftedOut = new DataOutputStream(crafted);
      craftedOut.write(java.util.HexFormat.of().parseHex(donor.portablePrefixFingerprint()));
      craftedOut.writeInt(1); // block count
      craftedOut.writeInt(0); // start
      craftedOut.writeInt(10); // end exceeds tokenCount (5)

      assertThatThrownBy(
              () ->
                  receiver.readPrefixBlocks(
                      new DataInputStream(new ByteArrayInputStream(crafted.toByteArray())),
                      5,
                      crafted.size()))
          .isInstanceOf(java.io.IOException.class)
          .hasMessageContaining("invalid");
    }
  }

  @Test
  void blockExportRejectsInvalidRange(@TempDir Path dir) throws Exception {
    var file = PureJavaBackendTest.buildNanoModelFile(dir, new Random(42));
    try (var donor = PureJavaBackend.load(file)) {
      var session = donor.openSession();
      donor.prefill(session, new int[] {2, 3, 4, 5, 6}, 0);
      var prefix = donor.freezePrefix(session);

      assertThatThrownBy(
              () ->
                  donor.writePrefixBlock(
                      prefix, -1, 3, new DataOutputStream(new ByteArrayOutputStream())))
          .isInstanceOf(IllegalArgumentException.class);

      assertThatThrownBy(
              () ->
                  donor.writePrefixBlock(
                      prefix, 3, 2, new DataOutputStream(new ByteArrayOutputStream())))
          .isInstanceOf(IllegalArgumentException.class);

      assertThatThrownBy(
              () ->
                  donor.writePrefixBlock(
                      prefix, 0, 10, new DataOutputStream(new ByteArrayOutputStream())))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }
}
