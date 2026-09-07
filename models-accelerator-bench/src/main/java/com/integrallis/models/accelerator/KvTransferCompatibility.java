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
package com.integrallis.models.accelerator;

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/** Screens two GGUF models before attempting calibrated cross-model KV transfer. */
public final class KvTransferCompatibility {

  /** KV and tokenizer properties needed by the first aligned-token calibration experiment. */
  public record Layout(
      String architecture,
      int layers,
      int embeddingLength,
      int queryHeads,
      int kvHeads,
      int keyLength,
      int valueLength,
      float ropeFrequencyBase,
      int vocabularySize,
      String vocabularyFingerprint) {

    public Layout {
      Objects.requireNonNull(architecture, "architecture");
      Objects.requireNonNull(vocabularyFingerprint, "vocabularyFingerprint");
    }
  }

  /** Screening result. Direct reuse remains false even when tensor shapes match. */
  public record Result(
      Layout source,
      Layout target,
      boolean calibrationCandidate,
      boolean directReuseSafe,
      List<String> reasons) {

    public Result {
      Objects.requireNonNull(source, "source");
      Objects.requireNonNull(target, "target");
      reasons = List.copyOf(reasons);
      if (directReuseSafe) {
        throw new IllegalArgumentException("screening cannot prove raw KV reuse safe");
      }
    }
  }

  private KvTransferCompatibility() {}

  public static void main(String[] args) throws Exception {
    if (args.length != 2) {
      throw new IllegalArgumentException("usage: KvTransferCompatibility SOURCE.gguf TARGET.gguf");
    }
    try (Arena sourceArena = Arena.ofConfined();
        Arena targetArena = Arena.ofConfined()) {
      Result result =
          screen(
              GgufParser.parse(Path.of(args[0]), sourceArena).metadata(),
              GgufParser.parse(Path.of(args[1]), targetArena).metadata());
      System.out.printf(
          Locale.ROOT,
          "source=%s%n target=%s%n calibrationCandidate=%s%n directReuseSafe=%s%n",
          result.source(),
          result.target(),
          result.calibrationCandidate(),
          result.directReuseSafe());
      result.reasons().forEach(reason -> System.out.println(" - " + reason));
    }
  }

  /**
   * Applies conservative prerequisites for a per-head, aligned-token calibration experiment.
   * Matching is a reason to measure, never evidence that one model can consume another's raw KV.
   */
  public static Result screen(GgufMetadata sourceMetadata, GgufMetadata targetMetadata) {
    Layout source = layout(sourceMetadata);
    Layout target = layout(targetMetadata);
    List<String> reasons = new ArrayList<>();
    boolean candidate = true;

    candidate &= match("architecture", source.architecture(), target.architecture(), reasons);
    candidate &= match("layer count", source.layers(), target.layers(), reasons);
    candidate &= match("query head count", source.queryHeads(), target.queryHeads(), reasons);
    candidate &= match("KV head count", source.kvHeads(), target.kvHeads(), reasons);
    candidate &= match("key length", source.keyLength(), target.keyLength(), reasons);
    candidate &= match("value length", source.valueLength(), target.valueLength(), reasons);
    candidate &=
        match(
            "RoPE frequency base", source.ropeFrequencyBase(), target.ropeFrequencyBase(), reasons);

    if (!source.vocabularyFingerprint().equals(target.vocabularyFingerprint())) {
      reasons.add("tokenizer vocabulary differs; aligned-token calibration is not valid");
      candidate = false;
    }

    if (candidate) {
      reasons.add(
          "matched "
              + source.layers()
              + "-layer, "
              + source.kvHeads()
              + "-head, "
              + source.keyLength()
              + "x"
              + source.valueLength()
              + " KV geometry");
      reasons.add("tokenizer vocabulary is identical");
      reasons.add(
          "raw KV values remain model-specific; calibration and quality gates are required");
    }
    return new Result(source, target, candidate, false, reasons);
  }

  private static Layout layout(GgufMetadata metadata) {
    String architecture =
        metadata
            .getString("general.architecture")
            .orElseThrow(() -> new IllegalArgumentException("missing general.architecture"));
    String prefix = architecture + ".";
    int embeddingLength = requiredInt(metadata, prefix + "embedding_length");
    int queryHeads = requiredInt(metadata, prefix + "attention.head_count");
    List<String> vocabulary =
        metadata
            .getStringArray("tokenizer.ggml.tokens")
            .orElseThrow(
                () -> new IllegalArgumentException("missing tokenizer.ggml.tokens string array"));
    return new Layout(
        architecture,
        requiredInt(metadata, prefix + "block_count"),
        embeddingLength,
        queryHeads,
        requiredInt(metadata, prefix + "attention.head_count_kv"),
        metadata.getUint32(prefix + "attention.key_length").orElse(embeddingLength / queryHeads),
        metadata.getUint32(prefix + "attention.value_length").orElse(embeddingLength / queryHeads),
        metadata.getFloat32(prefix + "rope.freq_base").orElse(10_000.0f),
        vocabulary.size(),
        vocabularyFingerprint(vocabulary));
  }

  private static int requiredInt(GgufMetadata metadata, String key) {
    return metadata
        .getUint32(key)
        .or(() -> metadata.getInt32(key))
        .orElseThrow(() -> new IllegalArgumentException("missing " + key));
  }

  private static boolean match(String label, Object source, Object target, List<String> reasons) {
    if (source.equals(target)) {
      return true;
    }
    reasons.add(label + " differs: " + source + " != " + target);
    return false;
  }

  private static String vocabularyFingerprint(List<String> vocabulary) {
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      for (String token : vocabulary) {
        byte[] bytes = token.getBytes(StandardCharsets.UTF_8);
        digest.update(ByteBuffer.allocate(Integer.BYTES).putInt(bytes.length).array());
        digest.update(bytes);
      }
      return HexFormat.of().formatHex(digest.digest());
    } catch (NoSuchAlgorithmException exception) {
      throw new IllegalStateException("SHA-256 is required by the Java platform", exception);
    }
  }
}
