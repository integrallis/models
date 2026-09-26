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

import com.integrallis.models.backend.purejava.gguf.GgufHeaderParser;
import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import com.integrallis.models.backend.purejava.gguf.GgufMetadataValue;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Gate G0 identity of a GGUF tokenizer: a sha256 per tokenizer metadata field and one over all of
 * them. Vocabulary, merges, token types, scores, special-token ids and the derived special-token
 * list are compared; the chat template is hashed separately and reported, not gated.
 *
 * @param fields per-field sha256 ("absent" when a key is missing)
 * @param tokenizerSha256 sha256 over the sorted field hashes
 * @param chatTemplateSha256 sha256 of {@code tokenizer.chat_template}, or null
 * @param architecture {@code general.architecture}
 * @param fileType {@code general.file_type}, or -1
 * @param vocabularySize number of tokens
 */
public record TokenizerIdentity(
    Map<String, String> fields,
    String tokenizerSha256,
    String chatTemplateSha256,
    String architecture,
    int fileType,
    int vocabularySize) {

  static final List<String> COMPARED_KEYS =
      List.of(
          "tokenizer.ggml.model",
          "tokenizer.ggml.pre",
          "tokenizer.ggml.tokens",
          "tokenizer.ggml.token_type",
          "tokenizer.ggml.merges",
          "tokenizer.ggml.scores",
          "tokenizer.ggml.bos_token_id",
          "tokenizer.ggml.eos_token_id",
          "tokenizer.ggml.eot_token_id",
          "tokenizer.ggml.eom_token_id",
          "tokenizer.ggml.unknown_token_id",
          "tokenizer.ggml.padding_token_id",
          "tokenizer.ggml.seperator_token_id",
          "tokenizer.ggml.add_bos_token",
          "tokenizer.ggml.add_eos_token",
          "tokenizer.ggml.add_space_prefix");
  static final String SPECIAL_TOKENS = "derived.special_tokens";

  public TokenizerIdentity {
    fields = Map.copyOf(fields);
  }

  static TokenizerIdentity read(Path gguf) {
    try (Arena arena = Arena.ofConfined();
        FileChannel channel = FileChannel.open(gguf, StandardOpenOption.READ)) {
      MemorySegment segment = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size(), arena);
      return of(GgufHeaderParser.parse(segment).metadata());
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  static TokenizerIdentity of(GgufMetadata metadata) {
    Map<String, String> fields = new LinkedHashMap<>();
    for (String key : COMPARED_KEYS) {
      fields.put(key, metadata.get(key).map(TokenizerIdentity::hash).orElse("absent"));
    }
    List<String> tokens = metadata.getStringArray("tokenizer.ggml.tokens").orElse(List.of());
    Optional<List<Integer>> types = metadata.getInt32Array("tokenizer.ggml.token_type");
    StringBuilder special = new StringBuilder();
    if (types.isPresent()) {
      for (int id = 0; id < Math.min(tokens.size(), types.get().size()); id++) {
        int type = types.get().get(id);
        if (type == 3 || type == 4) {
          special.append(id).append('\t').append(tokens.get(id)).append('\n');
        }
      }
    }
    fields.put(SPECIAL_TOKENS, Digests.sha256(special.toString()));
    StringBuilder all = new StringBuilder();
    fields.entrySet().stream()
        .sorted(Map.Entry.comparingByKey())
        .forEach(e -> all.append(e.getKey()).append('=').append(e.getValue()).append('\n'));
    return new TokenizerIdentity(
        fields,
        Digests.sha256(all.toString()),
        metadata.getString("tokenizer.chat_template").map(Digests::sha256).orElse(null),
        metadata.getString("general.architecture").orElse("unknown"),
        metadata.getUint32("general.file_type").orElse(-1),
        tokens.size());
  }

  /** Field names whose hashes differ between two identities. */
  List<String> differingFields(TokenizerIdentity other) {
    return fields.keySet().stream()
        .sorted()
        .filter(key -> !fields.get(key).equals(other.fields.get(key)))
        .toList();
  }

  static String hash(GgufMetadataValue value) {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    try (DataOutputStream out = new DataOutputStream(bytes)) {
      write(out, value);
    } catch (IOException impossible) {
      throw new UncheckedIOException(impossible);
    }
    return Digests.sha256(bytes.toByteArray());
  }

  private static void write(DataOutputStream out, GgufMetadataValue value) throws IOException {
    switch (value) {
      case GgufMetadataValue.StringValue s -> {
        byte[] utf8 = s.value().getBytes(StandardCharsets.UTF_8);
        out.writeByte('s');
        out.writeInt(utf8.length);
        out.write(utf8);
      }
      case GgufMetadataValue.ArrayValue a -> {
        out.writeByte('a');
        out.writeUTF(a.elementType().name());
        out.writeInt(a.elements().size());
        for (GgufMetadataValue element : a.elements()) {
          write(out, element);
        }
      }
      case GgufMetadataValue.Uint8Value v -> out.writeLong(v.value());
      case GgufMetadataValue.Int8Value v -> out.writeLong(v.value());
      case GgufMetadataValue.Uint16Value v -> out.writeLong(v.value());
      case GgufMetadataValue.Int16Value v -> out.writeLong(v.value());
      case GgufMetadataValue.Uint32Value v -> out.writeLong(Integer.toUnsignedLong(v.value()));
      case GgufMetadataValue.Int32Value v -> out.writeLong(v.value());
      case GgufMetadataValue.Uint64Value v -> out.writeLong(v.value());
      case GgufMetadataValue.Int64Value v -> out.writeLong(v.value());
      case GgufMetadataValue.Float32Value v -> out.writeDouble(v.value());
      case GgufMetadataValue.Float64Value v -> out.writeDouble(v.value());
      case GgufMetadataValue.BoolValue v -> out.writeBoolean(v.value());
    }
  }
}
