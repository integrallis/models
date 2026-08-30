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
package com.integrallis.models.backend.purejava.tokenizer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.gptoss.GptOssHuggingFaceConfig;
import com.integrallis.models.backend.purejava.huggingface.Qwen2HuggingFaceConfig;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class HuggingFaceTokenizerTest {

  @Test
  void buildsGptOssHarmonyTokenizerWithItsExactPretokenizer(@TempDir Path directory)
      throws IOException {
    String pattern =
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
            + "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
            + "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
            + "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
            + "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|"
            + "\\s+(?!\\S)|\\s+";
    Path tokenizer = directory.resolve("tokenizer.json");
    Files.writeString(
        tokenizer,
        """
        {
          "version":"1.0",
          "normalizer":null,
          "pre_tokenizer":{
            "type":"Sequence",
            "pretokenizers":[
              {
                "type":"Split",
                "pattern":{"Regex":"%s"},
                "behavior":"Isolated",
                "invert":false
              },
              {
                "type":"ByteLevel",
                "add_prefix_space":false,
                "trim_offsets":true,
                "use_regex":false
              }
            ]
          },
          "decoder":{"type":"ByteLevel"},
          "model":{
            "type":"BPE",
            "unk_token":null,
            "vocab":{"a":0,"b":1,"ab":2,"Ġ":3,"Ġa":4,"Ġab":5},
            "merges":["Ġ a","Ġa b","a b"]
          },
          "added_tokens":[
            {"id":6,"content":"<|startoftext|>","special":true},
            {"id":7,"content":"<|return|>","special":true},
            {"id":8,"content":"<|end|>","special":true},
            {"id":9,"content":"<|call|>","special":true}
          ]
        }
        """
            .formatted(pattern.replace("\\", "\\\\")));
    Path tokenizerConfig = directory.resolve("tokenizer_config.json");
    Files.writeString(
        tokenizerConfig,
        """
        {"bos_token":"<|startoftext|>","eos_token":"<|return|>"}
        """);

    Tokenizer parsed =
        HuggingFaceTokenizer.fromGptOss(tokenizer, tokenizerConfig, tinyGptOssConfig());

    assertThat(parsed.encode("ab")).containsExactly(2);
    assertThat(parsed.encode(" ab")).containsExactly(5);
    assertThat(parsed.encodeControl("<|startoftext|>ab<|end|>")).containsExactly(6, 2, 8);
    assertThat(parsed.encode("<|end|>")).doesNotContain(8);
    assertThat(parsed.isEndOfGeneration(7)).isTrue();
    assertThat(parsed.isEndOfGeneration(9)).isTrue();
    assertThat(parsed.isEndOfGeneration(8)).isFalse();
  }

  @Test
  void buildsQwen2ByteLevelBpeWithoutGivingOrdinaryTextControlPrivileges(@TempDir Path directory)
      throws IOException {
    Path config = writeSyntheticConfig(directory);
    Path tokenizer = directory.resolve("tokenizer.json");
    String qwenPattern =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|"
            + " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    Files.writeString(
        tokenizer,
        """
        {
          "version":"1.0",
          "normalizer":{"type":"NFC"},
          "pre_tokenizer":{
            "type":"Sequence",
            "pretokenizers":[
              {
                "type":"Split",
                "pattern":{"Regex":"%s"},
                "behavior":"Isolated",
                "invert":false
              },
              {
                "type":"ByteLevel",
                "add_prefix_space":false,
                "trim_offsets":false,
                "use_regex":false
              }
            ]
          },
          "decoder":{
            "type":"ByteLevel",
            "add_prefix_space":false,
            "trim_offsets":false,
            "use_regex":false
          },
          "model":{
            "type":"BPE",
            "unk_token":null,
            "vocab":{"a":0,"b":1,"ab":2,"Ġ":3,"Ġa":4,"Ġab":5},
            "merges":["Ġ a","Ġa b","a b"]
          },
          "added_tokens":[
            {"id":6,"content":"<|im_end|>","special":true},
            {"id":7,"content":"<|im_start|>","special":true}
          ]
        }
        """
            .formatted(qwenPattern.replace("\\", "\\\\")));
    Path tokenizerConfig = directory.resolve("tokenizer_config.json");
    Files.writeString(
        tokenizerConfig,
        """
        {
          "bos_token":null,
          "eos_token":"<|im_end|>",
          "unk_token":null,
          "add_bos_token":false,
          "add_eos_token":false
        }
        """);

    Tokenizer parsed =
        HuggingFaceTokenizer.fromQwen2(
            tokenizer, tokenizerConfig, Qwen2HuggingFaceConfig.parse(config));

    assertThat(parsed.vocabSize()).isEqualTo(8);
    assertThat(parsed.bosToken()).isEqualTo(6);
    assertThat(parsed.eosToken()).isEqualTo(6);
    assertThat(parsed.encode("ab")).containsExactly(2);
    assertThat(parsed.encode(" ab")).containsExactly(5);
    assertThat(parsed.encodeControl("<|im_start|>ab<|im_end|>")).containsExactly(7, 2, 6);
    assertThat(parsed.encode("<|im_start|>")).doesNotContain(7);
    assertThat(parsed.decode(new int[] {2})).isEqualTo("ab");
  }

  @Test
  void matchesPinnedOfficialQwen25Tokenizer() throws Exception {
    String configured = System.getProperty("models.fixtures.qwen25HuggingFaceDirectory");
    assumeTrue(configured != null, "set models.fixtures.qwen25HuggingFaceDirectory");
    Path directory = Path.of(configured).toAbsolutePath().normalize();
    Path config = directory.resolve("config.json");
    Path tokenizerJson = directory.resolve("tokenizer.json");
    Path tokenizerConfig = directory.resolve("tokenizer_config.json");
    assumeTrue(
        Files.isRegularFile(config)
            && Files.isRegularFile(tokenizerJson)
            && Files.isRegularFile(tokenizerConfig),
        "Qwen 2.5 tokenizer fixture is incomplete");

    assertThat(sha256(config))
        .isEqualTo("18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45");
    assertThat(sha256(tokenizerJson))
        .isEqualTo("c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539");
    assertThat(sha256(tokenizerConfig))
        .isEqualTo("5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583");

    Tokenizer parsed =
        HuggingFaceTokenizer.fromQwen2(
            tokenizerJson, tokenizerConfig, Qwen2HuggingFaceConfig.parse(config));

    assertThat(parsed.vocabSize()).isEqualTo(151936);
    assertThat(parsed.encode("Hello, world!")).containsExactly(9707, 11, 1879, 0);
    assertThat(parsed.encode("Name one JVM language.")).containsExactly(675, 825, 72379, 4128, 13);
    assertThat(parsed.encode("Phoenix → 東京 café"))
        .containsExactly(78825, 11397, 60596, 109, 46553, 51950);
    assertThat(parsed.encode("cafe\u0301")).containsExactly(924, 58858);
    assertThat(parsed.encode("Line 1\nLine 22\n"))
        .containsExactly(2460, 220, 16, 198, 2460, 220, 17, 17, 198);
    assertThat(parsed.decode(new int[] {9707, 11, 1879, 0})).isEqualTo("Hello, world!");
    assertThat(
            parsed.encodeControl(
                "<|im_start|>user\nName one JVM language.<|im_end|>\n" + "<|im_start|>assistant\n"))
        .containsExactly(
            151644, 872, 198, 675, 825, 72379, 4128, 13, 151645, 198, 151644, 77091, 198);
    assertThat(parsed.encodeControl("<tool_call>")).containsExactly(151657);
    assertThat(parsed.encode("<tool_call>")).doesNotContain(151657);
    assertThat(parsed.tokenId("<tool_call>")).isEqualTo(151657);
    assertThat(parsed.isEndOfGeneration(151643)).isTrue();
    assertThat(parsed.isEndOfGeneration(151645)).isTrue();
  }

  @Test
  void matchesPinnedOfficialGptOss20BTokenizer() throws Exception {
    String configured = System.getProperty("models.fixtures.gptOssHuggingFaceDirectory");
    assumeTrue(configured != null, "set models.fixtures.gptOssHuggingFaceDirectory");
    Path directory = Path.of(configured).toAbsolutePath().normalize();
    Path tokenizerJson = directory.resolve("tokenizer.json");
    Path tokenizerConfig = directory.resolve("tokenizer_config.json");
    assumeTrue(
        Files.isRegularFile(tokenizerJson) && Files.isRegularFile(tokenizerConfig),
        "GPT-OSS tokenizer fixture is incomplete");

    assertThat(sha256(tokenizerJson))
        .isEqualTo("0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3");
    assertThat(sha256(tokenizerConfig))
        .isEqualTo("9279e942392b742d633c7adbb89ebe002c98399db8926a7af5125c726f404070");
    Tokenizer parsed =
        HuggingFaceTokenizer.fromGptOss(tokenizerJson, tokenizerConfig, officialGptOssConfig());

    assertThat(parsed.vocabSize()).isEqualTo(201_088);
    assertThat(parsed.encode("Hello, world!")).containsExactly(13225, 11, 2375, 0);
    assertThat(parsed.encode("Name one JVM language."))
        .containsExactly(864, 1001, 162108, 6439, 13);
    assertThat(parsed.encode("Phoenix → 東京 café")).containsExactly(160441, 15155, 185244, 30469);
    assertThat(parsed.encode("cafe\u0301")).containsExactly(66, 6903, 13430);
    assertThat(parsed.encode("Line 1\nLine 2222\n"))
        .containsExactly(3665, 220, 16, 198, 3665, 220, 16427, 17, 198);
    assertThat(
            parsed.encodeControl(
                "<|start|>user<|message|>Name one JVM language.<|end|>"
                    + "<|start|>assistant<|channel|>final<|message|>"))
        .containsExactly(
            200006, 1428, 200008, 864, 1001, 162108, 6439, 13, 200007, 200006, 173781, 200005,
            17196, 200008);
    assertThat(parsed.isEndOfGeneration(200002)).isTrue();
    assertThat(parsed.isEndOfGeneration(200012)).isTrue();
    assertThat(parsed.isEndOfGeneration(200007)).isFalse();
    assertThat(
            parsed.encode(
                ChatTemplate.GPT_OSS.render(List.of(ChatMessage.user("Name one JVM language.")))))
        .containsExactly(
            200006, 17360, 200008, 3575, 553, 17554, 162016, 11, 261, 4410, 6439, 2359, 22203, 656,
            7788, 17527, 558, 87447, 100594, 25, 220, 1323, 19, 12, 3218, 279, 30377, 289, 25,
            14093, 279, 2, 13888, 18403, 25, 8450, 11, 49159, 11, 1721, 13, 21030, 2804, 413, 7360,
            395, 1753, 3176, 13, 200007, 200006, 1428, 200008, 864, 1001, 162108, 6439, 13, 200007,
            200006, 173781);
  }

  private static GptOssHuggingFaceConfig tinyGptOssConfig() {
    return new GptOssHuggingFaceConfig(
        List.of("GptOssForCausalLM"),
        32,
        1,
        1,
        1,
        32,
        10,
        8,
        4,
        32,
        1,
        1,
        0,
        1.0e-5f,
        10_000.0f,
        2.0f,
        32.0f,
        1.0f,
        4,
        7.0f,
        1.702f,
        "silu",
        "mxfp4",
        true,
        false,
        7,
        0,
        List.of("full_attention"));
  }

  private static GptOssHuggingFaceConfig officialGptOssConfig() throws Exception {
    Path config =
        Path.of(
            HuggingFaceTokenizerTest.class
                .getResource("/huggingface/gpt-oss-20b-config.json")
                .toURI());
    return GptOssHuggingFaceConfig.parse(config);
  }

  private static Path writeSyntheticConfig(Path directory) throws IOException {
    Path config = directory.resolve("config.json");
    Files.writeString(
        config,
        """
        {
          "architectures":["Qwen2ForCausalLM"],
          "bos_token_id":6,
          "eos_token_id":6,
          "hidden_act":"silu",
          "hidden_size":4,
          "intermediate_size":8,
          "max_position_embeddings":32,
          "model_type":"qwen2",
          "num_attention_heads":1,
          "num_hidden_layers":1,
          "rms_norm_eps":0.00001,
          "vocab_size":8
        }
        """);
    return config;
  }

  private static String sha256(Path path) throws IOException, NoSuchAlgorithmException {
    return HexFormat.of()
        .formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path)));
  }
}
