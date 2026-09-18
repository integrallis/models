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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Renders study prompts from a versioned JSON file through the Qwen3 chat template.
 *
 * <p>Thinking on uses {@link ChatTemplate#CHATML} (generation starts at {@code
 * <|im_start|>assistant\n}); thinking off uses {@link ChatTemplate#CHATML_NO_THINK}, which appends
 * the empty think block that Qwen3's template emits for {@code enable_thinking=False}.
 */
final class PromptBuilder {
  static final String BUNDLED = "/com/integrallis/models/bench/fusion/prompts-v1.json";
  private static final ObjectMapper JSON = new ObjectMapper();

  private final JsonNode root;
  private final String sha256;
  private final String source;

  private PromptBuilder(byte[] bytes, String source) {
    try {
      this.root = JSON.readTree(bytes);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
    this.sha256 = Digests.sha256(bytes);
    this.source = source;
  }

  static PromptBuilder bundled() {
    try (InputStream input = PromptBuilder.class.getResourceAsStream(BUNDLED)) {
      Objects.requireNonNull(input, "missing " + BUNDLED);
      return new PromptBuilder(input.readAllBytes(), "resource:" + BUNDLED);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  static PromptBuilder load(Path path) throws IOException {
    return new PromptBuilder(Files.readAllBytes(path), path.toAbsolutePath().toString());
  }

  String sha256() {
    return sha256;
  }

  String source() {
    return source;
  }

  static ChatTemplate template(boolean thinking) {
    return thinking ? ChatTemplate.CHATML : ChatTemplate.CHATML_NO_THINK;
  }

  /** sha256 of the template id and its rendering of a fixed probe conversation. */
  String templateSha256(boolean thinking) {
    ChatTemplate template = template(thinking);
    return Digests.sha256(
        template.id() + "\n" + template.render(List.of(ChatMessage.user("{probe}"))).text());
  }

  ModelPrompt render(DatasetKind dataset, String scorer, DatasetItem item, boolean thinking) {
    String user =
        switch (dataset) {
          case GSM8K -> fill(root.path("gsm8k").path("user").asText(), item, "arc");
          case MATH500 -> fill(root.path("math500").path("user").asText(), item, "arc");
          case ARC -> fill(root.path("arc").path("user").asText(), item, "arc");
          case GENERIC -> fill(root.path("generic").path(scorer).asText(), item, "arc");
        };
    return template(thinking).render(messages(user));
  }

  /** Prompt used for option log-likelihood scoring; the assistant prefix is appended as text. */
  ModelPrompt renderLoglik(DatasetItem item) {
    String user = fill(root.path("arcLoglik").path("user").asText(), item, "arcLoglik");
    return template(false).render(messages(user));
  }

  String loglikAssistantPrefix() {
    return root.path("arcLoglik").path("assistantPrefix").asText();
  }

  String loglikContinuation(String label) {
    return root.path("arcLoglik").path("continuation").asText().replace("{label}", label);
  }

  private List<ChatMessage> messages(String user) {
    List<ChatMessage> messages = new ArrayList<>();
    JsonNode system = root.path("system");
    if (system.isTextual() && !system.asText().isEmpty()) {
      messages.add(ChatMessage.system(system.asText()));
    }
    messages.add(ChatMessage.user(user));
    return messages;
  }

  private String fill(String template, DatasetItem item, String choiceSection) {
    if (template.isEmpty()) {
      throw new IllegalArgumentException("prompt file has no template for this dataset/scorer");
    }
    String choiceFormat = root.path(choiceSection).path("choice").asText("{label}. {text}");
    StringBuilder choices = new StringBuilder();
    for (DatasetItem.Choice choice : item.choices()) {
      if (!choices.isEmpty()) {
        choices.append('\n');
      }
      choices.append(
          choiceFormat.replace("{label}", choice.label()).replace("{text}", choice.text()));
    }
    return template.replace("{choices}", choices.toString()).replace("{question}", item.question());
  }
}
