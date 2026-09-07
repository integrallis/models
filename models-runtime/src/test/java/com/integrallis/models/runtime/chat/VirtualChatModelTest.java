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
package com.integrallis.models.runtime.chat;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.BatchInferenceBackend;
import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.runtime.InferencePipeline;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;

class VirtualChatModelTest {

  private static final SamplingOptions OPTIONS =
      SamplingOptions.builder().temperature(0).maxTokens(256).build();
  private static final List<ToolSpec> TOOLS =
      List.of(
          new ToolSpec(
              "remember",
              "Store one fact",
              "{\"type\":\"object\",\"properties\":{\"fact\":{\"type\":\"string\"}}}"));

  @Test
  void presentsTwoPhysicalModelsAsOneSemanticConversation() {
    RecordingBackend chatBackend = new RecordingBackend("chat", "CHAT", "mem-2048");
    RecordingBackend toolBackend =
        new RecordingBackend(
            "tools",
            "<tool_call>\n{\"name\":\"remember\",\"arguments\":{\"fact\":\"aisle\"}}\n</tool_call>");
    AtomicBoolean toolConstraintCreated = new AtomicBoolean();

    try (InferencePipeline chatPipeline = new InferencePipeline(chatBackend);
        InferencePipeline toolPipeline = new InferencePipeline(toolBackend)) {
      VirtualChatModel model =
          VirtualChatModel.builder()
              .member(
                  "chat",
                  Set.of("chat"),
                  ChatTemplate.CHATML_NO_THINK,
                  chatPipeline::openGenerationSession)
              .member(
                  "tools",
                  Set.of("tool-use"),
                  ChatTemplate.CHATML_NO_THINK,
                  toolPipeline::openGenerationSession,
                  (session, turn) -> {
                    toolConstraintCreated.set(true);
                    return Optional.of(
                        com.integrallis.models.runtime.TokenConstraint.unrestricted());
                  })
              .build();

      try (VirtualChatModel.Session conversation = model.openSession()) {
        VirtualChatModel.Response first =
            conversation.generate("chat", ChatMessage.user("hello"), List.of(), OPTIONS);
        VirtualChatModel.Response second =
            conversation.generate("tool-use", ChatMessage.user("remember aisle"), TOOLS, OPTIONS);
        VirtualChatModel.Response third =
            conversation.generate(
                "chat",
                ChatMessage.tool("remember", "{\"stored\":true,\"memoryId\":\"mem-2048\"}"),
                TOOLS,
                OPTIONS);

        assertThat(first.memberId()).isEqualTo("chat");
        assertThat(first.boundary()).isEqualTo(VirtualChatModel.Boundary.COLD);
        assertThat(first.content()).isEqualTo("CHAT");
        assertThat(second.memberId()).isEqualTo("tools");
        assertThat(second.boundary()).isEqualTo(VirtualChatModel.Boundary.FIRST_SWITCH);
        assertThat(second.toolCalls())
            .singleElement()
            .satisfies(
                call -> {
                  assertThat(call.name()).isEqualTo("remember");
                  assertThat(call.argumentsJson()).isEqualTo("{\"fact\":\"aisle\"}");
                });
        assertThat(second.metrics().promptCache().cacheReadInputTokens()).isZero();
        assertThat(toolConstraintCreated).isTrue();
        assertThat(third.memberId()).isEqualTo("chat");
        assertThat(third.boundary()).isEqualTo(VirtualChatModel.Boundary.SWITCH_BACK);
        assertThat(third.content()).isEqualTo("mem-2048");
        assertThat(third.metrics().promptCache().cacheReadInputTokens()).isPositive();

        assertThat(toolBackend.prompts().getFirst()).contains("CHAT").contains("remember aisle");
        assertThat(chatBackend.prompts().get(1))
            .contains("<tool_call>")
            .contains("memoryId")
            .contains("mem-2048");
        assertThat(conversation.history())
            .extracting(ChatMessage::role)
            .containsExactly(
                ChatRole.USER,
                ChatRole.ASSISTANT,
                ChatRole.USER,
                ChatRole.ASSISTANT,
                ChatRole.TOOL,
                ChatRole.ASSISTANT);
      }
    }
  }

  @Test
  void backgroundPrefillWarmsOnlyTheInactiveMembersOwnPrefix() {
    RecordingBackend chatBackend = new RecordingBackend("chat", "CHAT", "chat-shadow");
    RecordingBackend toolBackend = new RecordingBackend("tools", "tool-shadow", "TOOL");

    try (InferencePipeline chatPipeline = new InferencePipeline(chatBackend);
        InferencePipeline toolPipeline = new InferencePipeline(toolBackend)) {
      VirtualChatModel model =
          VirtualChatModel.builder()
              .member(
                  "chat",
                  Set.of("chat"),
                  ChatTemplate.CHATML_NO_THINK,
                  chatPipeline::openGenerationSession)
              .member(
                  "tools",
                  Set.of("tool-use"),
                  ChatTemplate.CHATML_NO_THINK,
                  toolPipeline::openGenerationSession)
              .selector(
                  turn ->
                      new VirtualChatModel.Selection(
                          turn.taskType().equals("tool-use") ? "tools" : "chat",
                          "task=" + turn.taskType()))
              .backgroundPrefill(Runnable::run)
              .build();

      try (VirtualChatModel.Session conversation = model.openSession("prefill-test")) {
        conversation.generate("chat", ChatMessage.user("hello"), TOOLS, OPTIONS);

        assertThat(conversation.memberStates().get("tools").prefilled()).isTrue();
        assertThat(conversation.memberStates().get("tools").cachedPromptTokens()).isPositive();

        VirtualChatModel.Response toolTurn =
            conversation.generate("tool-use", ChatMessage.user("remember aisle"), TOOLS, OPTIONS);

        assertThat(toolTurn.memberId()).isEqualTo("tools");
        assertThat(toolTurn.metrics().promptCache().cacheReadInputTokens()).isPositive();
        assertThat(toolTurn.metrics().promptCache().cacheWriteInputTokens()).isPositive();
      }
    }
  }

  private static final class RecordingBackend implements BatchInferenceBackend {
    private static final int VOCABULARY_SIZE = 130;
    private final String name;
    private final ArrayDeque<String> outputs;
    private final List<String> prompts = new ArrayList<>();
    private final List<State> sessions = new ArrayList<>();

    private RecordingBackend(String name, String... outputs) {
      this.name = name;
      this.outputs = new ArrayDeque<>(List.of(outputs));
    }

    List<String> prompts() {
      return List.copyOf(prompts);
    }

    @Override
    public String name() {
      return name;
    }

    @Override
    public ModelMetadata metadata() {
      return new ModelMetadata("fixture", name, 4096, VOCABULARY_SIZE, 16, 1, 1, 1);
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable(name);
    }

    @Override
    public Tokenizer tokenizer() {
      return new Tokenizer() {
        @Override
        public int[] encode(String text) {
          return text.chars().map(value -> value + 1).toArray();
        }

        @Override
        public int[] encode(ModelPrompt prompt) {
          prompts.add(prompt.text());
          return encode(prompt.text());
        }

        @Override
        public String decode(int[] tokens) {
          StringBuilder decoded = new StringBuilder();
          for (int token : tokens) {
            decoded.append(decode(token));
          }
          return decoded.toString();
        }

        @Override
        public String decode(int token) {
          return token == 0 ? "" : Character.toString(token - 1);
        }

        @Override
        public int vocabSize() {
          return VOCABULARY_SIZE;
        }

        @Override
        public int bosToken() {
          return 1;
        }

        @Override
        public int eosToken() {
          return 0;
        }
      };
    }

    @Override
    public float[] forward(int token, int position) {
      throw new AssertionError("default state must not serve a virtual conversation");
    }

    @Override
    public int maxBatchSize() {
      return 4;
    }

    @Override
    public InferenceSession openSession() {
      State state = new State();
      sessions.add(state);
      return state;
    }

    @Override
    public float[] prefill(InferenceSession session, int[] tokens, int startPosition) {
      State state = state(session);
      state.position = startPosition + tokens.length;
      state.output = outputs.removeFirst();
      state.outputIndex = 1;
      return logits(state.output.isEmpty() ? 0 : state.output.charAt(0) + 1);
    }

    @Override
    public float[] forward(InferenceSession session, int token, int position) {
      State state = state(session);
      state.position = position + 1;
      if (state.outputIndex >= state.output.length()) {
        return logits(0);
      }
      return logits(state.output.charAt(state.outputIndex++) + 1);
    }

    @Override
    public LogitBatch forwardBatch(InferenceSession[] sessions, int[] tokens) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void rewind(InferenceSession session, int checkpoint) {
      state(session).position = checkpoint;
    }

    @Override
    public void reset(InferenceSession session) {
      state(session).position = 0;
    }

    @Override
    public void close() {}

    private State state(InferenceSession session) {
      return (State) session;
    }

    private static float[] logits(int token) {
      float[] logits = new float[VOCABULARY_SIZE];
      logits[token] = 100;
      return logits;
    }

    private static final class State implements InferenceSession {
      private int position;
      private int outputIndex;
      private String output = "";
      private boolean closed;

      @Override
      public int checkpoint() {
        return position;
      }

      @Override
      public boolean isClosed() {
        return closed;
      }

      @Override
      public void close() {
        closed = true;
      }
    }
  }
}
