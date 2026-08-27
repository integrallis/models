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

import com.integrallis.models.api.AuxiliaryTextGenerationModel;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.ToolSpec;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ToolSpecSelectorTest {

  @Test
  void leavesSmallToolSetsAloneWithoutRunningTheContrastiveHead() {
    RecordingAuxiliaryModel model = new RecordingAuxiliaryModel();
    ToolSpecSelector selector = new ToolSpecSelector(model);
    List<ToolSpec> tools = tools(5);

    assertThat(selector.select("use tool-4", tools)).containsExactlyElementsOf(tools);
    assertThat(model.encoded).isEmpty();
  }

  @Test
  void selectsFiveToolsAndReusesTheIndexedSchemaEmbeddings() {
    RecordingAuxiliaryModel model = new RecordingAuxiliaryModel();
    ToolSpecSelector selector = new ToolSpecSelector(model);
    List<ToolSpec> tools = tools(7);

    assertThat(selector.select("use tool-6", tools))
        .extracting(ToolSpec::name)
        .startsWith("tool-6")
        .hasSize(5);
    assertThat(selector.select("use tool-5", List.copyOf(tools)))
        .extracting(ToolSpec::name)
        .startsWith("tool-5")
        .hasSize(5);

    assertThat(model.encoded).hasSize(9);
    assertThat(model.encoded.subList(0, 7))
        .allSatisfy(value -> assertThat(value).contains("A numbered test tool"));
    assertThat(model.encoded.subList(7, 9)).containsExactly("use tool-6", "use tool-5");
  }

  private static List<ToolSpec> tools(int count) {
    List<ToolSpec> tools = new ArrayList<>(count);
    for (int index = 0; index < count; index++) {
      tools.add(
          new ToolSpec(
              "tool-" + index,
              "A numbered test tool " + index,
              "{\"type\":\"object\",\"properties\":{}}"));
    }
    return List.copyOf(tools);
  }

  private static final class RecordingAuxiliaryModel implements AuxiliaryTextGenerationModel {
    private final List<String> encoded = new ArrayList<>();

    @Override
    public boolean supportsContrastiveEncoding() {
      return true;
    }

    @Override
    public int contrastiveDimension() {
      return 8;
    }

    @Override
    public float[] encodeContrastive(ModelPrompt prompt) {
      String text = prompt.text();
      encoded.add(text);
      float[] vector = new float[contrastiveDimension()];
      for (int index = 0; index < vector.length; index++) {
        if (text.contains("tool-" + index)) {
          vector[index] = 1.0f;
        }
      }
      if (text.startsWith("use tool-")) {
        vector[Integer.parseInt(text.substring("use tool-".length()))] = 2.0f;
      }
      return vector;
    }

    @Override
    public boolean supportsConfidenceScoring() {
      return false;
    }

    @Override
    public float scoreConfidence(ModelPrompt sequence) {
      throw new UnsupportedOperationException();
    }

    @Override
    public String modelName() {
      return "test";
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable("test");
    }

    @Override
    public void generate(String prompt, SamplingOptions options, TokenStream stream) {
      stream.onComplete();
    }
  }
}
