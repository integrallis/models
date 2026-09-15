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
package com.integrallis.models.bench;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.ActivatedToolCallingModel;
import com.integrallis.models.runtime.ActivatedToolTurn;
import com.integrallis.models.runtime.TokenConstraint;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;

/**
 * Small, reproducible smoke for an activated query-rewrite specialist.
 *
 * <p>The conversation is the public IBM model-card example. It deliberately uses the exact Granite
 * control tokens and appends the adapter-provided invocation as a control segment, so tokenization
 * cannot silently turn the activation marker into ordinary prose.
 */
final class ActivatedQueryRewriteQualificationCli {
  private static final Set<String> OPTIONS = Set.of("model", "adapter", "models-revision");
  private static final ObjectMapper JSON = new ObjectMapper();

  private ActivatedQueryRewriteQualificationCli() {}

  record Configuration(Path model, Path adapter, String modelsRevision) {}

  static Configuration parse(String[] args) {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);
    Path model = Path.of(required(values, "model"));
    Path adapter = Path.of(required(values, "adapter"));
    String revision = required(values, "models-revision");
    if (!revision.matches("[0-9a-f]{40}")) {
      throw new IllegalArgumentException("--models-revision must be an exact 40-character Git SHA");
    }
    return new Configuration(model, adapter, revision);
  }

  static ModelPrompt modelCardPrompt(String invocation) {
    return ModelPrompt.builder()
        .control("<|start_of_role|>system<|end_of_role|><|end_of_text|>\n")
        .control("<|start_of_role|>user<|end_of_role|>Who is the CEO of Apple?<|end_of_text|>\n")
        .control(
            "<|start_of_role|>assistant<|end_of_role|>Tim Cook is the CEO of Apple.<|end_of_text|>\n")
        .control("<|start_of_role|>user<|end_of_role|>and for Microsoft?<|end_of_text|>\n")
        .control(invocation)
        .build();
  }

  static int run(String[] args) throws Exception {
    Configuration configuration = parse(args);
    try (PureJavaBackend backend =
            PureJavaBackend.loadActivatedAdapter(configuration.model(), configuration.adapter());
        ActivatedToolCallingModel model = new ActivatedToolCallingModel(backend, 1);
        ActivatedToolTurn turn =
            model.openToolTurn(
                modelCardPrompt(model.adapter().invocationText()),
                ActivatedToolCallingModel.PrefixStrategy.SHARED)) {
      String output =
          turn.generateToolCall(
              SamplingOptions.builder().temperature(0).maxTokens(64).build(),
              TokenConstraint.unrestricted());
      JsonNode parsed = JSON.readTree(output);
      String rewrite = parsed.path("rewritten_question").asText();
      boolean qualified =
          turn.physicallySharesPrefix()
              && rewrite.toLowerCase(java.util.Locale.ROOT).contains("microsoft")
              && rewrite.toLowerCase(java.util.Locale.ROOT).contains("ceo");
      System.out.printf(
          "%s shared=%s prefix=%d rewrite=%s%n",
          qualified ? "PASS" : "FAIL",
          turn.physicallySharesPrefix(),
          turn.sharedPrefixTokens(),
          output);
      return qualified ? 0 : 1;
    }
  }

  private static String required(Map<String, String> values, String name) {
    String value = values.get(name);
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("--" + name + " is required");
    }
    return value;
  }
}
