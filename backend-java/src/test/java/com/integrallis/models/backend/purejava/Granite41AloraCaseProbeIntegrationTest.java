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

import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.SharedInferencePrefix;
import com.integrallis.models.api.SharedPrefixInferenceBackend.Branch;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * Diagnostic probe: replays one dumped qualification prompt through the activated-adapter branch
 * along three explicit paths and prints the top-5 next-token logits at every decode step, so the
 * distribution can be compared with an external reference implementation. Enabled only when {@code
 * models.fixtures.granite41AloraProbeDump} names a runner prompt dump.
 */
@Tag("integration")
class Granite41AloraCaseProbeIntegrationTest {
  static final String DUMP_PROPERTY = "models.fixtures.granite41AloraProbeDump";
  private static final int STEPS = 5;

  @Test
  @EnabledIfSystemProperty(named = DUMP_PROPERTY, matches = ".+")
  void printsTopLogitsAlongEveryActivationPath() throws Exception {
    String dump = Files.readString(Path.of(System.getProperty(DUMP_PROPERTY)));
    String caseId = dump.replaceAll("(?s).*\"caseId\"\\s*:\\s*\"([^\"]+)\".*", "$1");
    String tokenList = dump.replaceAll("(?s).*\"tokens\"\\s*:\\s*\\[([^\\]]*)\\].*", "$1");
    int[] prompt =
        Arrays.stream(tokenList.split(",")).map(String::trim).mapToInt(Integer::parseInt).toArray();
    try (PureJavaBackend backend =
        PureJavaBackend.loadActivatedAdapter(
            Path.of(System.getProperty(Granite41AloraIntegrationTest.BASE_PROPERTY)),
            Path.of(System.getProperty(Granite41AloraIntegrationTest.ADAPTER_PROPERTY)))) {
      int[] invocation =
          backend.activatedAdapter().orElseThrow().invocationTokens().stream()
              .mapToInt(Integer::intValue)
              .toArray();
      int prefixLength = prompt.length - invocation.length;
      if (!Arrays.equals(Arrays.copyOfRange(prompt, prefixLength, prompt.length), invocation)) {
        throw new IllegalStateException("dump does not end with the invocation tokens");
      }
      int[] prefix = Arrays.copyOf(prompt, prefixLength);
      System.out.printf("PROBE case=%s prompt=%d prefix=%d%n", caseId, prompt.length, prefixLength);
      java.util.Map<Integer, float[][]> observed = new java.util.TreeMap<>();
      String hiddenOutput = System.getProperty("models.fixtures.granite41AloraProbeHiddenOutput");
      if (hiddenOutput != null) {
        LayerCallback callback =
            (layer, position, state, offset, length) -> {
              if (position == prompt.length - 1 || position == prompt.length) {
                float[][] layers = observed.computeIfAbsent(position, ignored -> new float[64][]);
                layers[layer] = Arrays.copyOfRange(state, offset, offset + length);
              }
            };
        installLayerObserver(backend, callback);
      }

      // Path A: one session, prefix on the base, activate, invocation one token at a time.
      try (InferenceSession session = backend.openSession()) {
        backend.prefill(session, prefix, 0);
        backend.activateAdapter(session);
        float[] logits = null;
        for (int i = 0; i < invocation.length; i++) {
          logits = backend.forward(session, invocation[i], prefixLength + i);
        }
        decode(backend, session, logits, prompt.length, "A:single-token-invocation");
      }
      if (hiddenOutput != null) {
        writeHidden(Path.of(hiddenOutput), caseId, observed);
        observed.clear();
      }

      // Path B: one session, prefix on the base, activate, invocation as a batched prefill.
      try (InferenceSession session = backend.openSession()) {
        backend.prefill(session, prefix, 0);
        backend.activateAdapter(session);
        float[] logits = backend.prefill(session, invocation, prefixLength);
        decode(backend, session, logits, prompt.length, "B:batched-invocation");
      }

      // Path C: the shared-prefix fork the specialist turn uses (frozen base prefix, activated
      // branch, invocation as a batched prefill).
      try (InferenceSession source = backend.openSession()) {
        backend.prefill(source, prefix, 0);
        SharedInferencePrefix shared = backend.freezePrefix(source);
        try (InferenceSession session = backend.fork(shared, Branch.ACTIVATED_ADAPTER)) {
          float[] logits = backend.prefill(session, invocation, prefixLength);
          decode(backend, session, logits, prompt.length, "C:shared-fork-batched-invocation");
        }
      }

      // Path D (control, wrong semantics on purpose): whole prompt on the base branch, no
      // adapter at all.
      try (InferenceSession session = backend.openSession()) {
        float[] logits = backend.prefill(session, prompt, 0);
        decode(backend, session, logits, prompt.length, "D:base-no-adapter");
      }
      if (hiddenOutput != null) {
        writeHidden(Path.of(hiddenOutput + ".base.json"), caseId, observed);
        installLayerObserver(backend, null);
      }
    }
  }

  private static void decode(
      PureJavaBackend backend,
      InferenceSession session,
      float[] logits,
      int position,
      String label) {
    List<Integer> generated = new ArrayList<>();
    for (int step = 0; step < STEPS; step++) {
      int[] top = topK(logits, 5);
      StringBuilder line = new StringBuilder();
      line.append("PROBE ").append(label).append(" step=").append(step).append(" top5=");
      for (int id : top) {
        line.append('[')
            .append(id)
            .append(' ')
            .append(escape(backend.tokenizer().decode(id)))
            .append(' ')
            .append(String.format("%.3f", logits[id]))
            .append("] ");
      }
      System.out.println(line);
      generated.add(top[0]);
      logits = backend.forward(session, top[0], position + step);
    }
    int[] ids = generated.stream().mapToInt(Integer::intValue).toArray();
    System.out.printf(
        "PROBE %s generated=%s text=%s%n",
        label, Arrays.toString(ids), escape(backend.tokenizer().decode(ids)));
  }

  /** Reflectively installs a per-layer observer on the loaded Llama forward pass. */
  private static void installLayerObserver(PureJavaBackend backend, LayerCallback observer)
      throws Exception {
    java.lang.reflect.Field decoderField = PureJavaBackend.class.getDeclaredField("decoder");
    decoderField.setAccessible(true);
    Object decoder = decoderField.get(backend);
    Object forwardPass = null;
    for (java.lang.reflect.Field field : decoder.getClass().getDeclaredFields()) {
      field.setAccessible(true);
      Object value = field.get(decoder);
      if (value != null && value.getClass().getName().endsWith("llama.LlamaForwardPass")) {
        forwardPass = value;
      }
    }
    if (forwardPass == null) {
      throw new IllegalStateException("no LlamaForwardPass inside " + decoder.getClass());
    }
    java.lang.reflect.Field observerField =
        forwardPass.getClass().getDeclaredField("layerObserver");
    observerField.setAccessible(true);
    Class<?> observerType = observerField.getType();
    Object proxy = null;
    if (observer != null) {
      LayerCallback callback = observer;
      proxy =
          java.lang.reflect.Proxy.newProxyInstance(
              observerType.getClassLoader(),
              new Class<?>[] {observerType},
              (ignored, method, args) -> {
                if (method.getName().equals("onLayerComplete")) {
                  callback.onLayerComplete(
                      (Integer) args[0],
                      (Integer) args[1],
                      (float[]) args[2],
                      (Integer) args[3],
                      (Integer) args[4]);
                  return null;
                }
                return method.getName().equals("toString") ? "probe" : null;
              });
    }
    observerField.set(forwardPass, proxy);
  }

  @FunctionalInterface
  interface LayerCallback {
    void onLayerComplete(int layer, int position, float[] state, int offset, int length);
  }

  private static void writeHidden(
      Path target, String caseId, java.util.Map<Integer, float[][]> observed) throws Exception {
    StringBuilder json = new StringBuilder();
    json.append("{\"caseId\":\"").append(caseId).append("\"");
    for (var entry : observed.entrySet()) {
      json.append(",\"position").append(entry.getKey()).append("\":[");
      boolean firstLayer = true;
      for (float[] layer : entry.getValue()) {
        if (layer == null) continue;
        if (!firstLayer) json.append(',');
        firstLayer = false;
        json.append('[');
        for (int i = 0; i < layer.length; i++) {
          if (i > 0) json.append(',');
          json.append(layer[i]);
        }
        json.append(']');
      }
      json.append(']');
    }
    json.append("}\n");
    Files.writeString(target, json.toString());
    System.out.println("PROBE hidden states written to " + target);
  }

  private static int[] topK(float[] logits, int k) {
    int[] best = new int[k];
    Arrays.fill(best, -1);
    for (int i = 0; i < logits.length; i++) {
      for (int slot = 0; slot < k; slot++) {
        if (best[slot] < 0 || logits[i] > logits[best[slot]]) {
          System.arraycopy(best, slot, best, slot + 1, k - slot - 1);
          best[slot] = i;
          break;
        }
      }
    }
    return best;
  }

  private static String escape(String text) {
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t");
  }
}
