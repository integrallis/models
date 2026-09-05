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
package com.integrallis.models.audio;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.api.AudioStream;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.PcmAudio;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.SpeechSynthesisOptions;
import com.integrallis.models.api.WavEncoder;
import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.models.runtime.Sampler;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("integration")
class SopranoTextToSpeechModelIntegrationTest {

  @Test
  void synthesizesAudibleWaveDataWithoutAnExternalRuntime(@TempDir Path output) throws Exception {
    String configured = System.getenv("MODELS_SOPRANO_GGUF");
    assumeTrue(configured != null && Files.isRegularFile(Path.of(configured)));
    SpeechSynthesisOptions options =
        SpeechSynthesisOptions.builder()
            .sampling(
                SamplingOptions.builder()
                    .temperature(0.3f)
                    .topP(0.95f)
                    .topK(8192)
                    .maxTokens(128)
                    .repetitionPenalty(1.2f)
                    .seed(42)
                    .build())
            .language("en")
            .build();

    TimingEngine engine = new TimingEngine(SopranoBackendEngine.load(Path.of(configured)));
    PcmAudio audio;
    try (SopranoTextToSpeechModel model = new SopranoTextToSpeechModel(engine)) {
      model.synthesize("The JVM can speak for itself.", options);
      engine.resetTimings();
      long started = System.nanoTime();
      audio = model.synthesize("The JVM can speak for itself.", options);
      engine.elapsedNanos = System.nanoTime() - started;
    }
    long elapsedMillis = engine.elapsedNanos / 1_000_000L;
    byte[] wave = WavEncoder.pcm16(audio);
    String configuredOutput = System.getenv("MODELS_SOPRANO_OUTPUT");
    Path wavePath =
        configuredOutput == null ? output.resolve("soprano-java.wav") : Path.of(configuredOutput);
    Files.write(wavePath, wave);

    assertThat(audio.sampleRate()).isEqualTo(32_000);
    assertThat(audio.channels()).isEqualTo(1);
    assertThat(audio.duration().toMillis()).isBetween(100L, 32_000L);
    assertThat(rootMeanSquare(audio.samples())).isGreaterThan(1.0e-5);
    assertThat(wave).startsWith((byte) 'R', (byte) 'I', (byte) 'F', (byte) 'F');
    System.out.printf(
        "Soprano Java: %d ms total, %d ms LM, %d ms vocoder, %.3f s audio, RTF %.3f%n",
        elapsedMillis,
        engine.languageNanos / 1_000_000L,
        engine.vocoderNanos / 1_000_000L,
        audio.duration().toNanos() / 1_000_000_000.0,
        elapsedMillis / (double) audio.duration().toMillis());
  }

  @Test
  void synthesizesThroughTheModelsOwnedNativeQ8Kernel(@TempDir Path output) throws Exception {
    String configured = System.getenv("MODELS_SOPRANO_GGUF");
    String nativeLibrary = System.getProperty("models.soprano.test.nativeLibrary");
    assumeTrue(configured != null && Files.isRegularFile(Path.of(configured)));
    assumeTrue(nativeLibrary != null && Files.isRegularFile(Path.of(nativeLibrary)));
    SpeechSynthesisOptions options =
        SpeechSynthesisOptions.builder()
            .sampling(
                SamplingOptions.builder()
                    .temperature(0.3f)
                    .topP(0.95f)
                    .topK(8192)
                    .maxTokens(128)
                    .repetitionPenalty(1.2f)
                    .seed(42)
                    .build())
            .language("en")
            .build();
    PcmAudio audio;
    try (SopranoTextToSpeechModel model =
        SopranoTextToSpeechModel.loadNative(Path.of(configured), Path.of(nativeLibrary))) {
      model.synthesize("The JVM can speak for itself.", options);
      long started = System.nanoTime();
      audio = model.synthesize("The JVM can speak for itself.", options);
      long elapsedNanos = System.nanoTime() - started;

      assertThat(model.diagnostics().environment().get("matrix-kernel"))
          .startsWith("rust-ffm-quantized");
      System.out.printf(
          "Soprano Java + Rust/FFM Q8: %d ms total, %.3f s audio, RTF %.3f%n",
          elapsedNanos / 1_000_000L,
          audio.duration().toNanos() / 1_000_000_000.0,
          (elapsedNanos / 1_000_000.0) / audio.duration().toMillis());
    }

    byte[] wave = WavEncoder.pcm16(audio);
    Files.write(output.resolve("soprano-java-rust-ffm.wav"), wave);
    assertThat(audio.sampleRate()).isEqualTo(32_000);
    assertThat(audio.channels()).isEqualTo(1);
    assertThat(rootMeanSquare(audio.samples())).isGreaterThan(1.0e-5);
    assertThat(wave).startsWith((byte) 'R', (byte) 'I', (byte) 'F', (byte) 'F');
  }

  @Test
  void nativeQ8KernelMatchesThePureJavaOfficialPromptBoundary() throws Exception {
    String configured = System.getenv("MODELS_SOPRANO_GGUF");
    String nativeLibrary = System.getProperty("models.soprano.test.nativeLibrary");
    assumeTrue(configured != null && Files.isRegularFile(Path.of(configured)));
    assumeTrue(nativeLibrary != null && Files.isRegularFile(Path.of(nativeLibrary)));

    try (SopranoEngine javaEngine = SopranoBackendEngine.load(Path.of(configured));
        SopranoEngine nativeEngine =
            SopranoBackendEngine.load(
                Path.of(configured), RustGgufBatchedMatrixKernel.open(Path.of(nativeLibrary)))) {
      int[] prompt = javaEngine.encodePrompt("The JVM can speak for itself.");
      assertThat(nativeEngine.encodePrompt("The JVM can speak for itself."))
          .containsExactly(prompt);
      SopranoEngine.Step javaStep = javaEngine.begin(prompt);
      SopranoEngine.Step nativeStep = nativeEngine.begin(prompt);
      Comparison logits = compare(javaStep.logits(), nativeStep.logits());
      Comparison hidden = compare(javaStep.hiddenState(), nativeStep.hiddenState());
      SamplingOptions sampling =
          SamplingOptions.builder()
              .temperature(0.0f)
              .topP(1.0f)
              .topK(8192)
              .maxTokens(128)
              .repetitionPenalty(1.2f)
              .build();
      List<Integer> history = new ArrayList<>(prompt.length);
      for (int token : prompt) {
        history.add(token);
      }
      int javaToken = new Sampler(sampling).sample(javaStep.logits(), history);
      int nativeToken = new Sampler(sampling).sample(nativeStep.logits(), history);

      System.out.printf(
          "Soprano native prompt parity: logits cosine=%.9f maxAbs=%.6f; hidden cosine=%.9f maxAbs=%.6f; token=%d/%d%n",
          logits.cosine(),
          logits.maxAbsoluteError(),
          hidden.cosine(),
          hidden.maxAbsoluteError(),
          javaToken,
          nativeToken);
      assertThat(logits.cosine()).isGreaterThan(0.999);
      assertThat(logits.maxAbsoluteError()).isLessThan(0.15);
      assertThat(hidden.cosine()).isGreaterThan(0.999);
      assertThat(hidden.maxAbsoluteError()).isLessThan(0.1);
      assertThat(nativeToken).isEqualTo(javaToken);
    }
  }

  @Test
  void nativeQ8OutputHeadPreservesThePureJavaFirstAcousticToken() throws Exception {
    String configured = System.getenv("MODELS_SOPRANO_GGUF");
    String nativeLibrary = System.getProperty("models.soprano.test.nativeLibrary");
    assumeTrue(configured != null && Files.isRegularFile(Path.of(configured)));
    assumeTrue(nativeLibrary != null && Files.isRegularFile(Path.of(nativeLibrary)));

    try (SopranoEngine javaEngine = SopranoBackendEngine.load(Path.of(configured));
        SopranoEngine nativeOutputHeadEngine =
            SopranoBackendEngine.load(
                Path.of(configured),
                new SelectiveQ8Kernel(
                    RustGgufBatchedMatrixKernel.open(Path.of(nativeLibrary)), true, false))) {
      int[] prompt = javaEngine.encodePrompt("The JVM can speak for itself.");
      SopranoEngine.Step javaStep = javaEngine.begin(prompt);
      SopranoEngine.Step nativeStep = nativeOutputHeadEngine.begin(prompt);
      int javaToken = greedyToken(javaStep.logits(), prompt);
      int nativeToken = greedyToken(nativeStep.logits(), prompt);

      System.out.printf(
          "Soprano native output-head isolation: logits cosine=%.9f maxAbs=%.6f; token=%d/%d%n",
          compare(javaStep.logits(), nativeStep.logits()).cosine(),
          compare(javaStep.logits(), nativeStep.logits()).maxAbsoluteError(),
          javaToken,
          nativeToken);
      assertThat(nativeToken).isEqualTo(javaToken);
    }
  }

  @Test
  void nativeQ8SingleTransformerProjectionsPreserveThePureJavaFirstAcousticToken()
      throws Exception {
    String configured = System.getenv("MODELS_SOPRANO_GGUF");
    String nativeLibrary = System.getProperty("models.soprano.test.nativeLibrary");
    assumeTrue(configured != null && Files.isRegularFile(Path.of(configured)));
    assumeTrue(nativeLibrary != null && Files.isRegularFile(Path.of(nativeLibrary)));
    SelectiveQ8Kernel nativeTransformer =
        new SelectiveQ8Kernel(
            RustGgufBatchedMatrixKernel.open(Path.of(nativeLibrary)), false, true);

    try (SopranoEngine javaEngine = SopranoBackendEngine.load(Path.of(configured));
        SopranoEngine nativeTransformerEngine =
            SopranoBackendEngine.load(Path.of(configured), nativeTransformer)) {
      int[] prompt = javaEngine.encodePrompt("The JVM can speak for itself.");
      SopranoEngine.Step javaStep = javaEngine.begin(prompt);
      SopranoEngine.Step nativeStep = nativeTransformerEngine.begin(prompt);
      int javaToken = greedyToken(javaStep.logits(), prompt);
      int nativeToken = greedyToken(nativeStep.logits(), prompt);

      System.out.printf(
          "Soprano native single-transformer isolation: calls=%d shapes=%s; logits cosine=%.9f maxAbs=%.6f; token=%d/%d%n",
          nativeTransformer.multiplyCalls,
          nativeTransformer.shapes,
          compare(javaStep.logits(), nativeStep.logits()).cosine(),
          compare(javaStep.logits(), nativeStep.logits()).maxAbsoluteError(),
          javaToken,
          nativeToken);
      assertThat(nativeToken).isEqualTo(javaToken);
    }
  }

  @Test
  void nativeQ8GroupedTransformerProjectionsPreserveThePureJavaFirstAcousticToken()
      throws Exception {
    String configured = System.getenv("MODELS_SOPRANO_GGUF");
    String nativeLibrary = System.getProperty("models.soprano.test.nativeLibrary");
    assumeTrue(configured != null && Files.isRegularFile(Path.of(configured)));
    assumeTrue(nativeLibrary != null && Files.isRegularFile(Path.of(nativeLibrary)));
    GroupedQ8Kernel nativeGrouped =
        new GroupedQ8Kernel(RustGgufBatchedMatrixKernel.open(Path.of(nativeLibrary)), true, true);

    try (SopranoEngine javaEngine = SopranoBackendEngine.load(Path.of(configured));
        SopranoEngine nativeGroupedEngine =
            SopranoBackendEngine.load(Path.of(configured), nativeGrouped)) {
      int[] prompt = javaEngine.encodePrompt("The JVM can speak for itself.");
      SopranoEngine.Step javaStep = javaEngine.begin(prompt);
      SopranoEngine.Step nativeStep = nativeGroupedEngine.begin(prompt);
      int javaToken = greedyToken(javaStep.logits(), prompt);
      int nativeToken = greedyToken(nativeStep.logits(), prompt);

      System.out.printf(
          "Soprano native grouped isolation: dual=%d triple=%d grouped=%d; logits cosine=%.9f maxAbs=%.6f; token=%d/%d%n",
          nativeGrouped.dualCalls,
          nativeGrouped.tripleCalls,
          nativeGrouped.groupedCalls,
          compare(javaStep.logits(), nativeStep.logits()).cosine(),
          compare(javaStep.logits(), nativeStep.logits()).maxAbsoluteError(),
          javaToken,
          nativeToken);
      assertThat(nativeToken).isEqualTo(javaToken);
    }
  }

  @Test
  void nativeQ8DualTransformerProjectionsPreserveThePureJavaFirstAcousticToken() throws Exception {
    assertGroupedProjectionParity(true, false);
  }

  @Test
  void nativeQ8TripleTransformerProjectionsPreserveThePureJavaFirstAcousticToken()
      throws Exception {
    assertGroupedProjectionParity(false, true);
  }

  private static void assertGroupedProjectionParity(boolean dual, boolean triple) throws Exception {
    String configured = System.getenv("MODELS_SOPRANO_GGUF");
    String nativeLibrary = System.getProperty("models.soprano.test.nativeLibrary");
    assumeTrue(configured != null && Files.isRegularFile(Path.of(configured)));
    assumeTrue(nativeLibrary != null && Files.isRegularFile(Path.of(nativeLibrary)));
    GroupedQ8Kernel nativeGrouped =
        new GroupedQ8Kernel(RustGgufBatchedMatrixKernel.open(Path.of(nativeLibrary)), dual, triple);

    try (SopranoEngine javaEngine = SopranoBackendEngine.load(Path.of(configured));
        SopranoEngine nativeGroupedEngine =
            SopranoBackendEngine.load(Path.of(configured), nativeGrouped)) {
      int[] prompt = javaEngine.encodePrompt("The JVM can speak for itself.");
      SopranoEngine.Step javaStep = javaEngine.begin(prompt);
      SopranoEngine.Step nativeStep = nativeGroupedEngine.begin(prompt);
      int javaToken = greedyToken(javaStep.logits(), prompt);
      int nativeToken = greedyToken(nativeStep.logits(), prompt);

      System.out.printf(
          "Soprano native %s isolation: dual=%d triple=%d dual-shapes=%s; logits cosine=%.9f maxAbs=%.6f; token=%d/%d%n",
          dual ? "dual" : "triple",
          nativeGrouped.dualCalls,
          nativeGrouped.tripleCalls,
          nativeGrouped.dualShapes,
          compare(javaStep.logits(), nativeStep.logits()).cosine(),
          compare(javaStep.logits(), nativeStep.logits()).maxAbsoluteError(),
          javaToken,
          nativeToken);
      assertThat(nativeToken).isEqualTo(javaToken);
    }
  }

  @Test
  void nativeQ8KernelMatchesThePureJavaGreedyAudioTimeline() throws Exception {
    String configured = System.getenv("MODELS_SOPRANO_GGUF");
    String nativeLibrary = System.getProperty("models.soprano.test.nativeLibrary");
    assumeTrue(configured != null && Files.isRegularFile(Path.of(configured)));
    assumeTrue(nativeLibrary != null && Files.isRegularFile(Path.of(nativeLibrary)));
    SpeechSynthesisOptions options =
        SpeechSynthesisOptions.builder()
            .sampling(
                SamplingOptions.builder()
                    .temperature(0.0f)
                    .topP(1.0f)
                    .topK(8192)
                    .maxTokens(128)
                    .repetitionPenalty(1.2f)
                    .build())
            .language("en")
            .build();
    PcmAudio javaAudio;
    try (SopranoTextToSpeechModel model = SopranoTextToSpeechModel.load(Path.of(configured))) {
      javaAudio = model.synthesize("The JVM can speak for itself.", options);
    }
    PcmAudio nativeAudio;
    try (SopranoTextToSpeechModel model =
        SopranoTextToSpeechModel.loadNative(Path.of(configured), Path.of(nativeLibrary))) {
      nativeAudio = model.synthesize("The JVM can speak for itself.", options);
    }

    assertThat(nativeAudio.sampleRate()).isEqualTo(javaAudio.sampleRate());
    assertThat(nativeAudio.samples()).hasSameSizeAs(javaAudio.samples());
    assertThat(cosine(javaAudio.samples(), nativeAudio.samples())).isGreaterThan(0.999);
  }

  @Test
  void streamsRealAudioBeforeCompletionAndMatchesTheBlockingTimeline() throws Exception {
    String configured = System.getenv("MODELS_SOPRANO_GGUF");
    assumeTrue(configured != null && Files.isRegularFile(Path.of(configured)));
    SpeechSynthesisOptions options =
        SpeechSynthesisOptions.builder()
            .sampling(
                SamplingOptions.builder()
                    .temperature(0.0f)
                    .topP(1.0f)
                    .topK(8192)
                    .maxTokens(12)
                    .repetitionPenalty(1.2f)
                    .build())
            .language("en")
            .build();

    List<PcmAudio> chunks = new ArrayList<>();
    long[] firstAudioNanos = {-1L};
    long started = System.nanoTime();
    PcmAudio blocking;
    try (SopranoTextToSpeechModel model = SopranoTextToSpeechModel.load(Path.of(configured))) {
      blocking = model.synthesize("The JVM can speak for itself.", options);
      started = System.nanoTime();
      model.synthesize(
          "The JVM can speak for itself.",
          options,
          new AudioStream() {
            @Override
            public void onAudio(PcmAudio audio) {
              if (firstAudioNanos[0] < 0) {
                firstAudioNanos[0] = System.nanoTime();
              }
              chunks.add(audio);
            }

            @Override
            public void onComplete() {}

            @Override
            public void onError(Throwable failure) {
              throw new AssertionError(failure);
            }
          });
    }
    long completed = System.nanoTime();
    float[] streamed = concatenate(chunks);
    double cosine = cosine(blocking.samples(), streamed);

    assertThat(chunks).hasSizeGreaterThan(1);
    assertThat(firstAudioNanos[0]).isBetween(started, completed - 1);
    assertThat(streamed).hasSameSizeAs(blocking.samples());
    assertThat(cosine).isGreaterThan(0.99);
    System.out.printf(
        "Soprano stream: %d chunks, %.1f ms TTFA, %.1f ms total, blocking cosine %.9f%n",
        chunks.size(),
        (firstAudioNanos[0] - started) / 1_000_000.0,
        (completed - started) / 1_000_000.0,
        cosine);
  }

  @Test
  void nativeQ8StreamsRealAudioBeforeCompletionAndMatchesItsBlockingTimeline() throws Exception {
    String configured = System.getenv("MODELS_SOPRANO_GGUF");
    String nativeLibrary = System.getProperty("models.soprano.test.nativeLibrary");
    assumeTrue(configured != null && Files.isRegularFile(Path.of(configured)));
    assumeTrue(nativeLibrary != null && Files.isRegularFile(Path.of(nativeLibrary)));
    SpeechSynthesisOptions options =
        SpeechSynthesisOptions.builder()
            .sampling(
                SamplingOptions.builder()
                    .temperature(0.0f)
                    .topP(1.0f)
                    .topK(8192)
                    .maxTokens(12)
                    .repetitionPenalty(1.2f)
                    .build())
            .language("en")
            .build();

    List<PcmAudio> chunks = new ArrayList<>();
    long[] firstAudioNanos = {-1L};
    long started;
    PcmAudio blocking;
    try (SopranoTextToSpeechModel model =
        SopranoTextToSpeechModel.loadNative(Path.of(configured), Path.of(nativeLibrary))) {
      blocking = model.synthesize("The JVM can speak for itself.", options);
      started = System.nanoTime();
      model.synthesize(
          "The JVM can speak for itself.",
          options,
          new AudioStream() {
            @Override
            public void onAudio(PcmAudio audio) {
              if (firstAudioNanos[0] < 0) {
                firstAudioNanos[0] = System.nanoTime();
              }
              chunks.add(audio);
            }

            @Override
            public void onComplete() {}

            @Override
            public void onError(Throwable failure) {
              throw new AssertionError(failure);
            }
          });
    }
    long completed = System.nanoTime();
    float[] streamed = concatenate(chunks);
    double cosine = cosine(blocking.samples(), streamed);

    assertThat(chunks).hasSizeGreaterThan(1);
    assertThat(firstAudioNanos[0]).isBetween(started, completed - 1);
    assertThat(streamed).hasSameSizeAs(blocking.samples());
    assertThat(cosine).isGreaterThan(0.99);
    System.out.printf(
        "Soprano native stream: %d chunks, %.1f ms TTFA, %.1f ms total, blocking cosine %.9f%n",
        chunks.size(),
        (firstAudioNanos[0] - started) / 1_000_000.0,
        (completed - started) / 1_000_000.0,
        cosine);
  }

  @Test
  void recordsFiveControlledStreamingQualificationTrials() throws Exception {
    String configured = System.getenv("MODELS_SOPRANO_GGUF");
    String backend = System.getenv("MODELS_SOPRANO_QUALIFICATION_BACKEND");
    String nativeLibrary = System.getProperty("models.soprano.test.nativeLibrary");
    assumeTrue(configured != null && Files.isRegularFile(Path.of(configured)));
    assumeTrue("pure-java".equals(backend) || "rust-ffm".equals(backend));
    assumeTrue(
        !"rust-ffm".equals(backend)
            || (nativeLibrary != null && Files.isRegularFile(Path.of(nativeLibrary))));
    SpeechSynthesisOptions options =
        SpeechSynthesisOptions.builder()
            .sampling(
                SamplingOptions.builder()
                    .temperature(0.0f)
                    .topP(1.0f)
                    .topK(8192)
                    .maxTokens(128)
                    .repetitionPenalty(1.2f)
                    .build())
            .language("en")
            .build();

    try (SopranoTextToSpeechModel model =
        "rust-ffm".equals(backend)
            ? SopranoTextToSpeechModel.loadNative(Path.of(configured), Path.of(nativeLibrary))
            : SopranoTextToSpeechModel.load(Path.of(configured))) {
      PcmAudio blocking = model.synthesize("The JVM can speak for itself.", options);
      assertThat(blocking.duration().toMillis()).isPositive();

      for (int trial = 1; trial <= 5; trial++) {
        List<PcmAudio> chunks = new ArrayList<>();
        long[] firstAudioNanos = {-1L};
        long started = System.nanoTime();
        model.synthesize(
            "The JVM can speak for itself.",
            options,
            new AudioStream() {
              @Override
              public void onAudio(PcmAudio audio) {
                if (firstAudioNanos[0] < 0) {
                  firstAudioNanos[0] = System.nanoTime();
                }
                chunks.add(audio);
              }

              @Override
              public void onComplete() {}

              @Override
              public void onError(Throwable failure) {
                throw new AssertionError(failure);
              }
            });
        long completed = System.nanoTime();
        float[] streamed = concatenate(chunks);
        double cosine = cosine(blocking.samples(), streamed);
        double totalMillis = (completed - started) / 1_000_000.0;
        double ttfaMillis = (firstAudioNanos[0] - started) / 1_000_000.0;
        double realTimeFactor = totalMillis / blocking.duration().toMillis();

        assertThat(chunks).hasSizeGreaterThan(1);
        assertThat(firstAudioNanos[0]).isBetween(started, completed - 1);
        assertThat(streamed).hasSameSizeAs(blocking.samples());
        assertThat(cosine).isGreaterThan(0.999999);
        System.out.printf(
            Locale.ROOT,
            "SOPRANO_QUALIFICATION_TRIAL backend=%s trial=%d ttfaMillis=%.3f totalMillis=%.3f audioMillis=%d realTimeFactor=%.6f chunks=%d blockingCosine=%.9f%n",
            backend,
            trial,
            ttfaMillis,
            totalMillis,
            blocking.duration().toMillis(),
            realTimeFactor,
            chunks.size(),
            cosine);
      }
    }
  }

  private static float[] concatenate(List<PcmAudio> chunks) {
    int length = chunks.stream().mapToInt(PcmAudio::frameCount).sum();
    float[] result = new float[length];
    int offset = 0;
    for (PcmAudio chunk : chunks) {
      float[] samples = chunk.samples();
      System.arraycopy(samples, 0, result, offset, samples.length);
      offset += samples.length;
    }
    return result;
  }

  private static int greedyToken(float[] logits, int[] prompt) {
    SamplingOptions sampling =
        SamplingOptions.builder()
            .temperature(0.0f)
            .topP(1.0f)
            .topK(8192)
            .maxTokens(128)
            .repetitionPenalty(1.2f)
            .build();
    List<Integer> history = new ArrayList<>(prompt.length);
    for (int token : prompt) {
      history.add(token);
    }
    return new Sampler(sampling).sample(logits, history);
  }

  private static final class SelectiveQ8Kernel implements GgufBatchedMatrixKernel {
    private final GgufBatchedMatrixKernel nativeKernel;
    private final boolean outputHeadOnly;
    private final boolean forceUngrouped;
    private final List<String> shapes = new ArrayList<>();
    private int multiplyCalls;

    private SelectiveQ8Kernel(
        GgufBatchedMatrixKernel nativeKernel, boolean outputHeadOnly, boolean forceUngrouped) {
      this.nativeKernel = nativeKernel;
      this.outputHeadOnly = outputHeadOnly;
      this.forceUngrouped = forceUngrouped;
    }

    @Override
    public String implementation() {
      return nativeKernel.implementation() + (outputHeadOnly ? "-output-head" : "-transformer");
    }

    @Override
    public Map<String, String> planRecommendations() {
      return forceUngrouped
          ? Map.of(PureJavaPlanConfiguration.GROUPED_PROJECTIONS_PROPERTY, "false")
          : Map.of();
    }

    @Override
    public boolean supports(GgufTensorType type) {
      return type == GgufTensorType.Q8_0 && nativeKernel.supports(type);
    }

    @Override
    public boolean isEligible(GgufTensorType type, int batchSize, int rows, int cols) {
      boolean outputHead = rows == 8192 && cols == 512;
      return supports(type)
          && outputHead == outputHeadOnly
          && nativeKernel.isEligible(type, batchSize, rows, cols);
    }

    @Override
    public void multiply(
        float[] output,
        float[] input,
        MemorySegment weights,
        GgufTensorType type,
        int batchSize,
        int rows,
        int cols) {
      if (isEligible(type, batchSize, rows, cols)) {
        multiplyCalls++;
        String shape = batchSize + "x" + rows + "x" + cols;
        if (!shapes.contains(shape)) {
          shapes.add(shape);
        }
        nativeKernel.multiply(output, input, weights, type, batchSize, rows, cols);
        return;
      }
      throw new IllegalArgumentException(
          "selective kernel was invoked for an ineligible projection "
              + batchSize
              + "x"
              + rows
              + "x"
              + cols);
    }

    @Override
    public void close() {
      nativeKernel.close();
    }
  }

  private static final class GroupedQ8Kernel implements GgufBatchedMatrixKernel {
    private final GgufBatchedMatrixKernel nativeKernel;
    private final boolean dualEnabled;
    private final boolean tripleEnabled;
    private int dualCalls;
    private int tripleCalls;
    private int groupedCalls;
    private final List<String> dualShapes = new ArrayList<>();

    private GroupedQ8Kernel(
        GgufBatchedMatrixKernel nativeKernel, boolean dualEnabled, boolean tripleEnabled) {
      this.nativeKernel = nativeKernel;
      this.dualEnabled = dualEnabled;
      this.tripleEnabled = tripleEnabled;
    }

    @Override
    public String implementation() {
      return nativeKernel.implementation() + "-grouped";
    }

    @Override
    public Map<String, String> planRecommendations() {
      return nativeKernel.planRecommendations();
    }

    @Override
    public boolean supports(GgufTensorType type) {
      return nativeKernel.supports(type);
    }

    @Override
    public boolean isEligible(GgufTensorType type, int batchSize, int rows, int cols) {
      return false;
    }

    @Override
    public boolean isDualEligible(
        GgufTensorType firstType,
        int firstRows,
        GgufTensorType secondType,
        int secondRows,
        int batchSize,
        int cols) {
      return dualEnabled
          && nativeKernel.isDualEligible(
              firstType, firstRows, secondType, secondRows, batchSize, cols);
    }

    @Override
    public void multiplyDual(
        float[] firstOutput,
        MemorySegment firstWeights,
        GgufTensorType firstType,
        int firstRows,
        float[] secondOutput,
        MemorySegment secondWeights,
        GgufTensorType secondType,
        int secondRows,
        float[] input,
        int batchSize,
        int cols) {
      dualCalls++;
      String shape =
          firstType
              + "/"
              + secondType
              + ":"
              + batchSize
              + "x"
              + firstRows
              + "+"
              + secondRows
              + "x"
              + cols;
      if (!dualShapes.contains(shape)) {
        dualShapes.add(shape);
      }
      nativeKernel.multiplyDual(
          firstOutput,
          firstWeights,
          firstType,
          firstRows,
          secondOutput,
          secondWeights,
          secondType,
          secondRows,
          input,
          batchSize,
          cols);
    }

    @Override
    public boolean isTripleEligible(
        GgufTensorType firstType,
        int firstRows,
        GgufTensorType secondType,
        int secondRows,
        GgufTensorType thirdType,
        int thirdRows,
        int batchSize,
        int cols) {
      return tripleEnabled
          && nativeKernel.isTripleEligible(
              firstType, firstRows, secondType, secondRows, thirdType, thirdRows, batchSize, cols);
    }

    @Override
    public void multiplyTriple(
        float[] firstOutput,
        MemorySegment firstWeights,
        GgufTensorType firstType,
        int firstRows,
        float[] secondOutput,
        MemorySegment secondWeights,
        GgufTensorType secondType,
        int secondRows,
        float[] thirdOutput,
        MemorySegment thirdWeights,
        GgufTensorType thirdType,
        int thirdRows,
        float[] input,
        int batchSize,
        int cols) {
      tripleCalls++;
      nativeKernel.multiplyTriple(
          firstOutput,
          firstWeights,
          firstType,
          firstRows,
          secondOutput,
          secondWeights,
          secondType,
          secondRows,
          thirdOutput,
          thirdWeights,
          thirdType,
          thirdRows,
          input,
          batchSize,
          cols);
    }

    @Override
    public boolean isGroupedEligible(
        GgufTensorType[] types, int[] rows, int matrixCount, int batchSize, int cols) {
      return nativeKernel.isGroupedEligible(types, rows, matrixCount, batchSize, cols);
    }

    @Override
    public void multiplyGrouped(
        float[][] outputs,
        MemorySegment[] weights,
        GgufTensorType[] types,
        int[] rows,
        int matrixCount,
        float[] input,
        int batchSize,
        int cols) {
      groupedCalls++;
      nativeKernel.multiplyGrouped(
          outputs, weights, types, rows, matrixCount, input, batchSize, cols);
    }

    @Override
    public void multiply(
        float[] output,
        float[] input,
        MemorySegment weights,
        GgufTensorType type,
        int batchSize,
        int rows,
        int cols) {
      throw new IllegalArgumentException("grouped-only kernel received an individual projection");
    }

    @Override
    public void close() {
      nativeKernel.close();
    }
  }

  private static double cosine(float[] left, float[] right) {
    assertThat(right).hasSameSizeAs(left);
    double dot = 0.0;
    double leftNorm = 0.0;
    double rightNorm = 0.0;
    for (int index = 0; index < left.length; index++) {
      dot += (double) left[index] * right[index];
      leftNorm += (double) left[index] * left[index];
      rightNorm += (double) right[index] * right[index];
    }
    return dot / Math.sqrt(leftNorm * rightNorm);
  }

  private static Comparison compare(float[] expected, float[] actual) {
    assertThat(actual).hasSameSizeAs(expected);
    double dot = 0.0;
    double expectedNorm = 0.0;
    double actualNorm = 0.0;
    double maxAbsoluteError = 0.0;
    for (int index = 0; index < expected.length; index++) {
      dot += (double) expected[index] * actual[index];
      expectedNorm += (double) expected[index] * expected[index];
      actualNorm += (double) actual[index] * actual[index];
      maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(expected[index] - actual[index]));
    }
    return new Comparison(dot / Math.sqrt(expectedNorm * actualNorm), maxAbsoluteError);
  }

  private static double rootMeanSquare(float[] samples) {
    double sum = 0.0;
    for (float sample : samples) {
      sum += sample * sample;
    }
    return Math.sqrt(sum / samples.length);
  }

  private record Comparison(double cosine, double maxAbsoluteError) {}

  private static final class TimingEngine implements SopranoEngine {
    private final SopranoEngine delegate;
    private long languageNanos;
    private long vocoderNanos;
    private long elapsedNanos;

    private TimingEngine(SopranoEngine delegate) {
      this.delegate = delegate;
    }

    private void resetTimings() {
      languageNanos = 0L;
      vocoderNanos = 0L;
      elapsedNanos = 0L;
    }

    @Override
    public int[] encodePrompt(String text) {
      return delegate.encodePrompt(text);
    }

    @Override
    public Step begin(int[] prompt) {
      long started = System.nanoTime();
      try {
        return delegate.begin(prompt);
      } finally {
        languageNanos += System.nanoTime() - started;
      }
    }

    @Override
    public Step advance(int token) {
      long started = System.nanoTime();
      try {
        return delegate.advance(token);
      } finally {
        languageNanos += System.nanoTime() - started;
      }
    }

    @Override
    public float[] decode(float[] features, int frames) {
      long started = System.nanoTime();
      try {
        return delegate.decode(features, frames);
      } finally {
        vocoderNanos += System.nanoTime() - started;
      }
    }

    @Override
    public String modelName() {
      return delegate.modelName();
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return delegate.diagnostics();
    }

    @Override
    public int eosToken() {
      return delegate.eosToken();
    }

    @Override
    public int vocabularySize() {
      return delegate.vocabularySize();
    }

    @Override
    public int hiddenSize() {
      return delegate.hiddenSize();
    }

    @Override
    public int contextLength() {
      return delegate.contextLength();
    }

    @Override
    public int checkpoint() {
      return delegate.checkpoint();
    }

    @Override
    public int sampleRate() {
      return delegate.sampleRate();
    }

    @Override
    public int samplesPerToken() {
      return delegate.samplesPerToken();
    }

    @Override
    public void close() {
      delegate.close();
    }
  }
}
