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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.AudioStream;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.PcmAudio;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.SpeechSynthesisOptions;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class SopranoTextToSpeechModelTest {

  @Test
  void boundsTheNativeWorkerRecommendationForSmallSpeechProjections() {
    assertThat(SopranoTextToSpeechModel.nativeThreadRecommendation(1)).isEqualTo(1);
    assertThat(SopranoTextToSpeechModel.nativeThreadRecommendation(6)).isEqualTo(6);
    assertThat(SopranoTextToSpeechModel.nativeThreadRecommendation(16)).isEqualTo(6);
  }

  @Test
  void decodesOnlyHiddenFramesThatPredictNonEosAcousticTokens() {
    FakeSopranoEngine engine = new FakeSopranoEngine();
    SpeechSynthesisOptions options =
        SpeechSynthesisOptions.builder()
            .sampling(
                SamplingOptions.builder().temperature(0.0f).topP(1.0f).topK(8).maxTokens(4).build())
            .language("en")
            .build();

    PcmAudio audio = new SopranoTextToSpeechModel(engine).synthesize("Hello JVM", options);

    assertThat(audio.sampleRate()).isEqualTo(32_000);
    assertThat(audio.channels()).isEqualTo(1);
    assertThat(audio.samples()).containsExactly(0.25f, -0.25f);
    assertThat(engine.decodedFrames).isEqualTo(1);
    assertThat(engine.decodedFeatures).containsExactly(1.0f, 2.0f);
    assertThat(Arrays.copyOf(engine.advancedTokens, engine.advancedTokenCount)).containsExactly(4);
  }

  @Test
  void rejectsControlsThatThisFixedEnglishVoiceCannotHonor() {
    SopranoTextToSpeechModel model = new SopranoTextToSpeechModel(new FakeSopranoEngine());

    assertThatThrownBy(
            () ->
                model.synthesize("Hello", SpeechSynthesisOptions.builder().language("es").build()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("English");
    assertThatThrownBy(
            () ->
                model.synthesize(
                    "Hello", SpeechSynthesisOptions.builder().voice("someone-else").build()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("voice");
    assertThatThrownBy(
            () -> model.synthesize("Hello", SpeechSynthesisOptions.builder().speed(1.2f).build()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("speed");
    assertThatThrownBy(
            () ->
                model.synthesize(
                    "Hello",
                    SpeechSynthesisOptions.builder()
                        .sampling(SamplingOptions.builder().stopSequence("stop").build())
                        .build()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("stop sequences");
  }

  @Test
  void closingTheModelClosesItsMappedEngine() {
    FakeSopranoEngine engine = new FakeSopranoEngine();

    new SopranoTextToSpeechModel(engine).close();

    assertThat(engine.closed).isTrue();
  }

  @Test
  void rejectsAnAcousticFrameWithTheWrongHiddenSize() {
    FakeSopranoEngine engine = new FakeSopranoEngine();
    engine.initialHidden = new float[] {1.0f};

    assertThatThrownBy(
            () ->
                new SopranoTextToSpeechModel(engine)
                    .synthesize(
                        "Hello",
                        SpeechSynthesisOptions.builder()
                            .sampling(
                                SamplingOptions.builder().temperature(0.0f).maxTokens(2).build())
                            .build()))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("hidden size");
  }

  @Test
  void synthesizesLongTextAsReferenceSizedChunksAndConcatenatesTheirAudio() {
    FakeSopranoEngine engine = new FakeSopranoEngine();
    String text = "First sentence. " + "word ".repeat(45) + "Final sentence.";

    PcmAudio audio =
        new SopranoTextToSpeechModel(engine)
            .synthesize(
                text,
                SpeechSynthesisOptions.builder()
                    .sampling(
                        SamplingOptions.builder()
                            .temperature(0.0f)
                            .topP(1.0f)
                            .topK(8)
                            .maxTokens(4)
                            .build())
                    .build());

    assertThat(engine.encodedTexts).hasSizeGreaterThan(1);
    assertThat(engine.encodedTexts)
        .allSatisfy(
            chunk -> assertThat(chunk.codePointCount(0, chunk.length())).isLessThanOrEqualTo(200));
    assertThat(audio.samples()).hasSize(engine.encodedTexts.size() * 2);
  }

  @Test
  void streamsEachCompletedTextChunkBeforeCompletion() {
    FakeSopranoEngine engine = new FakeSopranoEngine();
    List<String> events = new ArrayList<>();
    String text = "First sentence. " + "word ".repeat(45) + "Final sentence.";

    new SopranoTextToSpeechModel(engine)
        .synthesize(
            text,
            SpeechSynthesisOptions.builder()
                .sampling(
                    SamplingOptions.builder()
                        .temperature(0.0f)
                        .topP(1.0f)
                        .topK(8)
                        .maxTokens(4)
                        .build())
                .build(),
            new AudioStream() {
              @Override
              public void onAudio(PcmAudio audio) {
                events.add("audio:" + audio.frameCount());
              }

              @Override
              public void onComplete() {
                events.add("complete");
              }

              @Override
              public void onError(Throwable failure) {
                events.add("error:" + failure.getMessage());
              }
            });

    assertThat(events).hasSize(engine.encodedTexts.size() + 1);
    assertThat(events.subList(0, events.size() - 1)).allMatch("audio:2"::equals);
    assertThat(events.getLast()).isEqualTo("complete");
  }

  @Test
  void normalizesTextBeforeTokenizationLikeTheOfficialPipeline() {
    FakeSopranoEngine engine = new FakeSopranoEngine();

    new SopranoTextToSpeechModel(engine)
        .synthesize(
            "Dr. Smith uses 2 GPUs at 8:05.",
            SpeechSynthesisOptions.builder()
                .sampling(
                    SamplingOptions.builder()
                        .temperature(0.0f)
                        .topP(1.0f)
                        .topK(8)
                        .maxTokens(4)
                        .build())
                .build());

    assertThat(engine.encodedTexts)
        .containsExactly("doctor smith uses two g p u's at eight oh five.");
  }

  @Test
  void streamsReceptiveFieldWindowsBeforeAcousticGenerationFinishes() {
    StreamingSopranoEngine engine = new StreamingSopranoEngine(16);
    List<PcmAudio> chunks = new ArrayList<>();

    new SopranoTextToSpeechModel(engine)
        .synthesize(
            "Hello",
            SpeechSynthesisOptions.builder()
                .sampling(
                    SamplingOptions.builder()
                        .temperature(0.0f)
                        .topP(1.0f)
                        .topK(16)
                        .maxTokens(16)
                        .build())
                .build(),
            new AudioStream() {
              @Override
              public void onAudio(PcmAudio audio) {
                chunks.add(audio);
                engine.advancesAtFirstAudio =
                    engine.advancesAtFirstAudio < 0
                        ? engine.advanceCount
                        : engine.advancesAtFirstAudio;
              }

              @Override
              public void onComplete() {}

              @Override
              public void onError(Throwable failure) {
                throw new AssertionError(failure);
              }
            });

    assertThat(chunks).hasSizeGreaterThan(1);
    assertThat(engine.advancesAtFirstAudio).isLessThan(engine.generatedTokens);
    assertThat(engine.decodedWindows).containsExactly(16, 16);
    assertThat(concatenate(chunks)).containsExactly(engine.completeWave());
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

  private static final class FakeSopranoEngine implements SopranoEngine {
    private final int[] advancedTokens = new int[4];
    private int advancedTokenCount;
    private float[] decodedFeatures;
    private int decodedFrames;
    private boolean closed;
    private float[] initialHidden = new float[] {1.0f, 2.0f};
    private final List<String> encodedTexts = new ArrayList<>();

    @Override
    public int[] encodePrompt(String text) {
      encodedTexts.add(text);
      return new int[] {3, 1, 2};
    }

    @Override
    public Step begin(int[] prompt) {
      return new Step(logits(4), initialHidden);
    }

    @Override
    public Step advance(int token) {
      advancedTokens[advancedTokenCount++] = token;
      return new Step(logits(3), new float[] {3.0f, 4.0f});
    }

    @Override
    public float[] decode(float[] features, int frames) {
      decodedFeatures = features.clone();
      decodedFrames = frames;
      return new float[] {0.25f, -0.25f};
    }

    @Override
    public String modelName() {
      return "soprano-test";
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable("soprano-test");
    }

    @Override
    public int eosToken() {
      return 3;
    }

    @Override
    public int vocabularySize() {
      return 8;
    }

    @Override
    public int hiddenSize() {
      return 2;
    }

    @Override
    public int contextLength() {
      return 16;
    }

    @Override
    public int checkpoint() {
      return 3 + advancedTokenCount;
    }

    @Override
    public int sampleRate() {
      return 32_000;
    }

    @Override
    public int samplesPerToken() {
      return 2048;
    }

    @Override
    public void close() {
      closed = true;
    }

    private static float[] logits(int selected) {
      float[] logits = new float[8];
      Arrays.fill(logits, -1.0f);
      logits[selected] = 1.0f;
      return logits;
    }
  }

  private static final class StreamingSopranoEngine implements SopranoEngine {
    private static final int SAMPLES_PER_TOKEN = 2048;

    private final int generatedTokens;
    private final List<Integer> decodedWindows = new ArrayList<>();
    private int advanceCount;
    private int advancesAtFirstAudio = -1;

    private StreamingSopranoEngine(int generatedTokens) {
      this.generatedTokens = generatedTokens;
    }

    @Override
    public int[] encodePrompt(String text) {
      return new int[] {3, 1, 2};
    }

    @Override
    public Step begin(int[] prompt) {
      return new Step(logits(4), new float[] {0.0f});
    }

    @Override
    public Step advance(int token) {
      advanceCount++;
      int selected = advanceCount < generatedTokens ? 4 : 3;
      return new Step(logits(selected), new float[] {advanceCount});
    }

    @Override
    public float[] decode(float[] features, int frames) {
      decodedWindows.add(frames);
      int length = Math.max(0, frames - 1) * SAMPLES_PER_TOKEN;
      int start = Math.round(features[0]) * SAMPLES_PER_TOKEN;
      float[] result = new float[length];
      for (int index = 0; index < length; index++) {
        result[index] = (start + index) / 100_000.0f;
      }
      return result;
    }

    private float[] completeWave() {
      float[] features = new float[generatedTokens];
      for (int index = 0; index < features.length; index++) {
        features[index] = index;
      }
      return wave(features, generatedTokens);
    }

    private float[] wave(float[] features, int frames) {
      int length = Math.max(0, frames - 1) * SAMPLES_PER_TOKEN;
      int start = Math.round(features[0]) * SAMPLES_PER_TOKEN;
      float[] result = new float[length];
      for (int index = 0; index < length; index++) {
        result[index] = (start + index) / 100_000.0f;
      }
      return result;
    }

    @Override
    public String modelName() {
      return "soprano-stream-test";
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable("soprano-stream-test");
    }

    @Override
    public int eosToken() {
      return 3;
    }

    @Override
    public int vocabularySize() {
      return 16;
    }

    @Override
    public int hiddenSize() {
      return 1;
    }

    @Override
    public int contextLength() {
      return 32;
    }

    @Override
    public int checkpoint() {
      return 3 + advanceCount;
    }

    @Override
    public int sampleRate() {
      return 32_000;
    }

    @Override
    public int samplesPerToken() {
      return SAMPLES_PER_TOKEN;
    }

    @Override
    public void close() {}

    private static float[] logits(int selected) {
      float[] logits = new float[16];
      Arrays.fill(logits, -1.0f);
      logits[selected] = 1.0f;
      return logits;
    }
  }
}
