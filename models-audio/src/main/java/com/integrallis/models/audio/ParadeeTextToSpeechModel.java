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

import com.integrallis.models.api.AudioStream;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.PcmAudio;
import com.integrallis.models.api.SpeechSynthesisOptions;
import com.integrallis.models.api.TextToSpeechModel;
import com.integrallis.models.audio.phonemizer.MisakiEnglishPhonemizer;
import com.integrallis.models.audio.phonemizer.MisakiLexicon;
import com.integrallis.models.backend.onnx.NoiseSource;
import com.integrallis.models.backend.onnx.OnnxGraphExecutor;
import com.integrallis.models.backend.onnx.OnnxModel;
import com.integrallis.models.backend.onnx.OnnxNode;
import com.integrallis.models.backend.onnx.Tensor;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import java.util.regex.Pattern;

/**
 * Paradee-8M, an English text-to-speech model distilled from Kokoro-82M, run from the publisher's
 * ONNX graph by the pure-Java executor with a Java port of misaki for grapheme-to-phoneme.
 *
 * <p>The pipeline is the publisher's own ({@code paradee/tts.py}): text is split into sentences,
 * each sentence is phonemized with the American or British dictionary according to the voice,
 * characters outside the model vocabulary are dropped, the phoneme ids are capped at 510 and
 * wrapped in the pad id 0, and the graph is run once per sentence with the speed and the voice
 * index. The graph draws its own noise for the unvoiced parts of the sound; a seed in the sampling
 * options makes that draw reproducible, and without one every call uses the same fixed seed.
 */
public final class ParadeeTextToSpeechModel implements TextToSpeechModel {

  /** Output sample rate of every Paradee release. */
  public static final int SAMPLE_RATE = 24_000;

  /** The text side has 512 positions, two of which are the pad tokens. */
  public static final int MAX_PHONEMES = 510;

  static final Pattern SENTENCE_SPLIT = Pattern.compile("(?<=[.!?…])\\s+|\\n+");
  private static final long DEFAULT_SEED = 0L;

  /** Runs the graph: phoneme ids with pads, speed, voice index and a noise source to a waveform. */
  @FunctionalInterface
  interface Synthesizer {
    float[] synthesize(long[] ids, float speed, int voiceIndex, NoiseSource noise);
  }

  /**
   * The publisher's {@code config.json}.
   *
   * @param version the release, for example {@code 1.2}
   * @param sampleRate the output sample rate
   * @param voices the voice names in graph order
   * @param vocab phoneme character to id
   */
  public record Config(
      String version, int sampleRate, List<String> voices, Map<String, Integer> vocab) {

    /** Defensive copies. */
    public Config {
      voices = List.copyOf(voices);
      vocab = Map.copyOf(vocab);
    }

    /** Parses the publisher's {@code config.json}. */
    @SuppressWarnings("unchecked")
    public static Config parse(String json) {
      Map<String, Object> root = ConfigJson.object(json);
      Object vocabValue = root.get("vocab");
      if (!(vocabValue instanceof Map<?, ?> vocabMap)) {
        throw new IllegalArgumentException("config.json has no vocab object");
      }
      Map<String, Integer> vocab = new LinkedHashMap<>();
      for (Map.Entry<?, ?> e : vocabMap.entrySet()) {
        vocab.put((String) e.getKey(), ((Number) e.getValue()).intValue());
      }
      Object voicesValue = root.get("voices");
      List<String> voices =
          voicesValue instanceof List<?> list ? (List<String>) list : List.of("af_heart");
      int sampleRate = root.get("sample_rate") instanceof Number n ? n.intValue() : SAMPLE_RATE;
      String version = String.valueOf(root.getOrDefault("version", "?"));
      return new Config(version, sampleRate, voices, vocab);
    }

    /** Reads and parses a {@code config.json} file. */
    public static Config load(Path path) throws IOException {
      return parse(Files.readString(path, StandardCharsets.UTF_8));
    }
  }

  private final Config config;
  private final Synthesizer synthesizer;
  private final MisakiEnglishPhonemizer american;
  private final MisakiEnglishPhonemizer british;
  private final BackendDiagnostics diagnostics;
  private boolean closed;

  ParadeeTextToSpeechModel(
      Config config,
      Synthesizer synthesizer,
      MisakiEnglishPhonemizer american,
      MisakiEnglishPhonemizer british,
      BackendDiagnostics diagnostics) {
    this.config = Objects.requireNonNull(config, "config");
    this.synthesizer = Objects.requireNonNull(synthesizer, "synthesizer");
    this.american = Objects.requireNonNull(american, "american");
    this.british = british;
    this.diagnostics = Objects.requireNonNull(diagnostics, "diagnostics");
  }

  /**
   * Loads the model from the publisher's ONNX graph and config plus a directory holding misaki's
   * {@code us_gold.json}, {@code us_silver.json}, {@code gb_gold.json} and {@code gb_silver.json}.
   */
  public static ParadeeTextToSpeechModel load(Path onnx, Path configJson, Path lexiconDirectory)
      throws IOException {
    MisakiLexicon us =
        MisakiLexicon.load(
            lexiconDirectory.resolve("us_gold.json"),
            lexiconDirectory.resolve("us_silver.json"),
            false);
    MisakiLexicon gb =
        MisakiLexicon.load(
            lexiconDirectory.resolve("gb_gold.json"),
            lexiconDirectory.resolve("gb_silver.json"),
            true);
    return load(
        onnx, configJson, new MisakiEnglishPhonemizer(us, ""), new MisakiEnglishPhonemizer(gb, ""));
  }

  /** Loads the graph and config with the given phonemizers; {@code british} may be {@code null}. */
  public static ParadeeTextToSpeechModel load(
      Path onnx, Path configJson, MisakiEnglishPhonemizer american, MisakiEnglishPhonemizer british)
      throws IOException {
    Config config = Config.load(configJson);
    OnnxModel model = OnnxModel.load(onnx);
    OnnxGraphExecutor executor = new OnnxGraphExecutor(model);
    Map<String, Integer> operators = new TreeMap<>();
    for (OnnxNode node : model.graph().nodes()) {
      operators.merge(node.opType(), 1, Integer::sum);
    }
    Map<String, String> environment = new LinkedHashMap<>();
    environment.put("backend", "pure-java");
    environment.put("graph", String.valueOf(onnx.getFileName()));
    environment.put("onnxOpset", Long.toString(model.defaultOpset()));
    environment.put("onnxNodes", Integer.toString(model.graph().nodes().size()));
    environment.put("onnxDynamicNodes", Integer.toString(executor.dynamicNodeCount()));
    environment.put("onnxOperators", Integer.toString(operators.size()));
    environment.put("voices", String.join(",", config.voices()));
    environment.put("phonemizer", "misaki-java fba1236, no fallback");
    BackendDiagnostics diagnostics =
        new BackendDiagnostics("pure-java", "paradee-onnx-eager-v1", environment, List.of());
    Synthesizer synthesizer =
        (ids, speed, voiceIndex, noise) -> {
          Map<String, Tensor> feeds = new LinkedHashMap<>();
          feeds.put("input_ids", Tensor.ofLongs(new long[] {1, ids.length}, ids));
          feeds.put("speed", Tensor.ofFloats(new long[] {1}, new float[] {speed}));
          if (config.voices().size() > 1) {
            feeds.put("voice", Tensor.ofLongs(new long[] {1}, new long[] {voiceIndex}));
          }
          Tensor waveform = executor.run(feeds, List.of("waveform"), null, noise).get("waveform");
          return waveform.floats();
        };
    return new ParadeeTextToSpeechModel(config, synthesizer, american, british, diagnostics);
  }

  /** The parsed publisher config. */
  public Config config() {
    return config;
  }

  @Override
  public String modelName() {
    return "Paradee-8M v" + config.version();
  }

  @Override
  public BackendDiagnostics diagnostics() {
    return diagnostics;
  }

  @Override
  public synchronized PcmAudio synthesize(String text, SpeechSynthesisOptions options) {
    requireOpen();
    Objects.requireNonNull(text, "text");
    Objects.requireNonNull(options, "options");
    String voice = resolveVoice(options);
    List<float[]> parts = new ArrayList<>();
    int total = 0;
    NoiseSource noise = noiseFor(options);
    for (String phonemes : phonemeChunks(text, voice)) {
      float[] part = synthesizePhonemes(phonemes, voice, options.speed(), noise);
      parts.add(part);
      total = Math.addExact(total, part.length);
    }
    return concatenate(parts, total);
  }

  /** Emits one PCM chunk per sentence, so a long text starts playing after its first sentence. */
  @Override
  public synchronized void synthesize(
      String text, SpeechSynthesisOptions options, AudioStream stream) {
    Objects.requireNonNull(stream, "stream");
    try {
      requireOpen();
      Objects.requireNonNull(text, "text");
      Objects.requireNonNull(options, "options");
      String voice = resolveVoice(options);
      NoiseSource noise = noiseFor(options);
      for (String phonemes : phonemeChunks(text, voice)) {
        float[] part = synthesizePhonemes(phonemes, voice, options.speed(), noise);
        stream.onAudio(new PcmAudio(config.sampleRate(), 1, part));
      }
    } catch (RuntimeException | Error failure) {
      stream.onError(failure);
      return;
    }
    stream.onComplete();
  }

  /**
   * Synthesizes an already-phonemized string, the publisher's {@code generate_from_phonemes}. This
   * is the route the qualification harness uses to feed the publisher's own phonemes.
   */
  public synchronized PcmAudio synthesizePhonemes(
      String phonemes, String voice, float speed, Long seed) {
    requireOpen();
    int voiceIndex = voiceIndex(voice);
    if (!(speed > 0f) || !Float.isFinite(speed)) {
      throw new IllegalArgumentException("speed must be finite and positive: " + speed);
    }
    NoiseSource noise = NoiseSource.seeded(seed == null ? DEFAULT_SEED : seed);
    float[] samples = synthesizer.synthesize(ids(phonemes), speed, voiceIndex, noise);
    return new PcmAudio(config.sampleRate(), 1, samples);
  }

  private float[] synthesizePhonemes(
      String phonemes, String voice, float speed, NoiseSource noise) {
    return synthesizer.synthesize(ids(phonemes), speed, voiceIndex(voice), noise);
  }

  /** Phoneme ids with the pad token at each end, keeping only characters in the vocabulary. */
  long[] ids(String phonemes) {
    List<Long> ids = new ArrayList<>();
    ids.add(0L);
    int kept = 0;
    for (int i = 0; i < phonemes.length() && kept < MAX_PHONEMES; i++) {
      Integer id = config.vocab().get(String.valueOf(phonemes.charAt(i)));
      if (id != null) {
        ids.add((long) id);
        kept++;
      }
    }
    ids.add(0L);
    long[] array = new long[ids.size()];
    for (int i = 0; i < array.length; i++) {
      array[i] = ids.get(i);
    }
    return array;
  }

  /** The publisher's {@code _chunks}: sentences, each phonemized, long ones cut at a space. */
  List<String> phonemeChunks(String text, String voice) {
    MisakiEnglishPhonemizer phonemizer = phonemizerFor(voice);
    List<String> chunks = new ArrayList<>();
    for (String sentence : SENTENCE_SPLIT.split(text.strip())) {
      if (sentence.isBlank()) {
        continue;
      }
      String ps = phonemizer.phonemize(sentence);
      while (ps.length() > MAX_PHONEMES) {
        int cut = ps.lastIndexOf(' ', MAX_PHONEMES);
        cut = cut > 0 ? cut : MAX_PHONEMES;
        chunks.add(ps.substring(0, cut));
        ps = ps.substring(cut).stripLeading();
      }
      if (!ps.isEmpty()) {
        chunks.add(ps);
      }
    }
    return chunks;
  }

  /** British dictionary for the {@code b*} voices, American for the rest, as the publisher does. */
  MisakiEnglishPhonemizer phonemizerFor(String voice) {
    if (voice.startsWith("b")) {
      if (british == null) {
        throw new IllegalStateException(
            "voice " + voice + " needs the British dictionary, which was not loaded");
      }
      return british;
    }
    return american;
  }

  private String resolveVoice(SpeechSynthesisOptions options) {
    String voice = options.voice() == null ? config.voices().get(0) : options.voice();
    voiceIndex(voice);
    return voice;
  }

  int voiceIndex(String voice) {
    int index = config.voices().indexOf(voice);
    if (index < 0) {
      throw new IllegalArgumentException(
          modelName() + " has no voice '" + voice + "'; its voices are " + config.voices());
    }
    return index;
  }

  private static NoiseSource noiseFor(SpeechSynthesisOptions options) {
    Long seed = options.sampling() == null ? null : options.sampling().seed();
    return NoiseSource.seeded(seed == null ? DEFAULT_SEED : seed);
  }

  private PcmAudio concatenate(List<float[]> parts, int total) {
    float[] merged = new float[total];
    int offset = 0;
    for (float[] part : parts) {
      System.arraycopy(part, 0, merged, offset, part.length);
      offset += part.length;
    }
    return new PcmAudio(config.sampleRate(), 1, merged);
  }

  @Override
  public synchronized void close() {
    closed = true;
  }

  private void requireOpen() {
    if (closed) {
      throw new IllegalStateException("model is closed");
    }
  }
}
