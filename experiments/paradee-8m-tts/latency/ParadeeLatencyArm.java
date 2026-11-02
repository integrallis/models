///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//RUNTIME_OPTIONS --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//DEPS com.integrallis:models-audio:${models.version:0.5.11}

import com.integrallis.models.api.AudioStream;
import com.integrallis.models.api.PcmAudio;
import com.integrallis.models.api.SpeechSynthesisOptions;
import com.integrallis.models.api.WavEncoder;
import com.integrallis.models.audio.ParadeeTextToSpeechModel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The candidate arm against the RELEASED library, for the latency block and the throughput
 * ratio on a quiet machine: {@code ParadeeTextToSpeechModel.load(onnx, config)} and the public
 * streaming {@code synthesize}, the route a ModelJars user gets (bundled dictionaries, dictionary
 * tier and fallback model). One untimed warm-up on the first sentence, then per sentence the time
 * to the first audio callback and the time to completion. Writes NNN.wav, NNN.txt and pass.json
 * in the layout derive.py reads; the phonemizer time is inside the first-audio time here, as it
 * is for a user.
 *
 * usage: jbang -Dmodels.version=0.5.11 ParadeeLatencyArm.java ONNX CONFIG CORPUS.json VOICE PASS OUT_DIR
 */
public class ParadeeLatencyArm {
  public static void main(String[] args) throws Exception {
    Path onnx = Path.of(args[0]);
    Path config = Path.of(args[1]);
    String corpusJson = Files.readString(Path.of(args[2]), StandardCharsets.UTF_8);
    String voice = args[3];
    int pass = Integer.parseInt(args[4]);
    Path out = Path.of(args[5]);
    Files.createDirectories(out);
    List<String> sentences = sentences(corpusJson);
    String corpusName = field(corpusJson, "name");
    SpeechSynthesisOptions options = SpeechSynthesisOptions.builder().voice(voice).build();
    try (ParadeeTextToSpeechModel model = ParadeeTextToSpeechModel.load(onnx, config)) {
      model.synthesize(sentences.get(0), options);
      List<Map<String, Object>> rows = new ArrayList<>();
      double totalAudio = 0;
      double totalRun = 0;
      boolean allFinite = true;
      for (int i = 0; i < sentences.size(); i++) {
        String text = sentences.get(i);
        List<float[]> parts = new ArrayList<>();
        long[] firstAudioNanos = {-1L};
        long[] completeNanos = {-1L};
        Throwable[] failure = {null};
        long start = System.nanoTime();
        model.synthesize(text, options, new AudioStream() {
          @Override public void onAudio(PcmAudio audio) {
            if (firstAudioNanos[0] < 0) firstAudioNanos[0] = System.nanoTime();
            parts.add(audio.samples());
          }
          @Override public void onComplete() { completeNanos[0] = System.nanoTime(); }
          @Override public void onError(Throwable t) { failure[0] = t; completeNanos[0] = System.nanoTime(); }
        });
        if (failure[0] != null) throw new IllegalStateException("sentence " + i + " failed", failure[0]);
        double runSeconds = (completeNanos[0] - start) / 1e9;
        Double firstChunk = firstAudioNanos[0] < 0 ? null : (firstAudioNanos[0] - start) / 1e9;
        double g2pSeconds = 0.0;
        int total = 0;
        for (float[] part : parts) {
          total += part.length;
        }
        float[] merged = new float[total];
        int offset = 0;
        for (float[] part : parts) {
          System.arraycopy(part, 0, merged, offset, part.length);
          offset += part.length;
        }
        boolean finite = true;
        for (float s : merged) {
          finite &= Float.isFinite(s);
        }
        allFinite &= finite;
        PcmAudio audio = new PcmAudio(ParadeeTextToSpeechModel.SAMPLE_RATE, 1, merged);
        Files.write(out.resolve(String.format("%03d.wav", i)), WavEncoder.pcm16(audio));
        Files.writeString(out.resolve(String.format("%03d.txt", i)), text + "\n", StandardCharsets.UTF_8);
        double audioSeconds = merged.length / (double) ParadeeTextToSpeechModel.SAMPLE_RATE;
        totalAudio += audioSeconds;
        totalRun += runSeconds;
        Map<String, Object> row = new LinkedHashMap<>();
        row.put("index", i);
        row.put("text", text);
        row.put("audioSeconds", audioSeconds);
        row.put("runSeconds", runSeconds);
        row.put("g2pSeconds", g2pSeconds);
        row.put("timeToFirstAudioSeconds", g2pSeconds + (firstChunk == null ? 0.0 : firstChunk));
        row.put("realTimeFactor", audioSeconds > 0 ? runSeconds / audioSeconds : null);
        row.put("finite", finite);
        row.put("samples", merged.length);
        rows.add(row);
      }
      Map<String, Object> report = new LinkedHashMap<>();
      report.put("arm", "candidate");
      report.put("runtime", "released com.integrallis:models-audio:" + System.getProperty("models.version", "0.5.11") + " " + model.diagnostics().environment() + " via " + model.modelName());
      report.put("onnxruntimeVersion", null);
      report.put("intraOpThreads", 1);
      report.put("interOpThreads", 1);
      report.put("model", "onnx/paradee_int8.onnx");
      report.put("phonemeSource", "java misaki port (bundled dictionaries, CMUdict tier, fallback model); public streaming synthesize");
      report.put("voice", voice);
      report.put("voiceIndex", model.config().voices().indexOf(voice));
      report.put("corpus", corpusName);
      report.put("pass", pass);
      report.put("noiseSeed", "model default");
      report.put("java", System.getProperty("java.version"));
      report.put("machine", System.getProperty("os.name") + " " + System.getProperty("os.version") + " " + System.getProperty("os.arch") + " " + Runtime.getRuntime().availableProcessors() + " cpus");
      report.put("sentences", rows.size());
      report.put("totalAudioSeconds", totalAudio);
      report.put("totalRunSeconds", totalRun);
      report.put("realTimeMultiple", totalRun > 0 ? totalAudio / totalRun : null);
      report.put("allFinite", allFinite);
      report.put("rows", rows);
      Files.writeString(out.resolve("pass.json"), toJson(report), StandardCharsets.UTF_8);
      System.out.printf("PARADEE_ARM %s %s pass%d: %.1fs audio in %.2fs -> %.2fx realtime%n", corpusName, voice, pass, totalAudio, totalRun, totalAudio / totalRun);
    }
  }

  // A minimal reader for the corpus JSON: {"name": "...", "source": ..., "sentences": ["...", ...]}
  static List<String> sentences(String json) {
    int k = json.indexOf("\"sentences\"");
    int start = json.indexOf('[', k);
    List<String> out = new ArrayList<>();
    StringBuilder cur = null;
    for (int i = start + 1; i < json.length(); i++) {
      char c = json.charAt(i);
      if (cur == null) {
        if (c == '"') cur = new StringBuilder();
        else if (c == ']') break;
        continue;
      }
      if (c == '\\') {
        char n = json.charAt(++i);
        cur.append(switch (n) { case 'n' -> '\n'; case 't' -> '\t'; case 'u' -> (char) Integer.parseInt(json.substring(i + 1, i + 5), 16); default -> n; });
        if (n == 'u') i += 4;
      } else if (c == '"') {
        out.add(cur.toString());
        cur = null;
      } else {
        cur.append(c);
      }
    }
    return out;
  }

  static String field(String json, String name) {
    int k = json.indexOf("\"" + name + "\"");
    int q = json.indexOf('"', json.indexOf(':', k) + 1);
    return json.substring(q + 1, json.indexOf('"', q + 1));
  }

  static String toJson(Object v) {
    if (v == null) return "null";
    if (v instanceof String s) return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n") + "\"";
    if (v instanceof Number || v instanceof Boolean) return String.valueOf(v);
    if (v instanceof List<?> l) { StringBuilder b = new StringBuilder("["); for (int i = 0; i < l.size(); i++) { if (i > 0) b.append(","); b.append(toJson(l.get(i))); } return b.append("]").toString(); }
    if (v instanceof Map<?, ?> m) { StringBuilder b = new StringBuilder("{"); boolean first = true; for (Map.Entry<?, ?> e : m.entrySet()) { if (!first) b.append(","); first = false; b.append(toJson(String.valueOf(e.getKey()))).append(":").append(toJson(e.getValue())); } return b.append("}").toString(); }
    return toJson(String.valueOf(v));
  }
}
