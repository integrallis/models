///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//RUNTIME_OPTIONS --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//DEPS com.integrallis:models-audio:${models.version:0.5.11}

import com.integrallis.models.api.PcmAudio;
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
 * ratio on a quiet machine: {@code ParadeeTextToSpeechModel.load(onnx, config)}, the route a
 * ModelJars user gets (bundled dictionaries, dictionary tier and fallback model), one untimed
 * warm-up on the first sentence, then per-sentence graph time. Writes NNN.wav, NNN.txt and
 * pass.json in the same layout as ParadeeCandidateArmIntegrationTest so derive.py reads it.
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
    long seed = 20261101L + pass;
    try (ParadeeTextToSpeechModel model = ParadeeTextToSpeechModel.load(onnx, config)) {
      for (String chunk : model.phonemeChunks(sentences.get(0), voice)) {
        model.synthesizePhonemes(chunk, voice, 1f, seed);
      }
      List<Map<String, Object>> rows = new ArrayList<>();
      double totalAudio = 0;
      double totalRun = 0;
      boolean allFinite = true;
      for (int i = 0; i < sentences.size(); i++) {
        String text = sentences.get(i);
        long g2pStart = System.nanoTime();
        List<String> chunks = model.phonemeChunks(text, voice);
        double g2pSeconds = (System.nanoTime() - g2pStart) / 1e9;
        List<float[]> parts = new ArrayList<>();
        double runSeconds = 0;
        Double firstChunk = null;
        for (String chunk : chunks) {
          long start = System.nanoTime();
          PcmAudio audio = model.synthesizePhonemes(chunk, voice, 1f, seed);
          double dt = (System.nanoTime() - start) / 1e9;
          runSeconds += dt;
          if (firstChunk == null) {
            firstChunk = dt;
          }
          parts.add(audio.samples());
        }
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
        row.put("phonemes", chunks);
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
      report.put("phonemeSource", "java misaki port (bundled dictionaries, CMUdict tier, fallback model)");
      report.put("voice", voice);
      report.put("voiceIndex", model.voiceIndex(voice));
      report.put("corpus", corpusName);
      report.put("pass", pass);
      report.put("noiseSeed", seed);
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
