/*
 * Copyright 2025-2026 Integrallis Software, LLC
 * SPDX-License-Identifier: Apache-2.0
 */
package com.integrallis.models.bench;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.nativekernel.RustFfmBackend;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.*;

/** Remote-only real-weight parity probe; inference evidence is written even when parity fails. */
public final class FlintKernelParity {
  public static void main(String[] args) throws Exception {
    if(args.length!=3) throw new IllegalArgumentException("MODEL REFERENCE_PROPERTIES NEW_REPORT");
    Path output=Path.of(args[2]);
    if(Files.exists(output)) throw new IllegalArgumentException("output already exists");
    Properties reference=new Properties();
    try(var in=Files.newInputStream(Path.of(args[1]))) { reference.load(in); }
    System.setProperty("models.purejava.maxContextLength","2048");
    var cases=new ArrayList<Map<String,Object>>();
    var javaLogits=new HashMap<String,List<float[]>>();
    boolean passed=true;
    for(String backendName:List.of("pure-java","rust-ffm")) {
      try(InferenceBackend backend=backendName.equals("pure-java") ? PureJavaBackend.load(Path.of(args[0])) : RustFfmBackend.load(Path.of(args[0]))) {
        for(String kind:List.of("completion","chat")) {
          backend.reset();
          int[] expectedTokens=ints(reference,kind+".tokens");
          int[] expectedContinuation=ints(reference,kind+".continuation");
          int[] tokens=kind.equals("completion") ? backend.tokenizer().encode(reference.getProperty(kind+".prompt")) : backend.tokenizer().encode(ChatTemplate.FLINT.render(List.of(ChatMessage.user(reference.getProperty(kind+".prompt")))));
          boolean tokenizer=Arrays.equals(expectedTokens,tokens);
          long start=System.nanoTime();
          float[] logits=backend.prefill(tokens,0);
          var captured=new ArrayList<float[]>();
          var actual=new ArrayList<Integer>();
          double maxAbsolute=0, squaredError=0, squaredReference=0;
          boolean finite=true;
          for(int step=0;step<expectedContinuation.length;step++) {
            captured.add(logits.clone());
            int best=0;
            for(int i=0;i<logits.length;i++) {
              finite &= Float.isFinite(logits[i]);
              if(logits[i]>logits[best])best=i;
              if(backendName.equals("rust-ffm")) {
                double expected=javaLogits.get(kind).get(step)[i];
                double delta=logits[i]-expected;
                maxAbsolute=Math.max(maxAbsolute,Math.abs(delta));
                squaredError+=delta*delta;squaredReference+=expected*expected;
              }
            }
            actual.add(best);
            if(step+1<expectedContinuation.length) logits=backend.forward(expectedContinuation[step],tokens.length+step);
          }
          if(backendName.equals("pure-java"))javaLogits.put(kind,captured);
          double relativeRms=Math.sqrt(squaredError/Math.max(1e-30,squaredReference));
          boolean tokenMatch=Arrays.equals(actual.stream().mapToInt(Integer::intValue).toArray(),expectedContinuation);
          boolean pass=tokenizer && tokenMatch && finite && maxAbsolute<=0.05 && relativeRms<=0.002;
          passed &= pass;
          var entry=new LinkedHashMap<String,Object>();
          entry.put("backend",backendName);entry.put("case",kind);entry.put("tokenizerMatches",tokenizer);
          entry.put("continuationMatches",tokenMatch);entry.put("actualContinuation",actual);
          entry.put("teacherForcedReferenceContinuation",expectedContinuation);entry.put("finiteLogits",finite);
          entry.put("maxAbsoluteLogitDifferenceFromJava",maxAbsolute);entry.put("relativeRmsLogitDifferenceFromJava",relativeRms);
          entry.put("elapsedMillis",(System.nanoTime()-start)/1e6);entry.put("passed",pass);
          entry.put("backendDiagnostics",backend.diagnostics());cases.add(entry);
        }
      }
    }
    var result=new LinkedHashMap<String,Object>();result.put("kind","flint-native-parity-v1");
    result.put("maximumAbsoluteLogitDifference",0.05);result.put("maximumRelativeRmsLogitDifference",0.002);
    result.put("passed",passed);result.put("cases",cases);
    Files.write(output,new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT).writeValueAsBytes(result),StandardOpenOption.CREATE_NEW);
    if(!passed)throw new IllegalStateException("Flint numerical parity failed; inspect "+output);
  }
  private static int[] ints(Properties p,String key) { return Arrays.stream(p.getProperty(key).split(",")).map(String::trim).mapToInt(Integer::parseInt).toArray(); }
}
