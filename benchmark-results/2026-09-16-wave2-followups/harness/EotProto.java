import com.integrallis.models.backend.purejava.gguf.*;
import java.lang.foreign.Arena;
import java.nio.file.*;
import java.util.*;
import java.util.regex.*;
public class EotProto {
  record Occ(int pos, int end, int id, String text) {}
  public static void main(String[] a) throws Exception {
    for (String p : a) try (Arena arena = Arena.ofConfined()) {
      GgufMetadata m = GgufParser.parse(Path.of(p), arena).metadata();
      String ct = m.getString("tokenizer.chat_template").orElse(null);
      if (ct == null) continue;
      var toks = m.getStringArray("tokenizer.ggml.tokens").orElse(List.of());
      var types = m.getInt32Array("tokenizer.ggml.token_type").orElse(List.of());
      int eos = m.getUint32("tokenizer.ggml.eos_token_id").orElse(-1);
      List<Occ> occ = new ArrayList<>();
      for (int id=0; id<toks.size(); id++) {
        if (id>=types.size() || types.get(id)!=3) continue;
        String t = toks.get(id); if (t.length()<3) continue;
        for (int i=ct.indexOf(t); i>=0; i=ct.indexOf(t,i+1)) occ.add(new Occ(i,i+t.length(),id,t));
      }
      Matcher em = Pattern.compile("\\beos_token\\b").matcher(ct);
      while (em.find()) if (eos>=0) occ.add(new Occ(em.start(), em.end(), eos, "eos_token"));
      occ.sort(Comparator.comparingInt(Occ::pos).thenComparing(o -> -(o.end()-o.pos())));
      // drop overlapped
      List<Occ> clean = new ArrayList<>(); int lastEnd=-1;
      for (Occ o: occ) if (o.pos()>=lastEnd) { clean.add(o); lastEnd=o.end(); }
      Map<Integer,Integer> votes = new TreeMap<>();
      Matcher cm = Pattern.compile("\\bcontent\\b").matcher(ct);
      while (cm.find()) {
        for (Occ o: clean) if (o.pos()>=cm.end()) { votes.merge(o.id(),1,Integer::sum); break; }
      }
      int gp = ct.lastIndexOf("add_generation_prompt");
      String ruleB = "none";
      if (gp>=0) { Occ before=null, after=null; for (Occ o: clean) { if (o.pos()<gp) before=o; else if (after==null) after=o; }
        ruleB = "before="+(before==null?null:before.id()+"="+toks.get(before.id()))+" opener="+(after==null?null:after.id()+"="+toks.get(after.id())); }
      StringBuilder vs=new StringBuilder(); votes.forEach((k,v)->vs.append(k).append('=').append(toks.get(k)).append(':').append(v).append(' '));
      System.out.println(Path.of(p).getFileName()+"\n  A: "+vs+"\n  B: "+ruleB);
    }
  }
}
