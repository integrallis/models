import java.nio.file.*;
import java.util.*;
import jdk.jfr.consumer.*;

/** Reads recorded JFR events; does not load model weights or execute inference. */
public class SummarizeJfr {
  static void add(Map<String,Long> map,String key,long value) { map.merge(key,value,Long::sum); }
  static String json(Object value) {
    if (value instanceof Map<?,?> map) {
      var items=new ArrayList<String>();
      map.forEach((k,v)->items.add(json(k.toString())+":"+json(v)));
      return "{"+String.join(",",items)+"}";
    }
    if (value instanceof Number || value instanceof Boolean) return value.toString();
    return "\""+value.toString().replace("\\","\\\\").replace("\"","\\\"")+"\"";
  }
  public static void main(String[] args) throws Exception {
    var top=new TreeMap<String,Long>(); var groups=new TreeMap<String,Long>();
    var allocations=new TreeMap<String,Long>(); var sites=new TreeMap<String,Long>();
    var kinds=new TreeMap<String,Long>();
    long pauseNanos=0;
    try (var recording=new RecordingFile(Path.of(args[0]))) {
      while(recording.hasMoreEvents()) {
        var event=recording.readEvent(); String kind=event.getEventType().getName();
        if (!Set.of("jdk.ExecutionSample","jdk.NativeMethodSample","jdk.ObjectAllocationSample","jdk.GarbageCollection").contains(kind)) continue;
        add(kinds,kind,1);
        if (kind.equals("jdk.GarbageCollection")) { pauseNanos+=event.getDuration("sumOfPauses").toNanos();continue; }
        var trace=event.getStackTrace(); if(trace==null || trace.getFrames().isEmpty())continue;
        var frames=trace.getFrames().stream().map(f->f.getMethod().getType().getName()+"."+f.getMethod().getName()).toList();
        if (kind.equals("jdk.ObjectAllocationSample")) {
          long weight=event.getLong("weight");
          add(allocations,event.getClass("objectClass").getName(),weight);
          add(sites,frames.stream().filter(f->f.startsWith("com.integrallis")).findFirst().orElse(frames.getFirst()),weight);
        } else {
          add(top,frames.getFirst(),1);
          String group=frames.stream().anyMatch(f->f.contains("TensorOps.geluErf") || f.contains("TensorOps.erf")) ? "gelu"
            : frames.stream().anyMatch(f->f.contains("lambda$attend")) ? "attention"
            : frames.stream().anyMatch(f->f.contains("lambda$project")) ? "projection" : "other";
          add(groups,group,1);
        }
      }
    }
    var report=new LinkedHashMap<String,Object>();
    report.put("events",kinds);report.put("cpuSamplesByOperation",groups);report.put("topMethods",top);
    report.put("estimatedAllocationBytesByType",allocations);report.put("estimatedAllocationBytesBySite",sites);
    report.put("sumGcPauseMs",pauseNanos/1e6);report.put("catalogQualified",false);
    report.put("note","CPU sample counts include load and inference across multiple threads; they are not wall-time percentages. Allocation weights are sampling estimates.");
    System.out.println(json(report));
  }
}
