"""Per-node profile of onnxruntime on Paradee: same corpus slice, one thread, optimizations on/off."""
import json, sys, time, collections, pathlib
sys.path.insert(0, ".")
import onnxruntime as ort
import pins
pins.prepare_espeak()
from paradee import Paradee

level, out, n = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
Real = ort.InferenceSession
def Patched(path, so=None, providers=None):
    so = so or ort.SessionOptions()
    so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    so.graph_optimization_level = {"all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                                   "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL}[level]
    so.optimized_model_filepath = str(out / "optimized.onnx")
    so.enable_profiling = True; so.profile_file_prefix = str(out / "prof")
    return Real(path, so, providers=providers)
ort.InferenceSession = Patched
corpus = pins.load_corpus("../corpus/harvard-200.json")
tts = Paradee(version="1.2", quantized=True, model_path=str(pins.artifact("onnx/paradee_int8.onnx")),
              config_path=str(pins.artifact("config.json")), threads=1)
sents = corpus["sentences"][:n]
for ps in tts._chunks(sents[0], "af_heart"): tts.generate_from_phonemes(ps, 1.0, "af_heart")
audio = run = 0.0
for s in sents:
    for ps in tts._chunks(s, "af_heart"):
        t = time.perf_counter(); y = tts.generate_from_phonemes(ps, 1.0, "af_heart"); run += time.perf_counter() - t
        audio += len(y) / 24000
prof = tts.session.end_profiling()
ev = json.load(open(prof))
by_op = collections.Counter(); calls = collections.Counter()
for e in ev:
    if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
        op = e["args"].get("op_name", "?"); by_op[op] += e["dur"]; calls[op] += 1
tot = sum(by_op.values())
json.dump({"level": level, "sentences": n, "audioSeconds": audio, "runSeconds": run, "realTimeMultiple": audio/run,
           "kernelMicros": tot, "byOp": [{"op": k, "micros": v, "share": v/tot, "calls": calls[k]} for k, v in by_op.most_common()]},
          open(out / "summary.json", "w"), indent=1)
print(f"{level}: {audio:.1f}s audio in {run:.2f}s -> {audio/run:.1f}x; kernel share by op:")
for k, v in by_op.most_common(15): print(f"  {k:28s} {100*v/tot:5.1f}%  calls={calls[k]}")
