#!/usr/bin/env python3
"""Seeded-parity oracle: run the publisher's graph with its own noise draw captured.

For each sentence in corpus/card-samples-5.json and each voice, the fp32 and int8 graphs are run
by onnxruntime with the RandomNormalLike output exposed as an extra graph output, so the exact
noise tensor the oracle used is saved beside the waveform. The Java executor consumes that tensor
instead of drawing its own, which makes bit-for-bit parity a defined question. The graph's
arithmetic is not modified: exposing an intermediate as an output changes nothing upstream.

For the first sentence and voice only, every node output of the fp32 graph is also saved
(intermediates.npz), for per-operator comparison.

usage: oracle_dump.py --out DIR
"""
import argparse, hashlib, json, pathlib
import numpy as np, onnx, onnxruntime as ort
import pins

pins.prepare_espeak()
from paradee import Paradee  # noqa: E402


def session_with_outputs(model_path: pathlib.Path, extra_outputs, threads=1):
    model = onnx.load(str(model_path))
    existing = {o.name for o in model.graph.output}
    for name in extra_outputs:
        if name not in existing:
            model.graph.output.append(onnx.ValueInfoProto(name=name))
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"]), model


def noise_output_name(model: onnx.ModelProto) -> str:
    nodes = [n for n in model.graph.node if n.op_type == "RandomNormalLike"]
    assert len(nodes) == 1, f"expected one RandomNormalLike, found {len(nodes)}"
    return nodes[0].output[0]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    corpus = pins.load_corpus(str(pathlib.Path(__file__).resolve().parent.parent / "corpus" / "card-samples-5.json"))
    config_path = pins.artifact("config.json")
    manifest = {"revision": pins.REVISION, "onnxruntimeVersion": ort.__version__, "cases": []}
    g2p = Paradee(version="1.2", quantized=True, model_path=str(pins.artifact("onnx/paradee_int8.onnx")), config_path=str(config_path))
    for graph_name in ("onnx/paradee.onnx", "onnx/paradee_int8.onnx"):
        model_path = pins.artifact(graph_name)
        base = onnx.load(str(model_path))
        noise = noise_output_name(base)
        all_outputs = [o for n in base.graph.node for o in n.output if o]
        sess, _ = session_with_outputs(model_path, [noise])
        full_sess = None
        for vi, voice in enumerate(pins.VOICES):
            for si, text in enumerate(corpus["sentences"]):
                ps = g2p.phonemize(text, voice)
                ids = [0] + [g2p.vocab[c] for c in ps if c in g2p.vocab][:510] + [0]
                feed = {"input_ids": np.array([ids], np.int64), "speed": np.array([1.0], np.float32), "voice": np.array([vi], np.int64)}
                waveform, noise_tensor = sess.run(["waveform", noise], feed)
                tag = f"{graph_name.split('/')[-1].replace('.onnx','')}_{voice}_{si}"
                np.savez(out / f"{tag}.npz", input_ids=feed["input_ids"], speed=feed["speed"], voice=feed["voice"], noise=noise_tensor, waveform=waveform)
                case = {"tag": tag, "graph": graph_name, "voice": voice, "voiceIndex": vi, "sentence": text, "phonemes": ps,
                        "ids": ids, "noiseShape": list(noise_tensor.shape), "waveformSamples": int(waveform.shape[-1]),
                        "waveformSha256": hashlib.sha256(waveform.tobytes()).hexdigest()}
                if graph_name.endswith("paradee.onnx") and vi == 0 and si == 0:
                    full_sess, _ = session_with_outputs(model_path, all_outputs)
                    names = [o.name for o in full_sess.get_outputs()]
                    values = full_sess.run(names, feed)
                    np.savez_compressed(out / f"{tag}.intermediates.npz", **{n: v for n, v in zip(names, values) if isinstance(v, np.ndarray)})
                    case["intermediates"] = f"{tag}.intermediates.npz"; case["intermediateCount"] = len(names)
                manifest["cases"].append(case)
                print(tag, "ids", len(ids), "noise", noise_tensor.shape, "samples", waveform.shape[-1])
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
