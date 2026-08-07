#!/usr/bin/env python3
"""Compare embedding models as task classifiers on the held-out split.

For each candidate this embeds every prompt in benchmark-prompts.tsv with
llama.cpp, classifies each eval prompt by its nearest training neighbour
(cosine), and reports accuracy. llama.cpp is used because the equivalence gate
already proves our runtime reproduces its vectors, so model *selection* can run
on the fast path and only the shipped artifact needs our own runtime.

Also reports the distance at which correct and incorrect matches sit, which is
what the classifier's unclassified threshold has to separate.

Run:  python3 bakeoff.py [--models a,b] [--limit-train N]
"""

import argparse
import json
import math
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).parent
LLAMA = pathlib.Path(
    "/home/bsb/Code/java-ai/research/repos/llama.cpp/build-nollamafile/bin/llama-embedding")
MODELS_DIR = pathlib.Path.home() / ".jvllm" / "models"

# name -> (gguf path, pooling). Pooling matters: decoder-only embedding models
# need last-token pooling, BERT-style encoders need mean or cls.
CANDIDATES = {
    "qwen3-emb-0.6b":     (MODELS_DIR / "Qwen3-Embedding-0.6B-Q8_0.gguf", "last"),
    "qwen3-emb-4b":       (MODELS_DIR / "Qwen3-Embedding-4B-Q4_K_M.gguf", "last"),
    "embeddinggemma-300m": (MODELS_DIR / "bakeoff" / "embeddinggemma-300M-Q8_0.gguf", "mean"),
    "granite-107m-multi": (MODELS_DIR / "bakeoff"
                           / "granite-embedding-107m-multilingual-Q8_0.gguf", "cls"),
    "nomic-v1.5":         (MODELS_DIR / "bakeoff" / "nomic-embed-text-v1.5-Q8_0.gguf", "mean"),
    "minilm-l6-v2":       (MODELS_DIR / "bakeoff" / "all-MiniLM-L6-v2-Q8_0.gguf", "mean"),
    "jina-v2-code":       (MODELS_DIR / "bakeoff" / "jina-embeddings-v2-base-code-Q8_0.gguf", "mean"),
    "lfm2.5-350m":        (MODELS_DIR / "bakeoff" / "LFM2.5-Embedding-350M-Q8_0.gguf", "last"),
}


def load_corpus(limit_train=None):
    train, evaluation = [], []
    for line in (HERE / "benchmark-prompts.tsv").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        split, task, _source, prompt = line.split("\t", 3)
        (train if split == "train" else evaluation).append((task, prompt))
    if limit_train:
        capped, seen = [], {}
        for task, prompt in train:
            if seen.get(task, 0) < limit_train:
                capped.append((task, prompt))
                seen[task] = seen.get(task, 0) + 1
        train = capped
    return train, evaluation


def embed(model_path, pooling, prompts):
    """Runs llama-embedding over one prompt per line, returning float lists."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(prompts))
        prompt_file = handle.name
    command = [str(LLAMA), "-m", str(model_path), "-f", prompt_file,
               "--pooling", pooling, "--embd-output-format", "array",
               "-c", "2048", "-b", "2048", "--no-warmup", "-ngl", "0", "-t", "8"]
    done = subprocess.run(command, capture_output=True, text=True, timeout=7200)
    if done.returncode != 0:
        raise RuntimeError("llama-embedding failed: " + done.stderr[-600:])
    start = done.stdout.find("[")
    vectors = json.loads(done.stdout[start:])
    if len(vectors) != len(prompts):
        raise RuntimeError("got %d vectors for %d prompts" % (len(vectors), len(prompts)))
    return vectors


def unit(vector):
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector


def evaluate(name, train, evaluation):
    path, pooling = CANDIDATES[name]
    if not path.exists():
        print("  %-20s SKIP (missing %s)" % (name, path.name))
        return None
    prompts = [p for _, p in train] + [p for _, p in evaluation]
    vectors = [unit(v) for v in embed(path, pooling, prompts)]
    cut = len(train)
    train_vectors, eval_vectors = vectors[:cut], vectors[cut:]
    dimension = len(vectors[0])

    correct, correct_d, wrong_d = 0, [], []
    per_task = {}
    for index, (want, _prompt) in enumerate(evaluation):
        query = eval_vectors[index]
        best, best_task = -2.0, None
        for position, candidate in enumerate(train_vectors):
            score = sum(a * b for a, b in zip(query, candidate))
            if score > best:
                best, best_task = score, train[position][0]
        hit = best_task == want
        correct += hit
        (correct_d if hit else wrong_d).append(1.0 - best)
        bucket = per_task.setdefault(want, [0, 0])
        bucket[1] += 1
        bucket[0] += hit

    accuracy = correct / len(evaluation)
    correct_d.sort()
    print("  %-20s dim=%-5d accuracy=%.4f (%d/%d)  correct-dist p50=%.3f p95=%.3f"
          % (name, dimension, accuracy, correct, len(evaluation),
             correct_d[len(correct_d) // 2] if correct_d else float("nan"),
             correct_d[int(len(correct_d) * 0.95)] if correct_d else float("nan")))
    return {"model": name, "dimension": dimension, "accuracy": accuracy,
            "correct": correct, "total": len(evaluation),
            "per_task": {t: {"correct": c, "total": n} for t, (c, n) in sorted(per_task.items())},
            "correct_distance_p50": correct_d[len(correct_d) // 2] if correct_d else None,
            "correct_distance_p95": correct_d[int(len(correct_d) * 0.95)] if correct_d else None,
            "wrong_distance_min": min(wrong_d) if wrong_d else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(CANDIDATES))
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--out", default=str(HERE / "bakeoff-results.json"))
    args = parser.parse_args()

    train, evaluation = load_corpus(args.limit_train)
    print("corpus: %d train, %d eval, %d tasks"
          % (len(train), len(evaluation), len({t for t, _ in train})))

    results = []
    for name in args.models.split(","):
        name = name.strip()
        if name not in CANDIDATES:
            print("  unknown candidate:", name, file=sys.stderr)
            continue
        try:
            result = evaluate(name, train, evaluation)
            if result:
                results.append(result)
                # Written after every model: each takes minutes, and a run interrupted
                # partway would otherwise report nothing at all.
                pathlib.Path(args.out).write_text(json.dumps(
                    {"train": len(train), "eval": len(evaluation),
                     "results": sorted(results, key=lambda r: -r["accuracy"])}, indent=2) + "\n")
        except Exception as error:
            print("  %-20s ERROR %s" % (name, str(error)[:200]))

    results.sort(key=lambda r: -r["accuracy"])
    pathlib.Path(args.out).write_text(json.dumps(
        {"train": len(train), "eval": len(evaluation), "results": results}, indent=2) + "\n")
    print("\nranking:")
    for r in results:
        print("  %-20s %.4f  dim=%d" % (r["model"], r["accuracy"], r["dimension"]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
