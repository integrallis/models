"""Run CLM over our JevBench cohort and emit the arm format `score_arm.py` already reads.

One row per task: `id<TAB>latency_seconds<TAB>label=p<TAB>...`, exactly what the Java runner
writes, so the same scorer turns both into the same axes. Nothing about scoring lives here --
validity, argmax, ECE and the tier weights all come from JevBench's own modules downstream.

CLM is a bi-encoder, so the mapping from a task to its request is theirs and not ours: the state
head sees the state with the question's instructions appended, and the action head sees each
option's own description. `clm.schema.build_pairs` does that, and it is imported rather than
reimplemented for the same reason the scorer is.

Two arms are worth running:

* `clm-latest` -- their reference projection heads, the model they published
* `clm-raw`    -- their own no-head ablation, cosine in the encoder's own space, as the floor

Latency here is wall-clock per task against a warm local server on the same host, which is the
most favourable number their architecture can post. It is NOT comparable to our CPU latency and is
recorded only so the scorer has a column; any speed claim needs both arms on one host.
"""
import argparse
import json
import sys
import time


def request_of(task):
    """-> (question_key, wire-format question dict) for one cohort row."""
    question = task["question"]
    kind = question["type"]
    criteria = question.get("criteria")
    labels = task["labels"]

    # The rubric every adapter sends, never less of it -- the same normalisation
    # prepare_tasks.py applies for our own arms, so both models read identical declarations.
    if kind == "noul":
        given = criteria or {}
        criteria = {"false": given.get("false", "No"), "true": given.get("true", "Yes")}
    elif kind == "score":
        criteria = list(criteria) if criteria else list(labels)
    else:
        criteria = {key: (value or key) for key, value in criteria.items()}

    return {"type": kind, "instructions": question["instructions"], "criteria": criteria}


def probabilities(answer, task):
    """-> {label: p} over the cohort's own label names, in its own order."""
    labels = task["labels"]
    kind = task["question"]["type"]
    given = answer.get("probabilities")

    if isinstance(given, dict) and given:
        # A choice answers with its own keys. Noul and score answer positionally, so map back.
        if kind == "choice":
            return {label: float(given.get(label, 0.0)) for label in labels}
        keys = list(given)
        if len(keys) == len(labels):
            return {label: float(given[key]) for label, key in zip(labels, keys)}

    if kind == "noul":
        # A bare noul probability is P(true); our cohort's labels are [no, yes] in that order.
        p = float(answer["noul"])
        return {labels[0]: 1.0 - p, labels[1]: p}

    raise ValueError(f"cannot map answer {answer!r} onto labels {labels!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", help="jevbench-120.jsonl")
    parser.add_argument("out", help="arm TSV to write")
    parser.add_argument("--model", default="clm-latest", help="clm-latest or clm-raw")
    parser.add_argument("--emb-url", default="http://127.0.0.1:8090/v1/embeddings")
    parser.add_argument("--local-encoder", default=None,
                        help="run the encoder in-process with transformers instead of calling a "
                             "vLLM endpoint, e.g. Qwen/Qwen3-8B. See local_embedder.py for why "
                             "and for what it deviates from.")
    parser.add_argument("--warmup", type=int, default=3,
                        help="tasks run and discarded first; a cold action cache is not the "
                             "steady state their design is built around")
    arguments = parser.parse_args()

    from clm import Engine

    if arguments.local_encoder:
        from local_embedder import LocalQwenEmbedder

        print(f"  encoder {arguments.local_encoder}, in-process")
        engine = Engine(embedder=LocalQwenEmbedder(arguments.local_encoder))
    else:
        engine = Engine(emb_url=arguments.emb_url)
    if not engine.has(arguments.model):
        print(f"  note: '{arguments.model}' is not a loaded head; engine has "
              f"{sorted(getattr(engine, 'heads', {}))}", file=sys.stderr)

    tasks = [json.loads(line) for line in open(arguments.cohort) if line.strip()]
    print(f"  cohort {len(tasks)} tasks, model {arguments.model}")

    for task in tasks[: arguments.warmup]:
        engine.answer(task["state"], {"q": request_of(task)}, arguments.model)

    written = 0
    with open(arguments.out, "w") as out:
        for task in tasks:
            question = request_of(task)
            started = time.perf_counter()
            result = engine.answer(task["state"], {"q": question}, arguments.model)
            latency = time.perf_counter() - started
            answer = result["answers"]["q"]
            probs = probabilities(answer, task)
            cells = "\t".join(f"{label}={probs[label]!r}" for label in task["labels"])
            out.write(f"{task['id']}\t{latency:.9f}\t{cells}\n")
            written += 1
    print(f"  wrote {written} rows to {arguments.out}")


if __name__ == "__main__":
    main()
