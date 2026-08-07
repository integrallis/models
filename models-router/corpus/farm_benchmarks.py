#!/usr/bin/env python3
"""Farm task-classification prompts from public benchmark datasets.

Writes two files next to this script:

  benchmark-prompts.tsv   split<TAB>task<TAB>source<TAB>prompt
  sources.json            provenance and licence for every source used

Only the *input* side of each benchmark is taken; reference answers are
discarded. Sources whose rows are documents or sentence pairs rather than user
requests get an instruction wrapper, which `sources.json` records as
`wrapped: true` so the transformation stays visible.

Datasets are pulled as whole parquet files rather than paged through the rows
API: paging a category-filtered source only ever sees the first few pages, which
silently starves tasks whose rows sit late in the file.

Requires pyarrow.  Run:  python3 farm_benchmarks.py [--per-task 250]
"""

import argparse
import hashlib
import json
import pathlib
import random
import re
import sys
import urllib.parse
import urllib.request

import pyarrow.parquet as pq

PARQUET = "https://datasets-server.huggingface.co/parquet"
UA = {"User-Agent": "models-router-corpus/1.0"}
CACHE = pathlib.Path(__file__).parent / ".cache"

SUMMARY_WRAPPERS = [
    "Summarize the following: {}",
    "Give me a short summary of this text: {}",
    "TL;DR of the passage below: {}",
    "Condense this into a couple of sentences: {}",
]
TRANSLATE_WRAPPERS = [
    "Translate this into {lang}: {text}",
    "How do you say this in {lang}? {text}",
    "Render the following in {lang}: {text}",
    "Put this sentence into {lang}: {text}",
]
LANGS = {"es": "Spanish", "fr": "French", "it": "Italian",
         "pt": "Portuguese", "nl": "Dutch", "pl": "Polish",
         "sv": "Swedish", "ro": "Romanian"}


def download(url):
    """Fetches a URL into the local cache, returning the cached path."""
    CACHE.mkdir(exist_ok=True)
    target = CACHE / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".parquet")
    if target.exists():
        return target
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=300) as response:
        target.write_bytes(response.read())
    return target


def parquet_rows(dataset, config, split, columns=None):
    """Yields every row of a split as a dict, reading whole parquet shards."""
    query = urllib.parse.urlencode({"dataset": dataset})
    request = urllib.request.Request(PARQUET + "?" + query, headers=UA)
    with urllib.request.urlopen(request, timeout=120) as response:
        listing = json.load(response).get("parquet_files", [])
    shards = [f for f in listing if f["config"] == config and f["split"] == split]
    if not shards:
        print("  ! no parquet for %s/%s/%s" % (dataset, config, split), file=sys.stderr)
        return
    for shard in shards:
        table = pq.read_table(download(shard["url"]), columns=columns)
        for batch in table.to_batches(1024):
            for row in batch.to_pylist():
                yield row


WHITESPACE = re.compile(r"\s+")


def clean(text, low=20, high=400):
    """Collapses whitespace; returns None when the result is unusable."""
    if not isinstance(text, str):
        return None
    text = WHITESPACE.sub(" ", text).strip()
    if not low <= len(text) <= high:
        return None
    if text.count("http") > 2:
        return None
    return text


def mbpp(wanted):
    """MBPP task descriptions are already natural-language coding requests."""
    request = urllib.request.Request(
        "https://raw.githubusercontent.com/google-research/google-research/"
        "master/mbpp/mbpp.jsonl", headers=UA)
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read().decode("utf-8")
    out = []
    for line in body.splitlines():
        if line.strip():
            text = clean(json.loads(line).get("text"))
            if text:
                out.append(text)
        if len(out) >= wanted:
            break
    return out


def field(dataset, config, split, name, wanted, keep=None, low=20, high=400):
    """Pulls one text field, optionally gated by a per-row predicate."""
    out = []
    for row in parquet_rows(dataset, config, split):
        if keep and not keep(row):
            continue
        text = clean(row.get(name), low, high)
        if text:
            out.append(text)
        if len(out) >= wanted:
            break
    return out


def hermes(wanted):
    """First human turn of a function-calling conversation.

    Tool-use requests run long — they name the systems and parameters involved —
    so the upper length bound is raised rather than discarding most of the split.
    """
    out = []
    for config, split in (("func_calling", "train"),
                          ("func_calling_singleturn", "train"),
                          ("glaive_func_calling", "train")):
        for row in parquet_rows("NousResearch/hermes-function-calling-v1", config, split):
            turns = row.get("conversations") or []
            for turn in turns:
                if isinstance(turn, dict) and turn.get("from") in ("human", "user"):
                    text = clean(turn.get("value"), 20, 700)
                    if text:
                        out.append(text)
                    break
            if len(out) >= wanted:
                return out
    return out


def summarization(wanted, rng):
    """XSum documents wrapped in an explicit summarize instruction.

    Dolly's `summarization` category is not used: every one of its rows carries the
    text in a separate `context` field, and the instruction alone reads as ordinary
    question-answering ("What is a dispersive prism?"), which is a different task.
    """
    out = []
    for row in parquet_rows("EdinburghNLP/xsum", "default", "train"):
        snippet = clean(" ".join((row.get("document") or "").split()[:60]), 60, 400)
        if snippet:
            out.append(rng.choice(SUMMARY_WRAPPERS).format(snippet))
        if len(out) >= wanted:
            break
    return out, len(out)


EXTRACT_VERBS = ("extract", "list", "identify", "from the passage", "from the text",
                 "from this", "pull out", "give me the", "what are the names")


def extraction(wanted):
    """Dolly information-extraction requests, instruction *and* passage.

    Every row in this category carries its passage in `context`. Extraction is
    defined by having something to extract from, so dropping the passage leaves a
    prompt that is not an extraction request at all. The passage is truncated: the
    request is recognisable from its opening, and the embedding model has a limited
    window anyway.
    """
    out = []
    for row in parquet_rows("databricks/databricks-dolly-15k", "default", "train"):
        if row.get("category") != "information_extraction":
            continue
        instruction = (row.get("instruction") or "").strip()
        if not any(verb in instruction.lower() for verb in EXTRACT_VERBS):
            continue
        passage = " ".join((row.get("context") or "").split()[:60])
        text = clean(instruction + " " + passage, 40, 500)
        if text:
            out.append(text)
        if len(out) >= wanted:
            break
    return out


WRITE_VERBS = ("write", "compose", "draft", "create a", "come up with", "invent",
               "tell me a story", "make a story", "generate a poem")


def creative(wanted):
    """Dolly creative-writing rows that actually ask for something to be written.

    The category also holds open-ended musings ("Why is music so special?") and even
    arithmetic word problems, which belong to other tasks; requiring a writing verb
    keeps the ones that are requests to produce text.
    """
    out = []
    for row in parquet_rows("databricks/databricks-dolly-15k", "default", "train"):
        if row.get("category") != "creative_writing":
            continue
        instruction = (row.get("instruction") or "").strip()
        if not any(verb in instruction.lower() for verb in WRITE_VERBS):
            continue
        text = clean(instruction)
        if text:
            out.append(text)
        if len(out) >= wanted:
            break
    return out


def chat(wanted):
    """Dolly's open-ended question categories.

    OpenAssistant conversation roots are not used: they are arbitrary user requests
    spanning every task in this taxonomy — code, creative writing, tool use — so
    labelling them all `chat` teaches the classifier that any request is chat. Dolly's
    open_qa, general_qa and brainstorming rows carry no context and are genuinely
    open-ended general questions, which is what this task means here.
    """
    wanted_categories = {"open_qa", "general_qa", "brainstorming"}
    out = []
    for row in parquet_rows("databricks/databricks-dolly-15k", "default", "train"):
        if row.get("category") not in wanted_categories:
            continue
        if (row.get("context") or "").strip():
            continue
        text = clean(row.get("instruction"))
        if text:
            out.append(text)
        if len(out) >= wanted:
            break
    return out


def translation(wanted, rng):
    """OPUS-100 English sides wrapped into translation requests."""
    out = []
    per_lang = wanted // len(LANGS) + 1
    for code, name in LANGS.items():
        taken = 0
        for row in parquet_rows("Helsinki-NLP/opus-100", "en-" + code, "train"):
            pair = row.get("translation") or {}
            source = clean(pair.get("en"), 30, 300)
            if source:
                out.append(rng.choice(TRANSLATE_WRAPPERS).format(lang=name, text=source))
                taken += 1
            if taken >= per_lang:
                break
        if len(out) >= wanted:
            break
    return out[:wanted]


def build(per_task):
    rng = random.Random(20260807)
    corpus = {}
    manifest = []

    def record(task, prompts, dataset, name, licence, split="train",
               wrapped=False, note=""):
        corpus.setdefault(task, []).extend((dataset, p) for p in prompts)
        manifest.append({"task": task, "dataset": dataset, "split": split,
                         "field": name, "license": licence, "kept": len(prompts),
                         "wrapped": wrapped, "note": note})
        print("  %-14s %-46s %4d" % (task, dataset, len(prompts)))

    print("farming %d prompts per task" % per_task)

    record("code", mbpp(per_task), "google-research/mbpp", "text", "CC-BY-4.0")

    record("math", field("openai/gsm8k", "main", "train", "question", per_task),
           "openai/gsm8k", "question", "MIT")

    record("sql", field("gretelai/synthetic_text_to_sql", "default", "train",
                        "sql_prompt", per_task),
           "gretelai/synthetic_text_to_sql", "sql_prompt", "Apache-2.0")

    half = per_task // 2
    record("reasoning", field("allenai/ai2_arc", "ARC-Challenge", "train",
                              "question", half),
           "allenai/ai2_arc", "question", "CC-BY-SA-4.0")
    bbh = []
    bbh_configs = ["causal_judgement", "date_understanding", "logical_deduction_three_objects",
                   "navigate", "temporal_sequences", "boolean_expressions"]
    for config in bbh_configs:
        share = (per_task - half) // len(bbh_configs) + 1
        bbh.extend(field("lukaemon/bbh", config, "test", "input", share))
    record("reasoning", bbh[: per_task - half], "lukaemon/bbh", "input", "MIT", split="test",
           note="BIG-Bench Hard across " + ", ".join(bbh_configs))

    summaries, wrapped_count = summarization(per_task, rng)
    record("summarization", summaries, "EdinburghNLP/xsum", "document", "CC-BY-SA-4.0",
           wrapped=True,
           note="%d XSum documents wrapped in a summarize instruction; dolly's summarization "
                "category was dropped because its instructions read as question-answering once "
                "separated from their context" % wrapped_count)

    record("extraction", extraction(per_task), "databricks/databricks-dolly-15k",
           "instruction + context", "CC-BY-SA-3.0", wrapped=True,
           note="category=information_extraction; passage appended and truncated to 60 words, "
                "since extraction needs something to extract from")

    record("translation", translation(per_task, rng), "Helsinki-NLP/opus-100",
           "translation.en", "CC-BY-4.0 (OPUS; per-corpus terms vary)",
           wrapped=True, note="English side wrapped in a translate instruction")

    record("creative", creative(per_task), "databricks/databricks-dolly-15k",
           "instruction", "CC-BY-SA-3.0",
           note="category=creative_writing, filtered to rows asking for text to be written")

    record("tool-use", hermes(per_task), "NousResearch/hermes-function-calling-v1",
           "conversations[human][0]", "Apache-2.0",
           note="length bound raised to 700 characters for this source")

    record("chat", chat(per_task), "databricks/databricks-dolly-15k", "instruction",
           "CC-BY-SA-3.0",
           note="categories open_qa, general_qa and brainstorming with no context; "
                "OpenAssistant roots were dropped because they span every task")

    # Collapse repeats inside a task first: the same prompt arriving twice from
    # two configs of one source is a duplicate, not a labelling conflict.
    deduped = {}
    for task, entries in corpus.items():
        seen = {}
        for source, prompt in entries:
            seen.setdefault(prompt.lower(), (source, prompt))
        deduped[task] = list(seen.values())

    # A prompt that survives under two *different* tasks teaches the classifier to
    # contradict itself, so drop every copy rather than pick a winner.
    counts = {}
    for entries in deduped.values():
        for _, prompt in entries:
            counts[prompt.lower()] = counts.get(prompt.lower(), 0) + 1
    ambiguous = {p for p, n in counts.items() if n > 1}
    corpus = deduped

    lines = ["# Task-classification prompts farmed from public benchmarks.",
             "#",
             "# Format: split<TAB>task<TAB>source<TAB>prompt",
             "# Generated by corpus/farm_benchmarks.py. See corpus/sources.json for provenance",
             "# and licence of every source. Reference answers are never included.",
             "#"]
    summary = {}
    for task in sorted(corpus):
        seen = set()
        rows = []
        for source, prompt in corpus[task]:
            key = prompt.lower()
            if key in ambiguous or key in seen:
                continue
            seen.add(key)
            rows.append((source, prompt))
        rng.shuffle(rows)
        cut = max(1, int(len(rows) * 0.2))
        for index, (source, prompt) in enumerate(rows):
            split = "eval" if index < cut else "train"
            lines.append("%s\t%s\t%s\t%s" % (split, task, source, prompt))
        summary[task] = {"total": len(rows), "eval": cut, "train": len(rows) - cut}

    here = pathlib.Path(__file__).parent
    (here / "benchmark-prompts.tsv").write_text("\n".join(lines) + "\n")
    (here / "sources.json").write_text(json.dumps(
        {"generator": "corpus/farm_benchmarks.py", "per_task_target": per_task,
         "counts": summary, "sources": manifest}, indent=2) + "\n")

    print("\nwrote benchmark-prompts.tsv")
    for task in sorted(summary):
        s = summary[task]
        print("  %-14s total=%4d train=%4d eval=%4d"
              % (task, s["total"], s["train"], s["eval"]))
    print("total prompts:", sum(s["total"] for s in summary.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-task", type=int, default=250)
    build(parser.parse_args().per_task)
