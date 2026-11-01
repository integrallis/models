#!/usr/bin/env python3
"""Paired downstream-quality gates for a Qwen cross-model KV mapper.

The native and mapped arms consume identical target-token IDs.  The mapped arm prefills every
token except the last with the source model, translates that cache in content coordinates, and
recomputes the final context token with the target before scoring any continuation.  This is an
offline Hugging Face oracle; production inference remains Java-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Sequence

try:
    import torch
except ModuleNotFoundError:  # Pure scoring unit tests do not need the experiment environment.
    torch = None


def log_probability(logits: Any, token: int) -> float:
    """Return the normalized log probability for one vocabulary token."""
    if token < 0 or token >= len(logits):
        raise ValueError(f"token {token} is outside vocabulary size {len(logits)}")
    if torch is not None and isinstance(logits, torch.Tensor):
        return float(torch.log_softmax(logits.float(), dim=0)[token])
    maximum = max(float(value) for value in logits)
    normalizer = sum(math.exp(float(value) - maximum) for value in logits)
    return float(logits[token]) - maximum - math.log(normalizer)


def sequence_log_probability(logits: Any, tokens: Sequence[int]) -> float:
    """Score aligned next-token logits without averaging away sequence length."""
    if len(logits) != len(tokens) or not tokens:
        raise ValueError("logit rows and tokens must be nonempty and aligned")
    if torch is not None and isinstance(logits, torch.Tensor):
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=logits.device)
        rows = torch.arange(len(tokens), device=logits.device)
        return float(torch.log_softmax(logits.float(), dim=-1)[rows, token_tensor].sum())
    return sum(log_probability(row, token) for row, token in zip(logits, tokens))


def choice_metrics(scores: Sequence[tuple[float, int]], label: int) -> dict[str, Any]:
    if not scores or any(length < 1 for _, length in scores):
        raise ValueError("choice scores must be nonempty with positive token counts")
    if label < 0 or label >= len(scores):
        raise ValueError("label is outside the choices")
    raw_winner = max(range(len(scores)), key=lambda index: scores[index][0])
    normalized_winner = max(
        range(len(scores)), key=lambda index: scores[index][0] / scores[index][1]
    )
    return {
        "rawWinner": raw_winner,
        "normalizedWinner": normalized_winner,
        "rawCorrect": raw_winner == label,
        "normalizedCorrect": normalized_winner == label,
    }


def perplexity_summary(
    native_nll: Sequence[float],
    mapped_nll: Sequence[float],
    token_counts: Sequence[int],
    maximum_ratio: float,
) -> dict[str, Any]:
    if not native_nll or len(native_nll) != len(mapped_nll) or len(native_nll) != len(token_counts):
        raise ValueError("native NLL, mapped NLL, and token counts must be nonempty and aligned")
    if any(count < 1 for count in token_counts):
        raise ValueError("token counts must be positive")
    total_tokens = sum(token_counts)
    native_mean = sum(native_nll) / total_tokens
    mapped_mean = sum(mapped_nll) / total_tokens
    ratio = math.exp(mapped_mean - native_mean)
    return {
        "tokens": total_tokens,
        "native": {"totalNll": sum(native_nll), "meanNll": native_mean, "ppl": math.exp(native_mean)},
        "mapped": {"totalNll": sum(mapped_nll), "meanNll": mapped_mean, "ppl": math.exp(mapped_mean)},
        "pplRatio": ratio,
        "gate": {"maximum": maximum_ratio, "value": ratio, "pass": ratio <= maximum_ratio},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_hellaswag(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def handoff_partition(
    context: list[int], native_target_sink: int, recomputed_target_tail: int
) -> tuple[list[int], list[int], list[int]]:
    """Split context into exact target sink, translated middle, and exact target recency."""
    if native_target_sink < 0:
        raise ValueError("native target sink must not be negative")
    if recomputed_target_tail < 1:
        raise ValueError("recomputed target tail must be positive")
    if native_target_sink + recomputed_target_tail >= len(context):
        raise ValueError("native target sink plus recomputed target tail must be smaller than the context")
    middle_end = len(context) - recomputed_target_tail
    return (
        context[:native_target_sink],
        context[native_target_sink:middle_end],
        context[middle_end:],
    )


def load_hellaswag(count: int, seed: int, revision: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("Rowan/hellaswag", split="validation", revision=revision)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    result = []
    for index in indices[:count]:
        row = dataset[index]
        result.append(
            {
                "index": index,
                "query": _clean_hellaswag(
                    f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b'].capitalize()}"
                ),
                "endings": [_clean_hellaswag(ending) for ending in row["endings"]],
                "label": int(row["label"]),
            }
        )
    return result


def load_wikitext_chunks(
    tokenizer: Any, count: int, prefix_tokens: int, tail_tokens: int, revision: str
) -> list[list[int]]:
    from datasets import load_dataset

    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="test", revision=revision
    )
    lines = [
        line.strip()
        for line in dataset["text"]
        if line.strip() and not line.strip().startswith(" =")
    ]
    ids = tokenizer(" ".join(lines), add_special_tokens=False)["input_ids"]
    length = prefix_tokens + tail_tokens
    chunks = [ids[offset : offset + length] for offset in range(0, len(ids) - length + 1, length)]
    if len(chunks) < count:
        raise ValueError(f"WikiText supplies {len(chunks)} disjoint chunks; requested {count}")
    return chunks[:count]


class Oracle:
    def __init__(self, args: argparse.Namespace):
        if torch is None:
            raise RuntimeError("PyTorch is required to run the quality experiment")
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from cross_model_kv.rope import RopeSpec

        self._device = args.device
        self._source_snapshot = snapshot_download(args.source_model, revision=args.source_revision)
        self._target_snapshot = snapshot_download(args.target_model, revision=args.target_revision)
        self.tokenizer = AutoTokenizer.from_pretrained(self._target_snapshot)
        self._source = AutoModelForCausalLM.from_pretrained(
            self._source_snapshot,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=args.attention_implementation,
        ).to(self._device).eval()
        self._target = AutoModelForCausalLM.from_pretrained(
            self._target_snapshot,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=args.attention_implementation,
        ).to(self._device).eval()
        self._source_rope = RopeSpec.from_hf_config(self._source.config.to_dict())
        self._target_rope = RopeSpec.from_hf_config(self._target.config.to_dict())
        self._native_target_sink = args.native_target_sink
        self._recomputed_target_tail = args.recomputed_target_tail
        self._mapping_token_chunk = args.mapping_token_chunk
        mapper = load_file(str(args.mapper), device=self._device)
        self._selection = mapper["selection"].to(torch.long)
        self._weight = mapper["weight"]
        self._bias = mapper["bias"]

    def _mapped_prefix(self, context: list[int]) -> tuple[Any, list[int]]:
        from evaluate_hf_mapper import _mapped_target_cache

        effective_sink = min(self._native_target_sink, max(0, len(context) - 2))
        effective_tail = min(self._recomputed_target_tail, len(context) - effective_sink - 1)
        target_sink, mapped_middle, target_tail = handoff_partition(
            context, effective_sink, effective_tail
        )
        source_prefix = target_sink + mapped_middle
        prefix = torch.tensor([source_prefix], dtype=torch.long, device=self._device)
        with torch.inference_mode():
            source_output = self._source(
                input_ids=prefix, use_cache=True, logits_to_keep=1, return_dict=True
            )
        native_sink_cache = None
        if target_sink:
            sink_tokens = torch.tensor([target_sink], dtype=torch.long, device=self._device)
            with torch.inference_mode():
                native_sink_cache = self._target(
                    input_ids=sink_tokens, use_cache=True, logits_to_keep=1, return_dict=True
                ).past_key_values
        mapped_cache = _mapped_target_cache(
            source_output.past_key_values,
            self._source_rope,
            self._selection,
            self._weight,
            self._bias,
            self._target_rope,
            source_token_from=effective_sink,
            token_chunk=self._mapping_token_chunk,
            prefix_cache=native_sink_cache,
        )
        del source_output, native_sink_cache
        return mapped_cache, target_tail

    def _native_score(self, context: list[int], continuation: list[int]) -> float:
        inputs = context + continuation[:-1]
        with torch.inference_mode():
            output = self._target(
                input_ids=torch.tensor([inputs], dtype=torch.long, device=self._device),
                use_cache=False,
                logits_to_keep=len(continuation),
                return_dict=True,
            )
        return sequence_log_probability(output.logits[0, -len(continuation) :], continuation)

    def _mapped_score(
        self,
        context: list[int],
        continuation: list[int],
        mapped_cache: Any,
        target_tail: list[int],
    ) -> float:
        from transformers import DynamicCache

        if not continuation:
            raise ValueError("continuation must not be empty")
        inputs = target_tail + continuation[:-1]
        start = len(context) - len(target_tail)
        positions = torch.arange(start, start + len(inputs), device=self._device)
        cache = DynamicCache.from_legacy_cache(
            tuple((key.clone(), value.clone()) for key, value in mapped_cache.to_legacy_cache())
        )
        with torch.inference_mode():
            output = self._target(
                input_ids=torch.tensor([inputs], dtype=torch.long, device=self._device),
                past_key_values=cache,
                position_ids=positions.unsqueeze(0),
                cache_position=positions,
                use_cache=False,
                logits_to_keep=len(continuation),
                return_dict=True,
            )
        return sequence_log_probability(output.logits[0, -len(continuation) :], continuation)

    def paired_scores_many(
        self, context: list[int], continuations: Sequence[list[int]]
    ) -> list[tuple[float, float]]:
        if not continuations or any(not continuation for continuation in continuations):
            raise ValueError("continuations must be nonempty")
        native = [self._native_score(context, continuation) for continuation in continuations]
        mapped_cache, target_tail = self._mapped_prefix(context)
        mapped = [
            self._mapped_score(
                context,
                continuation,
                mapped_cache,
                target_tail,
            )
            for continuation in continuations
        ]
        del mapped_cache
        torch.cuda.empty_cache()
        return list(zip(native, mapped))

    def paired_scores(self, context: list[int], continuation: list[int]) -> tuple[float, float]:
        return self.paired_scores_many(context, [continuation])[0]


def evaluate_hellaswag(
    oracle: Oracle, count: int, seed: int, revision: str, minimum_retention: float
) -> dict[str, Any]:
    examples = load_hellaswag(count, seed, revision)
    results = []
    started = time.time()
    for number, example in enumerate(examples, 1):
        context = oracle.tokenizer(example["query"], add_special_tokens=False)["input_ids"]
        endings = [
            oracle.tokenizer(" " + ending, add_special_tokens=False)["input_ids"]
            for ending in example["endings"]
        ]
        paired = oracle.paired_scores_many(context, endings)
        native_scores = [(score[0], len(tokens)) for score, tokens in zip(paired, endings)]
        mapped_scores = [(score[1], len(tokens)) for score, tokens in zip(paired, endings)]
        results.append(
            {
                "index": example["index"],
                "label": example["label"],
                "native": choice_metrics(native_scores, example["label"]),
                "mapped": choice_metrics(mapped_scores, example["label"]),
            }
        )
        if number % 10 == 0 or number == len(examples):
            print(f"hellaswag {number}/{len(examples)}", flush=True)
    native_correct = sum(row["native"]["normalizedCorrect"] for row in results)
    mapped_correct = sum(row["mapped"]["normalizedCorrect"] for row in results)
    retention = mapped_correct / native_correct if native_correct else 0.0
    return {
        "dataset": {"id": "Rowan/hellaswag", "revision": revision, "split": "validation"},
        "count": len(results),
        "indices": [row["index"] for row in results],
        "nativeNormalizedAccuracy": native_correct / len(results),
        "mappedNormalizedAccuracy": mapped_correct / len(results),
        "retention": retention,
        "elapsedSeconds": time.time() - started,
        "gate": {"minimum": minimum_retention, "value": retention, "pass": retention >= minimum_retention},
        "results": results,
    }


def evaluate_wikitext(
    oracle: Oracle,
    count: int,
    prefix_tokens: int,
    tail_tokens: int,
    revision: str,
    maximum_ratio: float,
) -> dict[str, Any]:
    chunks = load_wikitext_chunks(oracle.tokenizer, count, prefix_tokens, tail_tokens, revision)
    native_nll = []
    mapped_nll = []
    started = time.time()
    for number, chunk in enumerate(chunks, 1):
        context = chunk[:prefix_tokens]
        continuation = chunk[prefix_tokens:]
        native, mapped = oracle.paired_scores(context, continuation)
        native_nll.append(-native)
        mapped_nll.append(-mapped)
        print(f"wikitext {number}/{len(chunks)}", flush=True)
    result = perplexity_summary(
        native_nll, mapped_nll, [tail_tokens] * len(chunks), maximum_ratio
    )
    result.update(
        {
            "dataset": {
                "id": "Salesforce/wikitext",
                "configuration": "wikitext-2-raw-v1",
                "revision": revision,
                "split": "test",
            },
            "count": len(chunks),
            "prefixTokens": prefix_tokens,
            "tailTokensPerChunk": tail_tokens,
            "elapsedSeconds": time.time() - started,
            "nativeNll": native_nll,
            "mappedNll": mapped_nll,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--target-revision", required=True)
    parser.add_argument("--mapper", required=True, type=Path)
    parser.add_argument("--hellaswag-revision", required=True)
    parser.add_argument("--wikitext-revision", required=True)
    parser.add_argument("--hellaswag-n", type=int, default=100)
    parser.add_argument("--wikitext-n", type=int, default=20)
    parser.add_argument("--wikitext-prefix", type=int, default=1024)
    parser.add_argument("--wikitext-tail", type=int, default=64)
    parser.add_argument(
        "--native-target-sink",
        type=int,
        default=0,
        help="maximum initial context tokens computed exactly by the target before translation",
    )
    parser.add_argument(
        "--recomputed-target-tail",
        type=int,
        default=1,
        help="maximum recent context tokens recomputed exactly by the target after translation",
    )
    parser.add_argument(
        "--mapping-token-chunk",
        type=int,
        default=128,
        help="maximum cache positions materialized per mapper matrix operation",
    )
    parser.add_argument("--minimum-hellaswag-retention", type=float, default=0.95)
    parser.add_argument("--maximum-ppl-ratio", type=float, default=1.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--skip-hellaswag", action="store_true")
    parser.add_argument("--skip-wikitext", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    for name in (
        "hellaswag_n",
        "wikitext_n",
        "wikitext_prefix",
        "wikitext_tail",
        "recomputed_target_tail",
        "mapping_token_chunk",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.native_target_sink < 0:
        parser.error("--native-target-sink must not be negative")

    oracle = Oracle(args)
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "source": {"model": args.source_model, "revision": args.source_revision},
        "target": {"model": args.target_model, "revision": args.target_revision},
        "mapper": {"path": str(args.mapper), "sha256": _sha256(args.mapper)},
        "method": "paired Hugging Face target continuation scoring after mapped versus native prefix",
        "nativeTargetSinkTokens": args.native_target_sink,
        "recomputedTargetTailTokens": args.recomputed_target_tail,
        "mappingTokenChunk": args.mapping_token_chunk,
        "gates": {},
    }
    if not args.skip_hellaswag:
        report["hellaswag"] = evaluate_hellaswag(
            oracle,
            args.hellaswag_n,
            args.seed,
            args.hellaswag_revision,
            args.minimum_hellaswag_retention,
        )
        report["gates"]["hellaswagRetention"] = report["hellaswag"]["gate"]
    if not args.skip_wikitext:
        report["wikitext"] = evaluate_wikitext(
            oracle,
            args.wikitext_n,
            args.wikitext_prefix,
            args.wikitext_tail,
            args.wikitext_revision,
            args.maximum_ppl_ratio,
        )
        report["gates"]["wikitextPplRatio"] = report["wikitext"]["gate"]
    report["qualified"] = bool(report["gates"]) and all(
        gate["pass"] for gate in report["gates"].values()
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"gates": report["gates"], "qualified": report["qualified"]}, sort_keys=True))
    raise SystemExit(0 if report["qualified"] else 2)


if __name__ == "__main__":
    main()
