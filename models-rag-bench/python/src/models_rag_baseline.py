"""Controlled Python RAG baseline for Ollama and llama.cpp."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import bm25s
import httpx
import ollama


INSTRUCTIONS = (
    "You answer questions using only the supplied context.\n"
    "Rules:\n"
    "- If the context does not contain the answer, reply exactly INSUFFICIENT_CONTEXT.\n"
    "- Otherwise answer in one short sentence and cite every supporting source as "
    "[source-id].\n"
    "- Do not use prior knowledge.\n\n"
)
CITATION = re.compile(r"\[([a-z0-9][a-z0-9-]*)]", re.IGNORECASE)


@dataclass(frozen=True)
class RagDocument:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class RagCase:
    id: str
    question: str
    relevant_document_ids: tuple[str, ...]
    required_facts: tuple[str, ...]
    answerable: bool


@dataclass(frozen=True)
class RagCorpus:
    documents: tuple[RagDocument, ...]
    cases: tuple[RagCase, ...]

    def fingerprint(self) -> str:
        documents = [asdict(document) for document in self.documents]
        cases = [
            {
                "id": case.id,
                "question": case.question,
                "relevantDocumentIds": list(case.relevant_document_ids),
                "requiredFacts": list(case.required_facts),
                "answerable": case.answerable,
            }
            for case in self.cases
        ]
        digest = hashlib.sha256()
        digest.update(json.dumps(documents, separators=(",", ":")).encode())
        digest.update(b"\n")
        digest.update(json.dumps(cases, separators=(",", ":")).encode())
        return digest.hexdigest()


@dataclass(frozen=True)
class RetrievedDocument:
    document: RagDocument
    score: float
    rank: int


@dataclass(frozen=True)
class RagEvaluation:
    retrieval_recall: float
    reciprocal_rank: float
    fact_coverage: float
    citation_recall: float
    citation_precision: float
    abstained: bool
    correct: bool


@dataclass(frozen=True)
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    ttft_millis: float
    total_millis: float
    prefill_tokens_per_second: float
    load_millis: float
    peak_rss_bytes: int
    cpu_millis: float

    @property
    def tpot_millis(self) -> float:
        if self.output_tokens <= 1:
            return 0.0
        return (self.total_millis - self.ttft_millis) / (self.output_tokens - 1)

    @property
    def decode_tokens_per_second(self) -> float:
        return 1000.0 / self.tpot_millis if self.tpot_millis > 0 else 0.0


class GenerationClient(Protocol):
    backend: str
    model: str

    def generate(self, prompt: str, max_tokens: int) -> GenerationResult: ...

    def close(self) -> None: ...


def load_corpus(corpus_dir: Path) -> RagCorpus:
    documents_data = json.loads((corpus_dir / "documents.json").read_text())
    cases_data = json.loads((corpus_dir / "cases.json").read_text())
    documents = tuple(RagDocument(**item) for item in documents_data)
    cases = tuple(
        RagCase(
            id=item["id"],
            question=item["question"],
            relevant_document_ids=tuple(item["relevantDocumentIds"]),
            required_facts=tuple(item["requiredFacts"]),
            answerable=item["answerable"],
        )
        for item in cases_data
    )
    return RagCorpus(documents, cases)


class Bm25Retriever:
    def __init__(self, documents: Iterable[RagDocument]):
        self.documents = tuple(documents)
        searchable = [f"{document.title} {document.text}" for document in self.documents]
        self._retriever = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
        self._retriever.index(bm25s.tokenize(searchable, stopwords=None), show_progress=False)

    def retrieve(self, question: str, top_k: int) -> list[RetrievedDocument]:
        query = bm25s.tokenize([question], stopwords=None)
        documents, scores = self._retriever.retrieve(
            query, corpus=list(self.documents), k=len(self.documents), show_progress=False
        )
        positive = [
            (document, float(scores[0, index]))
            for index, document in enumerate(documents[0])
            if float(scores[0, index]) > 0
        ][:top_k]
        return [
            RetrievedDocument(document, score, index + 1)
            for index, (document, score) in enumerate(positive)
        ]


def render_prompt(
    question: str, hits: Iterable[RetrievedDocument], prompt_template: str = "raw"
) -> str:
    context = "".join(
        f"[{hit.document.id}] {hit.document.title}\n{hit.document.text}\n\n"
        for hit in hits
    )
    canonical = f"{INSTRUCTIONS}CONTEXT\n{context}QUESTION\n{question}\n\nANSWER\n"
    if prompt_template == "raw":
        return canonical
    if prompt_template == "chatml":
        return f"<|im_start|>user\n{canonical}<|im_end|>\n<|im_start|>assistant\n"
    raise ValueError(f"unsupported prompt template: {prompt_template}")


def evaluate(case: RagCase, hits: list[RetrievedDocument], answer: str) -> RagEvaluation:
    expected = {value.lower() for value in case.relevant_document_ids}
    retrieved = {hit.document.id.lower() for hit in hits}
    citations = {match.group(1).lower() for match in CITATION.finditer(answer)}
    retrieval_recall = _fraction_present(expected, retrieved) if expected else 1.0
    reciprocal_rank = 1.0 if not expected else 0.0
    if expected:
        for hit in hits:
            if hit.document.id.lower() in expected:
                reciprocal_rank = 1.0 / hit.rank
                break
    normalized_answer = _normalize_text(answer)
    fact_coverage = (
        sum(_normalize_text(fact) in normalized_answer for fact in case.required_facts)
        / len(case.required_facts)
        if case.required_facts
        else 1.0
    )
    citation_recall = (
        _fraction_present(expected, citations)
        if expected
        else (1.0 if not citations else 0.0)
    )
    citation_precision = (
        _fraction_present(citations, expected)
        if citations
        else (1.0 if not expected else 0.0)
    )
    abstained = answer.strip() == "INSUFFICIENT_CONTEXT"
    correct = (
        not abstained
        and fact_coverage == 1.0
        and citation_recall == 1.0
        and citation_precision == 1.0
        if case.answerable
        else abstained and not citations
    )
    return RagEvaluation(
        retrieval_recall,
        reciprocal_rank,
        fact_coverage,
        citation_recall,
        citation_precision,
        abstained,
        correct,
    )


class OllamaGenerationClient:
    backend = "ollama"

    def __init__(self, model: str, endpoint: str, context: int, threads: int, pid: int):
        self.model = model
        self.context = context
        self.threads = threads
        self.pid = pid
        self._client = ollama.Client(host=endpoint)
        self._observed_load_millis = 0.0

    def generate(self, prompt: str, max_tokens: int) -> GenerationResult:
        cpu_before = _process_cpu_seconds(self.pid)
        start = time.perf_counter_ns()
        first_token = 0
        text: list[str] = []
        final: Any = None
        stream = self._client.generate(
            model=self.model,
            prompt=prompt,
            stream=True,
            raw=True,
            keep_alive="5m",
            options={
                "temperature": 0,
                "top_k": 1,
                "top_p": 1,
                "seed": 42,
                "num_predict": max_tokens,
                "num_ctx": self.context,
                "num_thread": self.threads,
                "repeat_penalty": 1,
            },
        )
        for event in stream:
            content = _field(event, "response", "")
            if content and first_token == 0:
                first_token = time.perf_counter_ns()
            text.append(content)
            if _field(event, "done", False):
                final = event
        end = time.perf_counter_ns()
        if final is None or first_token == 0:
            raise RuntimeError("Ollama stream did not produce a final answer")
        input_tokens = int(_field(final, "prompt_eval_count", 0) or 0)
        output_tokens = int(_field(final, "eval_count", 0) or 0)
        prompt_nanos = int(_field(final, "prompt_eval_duration", 0) or 0)
        load_nanos = int(_field(final, "load_duration", 0) or 0)
        if load_nanos and not self._observed_load_millis:
            self._observed_load_millis = load_nanos / 1_000_000
        return GenerationResult(
            "".join(text),
            input_tokens,
            output_tokens,
            (first_token - start) / 1_000_000,
            (end - start) / 1_000_000,
            input_tokens * 1_000_000_000 / prompt_nanos if prompt_nanos else 0.0,
            self._observed_load_millis,
            _process_high_water_bytes(self.pid),
            max(0.0, (_process_cpu_seconds(self.pid) - cpu_before) * 1000),
        )

    def close(self) -> None:
        self._client.close()


class LlamaServerGenerationClient:
    backend = "llama.cpp"

    def __init__(self, model: str, endpoint: str, threads: int, pid: int):
        self.model = model
        self.threads = threads
        self.pid = pid
        self._client = httpx.Client(base_url=endpoint, timeout=1800)

    def generate(self, prompt: str, max_tokens: int) -> GenerationResult:
        cpu_before = _process_cpu_seconds(self.pid)
        start = time.perf_counter_ns()
        first_token = 0
        text: list[str] = []
        final: dict[str, Any] | None = None
        body = {
            "prompt": prompt,
            "stream": True,
            "n_predict": max_tokens,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "seed": 42,
            "n_threads": self.threads,
            "repeat_penalty": 1,
            "cache_prompt": False,
        }
        with self._client.stream("POST", "/completion", json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                payload = line.removeprefix("data:").strip()
                if not payload or payload == "[DONE]":
                    continue
                event = json.loads(payload)
                content = event.get("content", "")
                if content and first_token == 0:
                    first_token = time.perf_counter_ns()
                text.append(content)
                if event.get("stop") and event.get("timings"):
                    final = event
        end = time.perf_counter_ns()
        if final is None or first_token == 0:
            raise RuntimeError("llama.cpp stream did not produce final timings")
        timings = final["timings"]
        input_tokens = int(timings.get("prompt_n", 0))
        output_tokens = int(timings.get("predicted_n", 0))
        prompt_millis = float(timings.get("prompt_ms", 0))
        return GenerationResult(
            "".join(text),
            input_tokens,
            output_tokens,
            (first_token - start) / 1_000_000,
            (end - start) / 1_000_000,
            input_tokens * 1000 / prompt_millis if prompt_millis else 0.0,
            0.0,
            _process_high_water_bytes(self.pid),
            max(0.0, (_process_cpu_seconds(self.pid) - cpu_before) * 1000),
        )

    def close(self) -> None:
        self._client.close()


class LlamaCppPythonGenerationClient:
    backend = "llama.cpp-python"

    def __init__(self, model_path: Path, context: int, threads: int):
        try:
            from llama_cpp import Llama
        except ImportError as failure:
            raise RuntimeError("install the 'llama' extra for llama-cpp-python") from failure
        load_start = time.perf_counter_ns()
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=context,
            n_threads=threads,
            seed=42,
            verbose=False,
        )
        self._load_millis = (time.perf_counter_ns() - load_start) / 1_000_000
        self.model = model_path.name

    def generate(self, prompt: str, max_tokens: int) -> GenerationResult:
        start_cpu = time.process_time_ns()
        start = time.perf_counter_ns()
        first_token = 0
        text: list[str] = []
        for event in self._llm.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=0,
            top_k=1,
            top_p=1,
            repeat_penalty=1,
            seed=42,
            stream=True,
        ):
            content = event["choices"][0]["text"]
            if content and first_token == 0:
                first_token = time.perf_counter_ns()
            text.append(content)
        end = time.perf_counter_ns()
        answer = "".join(text)
        if first_token == 0:
            raise RuntimeError("llama-cpp-python produced no output token")
        input_tokens = len(self._llm.tokenize(prompt.encode(), add_bos=True))
        output_tokens = max(1, len(self._llm.tokenize(answer.encode(), add_bos=False)))
        return GenerationResult(
            answer,
            input_tokens,
            output_tokens,
            (first_token - start) / 1_000_000,
            (end - start) / 1_000_000,
            0.0,
            self._load_millis,
            _process_high_water_bytes(os.getpid()),
            (time.process_time_ns() - start_cpu) / 1_000_000,
        )

    def close(self) -> None:
        self._llm.close()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = quantile * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    fraction = index - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _run(args: argparse.Namespace) -> dict[str, Any]:
    corpus = load_corpus(args.corpus_dir)
    selected = [case for case in corpus.cases if not args.case or case.id in args.case]
    if not selected:
        raise ValueError("no RAG cases selected")
    unknown = set(args.case) - {case.id for case in corpus.cases}
    if unknown:
        raise ValueError(f"unknown RAG cases: {sorted(unknown)}")
    retriever = Bm25Retriever(corpus.documents)
    client = _client(args)
    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        for _ in range(args.warmups):
            for case in selected:
                hits = retriever.retrieve(case.question, args.top_k)
                client.generate(
                    render_prompt(case.question, hits, args.prompt_template), args.max_tokens
                )
        for iteration in range(args.iterations):
            for case in selected:
                try:
                    total_start = time.perf_counter_ns()
                    retrieval_start = time.perf_counter_ns()
                    hits = retriever.retrieve(case.question, args.top_k)
                    retrieval_ms = (time.perf_counter_ns() - retrieval_start) / 1_000_000
                    prompt = render_prompt(case.question, hits, args.prompt_template)
                    generation = client.generate(prompt, args.max_tokens)
                    end_to_end_ms = (time.perf_counter_ns() - total_start) / 1_000_000
                    evaluation = evaluate(case, hits, generation.text)
                    runs.append(
                        {
                            "iteration": iteration,
                            "caseId": case.id,
                            "retrievedIds": [hit.document.id for hit in hits],
                            "promptSha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "retrievalMillis": retrieval_ms,
                            "frameworkOverheadMillis": max(
                                0.0, end_to_end_ms - retrieval_ms - generation.total_millis
                            ),
                            "endToEndMillis": end_to_end_ms,
                            "generation": _generation_dict(generation),
                            "evaluation": _evaluation_dict(evaluation),
                        }
                    )
                    print(
                        f"iteration {iteration + 1}/{args.iterations} case={case.id} "
                        f"correct={evaluation.correct} retrieval={retrieval_ms:.1f}ms "
                        f"ttft={generation.ttft_millis:.1f}ms "
                        f"decode={generation.decode_tokens_per_second:.2f} tok/s "
                        f"e2e={end_to_end_ms:.1f}ms"
                    )
                except Exception as failure:  # The report must retain failed attempts.
                    failures.append(
                        {"iteration": iteration, "caseId": case.id, "error": repr(failure)}
                    )
    finally:
        client.close()
    total_attempts = len(selected) * args.iterations
    summary = _summary(runs, selected, total_attempts)
    return {
        "schemaVersion": 1,
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "framework": "python",
        "backend": client.backend,
        "backendVersion": args.backend_version,
        "modelId": args.model_id or Path(args.model).name.removesuffix(".gguf"),
        "model": args.model,
        "artifactSha256": _file_sha256(args.artifact) if args.artifact else None,
        "artifactSizeBytes": args.artifact.stat().st_size if args.artifact else 0,
        "settings": {
            "corpusSha256": corpus.fingerprint(),
            "caseIds": [case.id for case in selected],
            "promptTemplate": args.prompt_template,
            "retrievalTopK": args.top_k,
            "maxOutputTokens": args.max_tokens,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "contextLength": args.context,
            "threads": args.threads,
            "temperature": 0,
            "samplingTopK": 1,
            "topP": 1,
            "repetitionPenalty": 1,
            "seed": 42,
            "promptCacheEnabled": False,
        },
        "environment": {
            "hostname": socket.gethostname(),
            "osName": platform.system(),
            "osVersion": platform.release(),
            "architecture": platform.machine(),
            "cpuModel": platform.processor() or "unknown",
            "availableProcessors": os.cpu_count() or 0,
            "pythonVersion": platform.python_version(),
        },
        "summary": summary,
        "performanceTier": _classify(summary),
        "runs": runs,
        "failures": failures,
    }


def _client(args: argparse.Namespace) -> GenerationClient:
    if args.backend == "ollama":
        return OllamaGenerationClient(
            args.model, args.endpoint, args.context, args.threads, args.pid
        )
    if args.backend == "llama.cpp-server":
        return LlamaServerGenerationClient(args.model, args.endpoint, args.threads, args.pid)
    if args.artifact is None:
        raise ValueError("--artifact is required for llama.cpp-python")
    return LlamaCppPythonGenerationClient(args.artifact, args.context, args.threads)


def _summary(
    runs: list[dict[str, Any]], cases: list[RagCase], total_attempts: int
) -> dict[str, Any]:
    by_id = {case.id: case for case in cases}
    answerable = [run for run in runs if by_id[run["caseId"]].answerable]
    unanswerable = [run for run in runs if not by_id[run["caseId"]].answerable]

    def values(path: tuple[str, ...], source: list[dict[str, Any]] = runs) -> list[float]:
        result: list[float] = []
        for run in source:
            value: Any = run
            for key in path:
                value = value[key]
            result.append(float(value))
        return result

    def latency(path: tuple[str, ...]) -> dict[str, float]:
        samples = values(path)
        return {"p50": percentile(samples, 0.50), "p95": percentile(samples, 0.95)}

    def average(path: tuple[str, ...], source: list[dict[str, Any]]) -> float:
        samples = values(path, source)
        return sum(samples) / len(samples) if samples else 0.0

    load_values = values(("generation", "loadMillis"))
    rss_values = values(("generation", "peakRssBytes"))
    decode_values = values(("generation", "decodeTokensPerSecond"))
    prefill_values = values(("generation", "prefillTokensPerSecond"))
    correct = values(("evaluation", "correct"))
    abstentions = values(("evaluation", "correct"), unanswerable)
    return {
        "totalAttempts": total_attempts,
        "successfulAttempts": len(runs),
        "loadMillis": max(load_values, default=0.0),
        "retrievalMillis": latency(("retrievalMillis",)),
        "frameworkOverheadMillis": latency(("frameworkOverheadMillis",)),
        "ttftMillis": latency(("generation", "ttftMillis")),
        "tpotMillis": latency(("generation", "tpotMillis")),
        "endToEndMillis": latency(("endToEndMillis",)),
        "p50PrefillTokensPerSecond": percentile(prefill_values, 0.50),
        "p50DecodeTokensPerSecond": percentile(decode_values, 0.50),
        "peakRssBytes": int(max(rss_values, default=0.0)),
        "totalCpuMillis": sum(values(("generation", "cpuMillis"))),
        "retrievalRecall": average(("evaluation", "retrievalRecall"), answerable),
        "meanReciprocalRank": average(("evaluation", "reciprocalRank"), answerable),
        "factCoverage": average(("evaluation", "factCoverage"), answerable),
        "citationRecall": average(("evaluation", "citationRecall"), answerable),
        "citationPrecision": average(("evaluation", "citationPrecision"), answerable),
        "abstentionAccuracy": sum(abstentions) / len(abstentions) if abstentions else 0.0,
        "correctAnswerRate": sum(correct) / len(correct) if correct else 0.0,
    }


def _classify(summary: dict[str, Any]) -> str:
    if summary["successfulAttempts"] != summary["totalAttempts"]:
        return "FAILED_RUNTIME"
    if (
        summary["retrievalRecall"] < 0.95
        or summary["meanReciprocalRank"] < 0.90
        or summary["factCoverage"] < 0.90
        or summary["citationRecall"] < 0.90
        or summary["citationPrecision"] < 0.90
        or summary["abstentionAccuracy"] < 1.0
        or summary["correctAnswerRate"] < 0.90
    ):
        return "FAILED_QUALITY"
    production = (
        summary["retrievalMillis"]["p95"] <= 100
        and summary["ttftMillis"]["p95"] <= 1000
        and summary["tpotMillis"]["p95"] <= 100
        and summary["endToEndMillis"]["p95"] <= 5000
    )
    if production:
        return "PRODUCTION_READY"
    usable = (
        summary["retrievalMillis"]["p95"] <= 250
        and summary["ttftMillis"]["p95"] <= 2000
        and summary["tpotMillis"]["p95"] <= 200
        and summary["endToEndMillis"]["p95"] <= 10000
    )
    return "USABLE" if usable else "OFFLINE"


def _generation_dict(generation: GenerationResult) -> dict[str, Any]:
    return {
        "text": generation.text,
        "inputTokens": generation.input_tokens,
        "outputTokens": generation.output_tokens,
        "ttftMillis": generation.ttft_millis,
        "totalMillis": generation.total_millis,
        "prefillTokensPerSecond": generation.prefill_tokens_per_second,
        "loadMillis": generation.load_millis,
        "peakRssBytes": generation.peak_rss_bytes,
        "cpuMillis": generation.cpu_millis,
        "tpotMillis": generation.tpot_millis,
        "decodeTokensPerSecond": generation.decode_tokens_per_second,
    }


def _evaluation_dict(evaluation: RagEvaluation) -> dict[str, Any]:
    return {
        "retrievalRecall": evaluation.retrieval_recall,
        "reciprocalRank": evaluation.reciprocal_rank,
        "factCoverage": evaluation.fact_coverage,
        "citationRecall": evaluation.citation_recall,
        "citationPrecision": evaluation.citation_precision,
        "abstained": evaluation.abstained,
        "correct": evaluation.correct,
    }


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower().replace(",", ""))).strip()


def _fraction_present(required: set[str], actual: set[str]) -> float:
    return len(required & actual) / len(required)


def _field(value: Any, name: str, fallback: Any) -> Any:
    if isinstance(value, dict):
        return value.get(name, fallback)
    return getattr(value, name, fallback)


def _process_cpu_seconds(pid: int) -> float:
    if pid <= 0:
        return 0.0
    stat = Path(f"/proc/{pid}/stat")
    try:
        fields = stat.read_text().split()
        ticks = int(fields[13]) + int(fields[14])
        return ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return 0.0


def _process_high_water_bytes(pid: int) -> int:
    if pid <= 0:
        return 0
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return 0
    return 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("ollama", "llama.cpp-server", "llama.cpp-python"), required=True
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--backend-version", default="unknown")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--prompt-template", choices=("raw", "chatml"), default="raw")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "src" / "main" / "resources" / "rag",
    )
    parser.add_argument("--output", type=Path, default=Path("build/reports/rag/python.json"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.backend == "llama.cpp-server" and args.endpoint == "http://127.0.0.1:11434":
        args.endpoint = "http://127.0.0.1:8080"
    report = _run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    summary = report["summary"]
    print(
        f"python/{report['backend']}/{report['modelId']}: tier={report['performanceTier']} "
        f"success={summary['successfulAttempts']}/{summary['totalAttempts']} "
        f"p95-ttft={summary['ttftMillis']['p95']:.1f}ms "
        f"p50-decode={summary['p50DecodeTokensPerSecond']:.2f} tok/s "
        f"p95-e2e={summary['endToEndMillis']['p95']:.1f}ms\nreport: {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
