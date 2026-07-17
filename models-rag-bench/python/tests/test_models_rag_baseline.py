import hashlib
from pathlib import Path

import httpx
import models_rag_baseline as rag


CORPUS_DIR = Path(__file__).parents[2] / "src" / "main" / "resources" / "rag"


def test_bm25_ranks_every_answerable_source_first():
    corpus = rag.load_corpus(CORPUS_DIR)
    retriever = rag.Bm25Retriever(corpus.documents)

    for case in (case for case in corpus.cases if case.answerable):
        hits = retriever.retrieve(case.question, 3)
        assert hits[0].document.id == case.relevant_document_ids[0], case.id

    prompt_hashes = "\n".join(
        hashlib.sha256(
            rag.render_prompt(case.question, retriever.retrieve(case.question, 1)).encode()
        ).hexdigest()
        for case in corpus.cases
    )
    assert hashlib.sha256(prompt_hashes.encode()).hexdigest() == (
        "112000a6017861f70087349571fc5ec400f11499e1b6d9c96676f327558e357d"
    )


def test_prompt_and_deterministic_quality_match_java_contract():
    corpus = rag.load_corpus(CORPUS_DIR)
    case = corpus.cases[0]
    hits = rag.Bm25Retriever(corpus.documents).retrieve(case.question, 1)

    prompt = rag.render_prompt(case.question, hits)
    evaluation = rag.evaluate(
        case,
        hits,
        "The deadline is 30 calendar days and the deductible is $75 "
        "[claims-auto-glass].",
    )

    assert "reply exactly INSUFFICIENT_CONTEXT" in prompt
    assert "Copy each supporting source ID exactly" in prompt
    assert "[source-id]" not in prompt
    assert "[claims-auto-glass] Auto glass claims" in prompt
    assert evaluation.correct
    assert evaluation.fact_coverage == 1.0
    assert evaluation.citation_recall == 1.0
    assert corpus.fingerprint() == "6eeb61d5a4b48addb298889a2357cfbcbe7339c044308ba8cd23dcb27c603cb2"
    assert hashlib.sha256(prompt.encode()).hexdigest() == (
        "26fccba3e100ea107b382b25a9404ba8fb68bf9bd8c2003878ad6b6cefd87841"
    )

    chatml_prompt = rag.render_prompt(case.question, hits, "chatml")
    assert chatml_prompt.startswith("<|im_start|>user\nYou answer questions")
    assert chatml_prompt.endswith("ANSWER\n<|im_end|>\n<|im_start|>assistant\n")
    assert hashlib.sha256(chatml_prompt.encode()).hexdigest() == (
        "adeb518c096c5bbe124b3c8d60ab2001e6754d192938465f6bedea5ca5a62bad"
    )

    no_think_prompt = rag.render_prompt(case.question, hits, "chatml-no-think")
    assert no_think_prompt.endswith(
        "ANSWER\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert hashlib.sha256(no_think_prompt.encode()).hexdigest() == (
        "959555745e05099d76fd4af53d470399d4cabdaf0ed86bb9b7505b0d08c1cc4a"
    )


def test_linear_percentiles_match_java_report_math():
    assert rag.percentile([1250.0, 1500.0], 0.50) == 1375.0
    assert rag.percentile([1250.0, 1500.0], 0.95) == 1487.5


def test_transport_retry_only_retries_once_before_headers():
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.RemoteProtocolError("stale pooled connection")
        return "response"

    assert rag.with_single_transport_retry(operation) == "response"
    assert attempts == 2
