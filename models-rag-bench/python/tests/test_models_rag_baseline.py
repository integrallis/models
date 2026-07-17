import hashlib
from pathlib import Path

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
        "21da412524218914dc0ac680a8d92ee16c77d4254e2a18cd372f5134c6ab4709"
    )


def test_prompt_and_deterministic_quality_match_java_contract():
    corpus = rag.load_corpus(CORPUS_DIR)
    case = corpus.cases[0]
    hits = rag.Bm25Retriever(corpus.documents).retrieve(case.question, 1)

    prompt = rag.render_prompt(case.question, hits)
    evaluation = rag.evaluate(
        case,
        hits,
        "The deadline is 30 calendar days and the deductible is 75 dollars "
        "[claims-auto-glass].",
    )

    assert "reply exactly INSUFFICIENT_CONTEXT" in prompt
    assert "[claims-auto-glass] Auto glass claims" in prompt
    assert evaluation.correct
    assert evaluation.fact_coverage == 1.0
    assert evaluation.citation_recall == 1.0
    assert corpus.fingerprint() == "6eeb61d5a4b48addb298889a2357cfbcbe7339c044308ba8cd23dcb27c603cb2"
    assert hashlib.sha256(prompt.encode()).hexdigest() == (
        "8d208e5e0aa69d04a866624e5959a2d98af237b9d5d05fb40ad5146b42ccb2e7"
    )


def test_linear_percentiles_match_java_report_math():
    assert rag.percentile([1250.0, 1500.0], 0.50) == 1375.0
    assert rag.percentile([1250.0, 1500.0], 0.95) == 1487.5
