import pytest

import reranker_transformers_reference as reference


def test_build_report_retains_exact_pair_inputs_logits_and_order():
    report = reference.build_report(
        model_id="cross-encoder/example",
        requested_revision="0123456789abcdef",
        resolved_revision="0123456789abcdef",
        query="population?",
        documents=("Berlin population", "Paris museums"),
        encoded={
            "input_ids": [[101, 200, 102, 300, 102, 0], [101, 200, 102, 400, 102, 0]],
            "token_type_ids": [[0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0]],
            "attention_mask": [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]],
        },
        logits=(8.5, -3.25),
        dependency_versions={"torch": "test", "transformers": "test"},
    )

    assert report["reference"] == {
        "backend": "hugging-face-transformers",
        "modelId": "cross-encoder/example",
        "requestedRevision": "0123456789abcdef",
        "resolvedRevision": "0123456789abcdef",
        "dependencies": {"torch": "test", "transformers": "test"},
    }
    assert report["pairs"][0] == {
        "documentIndex": 0,
        "tokens": [101, 200, 102, 300, 102],
        "tokenTypes": [0, 0, 0, 1, 1],
        "logit": 8.5,
    }
    assert report["ranking"] == [0, 1]


@pytest.mark.parametrize(
    ("query", "documents", "message"),
    [
        ("", ("document",), "query"),
        ("query", (), "document"),
        ("query", ("",), "document"),
    ],
)
def test_validate_workload_rejects_incomplete_inputs(query, documents, message):
    with pytest.raises(ValueError, match=message):
        reference.validate_workload(query, documents)
