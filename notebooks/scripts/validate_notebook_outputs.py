#!/usr/bin/env python3

import json
from pathlib import Path
import re
import sys


output_dir = Path(sys.argv[1])
expected_output = {
    "01_getting_started.ipynb": [
        "model: ModelsNano",
        "family: llama",
        "vocabulary: 32",
        "logits: 32",
        "backend: pure-java",
    ],
    "02_framework_adapters.ipynb": [
        "plain java: local answer",
        "langchain4j: local answer",
        "spring ai: local answer",
        "spring stream: local answer",
        "diagnostics: notebook",
    ],
    "03_guarded_rag.ipynb": [
        "cited: MODEL_ANSWER",
        "derived: MODEL_ANSWER_WITH_DERIVED_CITATIONS",
        "weak retrieval: RETRIEVAL_ABSTENTION",
        "unsupported: EXTRACTIVE_FALLBACK",
    ],
}


def text_fragments(value):
    if isinstance(value, list):
        return [str(fragment) for fragment in value]
    if value is None:
        return []
    return [str(value)]


for name, expected_values in expected_output.items():
    file = output_dir / name
    if not file.exists():
        raise RuntimeError(f"Executed notebook is missing: {file}")

    with file.open(encoding="utf-8") as stream:
        notebook = json.load(stream)

    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    if not code_cells:
        raise RuntimeError(f"{name} has no code cells")

    output_text = []
    for index, cell in enumerate(code_cells, start=1):
        if not isinstance(cell.get("execution_count"), int):
            raise RuntimeError(f"{name} code cell {index} was not executed")

        for output in cell.get("outputs", []):
            output_type = output.get("output_type")
            if output_type == "error":
                raise RuntimeError(
                    f"{name} code cell {index}: "
                    f"{output.get('ename')}: {output.get('evalue')}"
                )
            if output_type == "stream":
                fragments = text_fragments(output.get("text"))
                if output.get("name") == "stderr":
                    raise RuntimeError(
                        f"{name} code cell {index} wrote to stderr: "
                        f"{''.join(fragments)}"
                    )
                output_text.extend(fragments)
            if output_type in {"execute_result", "display_data"}:
                output_text.extend(
                    text_fragments(output.get("data", {}).get("text/plain"))
                )

    combined = "".join(output_text)
    if re.search(r"\b(?:AssertionError|Exception|ERROR)\b", combined):
        raise RuntimeError(f"{name} contains suspicious exception/error output")
    if re.search(r"java\.io\.PrintStream@[0-9a-f]+", combined):
        raise RuntimeError(f"{name} contains an unintended PrintStream value")

    for expected in expected_values:
        if expected not in combined:
            raise RuntimeError(f"{name} is missing expected output: {expected}")

    print(
        f"Validated {name}: {len(code_cells)} code cells, "
        f"{len(output_text)} output fragment(s)"
    )
