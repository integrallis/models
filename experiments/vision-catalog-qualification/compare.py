#!/usr/bin/env python3
"""Fail-closed task verdict, with the pinned upstream answer scorer loaded externally."""
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path


def read(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(path.read_text(), parse_constant=invalid)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def measured(rows, case):
    expected = list(range(-case["warmups"], case["repetitions"]))
    require([row["run"] for row in rows] == expected, "Missing, reordered or duplicate runs: " + case["id"])
    return [row for row in rows if row["run"] >= 0]


def median(rows, field):
    values = [row[field] for row in rows]
    require(all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values), "Invalid timing: " + field)
    return statistics.median(values)


def score(evaluator, answer, answers):
    return evaluator._compute_answer_scores(answers).get(evaluator.answer_processor(answer), 0.0)


def evaluate(corpus, manifest, output, variant, evaluator):
    cases = corpus["cases"]
    require(len({case["id"] for case in cases}) == len(cases), "Duplicate case IDs")
    destination = output / variant
    candidate_config = read(destination / "candidate/configuration.json")
    receipt = read(output / "receipt.json")
    encoder_config = read(output / "encoder/configuration.json")
    require(candidate_config["host"] == encoder_config["host"] == receipt["host"], "Host mismatch")
    require(candidate_config["revision"] == receipt["revision"], "Candidate revision mismatch")
    require(encoder_config["mmprojSha256"] == manifest["files"]["llava-v1.5-7b-mmproj-model-f16.gguf"]["sha256"], "Projector mismatch")
    require(encoder_config["corpusSha256"] == receipt["corpusSha256"], "Encoder corpus mismatch")
    reference = [json.loads(line) for line in (destination / "reference.jsonl").read_text().splitlines()]
    reference_config = reference.pop(0)["configuration"]
    for config in [candidate_config, reference_config, receipt]:
        for key in ["threads", "context", "maximumOutputTokens"]:
            require(config[key] == manifest[key], "Execution setting mismatch: " + key)
    require(reference_config["kvType"] == receipt["kvType"] == "f32", "KV mismatch")
    if receipt.get("schema", 1) >= 2:
        require(reference_config.get("flashAttention") == receipt.get("referenceFlashAttention") == "auto",
                "Reference must preserve frozen AUTO attention execution")
        environment = candidate_config.get("diagnostics", {}).get("environment", {})
        require(environment.get("vision-projections", "").startswith("rust-")
                and environment.get("decoder-attention") == "NativeCausalAttention",
                "Frozen qualification requires native projections and attention")
    require(encoder_config["threads"] == manifest["threads"], "Encoder threads mismatch")
    require({row["id"] for row in reference} == {case["id"] for case in cases}, "Reference case coverage mismatch")
    case_results, timings, disagreements = [], [], []
    for case in cases:
        candidate = measured(read(destination / "candidate" / case["id"] / "runs.json"), case)
        decoder = measured([row for row in reference if row["id"] == case["id"]], case)
        encoder = measured(read(output / "encoder" / case["id"] / "runs.json"), case)
        for c, d in zip(candidate, decoder, strict=True):
            require(all(isinstance(t, int) and t >= 0 for t in c["tokens"] + d["tokens"]), "Invalid tokens")
            require(len(c["tokens"]) <= 32 and len(d["tokens"]) <= 32, "Output limit violated")
        require(all(row["tokens"] == candidate[0]["tokens"] for row in candidate), "Candidate nondeterminism")
        require(all(row["tokens"] == decoder[0]["tokens"] for row in decoder), "Reference nondeterminism")
        answer = candidate[0]["answer"]
        ref_answer = bytes.fromhex(decoder[0]["answerHex"]).decode("utf-8")
        match = candidate[0]["tokens"] == decoder[0]["tokens"]
        if not match:
            disagreements.append(case["id"])
        case_results.append(dict(id=case["id"], dataset=case["dataset"], blank=case["blank"],
                                 candidateAnswer=answer, referenceAnswer=ref_answer,
                                 candidateScore=score(evaluator, answer, case["answers"]),
                                 referenceScore=score(evaluator, ref_answer, case["answers"]),
                                 tokensMatch=match, candidateCapHit=len(candidate[0]["tokens"]) == 32,
                                 referenceCapHit=len(decoder[0]["tokens"]) == 32))
        if case["repetitions"] == manifest["performance"]["repetitions"]:
            require(case["warmups"] == manifest["performance"]["warmups"], "Warmup count mismatch")
            paired = [dict(prefillMs=d["prefillMs"] + e["encoderMs"], decodeMs=d["decodeMs"],
                           totalMs=d["prefillMs"] + e["encoderMs"] + d["decodeMs"])
                      for d, e in zip(decoder, encoder, strict=True)]
            ratios = {key: median(candidate, key) / median(paired, key) for key in ["prefillMs", "totalMs"]}
            if candidate[0]["tokens"] and decoder[0]["tokens"]:
                ratios["decodeMs"] = median(candidate, "decodeMs") / median(paired, "decodeMs")
            timings.append(dict(id=case["id"], dataset=case["dataset"], ratios=ratios,
                                candidateMedianMs={key: median(candidate, key) for key in ratios},
                                referenceMedianMs={key: median(paired, key) for key in ratios}))
    datasets = {}
    for dataset in ["vqav2", "textvqa"]:
        rows = [r for r in case_results if r["dataset"] == dataset and not r["blank"]]
        blanks = [r for r in case_results if r["dataset"] == dataset and r["blank"]]
        require(len(rows) == manifest["caseSelection"]["casesPerDataset"], "Dataset coverage differs")
        require(len(blanks) == manifest["accuracy"]["blankImageCasesPerDataset"], "Blank coverage differs")
        require(sum(t["dataset"] == dataset for t in timings) == manifest["performance"]["caseCountPerDataset"], "Performance coverage differs")
        normal = {r["id"]: r for r in rows}
        datasets[dataset] = dict(cases=len(rows), candidateScore=statistics.mean(r["candidateScore"] for r in rows),
                                 referenceScore=statistics.mean(r["referenceScore"] for r in rows),
                                 blankCases=len(blanks),
                                 imageContribution=statistics.mean(normal[b["id"].removesuffix("-blank")]["candidateScore"] - b["candidateScore"] for b in blanks))
    agreement = statistics.mean(r["tokensMatch"] for r in case_results if not r["blank"])
    gates = dict(accuracyLoss=all(d["referenceScore"]-d["candidateScore"] <= manifest["accuracy"]["maximumScoreLoss"] + 1e-12 for d in datasets.values()),
                 vqaFloor=datasets["vqav2"]["candidateScore"] >= manifest["accuracy"]["minimumVqaScore"],
                 textVqaFloor=datasets["textvqa"]["candidateScore"] >= manifest["accuracy"]["minimumTextVqaScore"],
                 imageContribution=datasets["vqav2"]["imageContribution"] >= manifest["accuracy"]["minimumVqaImageContribution"],
                 tokenAgreement=agreement >= manifest["accuracy"]["minimumTokenSequenceAgreement"],
                 performance=all(r <= manifest["performance"]["maximumMedianLatencyRatio"] for t in timings for r in t["ratios"].values()))
    return dict(variant=variant, gates=gates, taskQualified=all(gates.values()), datasets=datasets,
                tokenAgreement=agreement, disagreements=disagreements, performance=timings, cases=case_results)


def main():
    root, scorer = map(Path, sys.argv[1:])
    script = Path(__file__).resolve().parent
    manifest = read(script / "manifest.json")
    require(sha(scorer) == manifest["scorer"]["sha256"], "Upstream scorer digest mismatch")
    spec = importlib.util.spec_from_file_location("upstream_scorer", scorer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluator = module.TextVQAAccuracyEvaluator()
    corpus = read(root / "cases.json")
    receipt = read(root / "results/receipt.json")
    require(receipt["manifestSha256"] == corpus["manifestSha256"] == sha(script / "manifest.json"), "Manifest mismatch")
    require(receipt["corpusSha256"] == sha(root / "cases.json"), "Corpus mismatch")
    for case in corpus["cases"]:
        require(sha(root / case["image"]) == case["imageSha256"], "Image mismatch")
    results = [evaluate(corpus, manifest, root / "results", variant, evaluator) for variant in manifest["variants"]]
    report = dict(schema="llava-task-admission-v1", receipt=receipt, variants=results,
                  scope="branch task gate; integration and ModelJars runtime evidence also required")
    (root / "results/task-verdict.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in r.items() if k != "cases"} for r in results], indent=2))
    return 0 if all(r["taskQualified"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
