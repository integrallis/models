#!/usr/bin/env python3
"""Land every QUALIFIED fleet verdict: evidence, catalog entry, evidence test, pinned count.

Importing by hand took longer than measuring, and it repeats for every shard. This runs the whole cycle
and is idempotent, so it can be pointed at the same results twice without duplicating anything.

It writes; it does not validate. Run the models and ModelJars builds afterwards -- the evidence test it
generates re-derives each verdict from the committed reports, and that assertion is the point.
"""

import argparse
import glob
import json
import os
import pathlib
import re
import subprocess
import sys


def camel(slug):
    parts = re.split(r"[-_.]", slug)
    out = parts[0]
    for part in parts[1:]:
        out += part[:1].upper() + part[1:]
    return re.sub(r"[^A-Za-z0-9]", "", out)


def constant(slug):
    return re.sub(r"[^A-Z0-9]+", "_", slug.upper()).strip("_") + "_EVIDENCE"


def generate_tests(models_repo, date, test_path):
    """Add a case per certified directory that has none, pinning the verdict and both ratios."""
    source = test_path.read_text()
    root = models_repo / f"benchmark-results/certified-{date}/rag"
    added = []
    for directory in sorted(root.glob("*/")):
        slug = directory.name
        const = constant(slug)
        # Keyed on the resolved slug, not on the constant name: the first two directories were landed by
        # hand under different constant names, and matching on the name generated a second test for each.
        if f'resolve("{slug}")' in source:
            continue
        verdict = json.loads((directory / "qualification.json").read_text())["qualification"]
        if verdict["verdict"] != "QUALIFIED":
            continue
        comparison = verdict["comparisons"][0]
        candidate = next(p.name for p in directory.glob("*-rust-ffm-grounded.json"))
        comparator = next(p.name for p in directory.glob("*-ollama-grounded.json"))
        decode = comparison["decodeThroughputRatio"]
        latency = comparison["endToEndLatencyRatio"]
        anchor = '  private static final Path GEMMA_3_4B_EVIDENCE = TWO_ARM_EVIDENCE.resolve("gemma-3-4b-it-q4_k_m");\n'
        source = source.replace(
            anchor,
            anchor + f'  private static final Path {const} = TWO_ARM_EVIDENCE.resolve("{slug}");\n',
            1,
        )
        test = f'''  @Test
  void {camel(slug)}QualifiesAgainstItsSameHostOllamaComparator() throws Exception {{
    RagBenchmarkReport candidate = report({const}, "{candidate}");
    RagBenchmarkReport ollama = report({const}, "{comparator}");

    RagProductionQualification qualification =
        RagProductionQualificationPolicy.assess(candidate, List.of(ollama));

    assertThat(qualification.qualified()).isTrue();
    assertThat(qualification.verdict()).isEqualTo(RagQualificationVerdict.QUALIFIED);
    assertThat(qualification.qualifyingComparators()).containsExactly("ollama");
    assertThat(qualification.exclusions()).isEmpty();
    assertThat(qualification.modelAnswerRate())
        .isGreaterThanOrEqualTo(RagProductionQualificationPolicy.MINIMUM_MODEL_ANSWER_RATE);
    assertThat(qualification.modelAnswerCorrectRate())
        .isGreaterThanOrEqualTo(RagProductionQualificationPolicy.MINIMUM_MODEL_ANSWER_CORRECT_RATE);
    assertThat(qualification.comparisons())
        .singleElement()
        .satisfies(
            comparison -> {{
              assertThat(comparison.decodeThroughputRatio()).isBetween({round(decode-0.005,3)}, {round(decode+0.005,3)});
              assertThat(comparison.endToEndLatencyRatio()).isBetween({round(latency-0.01,2)}, {round(latency+0.01,2)});
            }});
  }}

'''
        cut = source.rstrip().rfind("  private RagBenchmarkReport report(")
        source = source[:cut] + test + source[cut:]
        added.append(slug)
    if added:
        test_path.write_text(source)
    return added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=pathlib.Path)
    parser.add_argument("--models-repo", type=pathlib.Path, required=True)
    parser.add_argument("--modeljars-repo", type=pathlib.Path, required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    importer = args.models_repo / "scripts/fleet/import-qualifications.py"
    for directory in args.results:
        if not directory.is_dir():
            continue
        subprocess.run(
            [sys.executable, str(importer), str(directory),
             "--models-repo", str(args.models_repo),
             "--modeljars-repo", str(args.modeljars_repo),
             "--date", args.date, "--apply"],
            check=True, stdout=subprocess.DEVNULL,
        )

    test_path = (args.models_repo
                 / "models-rag-bench/src/test/java/com/integrallis/models/rag/CertifiedRagEvidenceTest.java")
    added = generate_tests(args.models_repo, args.date, test_path)
    print(f"evidence tests added: {len(added)}")
    for slug in added:
        print(f"   {slug}")

    catalog = args.modeljars_repo / "catalog/qualifications.json"
    qualified = sum(1 for e in json.loads(catalog.read_text())["entries"] if e.get("qualified"))
    pinned = (args.modeljars_repo
              / "modeljars-core/src/test/java/org/modeljars/ModelRagQualificationRegistryTest.java")
    text = pinned.read_text()
    current = re.search(r"AGGREGATE_QUALIFIED_MODELS = (\d+);", text)
    if current and int(current.group(1)) != qualified:
        pinned.write_text(text.replace(current.group(0), f"AGGREGATE_QUALIFIED_MODELS = {qualified};"))
        print(f"pinned aggregate {current.group(1)} -> {qualified}")
    print(f"RAG qualifications now: {qualified}")


if __name__ == "__main__":
    sys.exit(main())
