"""The assembler must reproduce the entry ModelJars already publishes for a certified bundle."""

import json
import unittest
from pathlib import Path

from assemble_base_qualification_entry import assemble

MODELS_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = MODELS_ROOT / "benchmark-results/certified-20260902/rag/mobilemoe-s-qat-int4-g32"
# Copied verbatim from ModelJars catalog/qualifications.json (entry published in ModelJars #134).
PUBLISHED = json.loads(
    '{"modelId":"facebook_mobilemoe_s_qat_int4_g32","model":"Meta MobileMoE-S QAT INT4 G32","backend":"pure-java",'
    '"backendVersion":"models@13adbd12ad8d1593b9f76165ea62339bf05a0eef com.integrallis:vectors-core@46d462ae5bc3272d9785e9f0109b18c61eb16a92",'
    '"workload":"general","corpusSha256":"4b27eba8f166c84ef19c53de825445a6d0097f9bd8efa20b2d7013f34621f83c",'
    '"promptTemplate":"mobilemoe","groundingPolicy":"bounded-context-injection-screened-citation-safe-statement-grounding-v20",'
    '"artifactSha256":"1a54d8eb2adf19c296a1c129f9cd7d7395c21f5facfc6d13d2945e002d55e37d","artifactSizeBytes":713916240,'
    '"report":"benchmark-results/certified-20260902/rag/mobilemoe-s-qat-int4-g32/models-pure-java.json",'
    '"reportSha256":"2e0857a7b8def3b4d0999493f85039eac561e5fb3abd3d85eabdf56698776b85","performanceTier":"PRODUCTION_READY",'
    '"verdict":"QUALIFIED","qualified":true,"defaultConfigurationSmoke":{"configuration":"library-defaults","backend":"pure-java",'
    '"artifactSha256":"1a54d8eb2adf19c296a1c129f9cd7d7395c21f5facfc6d13d2945e002d55e37d",'
    '"report":"benchmark-results/certified-20260902/rag/mobilemoe-s-qat-int4-g32/default-correctness/models-pure-java.json",'
    '"reportSha256":"0b2a2f3d30259ec8c50439343b278ffffdaeb9f333dd6d9be5c1046e7bfa5751","totalAttempts":9,"successfulAttempts":9,'
    '"tuningSystemProperties":[]},"attempts":27,"p95RetrievalMillis":1.6134247,"p95TtftMillis":957.9658411,'
    '"p95TpotMillis":85.81608530000003,"p95EndToEndMillis":2315.5560206,"p50PrefillTokensPerSecond":78.33526344583672,'
    '"p50DecodeTokensPerSecond":21.834571524790327,"peakRssBytes":2554105856,"correctAnswerRate":1.0,"rawCorrectAnswerRate":0.0,'
    '"abstentionAccuracy":1.0,"modelAnswerRate":0.3333333333333333,"modelAnswerCorrectRate":1.0,'
    '"extractiveFallbackRate":0.5555555555555556,"environment":{"hostname":"vectors-bench","osName":"Linux",'
    '"osVersion":"6.8.0-124-generic","architecture":"amd64","cpuModel":"AMD EPYC-Milan Processor","availableProcessors":8,'
    '"totalMemoryBytes":32857444352,"maxHeapBytes":8216641536,"javaVersion":"25.0.3","javaVendor":"Eclipse Adoptium",'
    '"vmName":"OpenJDK 64-Bit Server VM"}}'
)


class AssembleBaseQualificationEntryTest(unittest.TestCase):
    def test_reproduces_the_published_mobilemoe_entry_field_for_field(self):
        entry = assemble(
            BUNDLE,
            "Meta MobileMoE-S QAT INT4 G32",
            "benchmark-results/certified-20260902/rag/mobilemoe-s-qat-int4-g32",
        )
        self.assertEqual(entry, PUBLISHED)
        self.assertEqual(list(entry), list(PUBLISHED))

    def test_refuses_a_bundle_whose_report_hash_drifted(self):
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "bundle"
            shutil.copytree(BUNDLE, copy)
            report = copy / "models-pure-java.json"
            report.write_text(report.read_text() + "\n")
            with self.assertRaises(SystemExit) as raised:
                assemble(copy, "x", "p")
            self.assertIn("does not match qualification.json candidate", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
