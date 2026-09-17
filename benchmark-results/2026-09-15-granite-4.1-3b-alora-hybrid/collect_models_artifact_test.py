import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from assemble_component_report import REQUIRED_MODELS_MODULES, validate_models_artifact
from collect_models_artifact import collect, main


def fake_central(requested):
    """A fetcher that serves deterministic bytes per URI and records every request."""

    def fetch(uri):
        requested.append(uri)
        return f"bytes of {uri}".encode()

    return fetch


class CollectModelsArtifactTest(unittest.TestCase):
    def test_shapes_every_required_module_jar_and_pom_in_the_gate_shape(self):
        requested = []
        gate = collect("0.3.42", fake_central(requested))

        validate_models_artifact(gate)
        self.assertTrue(gate["pass"])
        self.assertEqual("0.3.42", gate["version"])
        self.assertEqual(REQUIRED_MODELS_MODULES, [a["module"] for a in gate["artifacts"]])
        runtime = next(a for a in gate["artifacts"] if a["module"] == "models-runtime")
        uri = "https://repo1.maven.org/maven2/com/integrallis/models-runtime/0.3.42/models-runtime-0.3.42.jar"
        self.assertEqual("com.integrallis:models-runtime:0.3.42", runtime["coordinate"])
        self.assertEqual(
            {"uri": uri, "sha256": hashlib.sha256(f"bytes of {uri}".encode()).hexdigest(), "sizeBytes": len(f"bytes of {uri}")},
            runtime["jar"],
        )
        self.assertTrue(runtime["pom"]["uri"].endswith("/models-runtime-0.3.42.pom"))
        self.assertEqual(2 * len(REQUIRED_MODELS_MODULES), len(requested))

    def test_refuses_a_non_release_version_and_an_empty_download(self):
        with self.assertRaisesRegex(ValueError, "release version"):
            collect("0.3.42-SNAPSHOT", fake_central([]))
        with self.assertRaisesRegex(ValueError, "empty"):
            collect("0.3.42", lambda uri: b"")

    def test_command_line_writes_the_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "models-artifact.json"
            main(["--version", "0.3.42", "--output", str(output)], fetch=fake_central([]))
            written = json.loads(output.read_text())
        self.assertEqual(collect("0.3.42", fake_central([])), written)


if __name__ == "__main__":
    unittest.main()
