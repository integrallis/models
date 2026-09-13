import tempfile
import unittest
import hashlib
import json
from pathlib import Path

from evaluate_alora import (
    exact_calls_match,
    inference_device_name,
    parse_completion,
    resolve_qualification_window,
    resolve_adapter_identity,
    select_cases,
    verify_sha256,
)
from prepare_tool_data import canonical_json, query_fingerprint


class EvaluateAloraTest(unittest.TestCase):
    def test_resolves_and_verifies_the_exact_base_from_the_training_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "adapter"
            adapter.mkdir()
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"adapter")
            config = adapter / "adapter_config.json"
            config.write_text("{}")
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            config_digest = hashlib.sha256(config.read_bytes()).hexdigest()
            manifest = root / "training-manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "activated-lora-tool-specialist",
                        "base": {
                            "model": "Qwen/Qwen3-1.7B",
                            "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                        },
                        "adapter": {
                            "files": [
                                {
                                    "name": "adapter_model.safetensors",
                                    "bytes": weights.stat().st_size,
                                    "sha256": digest,
                                },
                                {
                                    "name": "adapter_config.json",
                                    "bytes": config.stat().st_size,
                                    "sha256": config_digest,
                                },
                            ]
                        },
                    }
                )
            )

            identity = resolve_adapter_identity(adapter, manifest)

            self.assertEqual(identity["model"], "Qwen/Qwen3-1.7B")
            self.assertEqual(
                identity["revision"], "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
            )
            self.assertEqual(identity["adapterSha256"], digest)

            weights.write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "adapter file SHA-256 differs"):
                resolve_adapter_identity(adapter, manifest)

            weights.write_bytes(b"adapter")
            config.write_text('{"tampered":true}')
            with self.assertRaisesRegex(ValueError, "adapter file size differs"):
                resolve_adapter_identity(adapter, manifest)

    def test_parses_one_or_multiple_strict_json_tool_blocks(self):
        self.assertEqual(
            parse_completion(
                '<tool_call>\n{"name":"weather","arguments":{"zip":"88252"}}\n</tool_call>\n'
                '<tool_call>{"name":"clock","arguments":{}}</tool_call>'
            ),
            [
                {"name": "weather", "arguments": {"zip": "88252"}},
                {"name": "clock", "arguments": {}},
            ],
        )
        self.assertEqual(parse_completion("<tool_call>\n[]\n</tool_call>"), [])

    def test_rejects_prose_trailing_content_and_invalid_call_shapes(self):
        for completion in (
            "I'll call it. <tool_call>{}</tool_call>",
            '<tool_call>{"name":"weather"}</tool_call> trailing',
            '<tool_call>{"name":"weather","arguments":[]}</tool_call>',
        ):
            with self.subTest(completion=completion):
                with self.assertRaises(ValueError):
                    parse_completion(completion)

    def test_exact_match_accepts_only_explicit_ground_truth_alternatives(self):
        expected = [
            {
                "weather": {
                    "zipcode": ["88252"],
                    "units": ["", "fahrenheit"],
                }
            }
        ]
        self.assertTrue(
            exact_calls_match(
                [{"name": "weather", "arguments": {"zipcode": "88252"}}], expected
            )
        )
        self.assertTrue(
            exact_calls_match(
                [
                    {
                        "name": "weather",
                        "arguments": {"zipcode": "88252", "units": "fahrenheit"},
                    }
                ],
                expected,
            )
        )
        self.assertFalse(
            exact_calls_match(
                [{"name": "weather", "arguments": {"zipcode": "10001"}}], expected
            )
        )
        self.assertFalse(
            exact_calls_match(
                [
                    {
                        "name": "weather",
                        "arguments": {"zipcode": "88252", "extra": True},
                    }
                ],
                expected,
            )
        )

    def test_exact_match_treats_parallel_calls_as_an_unordered_multiset(self):
        expected = [{"first": {"x": [1]}}, {"second": {"y": [2]}}]
        actual = [
            {"name": "second", "arguments": {"y": 2}},
            {"name": "first", "arguments": {"x": 1}},
        ]
        self.assertTrue(exact_calls_match(actual, expected))

    def test_exact_match_uses_bfcl_string_normalization(self):
        expected = [{"weather": {"city": ["New York, NY"]}}]
        actual = [{"name": "weather", "arguments": {"city": "new-york ny"}}]

        self.assertTrue(
            exact_calls_match(
                actual,
                expected,
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        },
                    }
                ],
            )
        )

    def test_exact_match_recursively_checks_bfcl_dictionary_alternatives(self):
        expected = [
            {
                "update_user_info": {
                    "user_id": [43523],
                    "update_info": [
                        {
                            "name": ["John Doe"],
                            "email": ["johndoe@email.com"],
                        }
                    ],
                    "database": ["CustomerInfo", ""],
                }
            }
        ]
        actual = [
            {
                "name": "update_user_info",
                "arguments": {
                    "user_id": 43523,
                    "update_info": {
                        "name": "john-doe",
                        "email": "johndoe@email.com",
                    },
                },
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "update_user_info",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "integer"},
                            "update_info": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "email": {"type": "string"},
                                },
                            },
                            "database": {"type": "string"},
                        },
                    },
                },
            }
        ]

        self.assertTrue(exact_calls_match(actual, expected, tools))

        actual[0]["arguments"]["update_info"]["email"] = "wrong@example.com"
        self.assertFalse(exact_calls_match(actual, expected, tools))

    def test_case_selection_is_stable_and_not_file_order_based(self):
        cases = [{"id": f"simple_{index}"} for index in range(20)]
        selected = select_cases(cases, count=5, seed=20260912)
        self.assertEqual(selected, select_cases(list(reversed(cases)), count=5, seed=20260912))
        self.assertEqual(len(selected), 5)

    def test_case_selection_offset_produces_a_disjoint_unseen_window(self):
        cases = [{"id": f"simple_{index}"} for index in range(200)]
        exposed = select_cases(cases, count=25, seed=20260912)
        unseen = select_cases(cases, count=100, seed=20260912, offset=25)

        self.assertEqual(
            unseen,
            select_cases(list(reversed(cases)), count=100, seed=20260912, offset=25),
        )
        self.assertTrue({case["id"] for case in exposed}.isdisjoint(case["id"] for case in unseen))

    def test_uses_cuda_for_the_gpu_qualification_run(self):
        self.assertEqual(inference_device_name(True), "cuda")
        self.assertEqual(inference_device_name(False), "cpu")

    def test_rejects_an_evaluation_file_that_differs_from_its_pinned_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cases.json"
            path.write_text("cases")
            verify_sha256(path, "352b84777d8dd96ac9c0b3c170ecb2c7cca7fc2dbae41a1ec1ed4286fb2c43db")
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                verify_sha256(path, "0" * 64)

    def test_resolves_exact_cases_from_a_frozen_query_bound_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "possible_answer").mkdir()
            cases = {}
            for kind in ("simple", "multiple", "irrelevance"):
                case = {
                    "id": f"{kind}_7",
                    "question": [[{"role": "user", "content": f"query-{kind}"}]],
                    "function": [
                        {
                            "name": "lookup",
                            "description": "Lookup a value",
                            "parameters": {
                                "type": "object",
                                "properties": {"key": {"type": "string"}},
                            },
                        }
                    ],
                }
                (root / f"BFCL_v3_{kind}.json").write_text(json.dumps(case) + "\n")
                if kind != "irrelevance":
                    answer = {"id": case["id"], "ground_truth": [{"lookup": {"key": [kind]}}]}
                    (root / "possible_answer" / f"BFCL_v3_{kind}.json").write_text(
                        json.dumps(answer) + "\n"
                    )
                cases[kind] = [
                    {
                        "id": case["id"],
                        "querySha256": [query_fingerprint(f"query-{kind}")],
                    }
                ]
            selection_digest = hashlib.sha256(canonical_json(cases).encode()).hexdigest()
            manifest = root / "window.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "status": "frozen-before-training",
                        "source": {
                            "repository": "ShishirPatil/gorilla",
                            "revision": "c15b2a151662cac9839c96d7dfb1493b5329c975",
                            "files": [
                                {"path": path, "sha256": digest}
                                for path, digest in sorted(
                                    {
                                        "BFCL_v3_simple.json": "fbc37b2ad252bf9af985582e0e07b456173fe627d957491472ea9cef5fb83158",
                                        "BFCL_v3_multiple.json": "aef168155ebd74b7ac2401198b201343bc7d16d7a3d7e0d4e6d8ee82c6969b2a",
                                        "BFCL_v3_irrelevance.json": "975f51c51f688649fd190078efd87081241e0a326f9114a2ea3c1ca2440d8690",
                                        "possible_answer/BFCL_v3_simple.json": "2911a2bc00df82c4f999ffa64fedb0164cb88e96212ffa5972087eb91fd496ee",
                                        "possible_answer/BFCL_v3_multiple.json": "244e00ce9395df948bcafc7bee64e8f9c87ef70887587d83cae45b13699f3047",
                                    }.items()
                                )
                            ],
                        },
                        "selection": {
                            "seed": 20260912,
                            "perKind": 1,
                            "caseSetSha256": selection_digest,
                        },
                        "cases": cases,
                    }
                )
            )

            selected, identity = resolve_qualification_window(root, manifest)

            self.assertEqual(
                [(case["kind"], case["id"]) for case in selected],
                [
                    ("simple", "simple_7"),
                    ("multiple", "multiple_7"),
                    ("irrelevance", "irrelevance_7"),
                ],
            )
            self.assertEqual(identity["caseSetSha256"], selection_digest)
            self.assertEqual(identity["manifestSha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())

            tampered = json.loads(manifest.read_text())
            tampered["cases"]["simple"][0]["querySha256"] = ["0" * 64]
            tampered["selection"]["caseSetSha256"] = hashlib.sha256(
                canonical_json(tampered["cases"]).encode()
            ).hexdigest()
            manifest.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, "query fingerprints differ"):
                resolve_qualification_window(root, manifest)


if __name__ == "__main__":
    unittest.main()
