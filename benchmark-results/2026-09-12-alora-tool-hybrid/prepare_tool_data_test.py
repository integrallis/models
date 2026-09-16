import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from prepare_tool_data import (
    HAMMER_QUERY_TOOL_AMBIGUITY_THRESHOLD,
    NO_TOOL_SENTINEL,
    apply_function_mask,
    external_irrelevance_record,
    external_irrelevance_records,
    filter_external_irrelevance,
    maximum_query_tool_similarity,
    normalize_tools,
    no_call_hardness,
    parse_calls,
    positive_call_index,
    query_fingerprint,
    render_calls,
    select_normalized_records,
    select_records,
    split_stratified,
    training_record,
    verify_source_sha256,
)


class PrepareToolDataTest(unittest.TestCase):
    def test_function_mask_replaces_names_consistently_without_changing_semantics(self):
        record = {
            "sourceLine": 7,
            "user": "Weather in 88252?",
            "tools": normalize_tools(
                [
                    {
                        "name": "get_weather",
                        "description": "Get weather for a postal code",
                        "parameters": {
                            "zipcode": {"type": "str", "description": "Postal code"},
                            "days": {"type": "int, optional", "default": 1},
                        },
                    }
                ]
            ),
            "calls": [
                {
                    "name": "get_weather",
                    "arguments": {"zipcode": "88252", "days": 1},
                }
            ],
            "assistant": "original",
        }

        masked = apply_function_mask([record], 1.0, 20260918)[0]

        function = masked["tools"][0]["function"]
        call = masked["calls"][0]
        self.assertNotEqual("get_weather", function["name"])
        self.assertEqual(function["name"], call["name"])
        self.assertEqual("Get weather for a postal code", function["description"])
        self.assertEqual("Weather in 88252?", masked["user"])
        self.assertEqual({"88252", 1}, set(call["arguments"].values()))
        self.assertEqual(
            set(function["parameters"]["properties"]), set(call["arguments"])
        )
        self.assertEqual(
            [
                next(
                    name
                    for name, value in call["arguments"].items()
                    if value == "88252"
                )
            ],
            function["parameters"]["required"],
        )
        self.assertEqual(render_calls(masked["calls"]), masked["assistant"])
        self.assertEqual("get_weather", record["tools"][0]["function"]["name"])
        self.assertRegex(function["name"], r"^f\d+$")
        self.assertTrue(
            all(re.fullmatch(r"p\d+", name) for name in call["arguments"])
        )

    def test_function_mask_is_deterministic_and_preserves_no_call_labels(self):
        records = [
            {
                "sourceLine": index,
                "user": f"Question {index}",
                "tools": normalize_tools(
                    [{"name": "lookup", "parameters": {"value": {"type": "str"}}}]
                ),
                "calls": [],
                "assistant": NO_TOOL_SENTINEL,
            }
            for index in range(1, 5)
        ]

        first = apply_function_mask(records, 2 / 3, 20260918)
        second = apply_function_mask(records, 2 / 3, 20260918)

        self.assertEqual(first, second)
        self.assertTrue(all(record["calls"] == [] for record in first))
        self.assertTrue(all(record["assistant"] == NO_TOOL_SENTINEL for record in first))
        self.assertEqual(records, apply_function_mask(records, 0.0, 20260918))

    def test_function_mask_rejects_an_invalid_fraction(self):
        with self.assertRaisesRegex(ValueError, "function mask fraction"):
            apply_function_mask([], 1.01, 20260918)

    def test_function_mask_accepts_equivalent_duplicate_tool_declarations(self):
        weather = normalize_tools(
            [{"name": "weather", "parameters": {"zipcode": {"type": "str"}}}]
        )[0]
        record = {
            "sourceLine": 17,
            "user": "Weather in 88252?",
            "tools": [weather, json.loads(json.dumps(weather))],
            "calls": [{"name": "weather", "arguments": {"zipcode": "88252"}}],
            "assistant": "original",
        }

        masked = apply_function_mask([record], 1.0, 20260918)[0]

        self.assertEqual(
            masked["tools"][0]["function"]["name"],
            masked["tools"][1]["function"]["name"],
        )
        self.assertEqual(masked["calls"][0]["name"], masked["tools"][0]["function"]["name"])

    def test_function_mask_accepts_duplicate_callables_with_different_descriptions(self):
        first = normalize_tools(
            [
                {
                    "name": "getuserbyname",
                    "description": "Fetches user information from the Petstore Blitz API.",
                    "parameters": {
                        "username": {
                            "type": "str",
                            "description": "The name of the user to fetch information for.",
                        }
                    },
                }
            ]
        )[0]
        second = normalize_tools(
            [
                {
                    "name": "getuserbyname",
                    "description": "Fetches user information by username.",
                    "parameters": {
                        "username": {
                            "type": "str",
                            "description": "The user name that needs to be fetched.",
                        }
                    },
                }
            ]
        )[0]
        record = {
            "sourceLine": 31686,
            "user": "Get the user named janedoe.",
            "tools": [first, second],
            "calls": [{"name": "getuserbyname", "arguments": {"username": "janedoe"}}],
            "assistant": "original",
        }

        masked = apply_function_mask([record], 1.0, 20260918)[0]

        self.assertEqual(
            masked["tools"][0]["function"]["name"],
            masked["tools"][1]["function"]["name"],
        )
        self.assertEqual(masked["calls"][0]["name"], masked["tools"][0]["function"]["name"])
        self.assertEqual(
            set(masked["calls"][0]["arguments"]),
            set(masked["tools"][0]["function"]["parameters"]["properties"]),
        )

    def test_normalize_tools_preserves_compatible_duplicate_callables(self):
        tools = normalize_tools(
            [
                {
                    "name": "lookup",
                    "description": "First description.",
                    "parameters": {
                        "query": {"type": "str", "description": "First wording."}
                    },
                },
                {
                    "name": "lookup",
                    "description": "Second description.",
                    "parameters": {
                        "query": {"type": "str", "description": "Second wording."}
                    },
                },
            ]
        )

        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0]["function"]["description"], "First description.")
        self.assertEqual(tools[1]["function"]["description"], "Second description.")

    def test_normalize_tools_rejects_ambiguous_duplicate_callables(self):
        with self.assertRaisesRegex(ValueError, "ambiguous duplicate tool declaration: search"):
            normalize_tools(
                [
                    {"name": "search", "parameters": {"query": {"type": "str"}}},
                    {"name": "search", "parameters": {"page": {"type": "int"}}},
                ]
            )

    def test_function_mask_rejects_conflicting_duplicate_tool_declarations(self):
        first = normalize_tools(
            [{"name": "weather", "parameters": {"zipcode": {"type": "str"}}}]
        )[0]
        second = normalize_tools(
            [{"name": "weather", "parameters": {"city": {"type": "str"}}}]
        )[0]
        record = {
            "sourceLine": 19,
            "user": "Weather?",
            "tools": [first, second],
            "calls": [],
            "assistant": NO_TOOL_SENTINEL,
        }

        with self.assertRaisesRegex(ValueError, "conflicting duplicate tool"):
            apply_function_mask([record], 1.0, 20260918)

    def test_verifies_pinned_source_content_before_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            source.write_text("pinned source")
            expected = hashlib.sha256(b"pinned source").hexdigest()

            verify_source_sha256(source, expected, "training source")
            source.write_text("changed source")

            with self.assertRaisesRegex(ValueError, "training source SHA-256 differs"):
                verify_source_sha256(source, expected, "training source")

    def test_parses_multiple_dotted_keyword_only_calls(self):
        self.assertEqual(
            parse_calls("[weather.now(zipcode='88252'), math.add(left=-2, right=3.5)]"),
            [
                {"name": "weather.now", "arguments": {"zipcode": "88252"}},
                {"name": "math.add", "arguments": {"left": -2, "right": 3.5}},
            ],
        )

    def test_rejects_executable_argument_syntax(self):
        with self.assertRaisesRegex(ValueError, "unsupported tool argument syntax"):
            parse_calls("[open_file(path=make_path())]")

    def test_normalizes_source_types_to_json_schema(self):
        tools = normalize_tools(
            [
                {
                    "name": "weather",
                    "description": "Weather",
                    "parameters": {
                        "zipcode": {"type": "str", "description": "Postal code"},
                        "days": {"type": "List[int], optional", "default": [1]},
                    },
                }
            ]
        )
        self.assertEqual(tools[0]["type"], "function")
        schema = tools[0]["function"]["parameters"]
        self.assertEqual(schema["properties"]["zipcode"]["type"], "string")
        self.assertEqual(
            schema["properties"]["days"], {"type": "array", "items": {"type": "integer"}}
        )
        self.assertEqual(schema["required"], ["zipcode"])

    def test_preserves_already_normalized_openai_tools(self):
        source = [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Weather",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "zipcode": {"type": "string", "enum": ["88252", "10001"]}
                        },
                        "required": ["zipcode"],
                    },
                },
            }
        ]
        tools = normalize_tools(source)
        self.assertEqual(tools[0]["function"]["parameters"]["type"], "object")
        self.assertEqual(
            tools[0]["function"]["parameters"]["properties"]["zipcode"]["enum"],
            ["88252", "10001"],
        )

    def test_normalizes_nested_collection_types(self):
        tools = normalize_tools(
            [
                {
                    "name": "matrix",
                    "parameters": {
                        "rows": {"type": "List[List[int]]"},
                        "range": {"type": "Tuple[float, float]"},
                        "lookup": {"type": "HashMap"},
                    },
                }
            ]
        )
        properties = tools[0]["function"]["parameters"]["properties"]
        self.assertEqual(
            properties["rows"],
            {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
        )
        self.assertEqual(properties["range"]["type"], "array")
        self.assertEqual(properties["lookup"]["type"], "object")

    def test_rejects_malformed_parameter_schema(self):
        with self.assertRaisesRegex(ValueError, "parameter schema"):
            normalize_tools([{"name": "bad", "parameters": {"value": "str"}}])

    def test_empty_call_list_has_an_explicit_fallback_sentinel(self):
        self.assertEqual(render_calls([]), NO_TOOL_SENTINEL)

    def test_training_record_preserves_tool_and_call_semantics(self):
        source = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are helpful.\nList of tools: "
                    + json.dumps(
                        [
                            {
                                "name": "weather",
                                "parameters": {"zipcode": {"type": "str"}},
                            }
                        ]
                    ),
                },
                {"role": "user", "content": "Weather in 88252?"},
                {"role": "assistant", "content": "[weather(zipcode='88252')]"},
            ]
        }
        record = training_record(json.dumps(source), 7)
        self.assertEqual(record["sourceLine"], 7)
        self.assertEqual(record["calls"][0]["arguments"], {"zipcode": "88252"})
        self.assertIn('"name":"weather"', record["assistant"])

    def test_training_record_rejects_calls_outside_the_declared_schema(self):
        source = {
            "messages": [
                {
                    "role": "system",
                    "content": "List of tools: "
                    + json.dumps([{"name": "weather", "parameters": {"zipcode": {"type": "str"}}}]),
                },
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": "[other(city='Jal')]"},
            ]
        }
        with self.assertRaisesRegex(ValueError, "undeclared tool"):
            training_record(json.dumps(source), 9)

    def test_training_record_rejects_missing_required_arguments(self):
        source = {
            "messages": [
                {
                    "role": "system",
                    "content": "List of tools: "
                    + json.dumps([{"name": "weather", "parameters": {"zipcode": {"type": "str"}}}]),
                },
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": "[weather()]"},
            ]
        }
        with self.assertRaisesRegex(ValueError, "missing required"):
            training_record(json.dumps(source), 11)

    def test_normalizes_a_pinned_hammer_irrelevance_record(self):
        record = external_irrelevance_record(
            {
                "query": "Who won the 2019 NCAA Final Four?",
                "tools": json.dumps(
                    [
                        {
                            "name": "get_nba_game",
                            "description": "Get one NBA game by id",
                            "parameters": {"game_id": {"type": "str"}},
                        }
                    ]
                ),
                "answers": "[]",
            },
            17,
        )

        self.assertEqual(record["sourceLine"], 17)
        self.assertEqual(record["calls"], [])
        self.assertEqual(record["assistant"], NO_TOOL_SENTINEL)
        self.assertEqual(record["tools"][0]["function"]["name"], "get_nba_game")

    def test_rejects_a_non_irrelevant_external_record(self):
        with self.assertRaisesRegex(ValueError, "must contain an empty answer"):
            external_irrelevance_record(
                {
                    "query": "Weather in Jal?",
                    "tools": "[]",
                    "answers": '[{"name":"weather","arguments":{}}]',
                },
                19,
            )

    def test_external_irrelevance_population_reports_and_skips_invalid_schemas(self):
        valid = {
            "query": "Who won the NCAA Final Four?",
            "tools": json.dumps(
                [{"name": "nba_game", "parameters": {"game": {"type": "str"}}}]
            ),
            "answers": "[]",
        }
        invalid = {
            "query": "Integrate this function",
            "tools": json.dumps(
                [
                    {
                        "name": "integrate",
                        "parameters": {"function": {"type": "Callable[[float], float]"}},
                    }
                ]
            ),
            "answers": "[]",
        }

        records, statistics = external_irrelevance_records([valid, invalid])

        self.assertEqual(len(records), 1)
        self.assertEqual(statistics, {"read": 2, "invalid": 1})

    def test_filters_ambiguous_or_unverifiable_function_masking_negatives(self):
        weather = normalize_tools(
            [
                {
                    "name": "weather_kentucky",
                    "description": "Fetch current gas prices for Kentucky",
                    "parameters": {},
                }
            ]
        )[0]["function"]
        near_duplicate = normalize_tools(
            [
                {
                    "name": "weather_georgia",
                    "description": "Fetch current gas prices for Georgia",
                    "parameters": {},
                }
            ]
        )
        unrelated = normalize_tools(
            [{"name": "nba_game", "description": "Fetch one NBA game", "parameters": {}}]
        )
        applicable_distractor = normalize_tools(
            [
                {
                    "name": "get_instagram_user",
                    "description": "Fetch an Instagram user profile",
                    "parameters": {"username": {"type": "str"}},
                }
            ]
        )
        geolocation_distractor = normalize_tools(
            [
                {
                    "name": "ip_geolocation",
                    "description": "Retrieves geolocation information for an IP address",
                    "parameters": {"ip": {"type": "str"}, "format": {"type": "str"}},
                },
                {
                    "name": "unrelated_catalog",
                    "description": " ".join(
                        f"unrelatedterm{index}" for index in range(80)
                    ),
                    "parameters": {},
                },
            ]
        )
        opaque_called_tool = normalize_tools(
            [{"name": "info", "description": "Call endpoint A", "parameters": {}}]
        )[0]["function"]
        records = [
            {
                "sourceLine": 1,
                "user": "Gas prices in Kentucky",
                "tools": near_duplicate,
                "calls": [],
                "assistant": NO_TOOL_SENTINEL,
            },
            {
                "sourceLine": 2,
                "user": "Gas prices in Kentucky",
                "tools": unrelated,
                "calls": [],
                "assistant": NO_TOOL_SENTINEL,
            },
            {
                "sourceLine": 3,
                "user": "Unknown query",
                "tools": unrelated,
                "calls": [],
                "assistant": NO_TOOL_SENTINEL,
            },
            {
                "sourceLine": 4,
                "user": "Fetch the Instagram profile for lewishamilton",
                "tools": applicable_distractor,
                "calls": [],
                "assistant": NO_TOOL_SENTINEL,
            },
            {
                "sourceLine": 5,
                "user": "Get geolocation details for IP address 8.8.8.8 in JSON format",
                "tools": geolocation_distractor,
                "calls": [],
                "assistant": NO_TOOL_SENTINEL,
            },
        ]
        positive_index = {
            query_fingerprint("Gas prices in Kentucky"): [
                {"callCount": 1, "calledFunctions": [weather]}
            ],
            query_fingerprint("Fetch the Instagram profile for lewishamilton"): [
                {"callCount": 1, "calledFunctions": [opaque_called_tool]}
            ],
            query_fingerprint(
                "Get geolocation details for IP address 8.8.8.8 in JSON format"
            ): [{"callCount": 1, "calledFunctions": [opaque_called_tool]}],
        }

        retained, statistics = filter_external_irrelevance(
            records,
            positive_index,
            0.45,
            HAMMER_QUERY_TOOL_AMBIGUITY_THRESHOLD,
        )

        self.assertEqual([record["sourceLine"] for record in retained], [2])
        self.assertEqual(statistics["ambiguousDistractor"], 1)
        self.assertEqual(statistics["queryToolAmbiguous"], 2)
        self.assertEqual(statistics["missingPositiveEvidence"], 1)

    def test_query_tool_similarity_is_not_diluted_by_unrelated_catalog_entries(self):
        exact = normalize_tools(
            [
                {
                    "name": "ip_geolocation",
                    "description": "Retrieves geolocation information for an IP address",
                    "parameters": {"ip": {"type": "str"}, "format": {"type": "str"}},
                },
                {
                    "name": "unrelated_catalog",
                    "description": " ".join(
                        f"unrelatedterm{index}" for index in range(80)
                    ),
                    "parameters": {},
                },
            ]
        )
        record = {
            "user": "Get geolocation details for IP address 8.8.8.8 in JSON format",
            "tools": exact,
            "calls": [],
        }

        self.assertGreater(maximum_query_tool_similarity(record), 0.25)
        self.assertLess(no_call_hardness(record), 0.15)

    def test_selects_normalized_records_deterministically_and_excludes_eval_queries(self):
        records = [
            {
                "sourceLine": index,
                "user": f"question {index}",
                "tools": [],
                "calls": [],
                "assistant": NO_TOOL_SENTINEL,
            }
            for index in range(8)
        ]
        excluded = {query_fingerprint("question 3")}

        first, statistics = select_normalized_records(records, excluded, 4, 41)
        second, _ = select_normalized_records(reversed(records), excluded, 4, 41)

        self.assertEqual(first, second)
        self.assertNotIn("question 3", [record["user"] for record in first])
        self.assertEqual(statistics["overlap"], 1)

    def test_query_fingerprint_ignores_case_and_whitespace(self):
        self.assertEqual(query_fingerprint(" Weather   NOW "), query_fingerprint("weather now"))

    def test_selection_is_deterministic_and_excludes_evaluation_queries(self):
        def line(index):
            return json.dumps(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "List of tools: "
                            + json.dumps([{"name": "f", "parameters": {}}]),
                        },
                        {"role": "user", "content": f"question {index}"},
                        {"role": "assistant", "content": "[f()]"},
                    ]
                }
            )

        lines = [line(index) for index in range(10)]
        excluded = {query_fingerprint("question 4")}
        first, statistics = select_records(lines, excluded, 4, 19)
        second, _ = select_records(lines, excluded, 4, 19)
        self.assertEqual(first, second)
        self.assertNotIn("question 4", [record["user"] for record in first])
        self.assertEqual(statistics["overlap"], 1)

    def test_selection_can_reserve_an_exact_no_call_population(self):
        def line(index, calls):
            assistant = "[]" if calls == 0 else "[f()]"
            return json.dumps(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "List of tools: "
                            + json.dumps([{"name": "f", "parameters": {}}]),
                        },
                        {"role": "user", "content": f"question {index}"},
                        {"role": "assistant", "content": assistant},
                    ]
                }
            )

        lines = [line(index, index % 3) for index in range(30)]

        selected, _ = select_records(lines, set(), 12, 23, no_call_count=4)

        self.assertEqual(sum(not record["calls"] for record in selected), 4)
        self.assertEqual(len(selected), 12)

    def test_hardness_prefers_irrelevant_queries_that_overlap_the_available_tool(self):
        weather_tool = normalize_tools(
            [
                {
                    "name": "get_weather_forecast",
                    "description": "Get forecast conditions for a city",
                    "parameters": {"city": {"type": "str"}},
                }
            ]
        )
        near_match = {
            "user": "Who first published the weather forecast for London?",
            "tools": weather_tool,
            "calls": [],
        }
        easy_mismatch = {
            "user": "Explain the history of abstract expressionism",
            "tools": weather_tool,
            "calls": [],
        }

        self.assertGreater(no_call_hardness(near_match), no_call_hardness(easy_mismatch))

    def test_selection_can_reserve_hard_no_call_examples(self):
        def line(index, user, tool_name):
            return json.dumps(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "List of tools: "
                            + json.dumps(
                                [
                                    {
                                        "name": tool_name,
                                        "description": "Weather forecast conditions by city",
                                        "parameters": {"city": {"type": "str"}},
                                    }
                                ]
                            ),
                        },
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": "[]"},
                    ]
                }
            )

        lines = [
            line(0, "Explain a cubist painting", "get_weather_forecast"),
            line(1, "Who published the first city weather forecast?", "get_weather_forecast"),
            line(2, "Write a poem about an oak tree", "get_weather_forecast"),
            line(3, "What does a city planner do?", "get_weather_forecast"),
        ]

        selected, statistics = select_records(
            lines,
            set(),
            count=2,
            seed=23,
            no_call_count=2,
            hard_no_call_count=1,
        )

        self.assertIn(
            "Who published the first city weather forecast?",
            [record["user"] for record in selected],
        )
        self.assertEqual(statistics["selectedHardNoCalls"], 1)

    def test_positive_call_index_retains_the_called_function_definition(self):
        source = json.dumps(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "List of tools: "
                        + json.dumps(
                            [
                                {
                                    "name": "weather_kentucky",
                                    "description": "Kentucky gas prices",
                                    "parameters": {},
                                },
                                {
                                    "name": "nba_game",
                                    "description": "One NBA game",
                                    "parameters": {},
                                },
                            ]
                        ),
                    },
                    {"role": "user", "content": "Gas prices in Kentucky"},
                    {"role": "assistant", "content": "[weather_kentucky()]"},
                ]
            }
        )

        index, statistics = positive_call_index([source])

        evidence = index[query_fingerprint("Gas prices in Kentucky")]
        self.assertEqual(evidence[0]["callCount"], 1)
        self.assertEqual(
            evidence[0]["calledFunctions"][0]["name"], "weather_kentucky"
        )
        self.assertEqual(statistics["positive"], 1)

    def test_stratified_split_keeps_the_declared_balance_in_both_sets(self):
        records = [
            {
                "sourceLine": index,
                "user": f"question {index}",
                "tools": [],
                "calls": [] if index < 3 else [{"name": "f", "arguments": {}}],
            }
            for index in range(12)
        ]

        train, validation = split_stratified(records, 8, 4, 0.25, 29)

        self.assertEqual(len(train), 8)
        self.assertEqual(len(validation), 4)
        self.assertEqual(sum(not record["calls"] for record in train), 2)
        self.assertEqual(sum(not record["calls"] for record in validation), 1)
        self.assertTrue(set(record["sourceLine"] for record in train).isdisjoint(
            record["sourceLine"] for record in validation
        ))


if __name__ == "__main__":
    unittest.main()
