import unittest

from generate_projection_oracle import MODULES, properties_text, tokenizer_probe


class GenerateProjectionOracleTest(unittest.TestCase):
    def test_covers_every_java_projection_once(self):
        self.assertEqual(
            [projection for projection, _ in MODULES],
            [
                "QUERY",
                "KEY",
                "VALUE",
                "ATTENTION_OUTPUT",
                "FFN_GATE",
                "FFN_UP",
                "FFN_DOWN",
            ],
        )
        self.assertEqual(len(set(module for _, module in MODULES)), len(MODULES))

    def test_properties_are_canonical_and_reject_line_injection(self):
        self.assertEqual(
            properties_text({"z": 2, "a": 1}),
            "# Generated aLoRA projection oracle; do not edit.\na=1\nz=2\n",
        )
        with self.assertRaisesRegex(ValueError, "line breaks"):
            properties_text({"bad": "one\ntwo"})

    def test_tokenizer_probe_uses_the_exact_qwen_tool_training_envelope(self):
        class FakeTokenizer:
            def __init__(self):
                self.render_arguments = None
                self.encode_arguments = None

            def apply_chat_template(self, messages, **arguments):
                self.render_arguments = (messages, arguments)
                return "rendered tool prompt"

            def encode(self, prompt, **arguments):
                self.encode_arguments = (prompt, arguments)
                return [151644, 8948, 198]

        tokenizer = FakeTokenizer()

        prompt, token_ids = tokenizer_probe(tokenizer)

        self.assertEqual(prompt, "rendered tool prompt")
        self.assertEqual(token_ids, [151644, 8948, 198])
        messages, render_arguments = tokenizer.render_arguments
        self.assertEqual(messages, [{"role": "user", "content": "What is the weather for 88252?"}])
        self.assertEqual(render_arguments["tokenize"], False)
        self.assertEqual(render_arguments["add_generation_prompt"], True)
        self.assertEqual(render_arguments["enable_thinking"], False)
        self.assertEqual(
            render_arguments["tools"][0]["function"]["name"],
            "get-weather-for-zipcode",
        )
        self.assertEqual(
            render_arguments["tools"][0]["function"]["parameters"]["required"],
            ["zipcode"],
        )
        self.assertEqual(
            tokenizer.encode_arguments,
            ("rendered tool prompt", {"add_special_tokens": False}),
        )


if __name__ == "__main__":
    unittest.main()
