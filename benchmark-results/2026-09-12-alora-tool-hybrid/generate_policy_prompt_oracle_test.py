import unittest

from generate_policy_prompt_oracle import INVOCATION_TOKENS, policy_tokenizer_probe


class GeneratePolicyPromptOracleTest(unittest.TestCase):
    def test_places_the_policy_once_before_tools_and_preserves_the_activation_sequence(self):
        class FakeTokenizer:
            def __init__(self):
                self.messages = None

            def apply_chat_template(self, messages, **arguments):
                self.messages = messages
                self.assertions = arguments
                return "strict policy\n# Tools\nrendered"

            def encode(self, prompt, **arguments):
                self.encoded = (prompt, arguments)
                return [7, *INVOCATION_TOKENS, 8]

        tokenizer = FakeTokenizer()

        prompt, token_ids = policy_tokenizer_probe(tokenizer, "strict policy")

        self.assertEqual(prompt.count("strict policy"), 1)
        self.assertLess(prompt.index("strict policy"), prompt.index("# Tools"))
        self.assertEqual(token_ids, [7, *INVOCATION_TOKENS, 8])
        self.assertEqual(
            tokenizer.messages[0], {"role": "system", "content": "strict policy"}
        )
        self.assertEqual(tokenizer.messages[1]["role"], "user")
        self.assertEqual(tokenizer.assertions["tools"][0]["function"]["name"], "get-weather-for-zipcode")
        self.assertEqual(tokenizer.encoded[1], {"add_special_tokens": False})

    def test_rejects_an_oracle_without_the_exact_activation_sequence(self):
        class MissingInvocationTokenizer:
            def apply_chat_template(self, messages, **arguments):
                return "strict policy\n# Tools"

            def encode(self, prompt, **arguments):
                return [1, 2, 3]

        with self.assertRaisesRegex(ValueError, "invocation sequence"):
            policy_tokenizer_probe(MissingInvocationTokenizer(), "strict policy")


if __name__ == "__main__":
    unittest.main()
