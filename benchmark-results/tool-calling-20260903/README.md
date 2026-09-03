# Small-model tool-calling qualification — 2026-09-03

This round applied one retained, checksum-pinned tool suite to six small models through the
Models pure-Java backend. The suite contains 14 scenarios derived from the Needle upstream
playground, including multiple calls, typed arguments, crowded tool lists, refusal, and the
Spring AI zipcode regression. Models whose published protocol forbids parallel calls were not
scored on those cases. Generative models also had to turn Craig Walls's weather-tool result into
grounded prose rather than return JSON or request the tool again.

The runtime commit under test was `cff14f65b2c36530e54c2812efa4576416ed3b11`. Every report
records the artifact checksum, suite checksum, prompt template, generation controls, backend
diagnostics, hardware, per-case output, and failure diagnostics.

## Results

| Artifact | Host | Cases passed | Selection | Schema | Arguments | Refusal | Result follow-up | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Needle 2 CQ2 Mixed | EPYC Linux | 12/14 | 100% | 100% | 91.9% | 100% | Host renderer | **PASS** |
| Qwen3 1.7B Q8_0 | EPYC Linux | 12/14 | 100% | 100% | 91.9% | 100% | PASS | **PASS** |
| Qwen3 0.6B Q4_0 | EPYC Linux | 8/14 | 85.7% | 78.6% | 75.7% | 100% | PASS | FAIL |
| MiniCPM5 1B Q4_K_M | Intel macOS | 5/14 | 71.4% | 71.4% | 48.6% | 100% | PASS | FAIL |
| SmolLM3 3B Q4_K_M | EPYC Linux | 11/14 | 92.9% | 85.7% | 86.5% | 100% | PASS | FAIL |
| Llama 3.2 3B Q4_K_M | EPYC Linux | 3/9 | 88.9% | 44.4% | 66.7% | 0% | FAIL | FAIL |

The policy requires exact selection, schema validity, and refusal; at least 90% argument
accuracy; and the required Spring zipcode regression. A model can therefore have individual
case failures and still qualify when they are exact-value differences rather than tool-selection
or schema failures.

## Adapter findings

- Qwen3 1.7B completed the full Spring AI and LangChain4j tool loops against real weights. Both
  frameworks invoked `get-weather-for-zipcode("88252")`, returned a structured Java value, and
  received a grounded natural-language answer without a custom result renderer.
- Needle 2 remains deliberately different: it is a compact tool selector, so applications use a
  typed host-side renderer for the final conversational response.
- MiniCPM5 required schema-aware recovery of its XML `<function>` and `<param>` protocol. The
  adapter now reconstructs JSON arguments with the types declared by the framework schema.
- SmolLM3 uses tagged JSON calls but returns tool results as plain user turns. Its dedicated
  template now follows that upstream convention.
- Protocol implementation did not make a weak artifact pass. MiniCPM5 and SmolLM3 both completed
  the zipcode round trip, but failed the broader selection or argument gates.
- Llama 3.2 frequently serialized numbers, booleans, and arrays as strings, invented a tool on the
  refusal case, and omitted the weather condition from its final response.

## Evidence

- [`needle2-linux-x86_64.json`](needle2-linux-x86_64.json)
- [`qwen3-1.7b-linux-x86_64.json`](qwen3-1.7b-linux-x86_64.json)
- [`qwen3-0.6b-linux-x86_64.json`](qwen3-0.6b-linux-x86_64.json)
- [`minicpm5-1b-macos-x86_64.json`](minicpm5-1b-macos-x86_64.json)
- [`smollm3-3b-linux-x86_64.json`](smollm3-3b-linux-x86_64.json)
- [`llama3.2-3b-linux-x86_64.json`](llama3.2-3b-linux-x86_64.json)
