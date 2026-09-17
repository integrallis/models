# Repetition-loop detector on greedy fixture outputs

- java: 25.0.3
- os.arch: x86_64
- maxTokens: 200
- contextLength: 512
- sampling: temperature=0 (greedy), repetitionPenalty=1.0, minP=0, no stop sequences
- detector configurations (maxSpan, minRepeats, minLoopTokens):
  - off: disabled (baseline)
  - A: (32, 4, 16)
  - B: (64, 3, 32)
  - C: (16, 10, 64)

- model `Qwen3-0.6B-Q4_0.gguf` sha256 `da2572f16c06133561ce56accaa822216f2391ef4d37fba427801cd6736417d4`
  - resolved end-of-generation ids: [128247, 151643, 151645]
- model `smollm2-360m-instruct-q8_0.gguf` sha256 `48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201`
  - resolved end-of-generation ids: [0, 2]
- model `qwen2.5-coder-0.5b-instruct-q4_0.gguf` sha256 `9739055e046d62a937e5b7879012209ef40ebea8a1569a96028de491f3f091d5`
  - resolved end-of-generation ids: [128247, 151643, 151645]
- model `tinyllama-1.1b-chat-v1.0.Q4_0.gguf` sha256 `da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556`
  - resolved end-of-generation ids: [2]

Prompts (written for this experiment, not a published dataset):

- P0: `The quick brown fox`
- P1: `Continue_only_the_exact_sequence_without_explanation:_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_`
- P2: `List the numbers from 1 to 50, one per line:\n1\n2\n`
- P3: `Write a short poem about the sea.\n`
- P4: `Repeat after me: I am a robot. I am a robot. I am a robot.`
- P5: `def fibonacci(n):`

| model | prompt | detector | stop reason | completion tokens | loop stops (loop counter) | prefix of baseline | last 60 chars at stop |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B-Q4_0.gguf | P0 | off | MAX_TOKENS | 200 | 0 | true | `x. The dog is a dog. The dog is a dog. The fox is a fox. The` |
| Qwen3-0.6B-Q4_0.gguf | P0 | A | REPETITION_LOOP | 107 | 1 | true | `g is a dog. The dog is a dog. The fox is a fox. The fox is a` |
| Qwen3-0.6B-Q4_0.gguf | P0 | B | REPETITION_LOOP | 83 | 1 | true | `g is a dog. The dog is a dog. The fox is a fox. The fox is a` |
| Qwen3-0.6B-Q4_0.gguf | P0 | C | MAX_TOKENS | 200 | 0 | true | `x. The dog is a dog. The dog is a dog. The fox is a fox. The` |
| Qwen3-0.6B-Q4_0.gguf | P1 | off | MAX_TOKENS | 200 | 0 | true | `3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_` |
| Qwen3-0.6B-Q4_0.gguf | P1 | A | REPETITION_LOOP | 32 | 1 | true | `1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_` |
| Qwen3-0.6B-Q4_0.gguf | P1 | B | REPETITION_LOOP | 32 | 1 | true | `1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_` |
| Qwen3-0.6B-Q4_0.gguf | P1 | C | REPETITION_LOOP | 80 | 1 | true | `3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_` |
| Qwen3-0.6B-Q4_0.gguf | P2 | off | MAX_TOKENS | 200 | 0 | true | `rst list is the numbers from 1 to 50, and the second list is` |
| Qwen3-0.6B-Q4_0.gguf | P2 | A | MAX_TOKENS | 200 | 0 | true | `rst list is the numbers from 1 to 50, and the second list is` |
| Qwen3-0.6B-Q4_0.gguf | P2 | B | MAX_TOKENS | 200 | 0 | true | `rst list is the numbers from 1 to 50, and the second list is` |
| Qwen3-0.6B-Q4_0.gguf | P2 | C | MAX_TOKENS | 200 | 0 | true | `rst list is the numbers from 1 to 50, and the second list is` |
| Qwen3-0.6B-Q4_0.gguf | P3 | off | MAX_TOKENS | 200 | 0 | true | ` me know,**\n**The sea never leaves me.**\n\n**Now, let me hear` |
| Qwen3-0.6B-Q4_0.gguf | P3 | A | MAX_TOKENS | 200 | 0 | true | ` me know,**\n**The sea never leaves me.**\n\n**Now, let me hear` |
| Qwen3-0.6B-Q4_0.gguf | P3 | B | MAX_TOKENS | 200 | 0 | true | ` me know,**\n**The sea never leaves me.**\n\n**Now, let me hear` |
| Qwen3-0.6B-Q4_0.gguf | P3 | C | MAX_TOKENS | 200 | 0 | true | ` me know,**\n**The sea never leaves me.**\n\n**Now, let me hear` |
| Qwen3-0.6B-Q4_0.gguf | P4 | off | MAX_TOKENS | 200 | 0 | true | `bot. I am a robot. I am a robot. I am a robot. I am a robot.` |
| Qwen3-0.6B-Q4_0.gguf | P4 | A | REPETITION_LOOP | 20 | 1 | true | ` I am a robot. I am a robot. I am a robot. I am a robot.` |
| Qwen3-0.6B-Q4_0.gguf | P4 | B | REPETITION_LOOP | 32 | 1 | true | `I am a robot. I am a robot. I am a robot. I am a robot. I am` |
| Qwen3-0.6B-Q4_0.gguf | P4 | C | REPETITION_LOOP | 64 | 1 | true | `obot. I am a robot. I am a robot. I am a robot. I am a robot` |
| Qwen3-0.6B-Q4_0.gguf | P5 | off | MAX_TOKENS | 200 | 0 | true | `manually. Let's compute the Fibonacci sequence up to n=10:\n\n` |
| Qwen3-0.6B-Q4_0.gguf | P5 | A | MAX_TOKENS | 200 | 0 | true | `manually. Let's compute the Fibonacci sequence up to n=10:\n\n` |
| Qwen3-0.6B-Q4_0.gguf | P5 | B | MAX_TOKENS | 200 | 0 | true | `manually. Let's compute the Fibonacci sequence up to n=10:\n\n` |
| Qwen3-0.6B-Q4_0.gguf | P5 | C | MAX_TOKENS | 200 | 0 | true | `manually. Let's compute the Fibonacci sequence up to n=10:\n\n` |
| smollm2-360m-instruct-q8_0.gguf | P0 | off | EOS | 6 | 0 | true | ` jumps over the lazy dog.` |
| smollm2-360m-instruct-q8_0.gguf | P0 | A | EOS | 6 | 0 | true | ` jumps over the lazy dog.` |
| smollm2-360m-instruct-q8_0.gguf | P0 | B | EOS | 6 | 0 | true | ` jumps over the lazy dog.` |
| smollm2-360m-instruct-q8_0.gguf | P0 | C | EOS | 6 | 0 | true | ` jumps over the lazy dog.` |
| smollm2-360m-instruct-q8_0.gguf | P1 | off | MAX_TOKENS | 200 | 0 | true | `3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_` |
| smollm2-360m-instruct-q8_0.gguf | P1 | A | REPETITION_LOOP | 32 | 1 | true | `1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_` |
| smollm2-360m-instruct-q8_0.gguf | P1 | B | REPETITION_LOOP | 32 | 1 | true | `1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_` |
| smollm2-360m-instruct-q8_0.gguf | P1 | C | REPETITION_LOOP | 80 | 1 | true | `3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_` |
| smollm2-360m-instruct-q8_0.gguf | P2 | off | EOS | 148 | 0 | true | `\n49\n50\n\nNote that the numbers are listed in ascending order.` |
| smollm2-360m-instruct-q8_0.gguf | P2 | A | EOS | 148 | 0 | true | `\n49\n50\n\nNote that the numbers are listed in ascending order.` |
| smollm2-360m-instruct-q8_0.gguf | P2 | B | EOS | 148 | 0 | true | `\n49\n50\n\nNote that the numbers are listed in ascending order.` |
| smollm2-360m-instruct-q8_0.gguf | P2 | C | EOS | 148 | 0 | true | `\n49\n50\n\nNote that the numbers are listed in ascending order.` |
| smollm2-360m-instruct-q8_0.gguf | P3 | off | EOS | 27 | 0 | true | `nd evocative, with a focus on the ocean's power and beauty.)` |
| smollm2-360m-instruct-q8_0.gguf | P3 | A | EOS | 27 | 0 | true | `nd evocative, with a focus on the ocean's power and beauty.)` |
| smollm2-360m-instruct-q8_0.gguf | P3 | B | EOS | 27 | 0 | true | `nd evocative, with a focus on the ocean's power and beauty.)` |
| smollm2-360m-instruct-q8_0.gguf | P3 | C | EOS | 27 | 0 | true | `nd evocative, with a focus on the ocean's power and beauty.)` |
| smollm2-360m-instruct-q8_0.gguf | P4 | off | MAX_TOKENS | 200 | 0 | true | `bot. I am a robot. I am a robot. I am a robot. I am a robot.` |
| smollm2-360m-instruct-q8_0.gguf | P4 | A | REPETITION_LOOP | 20 | 1 | true | ` I am a robot. I am a robot. I am a robot. I am a robot.` |
| smollm2-360m-instruct-q8_0.gguf | P4 | B | REPETITION_LOOP | 32 | 1 | true | `I am a robot. I am a robot. I am a robot. I am a robot. I am` |
| smollm2-360m-instruct-q8_0.gguf | P4 | C | REPETITION_LOOP | 64 | 1 | true | `obot. I am a robot. I am a robot. I am a robot. I am a robot` |
| smollm2-360m-instruct-q8_0.gguf | P5 | off | EOS | 197 | 0 | true | `the Fibonacci sequence: '[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]'.` |
| smollm2-360m-instruct-q8_0.gguf | P5 | A | EOS | 197 | 0 | true | `the Fibonacci sequence: '[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]'.` |
| smollm2-360m-instruct-q8_0.gguf | P5 | B | EOS | 197 | 0 | true | `the Fibonacci sequence: '[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]'.` |
| smollm2-360m-instruct-q8_0.gguf | P5 | C | EOS | 197 | 0 | true | `the Fibonacci sequence: '[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]'.` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P0 | off | MAX_TOKENS | 200 | 0 | true | `over the lazy dog once more, and then the dog jumps over the` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P0 | A | REPETITION_LOOP | 112 | 1 | true | `over the lazy dog. The fox jumps over the lazy dog once more` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P0 | B | REPETITION_LOOP | 92 | 1 | true | `over the lazy dog. The fox jumps over the lazy dog once more` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P0 | C | MAX_TOKENS | 200 | 0 | true | `over the lazy dog once more, and then the dog jumps over the` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P1 | off | MAX_TOKENS | 200 | 0 | true | `3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P1 | A | REPETITION_LOOP | 32 | 1 | true | `1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P1 | B | REPETITION_LOOP | 32 | 1 | true | `1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P1 | C | REPETITION_LOOP | 80 | 1 | true | `3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P2 | off | MAX_TOKENS | 200 | 0 | true | `int(number)\n'''\n\nThis code will output each number from 1 to` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P2 | A | MAX_TOKENS | 200 | 0 | true | `int(number)\n'''\n\nThis code will output each number from 1 to` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P2 | B | MAX_TOKENS | 200 | 0 | true | `int(number)\n'''\n\nThis code will output each number from 1 to` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P2 | C | MAX_TOKENS | 200 | 0 | true | `int(number)\n'''\n\nThis code will output each number from 1 to` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P3 | off | MAX_TOKENS | 200 | 0 | true | `eful and peaceful place, with no worries or worries. The sea` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P3 | A | REPETITION_LOOP | 107 | 1 | true | `as a peaceful and peaceful place, with no worries or worries` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P3 | B | REPETITION_LOOP | 92 | 1 | true | `as a peaceful and peaceful place, with no worries or worries` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P3 | C | REPETITION_LOOP | 197 | 1 | true | `as a peaceful and peaceful place, with no worries or worries` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P4 | off | MAX_TOKENS | 200 | 0 | true | ` a robot I am a robot I am a robot I am a robot I am a robot` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P4 | A | REPETITION_LOOP | 16 | 1 | true | ` I am a robot I am a robot I am a robot I am a robot` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P4 | B | REPETITION_LOOP | 32 | 1 | true | ` a robot I am a robot I am a robot I am a robot I am a robot` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P4 | C | REPETITION_LOOP | 64 | 1 | true | ` a robot I am a robot I am a robot I am a robot I am a robot` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P5 | off | EOS | 141 | 0 | true | `le usage demonstrates calculating the 10th Fibonacci number.` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P5 | A | EOS | 141 | 0 | true | `le usage demonstrates calculating the 10th Fibonacci number.` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P5 | B | EOS | 141 | 0 | true | `le usage demonstrates calculating the 10th Fibonacci number.` |
| qwen2.5-coder-0.5b-instruct-q4_0.gguf | P5 | C | EOS | 141 | 0 | true | `le usage demonstrates calculating the 10th Fibonacci number.` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P0 | off | MAX_TOKENS | 200 | 0 | true | `s over the lazy dog.\n\n13. The quick brown fox jumps over the` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P0 | A | MAX_TOKENS | 200 | 0 | true | `s over the lazy dog.\n\n13. The quick brown fox jumps over the` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P0 | B | MAX_TOKENS | 200 | 0 | true | `s over the lazy dog.\n\n13. The quick brown fox jumps over the` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P0 | C | MAX_TOKENS | 200 | 0 | true | `s over the lazy dog.\n\n13. The quick brown fox jumps over the` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P1 | off | MAX_TOKENS | 200 | 0 | true | `3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P1 | A | REPETITION_LOOP | 32 | 1 | true | `1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P1 | B | REPETITION_LOOP | 32 | 1 | true | `1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P1 | C | REPETITION_LOOP | 80 | 1 | true | `3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P2 | off | MAX_TOKENS | 200 | 0 | true | `3\n4\n5\n1\n2\n3\n4\n5\n1\n2\n3\n4\n5\n1\n2\n3\n4\n5\n1\n2\n3\n4\n5\n1\n2\n3\n4\n5\n1\n2\n` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P2 | A | REPETITION_LOOP | 40 | 1 | true | `3\n4\n5\n1\n2\n3\n4\n5\n1\n2\n3\n4\n5\n1\n2\n3\n4\n5\n1\n2\n` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P2 | B | REPETITION_LOOP | 32 | 1 | true | `3\n4\n5\n1\n2\n3\n4\n5\n1\n2\n3\n4\n5\n1\n2\n3\n` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P2 | C | REPETITION_LOOP | 100 | 1 | true | `3\n4\n5\n1\n2\n3\n4\n5\n1\n2\n3\n4\n5\n1\n2\n3\n4\n5\n1\n2\n3\n4\n5\n1\n2\n3\n4\n5\n1\n2\n` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P3 | off | MAX_TOKENS | 200 | 0 | true | `em by Walt Whitman\n15. The Sea - A poem by Emily Dickinson\n1` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P3 | A | MAX_TOKENS | 200 | 0 | true | `em by Walt Whitman\n15. The Sea - A poem by Emily Dickinson\n1` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P3 | B | MAX_TOKENS | 200 | 0 | true | `em by Walt Whitman\n15. The Sea - A poem by Emily Dickinson\n1` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P3 | C | MAX_TOKENS | 200 | 0 | true | `em by Walt Whitman\n15. The Sea - A poem by Emily Dickinson\n1` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P4 | off | MAX_TOKENS | 200 | 0 | true | `bot. I am a robot. I am a robot. I am a robot. I am a robot.` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P4 | A | REPETITION_LOOP | 20 | 1 | true | ` I am a robot. I am a robot. I am a robot. I am a robot.` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P4 | B | REPETITION_LOOP | 32 | 1 | true | `I am a robot. I am a robot. I am a robot. I am a robot. I am` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P4 | C | REPETITION_LOOP | 64 | 1 | true | `obot. I am a robot. I am a robot. I am a robot. I am a robot` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P5 | off | EOS | 130 | 0 | true | `will return 0 for all values of 'n' less than or equal to 0.` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P5 | A | EOS | 130 | 0 | true | `will return 0 for all values of 'n' less than or equal to 0.` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P5 | B | EOS | 130 | 0 | true | `will return 0 for all values of 'n' less than or equal to 0.` |
| tinyllama-1.1b-chat-v1.0.Q4_0.gguf | P5 | C | EOS | 130 | 0 | true | `will return 0 for all values of 'n' less than or equal to 0.` |
