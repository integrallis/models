# Qwen2.5 0.5B Q5_K_M qualification

Released Models 0.5.9 and Vectors 0.1.32, with a separately identified measurement
harness, pass the unchanged RAG policy against same-host Ollama 0.40.13.
Configuration: native pool8, decode8, fused Java grouped attention, declared Java
and reference threads16. One warmup, three iterations, nine cases. The selected
screen and both counterbalanced confirmations pass. The final confirmation is
the canonical catalog report; all three are retained under repeatability.

Each pair also passed the untuned library-default correctness smoke. Raw answers,
output-token counts, prompt hashes and grounding/evaluation matched the control;
the harness does not retain token IDs. Model contribution is 66.7%, with 100%
correctness among contributed answers. This is evidence for this RAG workload and
host, not a broad accuracy or cross-hardware claim.

Profile recommendations use existing runtime implementations: native worker count8
and fusedGroupedAttention=true. Decode defaults to the native pool size, so no
separate decode-thread system property or new runtime branch is required.

The experimental F16C code did not show a repeatable gain and was reverted. All
negative and positive experiment evidence remains on the experiment branch; the
profile links both the separate fused-attention comparison and native-pool comparison.
