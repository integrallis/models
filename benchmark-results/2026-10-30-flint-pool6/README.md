# Flint public worker-pool diagnosis

Runtime `fb5d7a68248ca14da832ae886016b86d62060382`, worker `fca402830aed92443c88eff83a98457a4e320f1c`.
The exact frozen primary qualification payload was reused on a separate m6a.4xlarge host.
Sequential pool8, pool6, and Ollama16 arms used one auto-glass prompt, a 512-token limit,
one warmup, and three measurements. This is development evidence, not full qualification.

Median decode: pool8 11.8797 tokens/second, pool6 12.1181, Ollama16 14.0529 (see exact report values).
Pool6 improves 2.01% over pool8; all six native outputs are byte-identical and pass the visible
answer screen. The already-running full campaign remains frozen at pool8. No policy gate changes.

The worker completed with exit zero. Its independently bounded instance terminated and the
DeleteOnTermination root volume was confirmed absent. Raw reports, compressed logs, deployment
inputs and cleanup evidence are retained.
