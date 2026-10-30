# Flint schema tool comparison — failed

Frozen runtime and worker: `07fc6853f8bc596c826a37269164bc9f4c993bce`.
Both arms completed all 14 cases with 512 call tokens and 512 unconstrained follow-up tokens.
Default and schema-constrained arms both passed 10/14, with 30/37 exact arguments correct.
Schema/structure were already 100%; the grammar did not fix selection or semantic errors.
The unchanged failures were home (front vs front door), robot (omitted actions), device
(Hello! vs hello), and gallery (omitted move). No tool capability is qualified.

Reference parity passed first. The two arms ran sequentially on the same disposable
m6a.4xlarge. The worker completed at 2026-10-30T16:45:57Z; instance and root volume are gone.
The frozen worker, payload hashes, launch inputs, original reports, and compressed logs are retained.
The prior weather call/result in the follow-up test are scripted, not a live agent evaluation.
