# V14 result: rejected at the development screen

V14 completed its frozen 32-step treatment and passed every execution and
artifact-integrity gate. It completed 512 finite-gradient checks, preserved the
exact base logits with the adapter disabled, preserved live logits across save,
and reproduced every adapter tensor and sampled logit exactly after reload.

The treatment failed three predeclared quality gates:

| Metric | V13 control | V14 treatment | Required |
| --- | ---: | ---: | --- |
| Call decisions | 362/390 | 370/390 | lose no more than 7 |
| No-call decisions | 77/81 | 75/81 | no regression |
| Balanced accuracy | 0.939411 | 0.937322 | exceed control |
| Applicability loss | 0.183260 | 0.187295 | at least 5% lower |
| Language-model loss | 0.047282 | 0.047714 | no more than 10% higher |

The treatment improved call classification and kept ordinary language-model
loss within bounds, but it moved the decision boundary in the wrong direction
for no-call applicability. V14 is rejected. No fixed-endpoint, generation,
sealed, JVM, framework, or packaging evaluation was run, and no candidate is
eligible for publication.

Evidence is under `gpu-screen/`. The exact Vultr A16 instance
`94be59dd-2466-4248-b759-812c9c208e61` existed from
2026-09-13T20:05:14Z until approximately 2026-09-13T21:30Z. It was deleted
after evidence transfer, provider-wide Vultr inventory returned zero instances,
and the identity-bound local cleanup watchdog was unloaded and removed. At the
advertised USD 0.236/hour rate, the estimated runtime cost was about USD 0.34.
