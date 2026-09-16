# Qwen3 1.7B production-Q4 calibration V17 result

Status: **REJECTED**

V17 tested whether a decision boundary calibrated with the real Java 25 Q4 runtime could make the
unchanged V16 applicability head reliable enough for hybrid tool routing. The calibration phase
passed. The separately frozen, untouched screen did not.

## Calibration

- Observations: 30/30
- Correct call decisions: 20/20
- Correct no-call decisions: 9/10
- Balanced accuracy: 0.95
- Physical shared-prefix identity: 30/30
- Selected threshold: `2.7213982172616653`
- Calibration artifact SHA-256:
  `97c81ca7dcb3eab4114f0164c25506a9e6332c2339083cc1c72f3925ee18487d`

The artifact and its hash were committed before any screen observation was collected.

## Untouched screen

The run stopped after 43 of 45 cases because the frozen call gate had become mathematically
impossible to pass:

- Correct no-call decisions observed: 15/15
- Correct call decisions observed: 25/28
- Physical shared-prefix identity observed: 43/43
- Call misses:
  - `multiple_115`: score `2.72015`
  - `multiple_76`: score `-5.04900`
  - `simple_326`: score `-1.35086`
- Best possible final call result with the two remaining cases: 27/30
- Frozen required call result: 28/30

The run therefore rejected V17 immediately. It did not reinterpret the threshold, finish the two
unneeded observations, open the sealed 300-case window, or proceed to generation and framework
qualification.

## Finding

The Java runtime's physical prefix sharing is working: every observed calibration and screen case
shared the immutable KV prefix. The rejection is isolated to decision-boundary transfer. The
screen's highest no-call score was `2.27231`, while the near-boundary positive `multiple_115`
scored `2.72015`. That separation is useful evidence for a new, independently frozen calibration
experiment; it is not permission to change V17 after seeing its screen.

No Models or ModelJars artifact is qualified or publishable from V17.
