# Adjacent Q4_K scale conversion component checks

Rust 1.96.0 x86-64 assembly uses one F16C conversion instead of two in both
ordered and ordinary Q4_K row kernels. The ordinary function shrinks from 181
to 176 assembly lines; the ordered function from 176 to 174. This establishes
the intended instruction change, not a speed improvement.

All 63 active synthetic Rust tests and Clippy pass (3 tests ignored). The first
test run passed every numeric comparison but failed the existing conversion
counter: the new paired instruction was not instrumented. The final test records
one instruction for the pair, asserts that immediately after Q4_K, and expects
four total instructions across Q4_K/Q5_K/Q6_K instead of five. The failed log
is preserved alongside the corrected passing run. No numeric tolerance changed.

Real-model baseline/candidate/baseline timing and full qualification remain pending.
The experiment is not on the production or prepared release branch.
