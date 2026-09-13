# Hammer 19% preparation audit

Status: **trained and rejected; not qualified for release**.

This split combines positive calls from pinned `edbuildingstuff/bfcl-ft-data` revision
`a7ceb3b1e1605f609fda6f2befec704f575290de` with no-call examples from pinned
`MadeAgents/xlam-irrelevance-7.5k` revision
`34323bf09efc7e4a394998a0fa91ff997617c369`. The preparation manifest records both source hashes,
all three excluded BFCL evaluation hashes, the deterministic seed, row counts, and output hashes.

The Hammer source removes the function selected by a known positive example. Preparation then:

1. rejects schemas the Java tool contract cannot represent;
2. requires matching positive evidence and rejects every multi-call positive;
3. rejects a remaining function whose lexical/schema cosine to the removed function is 0.45 or
   greater; and
4. rejects a row when any single remaining function has query cosine 0.125 or greater.

The fourth check is deliberately per function. An earlier aggregate calculation allowed an exact
`ip_geolocation` or `search_news` function to hide inside a long list of unrelated tools. A
regression test now fixes that catalog-dilution defect. The stricter pass retains 2,569 candidates;
2,470 are selected for a declared 19% no-call share across the 12,000-row train and 1,000-row
validation splits. No evaluation query overlaps either source selection.

The 40 retained rows nearest the 0.125 boundary were inspected with their complete advertised tool
descriptions. None contained a function documented to fulfill the complete request. Close cases
were retained only when the advertised contract was materially narrower: for example, checking
whether a domain has a role account does not validate an email address, listing recent articles
does not calculate daily publisher counts, and a generic trading-symbol quote does not document a
GBP/JPY exchange-rate contract.

Frozen artifacts:

- manifest SHA-256: `c7da1f5f826bb1c6f10047e09ff1e7a69e3346dad2336400e5a277ef937090b9`
- train SHA-256: `3fe555c1e4a68b65b6715341cd1d1cbf9995e549cbdb267f15896b0a9b7edf77`
- validation SHA-256: `f12c4c34c875d929b252d075749468f2c53d919744eca711e907927d9ecd83ca`

The resulting rank-32 adapter has SHA-256
`437c6e1e098768d6d40e256044fdcbcd8d26f2e785f0b4734b9d1d56aa203106`. On the fixed
25-case-per-kind diagnostic slice it produced 98.67% parseable syntax, 94.67% schema validity, 76%
exact calls, and a 32% false-call rate. Because the syntax failure is already contained in the
fixed 100-case selection, the required 100% syntax gate is mathematically impossible for the full
run. The full run was therefore not performed, and this adapter did not proceed to Java behavior,
performance, packaging, catalog, or release qualification.
