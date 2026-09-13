# Rejected hard-negative preparation audit

Status: **rejected before training**.

The schema-2 lexical miner selected 2,438 of 3,250 no-call examples by query/tool overlap. A manual
audit of the highest-scoring rows exposed contradictory source labels, so the prepared train and
validation hashes in this directory must not be used for an adapter run.

Representative source errors from pinned `edbuildingstuff/bfcl-ft-data` revision
`a7ceb3b1e1605f609fda6f2befec704f575290de`:

- Source line 25,284 asks for a chi-square test and supplies `chi_square_independence_test`, but the
  assistant label is an empty call list.
- Source line 107,184 asks for Boxer and Poodle breed information and supplies
  `get_breed_information`, but the assistant label is empty.
- Source line 117,938 asks for the first non-repeating character in `minimum` and supplies
  `find_first_non_repeating_char`; the row also requests an unsupported flatten operation, yet the
  label rejects every available tool.

The miner behaved as designed: it surfaced close query/tool pairs. The input labels were not sound
enough to turn those pairs into training examples. Increasing the no-call percentage or selecting
harder rows cannot repair contradictory supervision.

The replacement sequence uses the pinned Hammer function-masking source, but its first two
preparations were also rejected during source auditing. The frozen successor uses per-function
ambiguity checks and a declared 19% no-call share; its manifest and audit are in
`../hammer19-per-function-v3/`.
