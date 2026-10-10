# Candidate catalog fixture

These seven JSON files are byte-for-byte catalog inputs from the public ModelJars repository at
the revision recorded in `source.json`. They are test data, never runtime dependencies or current
qualification claims. Models builds and CI must not depend on a checkout of its downstream project.

To refresh, export each listed `catalog/<filename>` from one reviewed ModelJars commit with
`git show <revision>:catalog/<filename>`, record that revision and each file's SHA-256 in
`source.json`, regenerate `docs/CANDIDATE-INVENTORY.md`, then run the fleet tests and integrity
mutation checks. Do not copy a dirty working tree or edit the historical reports to make tests pass.
