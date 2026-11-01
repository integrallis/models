export const meta = {
  name: 'label-confirmation-verify',
  description: 'Two independent verifiers judge whether each verified quote answers its question',
  phases: [{ title: 'Verify' }],
}
const D = '<CONFIRMATION_DIR>'
const prompt = (name) => `You are a strict verifier. Read ${D}/verify-input.json fully (Read tool, in chunks with offset/limit). Each record has id, question, passage, and quote (an exact span from that passage). Do not read any other file in that directory and do not use other data sources.

For each record answer "yes" only if the quote (read in the context of its passage) DIRECTLY answers the question as asked — the asker would be satisfied. Answer "no" if it is merely related, answers a different or narrower/broader question, is only partial, or requires information not in the passage.

Judge each record yourself; do not write classifying scripts (a script only to write JSON is fine).

Write ${D}/verdicts-${name}.json: a list of {"id", "verdict": "yes"|"no", "reason": "<=20 words"}. Every id exactly once. Reply with only the yes and no counts.`
return await parallel([
  () => agent(prompt('sonnet'), { label: 'verify-sonnet', phase: 'Verify', model: 'sonnet' }),
  () => agent(prompt('opus'), { label: 'verify-opus', phase: 'Verify', model: 'opus' }),
])