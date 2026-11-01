export const meta = {
  name: 'msmarco-blind-relabel',
  description: 'Blind two-judge answerability labelling of 133 MS MARCO window cases',
  phases: [{ title: 'Judge' }],
}
const S = '<AUDIT_DIR>'
const prompt = (part, judge) => `You are a careful annotator. Read ${S}/msm-relabel-part${part}.json (records with id, question = a web search query, documents = a Python-repr list of passages). Read it in chunks with the Read tool (offset/limit) until you have read every record fully. Do not use any other data source, do not search the web, and do not look at any other file in that directory.

For each record decide, from the documents alone:
- "answerable": at least one passage contains information that directly answers the query (a reasonable searcher would be satisfied).
- "unanswerable": no passage answers it (passages are off-topic, only tangential, or answer a different question).
- "ambiguous": the query is too vague to decide, or the passages only partially answer it.

Judge each record independently. Do not write any scripts that auto-classify; read and decide yourself.

Write the result as JSON to ${S}/msm-relabel-part${part}-judge${judge}.json: a list of {"id", "judgment", "evidence"} where evidence is a <=25-word quote or reason. Every record id must appear exactly once. Reply with only the count of each judgment.`
const jobs = []
for (const part of [0, 1, 2]) for (const [judge, model] of [['A', 'opus'], ['B', 'sonnet']])
  jobs.push(() => agent(prompt(part, judge), { label: `part${part}-judge${judge}`, phase: 'Judge', model }))
return await parallel(jobs)