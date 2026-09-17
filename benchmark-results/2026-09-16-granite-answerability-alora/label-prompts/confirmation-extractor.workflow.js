export const meta = {
  name: 'label-confirmation-extract',
  description: 'Extract verbatim answering quotes for 126 blind answerability cases',
  phases: [{ title: 'Extract' }],
}
const D = '<CONFIRMATION_DIR>'
const prompt = (part) => `You extract evidence. Read ${D}/extract-part${part}.json fully, in chunks with the Read tool (offset/limit). Each record has id, question, and passages (list of {doc_id, text}). Do not read any other file in that directory, and do not use any other data source.

For each record, find the SHORTEST contiguous span copied EXACTLY, character for character, from ONE passage that directly answers the question — a span a reasonable reader would accept as the answer by itself or with its sentence. Copy it verbatim (do not fix typos, do not paraphrase, do not join text from different places). If no passage directly answers the question (only related, tangential, or partial information), return null. Be strict: a passage about the topic is not an answer.

Do not write scripts that classify; read and decide yourself. You may use a script only to write the JSON file.

Write ${D}/extraction-part${part}.json: a list of {"id", "quote" (string or null), "doc_id" (int or null), "answer" (short answer in your words, or null)}. Every record id exactly once. Reply with only: count of quotes and count of nulls.`
return await parallel([0, 1, 2].map(p => () => agent(prompt(p), { label: `extract-part${p}`, phase: 'Extract', model: 'opus' })))