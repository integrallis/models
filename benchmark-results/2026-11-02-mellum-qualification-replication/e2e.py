import json
for n in (1,2,3):
    c=json.load(open(f"/root/rep{n}/candidate.json")); o=json.load(open(f"/root/rep{n}/comparator.json"))
    cs,os_=c["summary"],o["summary"]
    print(f"--- rep{n} ---")
    print("  ours   outTokens %5d  e2e p50 %8.1f ms  p95 %8.1f  ttft p95 %8.1f" % (
        cs["totalOutputTokens"], cs["endToEndMillis"]["p50"], cs["endToEndMillis"]["p95"], cs["ttftMillis"]["p95"]))
    print("  ollama outTokens %5d  e2e p50 %8.1f ms  p95 %8.1f  ttft p95 %8.1f" % (
        os_["totalOutputTokens"], os_["endToEndMillis"]["p50"], os_["endToEndMillis"]["p95"], os_["ttftMillis"]["p95"]))
    print("  token ratio ours/ollama %.3f   e2e p95 ratio %.3f" % (
        cs["totalOutputTokens"]/os_["totalOutputTokens"], cs["endToEndMillis"]["p95"]/os_["endToEndMillis"]["p95"]))
