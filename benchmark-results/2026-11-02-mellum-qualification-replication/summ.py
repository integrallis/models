import json, statistics
rows=[]
for n in (1,2,3):
    try:
        v=json.load(open(f"/root/rep{n}/verdict.json"))["qualification"]
        cm=v["comparisons"][0]
        cand=json.load(open(f"/root/rep{n}/candidate.json"))["summary"]["p50DecodeTokensPerSecond"]
        comp=json.load(open(f"/root/rep{n}/comparator.json"))["summary"]["p50DecodeTokensPerSecond"]
        rows.append((n,cand,comp,cm["decodeThroughputRatio"],cm["endToEndLatencyRatio"],v["verdict"],v["modelAnswerRate"],v["modelAnswerCorrectRate"]))
    except Exception as e:
        rows.append((n,None,None,None,None,f"ERROR {type(e).__name__}",None,None))
print("%4s %8s %8s %9s %9s  %s" % ("run","ours","ollama","decodeR","e2eR","verdict"))
for r in rows:
    if r[3] is None:
        print("%4d %8s %8s %9s %9s  %s" % (r[0],"-","-","-","-",r[5])); continue
    print("%4d %8.2f %8.2f %9.4f %9.4f  %s" % (r[0],r[1],r[2],r[3],r[4],r[5]))
g=[r for r in rows if r[3] is not None]
if len(g)>=2:
    rt=[r[3] for r in g]; ou=[r[1] for r in g]; co=[r[2] for r in g]
    m=statistics.mean(rt); sd=statistics.stdev(rt) if len(rt)>1 else 0.0
    print()
    print("  ours     mean %.2f  sd %.2f" % (statistics.mean(ou), statistics.stdev(ou) if len(ou)>1 else 0))
    print("  ollama   mean %.2f  sd %.2f" % (statistics.mean(co), statistics.stdev(co) if len(co)>1 else 0))
    print("  RATIO    mean %.4f  sd %.4f   min %.4f   floor 0.80" % (m, sd, min(rt)))
    print("  passes   %d of %d" % (sum(1 for x in rt if x>=0.80), len(rt)))
    print("  contrib  answerRate %.4f  correctRate %.4f" % (min(r[6] for r in g), min(r[7] for r in g)))
    print("  DECISION %s" % ("EARNED - replicated mean clears the floor" if m>=0.80 else "NOT EARNED - withdraw the entry"))
