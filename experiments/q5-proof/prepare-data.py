import pathlib,urllib.request,json,csv,io,hashlib,pyarrow.parquet as pq
out=pathlib.Path(__file__).parent/'data';out.mkdir(exist_ok=True);cases=[];provenance=[]
# sources.json is an INPUT when it exists, not only an output. Without this, fetch() resolved
# rev from the live HuggingFace API every run, so a re-run silently took whatever revision was
# current and then wrote that as the provenance -- the pin was a record of the past run rather
# than a constraint on the next one, and the experiment could not reproduce its own data. The
# recorded sha256 is verified after download, so a moved revision fails loudly instead of
# quietly changing the corpus.
PINNED={s['url']:s for d in (json.loads((out/'sources.json').read_text()) if (out/'sources.json').exists() else []) for s in d.get('sources',[])}
PINNED_BY_PATH={u.split('/resolve/',1)[1].split('/',1)[1]:s for u,s in PINNED.items()}
def fetch(repo,path):
 pin=PINNED_BY_PATH.get(path) if PINNED_BY_PATH.get(path,{}).get('repository')==repo else None
 if pin:rev=pin['revision'];url=pin['url']
 else:meta=json.load(urllib.request.urlopen('https://huggingface.co/api/datasets/'+repo));rev=meta['sha'];url='https://huggingface.co/datasets/'+repo+'/resolve/'+rev+'/'+path
 dest=out/(repo.replace('/','-')+'-'+path.replace('/','-'))
 if not dest.exists():urllib.request.urlretrieve(url,dest)
 digest=hashlib.sha256(dest.read_bytes()).hexdigest()
 if pin and digest!=pin['sha256']:raise SystemExit('pinned sha256 mismatch for '+url+': expected '+pin['sha256']+' got '+digest)
 return dest,dict(repository=repo,revision=rev,url=url,sha256=digest)
for name in ['scifact','nfcorpus']:
 cp,cs=fetch('BeIR/'+name,'corpus/corpus-00000-of-00001.parquet');qp,qs=fetch('BeIR/'+name,'queries/queries-00000-of-00001.parquet');rp,rs=fetch('BeIR/'+name+'-qrels','test.tsv')
 corpus={str(r['_id']):r for r in pq.read_table(cp).to_pylist()};queries={str(r['_id']):r for r in pq.read_table(qp).to_pylist()};qrels=list(csv.DictReader(io.StringIO(rp.read_text()),delimiter='\t'))
 ids=sorted({r['query-id'] for r in qrels})[:8];docs=sorted({key for qid in ids for key in sorted({r['corpus-id'] for r in qrels if r['query-id']==qid and int(r['score'])>0})[:4]});docs += [k for k in sorted(corpus) if k not in docs][:16]
 for key in ids:cases.append(dict(id=name+'/query/'+key,dataset=name,kind='query',text=queries[key]['text']))
 for key in docs:cases.append(dict(id=name+'/document/'+key,dataset=name,kind='document',text=(corpus[key].get('title','')+' '+corpus[key]['text']).strip()))
 provenance.append(dict(dataset=name,sources=[cs,qs,rs],queries=ids,documents=docs,qrels=[r for r in qrels if r['query-id'] in ids]))
(out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n');(out/'sources.json').write_text(json.dumps(provenance,indent=2)+'\n');print(len(cases))
