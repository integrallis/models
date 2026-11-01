"""Render pinned upstream Jinja into text fixtures; no model inference or measurements."""
import hashlib,json,pathlib,urllib.request
from jinja2 import Environment,StrictUndefined
REV='ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7'
raw=urllib.request.urlopen(f'https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/{REV}/chat_template.jinja').read()
# The HF generation extension marks assistant spans but does not alter rendered text.
s=raw.decode().replace('{%- generation -%}','').replace('{%- endgeneration -%}','')
e=Environment(trim_blocks=True,lstrip_blocks=True)
e.filters['tojson']=lambda value:json.dumps(value,ensure_ascii=False)
e.globals['raise_exception']=lambda message:(_ for _ in ()).throw(ValueError(message))
t=e.from_string(s)
fn={'name':'echo','description':'Return exact text.','parameters':{'type':'object','properties':{'text':{'type':'string'},'count':{'type':'integer'}},'required':['text'],'additionalProperties':False}}
cases=[
 {'id':'chat','messages':[{'role':'user','content':'Hello'}],'tools':[],'thinking':True},
 {'id':'direct','messages':[{'role':'system','content':'Be concise.'},{'role':'user','content':'Hello'}],'tools':[],'thinking':False},
 {'id':'tools','messages':[{'role':'user','content':'Echo MiXeD: 007! 🌧️'}],'tools':[fn],'thinking':False},
 {'id':'tool-roundtrip','messages':[{'role':'user','content':'Echo MiXeD: 007! 🌧️'},{'role':'assistant','content':'<think>Use echo.</think>','tool_calls':[{'function':{'name':'echo','arguments':{'text':' MiXeD: 007! 🌧️ ','count':2}}}]},{'role':'tool','content':' MiXeD: 007! 🌧️ '}],'tools':[fn],'thinking':True},
 {'id':'old-thought','messages':[{'role':'user','content':'One'},{'role':'assistant','content':'<think>old thought</think> First answer '},{'role':'user','content':'Two'}],'tools':[],'thinking':True},
 {'id':'assistant-history','messages':[{'role':'user','content':'One'},{'role':'assistant','content':'First answer'},{'role':'user','content':'Two'}],'tools':[],'thinking':False},
]
for c in cases:c['expected']=t.render(messages=c['messages'],tools=c['tools'],bos_token='<|startoftext|>',add_generation_prompt=True,enable_thinking=c['thinking'])
out={'modelRevision':REV,'templateSha256':hashlib.sha256(raw).hexdigest(),'cases':cases}
p=pathlib.Path('models-runtime/src/test/resources/lfm3/template-fixtures.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
