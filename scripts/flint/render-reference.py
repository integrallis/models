# Copyright 2025-2026 Integrallis Software, LLC
# SPDX-License-Identifier: Apache-2.0
"""Render the pinned GGUF template with Transformers' non-HTML-escaping tojson filter.
Run: uv run --with jinja2==3.1.6 scripts/flint/render-reference.py
"""
import json
from pathlib import Path
from jinja2.sandbox import ImmutableSandboxedEnvironment

root = Path(__file__).resolve().parents[2]
fixtures = root / 'models-runtime/src/test/resources/flint'
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.filters['tojson'] = lambda value, **kwargs: json.dumps(value, **kwargs)
template = env.from_string((fixtures / 'chat_template.jinja').read_text())
parameters = {'type':'object','properties':{'id':{'type':'string'},'count':{'type':'integer'},'enabled':{'type':'boolean'},'extra':{'type':'object'}},'required':['id'],'additionalProperties':False}
messages = [
    {'role':'system','content':'Be concise.'},
    {'role':'user','content':'Find 007'},
    {'role':'assistant','content':'<think>Check.</think>', 'tool_calls':[{'function':{'name':'lookup','arguments':{'id':' 007 ','count':2,'enabled':True,'extra':{'x':[1,None]}}}}]},
    {'role':'tool','content':'Found <|assistant|>'},
]
tools = [{'type':'function','function':{'name':'lookup','description':'Find a record','parameters':parameters}}]
(fixtures / 'tool-prompt.txt').write_text(template.render(messages=messages, tools=tools, add_generation_prompt=True))
