import json, sys
r = json.load(open(sys.argv[1]))
s = r.get('summary') or {}
env = (r.get('backendDiagnostics') or {}).get('environment') or {}
st = r.get('settings') or {}
reasons = []
if s.get('correctAnswerRate') != 1:   reasons.append(f"correctAnswerRate={s.get('correctAnswerRate')}")
if s.get('abstentionAccuracy') != 1:  reasons.append(f"abstentionAccuracy={s.get('abstentionAccuracy')}")
if r.get('failures') != []:           reasons.append(f"failures={r.get('failures')}")
if st.get('warmups') != 0:            reasons.append(f"warmups={st.get('warmups')}")
if st.get('iterations') != 1:         reasons.append(f"iterations={st.get('iterations')}")
gc = st.get('generationControls') or {}
if gc.get('promptCache') != 'longest-common-prefix':
    reasons.append(f"promptCache={gc.get('promptCache')}")
if r.get('backend') == 'rust-ffm' and env.get('native-quantized-decode') != 'true':
    # 0.3.54 removed the setting: the shim always serves decode, so "false" means it fell back.
    reasons.append(f"native-quantized-decode={env.get('native-quantized-decode')}")
print('PASS' if not reasons else 'FAIL:' + ';'.join(reasons))
