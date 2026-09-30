# gemma3n: verified specification

Transcribed from llama.cpp `src/models/gemma3n.cpp` and the published
`ggml-org/gemma-3n-E2B-it-GGUF` header on 2026-09-29. Every shape below was read from one of those
two, not inferred.

## The thing that makes this big

**AltUp carries `n_altup = 4` parallel residual streams through every layer.** The state is
`x[a][d]`, not `x[d]`. That is not an additive mechanism bolted onto a Gemma 4 layer -- it changes the
shape of the residual stream, and therefore of every buffer and every residual add in the forward
pass. Budget accordingly.

E2B constants: `n_altup = 4`, `altup.active_idx = 0`, `n_embd = 2048`, `n_embd_altup = 256`,
`laurel_rank = 64`, `n_layer = 30`, heads 8/2, head dim 256, sliding window 512.

## Metadata

| key | value | note |
|---|---|---|
| `gemma3n.altup.num_inputs` | 4 | |
| `gemma3n.altup.active_idx` | 0 | |
| `gemma3n.embedding_length_per_layer_input` | 256 | as Gemma 4 E-series |
| `gemma3n.attention.shared_kv_layers` | **10.0** | **a float.** Gemma 4 reads its equivalent with `getUint32`; that will not work here. llama.cpp ignores the key and hard-codes `n_layer_kv_from_start = 20`, which for 30 layers is the same as `n_layer - 10` -- so deriving it is equivalent here and more general |
| `gemma3n.activation_sparsity_scale` | float array[30] | **1.6448533535003662 for layers 0-9 and `-inf` for 10-29.** Not zero -- see trap 10 |
| `gemma3n.attention.sliding_window_pattern` | bool array[30] | `SSSS.SSSS.SSSS.SSSS.SSSS.SSSS.` -- every fifth layer is full attention. llama.cpp derives this from a period of 5; the published array says the same thing and is the more faithful source |
| `gemma3n.context_length` | 32768 | |
| *(no `laurel_rank` key)* | | llama.cpp hard-codes 64; derive it from `laurel_l`'s row count instead |
| *(no `vocab_size` key)* | | fall back to the tokenizer token count (262144) |
| *(no `rope.freq_base_swa`)* | | optional in the reference, absent here, so sliding layers share the 1e6 base |
| *(no attention-scale key)* | | llama.cpp hard-codes `f_attention_scale = 1.0` for gemma3n: the softmax scale is **1.0**, not 1/sqrt(head_dim). The Q norm absorbs it |

No `output.weight`: gemma3n **ties** its head to `token_embd`.

## Model-level tensors

| tensor | dims | type |
|---|---|---|
| `token_embd.weight` | [2048, 262144] | Q8_0 |
| `per_layer_token_embd.weight` | [7680, 262144] | Q8_0 (7680 = 256 x 30) |
| `per_layer_model_proj.weight` | [2048, 7680] | F16 |
| `per_layer_proj_norm.weight` | [256] | F32 |
| `altup_proj.weight` | [2048, 2048, **3**] | F16 (3 = n_altup - 1) |
| `altup_unembd_proj.weight` | [2048, 2048, **3**] | F16 |
| `output_norm.weight` | [2048] | F32 |

## Per-layer tensors beyond a Gemma 4 E-series layer

`altup_router` [2048, 4] F16, `altup_router_norm` [2048], `altup_predict_coef` [4, **16**],
`altup_correct_coef` [4, 4], `altup_correct_scale` [2048], `laurel_l` [2048, 64] F16,
`laurel_r` [64, 2048] F16, `laurel_post_norm` [2048].

`altup_router` and both `laurel_*` are **F16** while the rest of the layer is Q8_0.

## Algorithm (single token; drop the token dimension)

Helpers:

```
magnitude(v)      = sqrt(sum_d v[d]^2)                      # L2 over the embedding
modalities(v, il) = tanh( altup_router[il] . ( rmsnorm(v, altup_router_norm[il]) * (1/n_embd) ) )
                    # NOTE the scale is 1/n_embd, NOT 1/sqrt(n_embd)      -> 4 values
gaussian_topk(z)  = relu( z - (mean(z) + std(z) * sparsity_scale[il]) )
                    # std uses the (n-1) denominator
laurel(v, il)     = v + rmsnorm( laurel_r[il] . (laurel_l[il] . v), laurel_post_norm[il] )
```

### Initialisation

```
x[0]         = scaled token embedding                       # the usual Gemma embedding scale
target_mag   = magnitude(x[0])
for a in 1..3:
    x[a] = altup_proj[a-1] . x[0]
    x[a] = x[a] * target_mag / magnitude(x[a])               # renormalise to the active stream
```

Per-layer inputs, as Gemma 4 E-series but with two scales:

```
p            = (per_layer_model_proj . embedding) * (1/sqrt(n_embd))     # [256 x 30]
p            = rmsnorm(p, per_layer_proj_norm)                          # over each 256-wide slice
per_layer[il]= (p[il] + per_layer_token_embd_slice[il]) * (1/sqrt(2))
```

### Each layer `il`

```
# 1. predict
m       = modalities(x[i_act], il)
coefs   = altup_predict_coef[il] . m                       # 16 values
# laid out with n_altup as the FASTEST dim: coefs[b][a] = flat[a*4 + b]
for a, d:  pred[a][d] = x[a][d] + sum_b coefs[b][a] * x[b][d]

act_pred = pred[i_act]

# 2. attention and LAuReL, both on the NORMED active prediction
cur        = rmsnorm(act_pred, attn_norm[il])
laurel_out = laurel(cur, il)                               # takes the NORMED cur, not act_pred

Q = attn_q . cur ; Q = rmsnorm(Q, attn_q_norm) ; Q = rope(Q)
if layer owns its KV:
    K = attn_k . cur ; K = rmsnorm(K, attn_k_norm) ; K = rope(K)
    V = attn_v . cur ; V = rms_norm(V, eps)                # !! RMS norm with NO weight vector
else:
    reuse the KV another layer wrote (as Gemma 4 shared-KV)
cur = attn_output . attend(Q, K, V) * f_attention_scale

cur         = rmsnorm(cur, post_attention_norm[il])
cur         = cur + act_pred
attn_laurel = (cur + laurel_out) * (1/sqrt(2))

# 3. feed-forward
cur  = rmsnorm(attn_laurel, ffn_norm[il])
up   = ffn_up . cur
gate = ffn_gate . cur
if il < n_layer_sparsity:  gate = gaussian_topk(gate)
gate = gelu(gate)                                          # GELU, not SwiGLU's silu
cur  = ffn_down . (up * gate)
cur  = rmsnorm(cur, post_ffw_norm[il])
gated = cur + attn_laurel

# 4. correct
m2        = modalities(gated, il)
innov     = gated - pred[i_act]
c         = altup_correct_coef[il] . m2 + 1.0              # 4 values, the +1.0 matters
for a:  corrected[a] = pred[a] + innov * c[a]

# 5. per-layer input injection
fp = corrected[i_act] * altup_correct_scale[il]
fp = gelu(per_layer_inp_gate[il] . fp)                     # -> 256 wide
fp = fp * per_layer[il]
fp = per_layer_proj[il] . fp                               # -> 2048 wide
fp = rmsnorm(fp, per_layer_post_norm[il])
for a in 1..3:  corrected[a] += fp                         # streams 1..3 ONLY, not stream 0

x = corrected
```

### After the loop

```
target_mag = magnitude(x[i_act])
for a in 0..2:
    u[a] = altup_unembd_proj[a] . x[a+1]
    u[a] = u[a] * target_mag / magnitude(u[a])
out = ( x[0] + u[0] + u[1] + u[2] ) / n_altup               # a mean over the 4 streams
out = rmsnorm(out, output_norm)
logits = token_embd . out                                   # tied head
```

## Traps, each already verified

1. `shared_kv_layers` is a **float** in the metadata.
2. The AltUp router scale is **1/n_embd**, not 1/sqrt(n_embd).
3. `altup_predict_coef` output is indexed `[b][a] = flat[a*4 + b]` -- n_altup is the fastest dim. The
   transpose produces finite, plausible output.
4. The correction coefficients have **+1.0** added before use.
5. The per-layer injection is added to streams **1..3 only**; stream 0 is left alone.
6. `V` is RMS-normed **with no weight vector** -- a plain normalisation, not `rmsnorm(V, some_weight)`.
7. LAuReL consumes the **attn-normed** `cur`, not the raw active prediction.
8. The FFN gate is **GELU**, and sparsity is applied **before** the GELU.
9. `altup_router` and the `laurel_*` matrices are F16 among Q8_0 -- the F16 matmul fix of 2026-09-29
   is a prerequisite.
10. **The sparsity scale is `-inf`, not 0, on the layers that have no sparsity.** Feeding it into
    `mean + std * scale` gives a cutoff of `-inf`, and `relu(x - -inf)` is `+inf` -- every activation
    saturates. llama.cpp never hits this because it ignores the array and hard-codes "the first 10
    layers". Reading the array is the faithful choice, but it MUST skip non-finite entries.
11. The attention softmax scale is **1.0**, not 1/sqrt(head_dim) (see the table above). Applying the
    usual scale as well would halve every logit's spread.
12. E2B is 30 layers, E4B is 35. Nothing below assumes 30.
