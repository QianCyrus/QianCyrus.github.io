---
title: "My_Sglang (II): From Understanding the Model to Day 0 Support and Incremental Optimization"
description: "Retracing how we integrated Qwen3.5 into My_Sglang: inspect configuration, weights, and reference code; build generic operators and a minimal inference loop; add request state and MTP; then use profiling to guide individual fusions and verify their benefits."
date: 2026-09-24T00:00:00Z
lang: en
translationKey: my-sglang-02-qwen35-model
tags: [LLM Inference, Model Integration, Qwen3.5, My_Sglang]
draft: false
---

Adding support for a new model to an inference framework usually has a small entry point: add a class name to the registry. Once implementation begins, however, more concrete questions quickly arise: can the weights map directly to existing layers, which historical states must the model retain, where does the second forward pass resume, and do the framework's existing scheduling and caching assumptions still hold?

This article retraces how we **integrated the Qwen3.5-4B text path into My_Sglang**. The explanation follows implementation dependencies: identify the differences, implement generic computation, establish a minimal generation loop, let the framework manage requests, and finally use measurements to choose the next optimization. We examine model structure where it affects implementation decisions.

In this article, **Day 0** means only a “minimal native offline loop for ordinary text inference”: backbone weights load strictly, ordinary prefill/decode runs continuously, and fixed greedy cases match an independent reference. **MTP is neither a Day 0 feature nor a Day 0 acceptance criterion**; candidate generation, target verification, and state commit are a separate extension built after the ordinary inference baseline. Day 0 is our name for a milestone, not a claim that integration was completed on release day or that every feature has passed acceptance. In the actual Git history, the initial model, Engine, scheduling, and MTP implementations were committed together; sharing a commit does not place these capabilities in the same acceptance stage.

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="Integration route from model audit to Day 0 and individual optimizations; scroll horizontally on narrow screens">
<img src="/images/my-sglang/en/qwen35-support-roadmap.svg" alt="Configuration, weights, and reference code produce a list of differences, followed by generic operators, minimal Day 0 inference, request state and MTP integration, then a loop of profiling, one change, correctness checks, and performance comparisons." loading="lazy" />
</div>
<figcaption>Figure 1: The integration route followed in this article. Arrows indicate implementation dependencies, not the actual time spent on each step. All mechanism diagrams were redrawn around this project's content; their white backgrounds, hand-drawn lines, colored outlines, and short labels take visual inspiration from <a href="https://vllm.ai/blog/2025-09-05-anatomy-of-vllm">Inside vLLM</a>. <a href="/images/my-sglang/en/qwen35-support-roadmap.svg" target="_blank" rel="noopener">View the full SVG</a>.</figcaption>
</figure>

This article uses code snapshot `566bf693` and the pinned model revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Code paths below are relative to `python/minisgl/`; experiment files are relative to `docs/experiments/qwen35-sm120/results/`. All measurements come from the earlier 5090 experiments; no GPU benchmarks were rerun while writing this article.

## 1. What to Read When a New Model Arrives

I first divided the inspection into three sources: **configuration describes structure, the checkpoint describes actual parameters, and the reference implementation describes computation order.** Together, they tell us which parts of the framework can be reused and which must be added.

### 1.1 Start with the Config and List the Differences

The first source is the [official config.json at the pinned revision](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json). The top-level architecture is `Qwen3_5ForConditionalGeneration`, while the language model parameters live inside `text_config`. Seeing `vision_config`, we first established a text-only scope so that vision encoding, image tokens, and multimodal position encoding would not all enter the first integration round.

Reading through `text_config`, the differences with the greatest implementation impact were not about parameter count:

| Configuration we found | Direct impact on integration |
| --- | --- |
| `layer_types`: three linear-attention layers followed by one full-attention layer, repeated eight times | The 32 layers cannot all be instantiated as the same Attention type; 24 need GDN state |
| `hidden_size=2560`, `head_dim=256`, Q heads=16 | Attention's internal width is 4096; head dimension cannot be inferred as `2560/16=160` |
| `partial_rotary_factor=0.25` | Only the first 64 dimensions of each Q/K head are rotated; the remaining 192 are preserved |
| GDN QK heads=16, V heads=32, both with dimension 128, convolution width=4 | Add grouped QK expansion, a short convolution window, and an FP32 recurrent matrix |
| `tie_word_embeddings=true` | The output head reuses the `[248320,2560]` embedding weights |
| `mtp_num_hidden_layers=1` | The checkpoint includes a prediction module; native MTP can be added later without training a new prediction head |

This 4B configuration is a dense network. Every layer's FFN is a `2560 → 9216 → 2560` SwiGLU; no expert routing is required. The backbone contains 24 GDN mixers and 8 gated full-attention mixers, sharing the same pre-norm residual and FFN structure.

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="Network differences this model integration must handle; scroll horizontally on narrow screens">
<img src="/images/my-sglang/en/qwen35-blocks.svg" alt="Qwen3.5-4B's shared embedding, 32-layer hybrid backbone, and tied head; 24 GDN layers retain convolution and recurrent state, 8 gated-attention layers retain KV, and MTP uses an independent prediction layer and state." loading="lazy" />
</div>
<figcaption>Figure 2: The model map obtained from the configuration audit. It highlights the differences integration must handle; the supported model scope remains limited to the text path. <a href="/images/my-sglang/en/qwen35-blocks.svg" target="_blank" rel="noopener">View the full SVG</a>.</figcaption>
</figure>

### 1.2 Inspect the Checkpoint to See How the Config Appears in the Weights

The second source is the parameter names, shapes, and dtypes in safetensors. At this stage, we check which prefix holds the text weights, whether the Q projection includes an output gate, how GDN projections and state parameters are named, whether MTP parameters actually exist, and whether the tied head has a separate weight tensor.

Our eventual mapping was `model.language_model.* → model.*`, retaining `mtp.*` and skipping `model.visual.*` within the text-only scope. The hybrid branch is limited to TP=1 and initially bypasses the existing loader's QKV merging and sharding logic for other models, allowing local object names to correspond directly to the checkpoint.

Dtypes should also be recorded here. Our first strict load later failed precisely because we had missed two kinds of FP32 parameters; matching shapes does not mean the parameter declarations are complete.

### 1.3 Trace State and Rounding Through the Reference Forward Pass

The third source is the Qwen/Transformers reference computation. We focused on operation order, rather than merely copying module class names:

- How Attention's Q and gate are laid out, and whether Q/K norm comes before or after RoPE.
- Whether causal conv caches its inputs or outputs, and how wide a window must survive chunking.
- Whether GDN decays or updates first, and whether its output reads the old or new state.
- Which computations convert to FP32 and where values must first round back to BF16.
- Whether MTP's tokens and hidden states are shifted by one position, and which hidden states it uses.

Back in mini-sglang, this meant reading `ModelConfig`, model registration, `BaseOP` weight loading, `Engine.forward_batch`, and `Req.cached_len`. The resulting integration checklist was: add hybrid configuration, implement two mixers, load real parameters, define complete request state, and connect the Engine. The existing HTTP and scheduling skeleton could be reused, but the new state lifecycle needed explicit integration.

### 1.4 Pin This Starting Point in an Isolated Environment

The experiment began with a new branch and a separate worktree from upstream `20fcd7f`, avoiding existing uncommitted changes on the remote machine. The GPU was verified as an RTX 5090, SM120. We used a dedicated venv; when system Python lacked ensurepip, we used the existing uv installation to create an isolated environment with system-site-packages disabled. Software versions and model hashes were recorded, and every subsequent operator change used the same model snapshot.

At this stage, we also corrected capability detection for SM120 versus SM100: belonging to Blackwell does not automatically make the same backend applicable. We first used generic PyTorch CUDA computation so that initial model support would not depend on a particular specialized kernel happening to run.

## 2. Start Coding: Extend Configuration, Then Build Generic Computation We Can Compare

### 2.1 Keep Configuration Changes Small and Pass the Structure Through

`e8c60a6` extended `ModelConfig` with `layer_types`, GDN heads/dimensions/conv width, and MTP layer count, and added partial RoPE parsing. Unwrapping nested `text_config` and handling explicit `head_dim` already existed; we reused and checked them. The addition was to carry the missing hybrid-model information through to the constructors.

Configuration tests cover both the new model and the existing dense path: the new model should receive 64-dimensional RoPE, the correct layer types, and GDN parameters, while existing models retain their original RoPE configuration. This allows later shape errors to be traced early to configuration or layer construction instead of all surfacing during real weight loading.

### 2.2 Implement Reference Operators First and Make State Ownership Explicit

`4f501ec` added three generic operators. Convolution and recurrence explicitly receive old state and return new state; gated norm is a stateless transformation:

```text
causal_conv1d(x, weight, initial_state)        → output, next_conv
recurrent_gated_delta_rule(..., initial_state) → output, next_recurrent
rms_norm_gated(output, z, weight)              → gated_output
```

They use PyTorch tensor operations on the input device. In the GPU experiments, inputs, states, and computation were all on CUDA; the first implementation had no hardware specialization or autotuning.

We made the operators avoid overwriting incoming old state in place. GDN starts from a clone of the old state, and convolution returns an independent new window. The model layer decides when to replace request-owned state; the operator does not need to know the request UID. This makes continuity easier to check and lets later MTP execution discard temporary branches.

### 2.3 For GDN, First Translate the Recurrence into Code We Can Check Line by Line

GDN inputs come from four projections: qkv is `[N,8192]`, z is `[N,4096]`, and a/b are each `[N,32]`, where N is the total token count in the current forward pass. qkv passes through a width-4 depthwise causal conv and SiLU, then splits into Q, K, and V; z and a/b do not pass through the convolution.

The convolution cache retains the latest four **projection inputs before convolution**. To compute the next chunk, it concatenates the old window and current input, performs convolution, takes the final T outputs corresponding to the current chunk, and retains the last four raw inputs. If it stored post-SiLU results instead, the first chunk might look correct, but the second would continue convolution with the wrong history.

Q/K initially have 16 heads, each repeated twice to match V's 32 heads. Each head's recurrent state is `128×128`, organized along key and value axes. Q/K undergo L2 normalization, and query is additionally multiplied by `128^(-1/2)`; the decay and update gates are:

```text
β = sigmoid(b)
g = -exp(A_log) * softplus(a + dt_bias)
```

Each token is computed in this order:

```python
state = state * g[:, t].exp()[..., None, None]
memory = (state * k_t.unsqueeze(-1)).sum(-2)
delta = (v_t - memory) * beta[:, t].unsqueeze(-1)
state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
output[:, t] = (state * q_t.unsqueeze(-1)).sum(-2)
```

This code fixes the semantics that are easy to confuse: memory is read from decayed history; delta corrects the value associated with the current key; output is read from the updated state. Recurrent accumulation and state use FP32, with output cast back to the input dtype. Q/K normalization follows the reference path's input-dtype operation order; the entire layer cannot simply be described as “all FP32.”

### 2.4 For Attention, Resolve Layout First, Then Access to History

The Q projection produces both query and an output gate. Its output is `[N,8192]`, which must first be reshaped by head and then split within each head:

```python
qg = q_proj(x).view(N, 16, 512)
q, gate = qg.chunk(2, dim=-1)   # Both are [N,16,256]
```

Splitting the 8192 dimensions directly in half may still produce the expected final shapes, but changes the Q/gate mapping within each head. Q/K then receive their own norms, RoPE rotates only the first 64 dimensions, and the remaining 192 retain their values. Each of K/V's four heads is repeated four times to match the 16 Q heads.

We initially used PyTorch SDPA, with scale based on the full head dimension, `256^(-1/2)`. Its output is multiplied by `sigmoid(gate)`, flattened to 4096 dimensions, and projected back to 2560. The actual KV cache still retains four heads; expansion is an intermediate computation result.

### 2.5 Implement the Two Norm Variants Separately to Avoid Incorrect Reuse

The general `_Norm` normalizes in FP32, multiplies by `1 + weight`, and finally casts back to the input dtype. GDN's gated norm instead multiplies directly by weight and requires an intermediate BF16 rounding step:

```python
x = hidden.float()
x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
x = weight * x.to(hidden.dtype)
out = (x * F.silu(gate.float())).to(hidden.dtype)
```

That cast is a boundary subsequent fusion must preserve. Keeping every step in FP32 until the end may still resemble the same mathematical formula, but it changes the floating-point execution path.

The shared decoder structure is straightforward:

```text
u      = x + mixer(input_norm(x))
x_next = u + down_proj(silu(gate_proj(post_norm(u))) * up_proj(post_norm(u)))
```

Before loading the full model, generic GPU operator tests had already passed **30/30**; independent HF GDN comparisons covered eight combinations of B=1/4, T=1/16, and FP32/BF16, with recorded maximum absolute errors of 0 for both output and final state. This established the computation and state contracts for the specified operators, not yet the behavior of the whole model.

During development, we also encountered shape-dependent rounding in CPU BF16 `rsqrt`: full-sequence and chunked calls differed by 1 ULP. We tested “normalization numerics” separately from “subsequent recurrence continuity” and retained the failure record; GPU tests still covered the complete normalization-plus-recurrence path. This distinction kept the comparison target clear instead of obscuring it just to make tests pass.

## 3. Complete Day 0: Strict Loading, Then a Working Prefill → Decode Loop

### 3.1 Build a Native Model with BaseOP Instead of Calling HF Generate

We reused mini's `BaseOP` to organize `_TextModel`, `_DecoderLayer`, `_Attention`, `_GatedDeltaNet`, and `_MLP`, and mapped the checkpoint architecture name to the local `Qwen3_5ForCausalLM` in the registry. The model first creates parameter placeholders on the meta device, then loads actual tensors, avoiding allocation of a complete randomly initialized copy of the GPU weights.

`BaseOP.load_state_dict` consumes expected keys one by one, checks shape/dtype, and rejects any remaining keys. Our goal was to make structural mismatches fail explicitly at startup.

The first load did fail: `A_log` and `linear_attn.norm.weight` were FP32 in the checkpoint, while the original declarations inherited the BF16 default dtype. The fix was to declare these two parameter types explicitly as FP32 and make the Engine load each parameter with its expected dtype; we did not cast all parameters uniformly down to BF16.

### 3.2 First Account for the 426 Backbone Keys Required by Ordinary Inference

Day 0 needs to check the ordinary text backbone. Its required 426 keys can be derived from the network structure:

| Component | Key-count derivation |
| --- | ---: |
| 24 GDN decoders | 24 × 14 = 336 |
| 8 Attention decoders | 8 × 11 = 88 |
| Text embedding and final norm | 2 |
| Ordinary text backbone total | **426** |

A GDN layer's 14 keys comprise four input projections, conv, A_log, dt_bias, gated norm, and out projection, plus two decoder norms and three FFN projections. An Attention layer's 11 keys comprise q/k/v/o, Q/K norms, two decoder norms, and three FFN projections. Tied embedding does not create an additional LM head weight.

The count is only a structural cross-check; completeness still depends on every key's name, shape, and dtype. The integrated implementation also loaded 15 MTP parameters at startup, which is why the historical logs report 441 in total. That is the loader's implementation granularity, not a reason to include MTP execution or correctness in Day 0. The additional 15 keys and their execution flow are discussed in Section 4.4.

### 3.3 Isolate Model Problems with a Minimal Loop

I first used `forward_tokens` for offline generation, carrying one state through the prompt and subsequent tokens. The core logic is shown below; the actual test script also includes timing synchronization, warmup, and result persistence:

```python
state = Qwen35State()
out = model.forward_tokens(prompt_ids, state=state)  # Consume the entire prompt
generated = []

for step in range(output_tokens):
    token = out.logits[-1].argmax().view(1)
    generated.append(token)
    if step + 1 < output_tokens:
        out = model.forward_tokens(token, state=state)
```

At this point, checking whether the text resembles a sentence is not enough. We also need to check whether logits are finite, state length grows correctly, the next iteration actually consumes the token selected in the previous one, and repeated runs agree. The first output comes from the prompt's final-position logits; when it is selected, it is not yet part of the state. The next forward pass consumes it.

The second smoke run also exposed a problem in the test script itself: the tokenizer returned an object other than the Tensor the script expected, so calling `numel` failed. We changed the script to generate text with the chat template first, then explicitly obtain input IDs with `return_tensors="pt"`, avoiding misclassification of a client API error as a model computation failure.

### 3.4 What Evidence Lets Us Cross This Milestone?

The initial `02-model/smoke/model.json` records real loading of 441 keys—the 426 backbone keys plus 15 MTP keys loaded alongside them—three matching ordinary generation runs with 32 input tokens and 16 output tokens, and agreement between the first 16 greedy tokens of a fixed chat case and an independent HF model. This ordinary generation run executes no MTP candidate generation, verification, or commit. At this point, we called the minimal Day 0 loop passed and continued extending the framework interface.

A later audit added a correction here: although the first run recorded `chunked_greedy_equal=true`, the input was only 32 tokens and the script's chunk size was 64, so it still used only one chunk. Actual cross-chunk evidence came from a later experiment splitting 512 input tokens into eight 64-token chunks. The Day 0 smoke test cannot preemptively satisfy an acceptance check it never exercised.

These implementations were ultimately saved together with the initial Engine, scheduler, and MTP integration in `0e3bbf4`, rather than a series of artificially separated Day 0 commits.

## 4. Let the Framework Manage Requests: State, Chunking, Batches, and MTP

The offline loop established the model's input/output contract. The next step was to connect those conventions to mini's request lifecycle, so that the scheduler supplies only new tokens each round and the model finds the correct history.

### 4.1 Put the Three Kinds of History into One Request State

`Qwen35State` contains `length`, `position_offset`, and per-layer `kv`, `conv`, and `recurrent` state. The main tensors for an ordinary request are:

| State | Per-layer shape | Property |
| --- | --- | --- |
| K, V | Each `[L,4,256]`, BF16; 8 layers | Grows with the number L of consumed tokens |
| Convolution window | `[1,8192,4]`, BF16; 24 layers | Fixed window retaining pre-convolution inputs |
| GDN recurrent | `[1,32,128,128]`, FP32; 24 layers | Fixed size retaining the updated recurrent memory |

From the shapes, basic backbone state storage is approximately `49.5 MiB + L×32 KiB`. This excludes temporary tensors, MTP branches, Graph staging, and the allocator, but is sufficient to show why admission must reserve fixed GDN state for each request instead of estimating capacity solely by KV token count.

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="Request continuation across chunks and state ownership; scroll horizontally on narrow screens">
<img src="/images/my-sglang/en/qwen35-state-continuation.svg" alt="Two prefill chunks and subsequent decode consume tokens in order: KV appends, the convolution window rolls forward, and the GDN matrix continues recurrence; emitted pending tokens and consumed state are counted separately, and slot reuse checks the UID." loading="lazy" />
</div>
<figcaption>Figure 3: The three histories that the state interface must maintain together. Token blocks represent logical positions, not physical KV pages; the current implementation uses request-owned dynamic KV. <a href="/images/my-sglang/en/qwen35-state-continuation.svg" target="_blank" rel="noopener">View the full SVG</a>.</figcaption>
</figure>

### 4.2 Chunking Is Not Repeating Several Independent Prefills

After the first chunk, the next must use the same state. Positions begin at `state.length + position_offset`; Attention reads historical KV, convolution carries the old window, and GDN continues updating the old matrix.

Attention also requires a concrete change: with P historical tokens and T tokens in the current chunk, the keys visible to query row i satisfy `j ≤ P+i`. The code explicitly constructs an offset mask:

```python
rows = torch.arange(T, device=device)[:, None] + P
cols = torch.arange(P + T, device=device)[None, :]
mask = cols <= rows
```

For example, with P=3 and T=2, the two rows can see keys `0…3` and `0…4`, respectively. A triangle anchored at the upper-left corner without a history offset would mask valid history. For ordinary single-token decode with existing past, all positions in the current cache are valid.

Later, the 512-token input in `03-gated-norm/model-reference/model.json` genuinely crossed eight 64-token chunks, and the first 16 outputs from chunked execution matched the full-sequence path. This was among the first actual cross-chunk evidence, still within a limited scope.

### 4.3 Flatten Multiple Requests Without Mixing Their Histories

`forward_packed` receives `input_ids`, `lengths`, one state per request, and corresponding positions. For example, a three-token input chunk and one decode token can be packed as:

```text
input_ids = [a0, a1, a2, b0]
lengths   = [3, 1]
states    = [state_A, state_B]
positions = [L_A, L_A+1, L_A+2, L_B]
```

Projections, norm, and FFN share computation across the total token axis; Attention/GDN read and update each request's history by segment. The entry point rejects two requests sharing the same mutable state. By default, it returns only the last-position logits of each segment; MTP verification can request logits for every position.

The Scheduler calls `Engine.forward_batch` to reach model `forward()`. The model maintains `_request_states[table_idx]=(uid,state)`: new requests receive empty state, UID mismatches raise errors, and `state.length` must equal `Req.cached_len`. Completion or cancellation releases the corresponding state, establishes necessary stream dependencies if execution is in flight, and only then permits a new request to use the slot.

We did not present this as a completed unified Paged KV implementation. The model dynamically appends actual KV in the current hybrid path, the Engine has `kv_cache=None`, and page tables and page budgets still serve scheduling and capacity management. The current path is limited to TP=1, page_size=1, and naive prefix cache; matching KV without the corresponding convolution/GDN state cannot restore a correct prefix.

### 4.4 After Day 0: Integrate MTP Candidates, Verification, and Commit Separately

MTP is added after the ordinary inference baseline has been established. Although the initial code shared a commit with model integration, this speculative execution path requires separate implementation and acceptance. It loads 15 additional keys: FC, two input norms, one Attention decoder containing 11 keys, and final norm. Combined with the backbone's 426 keys, these produce the historical loading count of 441.

The MTP network separately normalizes the next-position token embedding and previous-position hidden state, concatenates them, and produces predictions through a `[2560,5120]` FC, an independent full-attention decoder, final norm, and the shared head. The current controller passes `hidden_states` after the target's final norm; this is an explicit implementation choice whose full numerical alignment still requires regression testing.

Prefill priming pairs `input_ids[1:]` with `target.hidden_states[:-1]`. MTP has `position_offset=1`, so its state length trails the target by one. Across chunks, the last hidden state from the previous chunk must also be retained for the first token of the new chunk; independently slicing each chunk must not discard that boundary pair.

`GreedyMTPController` rolls out candidates on temporary state, then asks the target to verify `[pending,candidates…]` on a separate clone. Acceptance is the contiguous matching prefix. When rejection occurs, GDN state cannot simply have its suffix sliced off like a token array, so we replay the truly accepted input prefix from the old target state, then use actual target hidden states to teacher-force the MTP state to be committed.

Session references are replaced together only after all new states and hidden states to be published are ready. Output consists of the accepted prefix plus the target bonus, which becomes pending for the next round. Online integration must also update the token pool, release reservations for rejected suffixes, and handle EOS and length budgets within an output block.

The early short regression in `04-mtp/mtp.json` covered B=1/3, ordinary/k=1/k=3, and three rounds per configuration, for 18 batch runs, with outputs matching their corresponding ordinary paths. However, it used only three short chat prompts, forced generation of 32 tokens, and ignored EOS. We treated it as an early check of state commit; the later real HTTP subset still exposed strict alignment problems.

## 5. After Day 0, Profile Before Choosing the Next Change

Only once the reference path runs do performance questions have a comparison target. We had already collected local traces during the generic-operator stage; after the model worked, we added `record_function` markers to each layer, token mixer, and MLP, and collected a CPU/CUDA trace for another native decode step.

At this point, I looked at two things together: where GPU time was concentrated, and how many small operations the CPU was launching onto the GPU. Total token/s alone cannot tell us whether to change projections, recurrence, state transfers, or submission.

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="Kernel hotspots from the first real decode trace; scroll horizontally on narrow screens">
<img src="/images/my-sglang/qwen35-day0-profile.png" alt="Kernel hotspot chart replotted from an actual Chrome trace; two GEMV families account for most cumulative execution time, with conversions, reductions, and elementwise operations among the remainder." loading="lazy" />
</div>
<figcaption>Figure 4: Statistics replotted from the earlier real trace, retaining the measurement chart's original presentation. The initial record contains 2777 GPU kernels; this is the distribution of cumulative kernel execution time, not full generation latency. <a href="/images/my-sglang/qwen35-day0-profile.png" target="_blank" rel="noopener">View the original image</a>.</figcaption>
</figure>

In this single-step trace, cumulative GPU kernel time was approximately 9.172 ms, with two GEMV families accounting for approximately 5.892 ms, or 64.2%. Conversions, reductions, and elementwise operations also generated many launches. This highlighted the importance of projections while suggesting that we could first choose a small region with clear numerical boundaries to test whether fusion benefits would reach the model level.

The first choice was gated RMSNorm: it has a well-defined reduction, weight multiplication, and SiLU gate, with no cross-token state dependence, making it suitable for a first test of the fusion approach. This choice was not a promise that overall latency would improve.

For formal comparisons, profiling and benchmarking ran separately. Profiling adds recording and instrumentation overhead; a CUDA event interval around a sequence of Python launches may also include gaps where the stream waits for the host. Model comparisons used three generation rounds without profiling, after warmup at the same shapes, and saved prefill, decode, and full offline generation times separately.

Both A/B experiments below fix B=1, 512 input tokens, 128 generated tokens, the same model/input hashes, and the greedy path. Full offline generation includes prefill, 127 subsequent decode passes, and token selection, excluding weight loading, HTTP, queuing, and networking. Each optimization remeasures its own reference; the tables report three-round means and sample standard deviations, and the raw sample excerpts can be [viewed directly as JSON](/data/my-sglang/qwen35-stages.en.json).

## 6. First Fusion: Faster Locally, but No Model-Level Benefit

`qwen35_fused.py` assigns one row of gated norm to a Triton program, combining mean-square calculation, rsqrt, weight multiplication, and the SiLU gate in one call. The implementation preserves the point where normalized values cast back to the input dtype; for non-FP32 weights, it also preserves the corresponding rounding after weight multiplication, and sets `enable_fp_fusion=False`.

I did not change GDN recurrence at the same time. This leaves only one replacement boundary to inspect if regression fails and attributes performance changes to this switch alone: `MINISGL_QWEN35_GATED_NORM=reference|triton`.

Local tests passed 12/12. In the B=1/T=1 microbenchmark, the median CUDA event interval fell from 54.606 μs to 9.918 μs; a single trace changed from 12 kernels to 1. The full model still needed to be rerun:

| Metric, ms | Reference: three-round mean ± standard deviation | Fused gated norm | Observation |
| --- | ---: | ---: | --- |
| Prefill | 516.042 ± 0.651 | 521.302 ± 0.659 | No reduction |
| Decode | 1769.918 ± 4.760 | 1796.328 ± 14.073 | No reduction |
| Full offline generation | 2285.960 ± 5.274 | 2317.630 ± 14.565 | Time increased by approximately 1.385% |

The two paths produced identical 128-token outputs across all three rounds, and their first 16 outputs with chunked inputs also matched. A separate model trace confirmed that all 24 layers actually executed the fused kernel, ruling out a simple explanation such as “the switch never took effect.”

We therefore retained the implementation and measurements while leaving reference as the default. The two model experiments ran in separate processes without interleaved A/B, so this approximately 1.4% regression cannot be attributed to a particular cache or hardware behavior from these data alone. What we can establish is that this attempt delivered no model-level benefit; an approximately 5× microbenchmark improvement cannot be reported as model acceleration.

This negative result meant the next step still needed an independent experiment, rather than enabling several fusions together and selecting a better overall number.

## 7. Second Fusion: Keep GDN's Temporal Recurrence Inside One Kernel

### 7.1 This Change Targets State Access and Kernel Launches

Reference GDN loops over T tokens in Python, invoking several elementwise and reduction operations for each token. This expresses the formula clearly, but repeatedly launches kernels and materializes intermediate state as tensors. Its execution cost accumulates rapidly as T grows.

`ce8acc4` introduced generic Triton recurrence. We first fixed the real shape K=V=128 and used `value_tile=32, num_warps=4`. One program corresponds to a batch/head/value tile, holds a `128×32` FP32 state block, and executes normalization, decay, memory reduction, delta, outer-product update, and query readout in token order inside the kernel.

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="Independent fusion approaches for gated norm and GDN recurrence; scroll horizontally on narrow screens">
<img src="/images/my-sglang/en/qwen35-fusion-process.svg" alt="Gated norm fuses multiple operations while preserving BF16 rounding boundaries; GDN tiles value columns, reduces over the full key axis, and advances state through time within one program, separating the generic implementation from the later SM120 parameter sweep." loading="lazy" />
</div>
<figcaption>Figure 5: The implementation approaches for two independent fusions. Boxes indicate computation and data ownership, not measured time proportions; projections, convolution, and the output projection remain outside the recurrence kernel. <a href="/images/my-sglang/en/qwen35-fusion-process.svg" target="_blank" rel="noopener">View the full SVG</a>.</figcaption>
</figure>

Why split along the value axis? Updating each value column requires memory/query reductions over the complete key axis, but different value columns can be handled independently. Tiling value columns retains the full K axis and places parallelism across batch, head, and value tile. Tokens still have recurrent dependencies and cannot be arbitrarily reordered for parallel execution.

Initial state is read-only, and final state is allocated separately, preserving the reference ownership contract. Q/K normalization is also inside the kernel, but explicitly retains BF16 conversion boundaries for square, sum output, epsilon addition, rsqrt, multiply, and related steps; an “all-FP32 rewrite” is not treated as an equivalent replacement for the original path.

This remains a generic Triton implementation with no SM120-specific instructions. Shapes that do not meet the supported conditions fall back to reference, while projections, causal convolution, gate-parameter preparation, output norm, and out projection remain on the original path.

### 7.2 Pass Numerical Checks, Then Return to Full Generation

All 23 GPU tests passed preset tolerances, covering B=1/4/8, T=1/16/65, FP32/BF16, nonzero initial states, chunking, and fallback. The maximum absolute BF16 output difference was `0.0001220703125`, and the maximum absolute state difference was approximately `1.7881393e-7`; this is agreement within tolerance, not bitwise identity.

The model experiment changed only `MINISGL_QWEN35_GDN`, kept gated norm at reference, and remeasured the baseline. In the current code, GDN still defaults to reference; fusion requires explicit opt-in and has not been globally enabled based on this one workload:

| Metric, ms | Reference: three-round mean ± standard deviation | GDN Triton | Time reduction |
| --- | ---: | ---: | ---: |
| Prefill | 512.128 ± 0.858 | 42.210 ± 0.039 | 91.758% |
| Decode | 1761.840 ± 4.195 | 1540.485 ± 22.505 | 12.564% |
| Full offline generation | 2273.968 ± 5.009 | 1582.695 ± 22.496 | **30.399%** |

The three reference full-generation times were `2271.372 / 2270.791 / 2279.743 ms`, versus `1608.604 / 1571.362 / 1568.119 ms` for fusion. The first fused round was slower and remained in the statistics. The two paths matched on 128 output tokens across three rounds and on the first 16 chunked outputs; the fused path also had a 16-token comparison against HF on a fixed chat case.

The large prefill reduction has a specific comparison target: we removed the dominant cost of generic PyTorch's per-token, multiple-launch execution. This is not evidence of outperforming mature projects' parallel GDN prefill, nor can it be generalized to “every request is 30% faster.” The evidence covers this ordinary, single-request, 512/128 offline workload; MTP/HTTP/Graph combinations still require separate measurements.

### 7.3 Profile Again to Confirm Where the Optimization Took Effect

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="Actual CPU and GPU timeline after GDN fusion; scroll horizontally on narrow screens">
<img src="/images/my-sglang/qwen35-gdn-profile.png" alt="CPU launch and GPU stream timeline replotted from a real single-step decode trace; projections and auxiliary computation remain after fusion, and the plotted span must not be treated as full generation time." loading="lazy" />
</div>
<figcaption>Figure 6: The actual post-fusion CPU/GPU trace, replotted. The reference from the same stage had 2785 kernels, versus 2185 with fusion, including 24 recurrence-kernel calls; the 30.399% reduction comes from three generation rounds without profiling, not from this chart's width. <a href="/images/my-sglang/qwen35-gdn-profile.png" target="_blank" rel="noopener">View the original image</a>.</figcaption>
</figure>

We also searched for `_recurrent_kernel` in Perfetto and confirmed that the call count matched the 24 GDN layers. This turns “fusion is enabled in the code” into “fusion executed in the actual trace.” A reduction of approximately 600 kernels matches the scope of the replacement, but the remaining projections, Attention, norms, and state operations continue to incur cost.

## 8. CPU Launches Still Cost Time, So We Tried CUDA Graph Separately

Fusion changes the granularity of kernel computation; CUDA Graph changes how repeated work is submitted. Because they address different problems, we kept Graph behind an independent switch, capturing ordinary decode with fixed-address KV/GDN staging, then loading request state, replaying, and committing it back.

This step first encountered a numerical problem. Dynamic Attention at its effective length and a masked path with static capacity may choose different numerical implementations. In diagnosis, static and Graph agreed while eager and static differed, so the problem should not be attributed directly to capture. After standardizing on math SDPA, three-step checks of logits, hidden, KV, conv, and recurrent state agreed at B=1, prefix=16, capacity=256; larger batches and capacities did not all pass.

This independent Graph A/B set both GDN and gated norm to reference and used math SDPA on both sides; it did not stack benefits on top of the previous section's fused GDN configuration. Within this limited scope, the forward wrapper including staging/replay/commit fell from `16.176 ± 0.029 ms` to `10.807 ± 0.001 ms`. It excludes the external test harness's cloning and token selection, capture has a separate cost, and this is not full HTTP generation. The logits check at capacity=8192 still failed.

We also measured math SDPA's additional cost separately: with Graph disabled, interleaved A/B of 512/128 generation over three rounds per side found math approximately 5.44% slower than auto. The current defaults therefore remain auto with Graph disabled; only an experimental B=1, capacity=256, math entry point is exposed, with no claim of completed 8K or MTP Graph support.

This experiment shows that extending a framework capability also requires checking the input layout and numerical path it changes. Successful capture is not the endpoint of acceptance.

## 9. Only Then Tune for the 5090: From a CTA Hypothesis to a Negative Result

Only after generic GDN fusion was stable did we start forming hypotheses from actual hardware information. At B=1, H=32, and value tile=32, the launch grid contains `32×4=128` CTAs, while the tested 5090 has 170 SMs. This suggested a testable hypothesis: could smaller value tiles increase the CTA count and improve parallel utilization at small batch sizes?

There is no guarantee. Smaller value tiles may repeat Q/K normalization and reads more often; larger tiles may increase register usage. We therefore performed an independent sweep:

```text
B       = 1, 4, 8
T       = 1, 16, 128
tile    = 16, 32, 64
warps   = 4, 8
Total   = 3 × 3 × 3 × 2 = 54 cases
```

Each case first compared output and final state against generic tile32/warp4, then ran three measurement rounds after passing. Measurements separated the Python wrapper from GPU intervals between consecutive calls inside a Graph. The latter reduces the effect of host launch gaps; **it is not a claim that Graph support for the complete model has passed**.

Of the nine input shapes, seven still had their lowest interval with generic tile32/warp4. Only B1/T1 and B8/T1 improved with tile16/warp4, reducing the local GPU interval by approximately 5.27% and 5.79%. At B1/T1, CTA count rose from 128 to 256 and the GPU interval fell from approximately 2.436 to 2.308 μs, but mean Python wrapper time increased from approximately 13.371 to 13.844 μs.

We also observed that at T>1, tile64/warp4 used substantially more registers than tile32/warp4, and increasing the warp count was usually slower. We did not add a default SM120-specialized dispatch on this basis: the full model calls GDN per request, so the B8 microbenchmark cannot be applied directly, and the local improvement has no new end-to-end evidence.

Mature backends were also included in comparisons. FlashInfer SM120 GDN's final state failed this project's preset tolerance at all nine tested shapes; the cause still needs investigation and cannot immediately be classified as a backend bug. That round and subsequent fixed-linear microbenchmarks also overlapped external GPU load, invalidating their timings; only numerical and call-structure records were retained. None of these results count as accepted optimizations.

## 10. Online Regression Brought Us Back to Correctness

After placing the model behind HTTP, SSE, and real batching, we fixed nine coding/math/reasoning requests from SPEED and measured ordinary execution and MTP k=1 separately at concurrency 1/4/8, three rounds each. This online comparison used reference GDN and gated norm, disabled Graph and overlap, and used the `prefill_first` scheduling policy. Both modes completed 81/81 requests; actual backend evidence covers counting, EOS, SSE assembly, and six cancellations of running requests per mode.

However, only **55/81** raw token hashes matched. The ordinary path itself also differed across batches. Passing the early three short chat prompts did not cover real arrival ordering and longer outputs; protocol success rate cannot substitute for greedy consistency.

Next, we fixed the same teacher-forced token history, compared activations layer by layer at B=1/4/8, and checked full weight hashes before and after. The earliest visible divergence appeared in layer 0 GDN's `in_proj_qkv`. Disabling BF16 reduced-precision reduction moved the first prefill divergence to the MLP down projection but did not eliminate it. All 17 prediction positions in this short diagnostic still had identical argmax results, so it narrowed the search rather than reproducing and explaining every online divergence.

This led to an opt-in BF16 linear path with a fixed reduction order (`89316b3`), and a fix for the internal consistency issue that rejection replay must carry its actual replay hidden states and bonus forward (`49d8eeb`). Limited operator/CPU checks have results; full GPU model and online regression after these fixes remain outstanding.

This is also the article's current endpoint: we have established the Day 0 baseline, integrated hybrid state and MTP, and obtained local and offline benefits on some workloads; full Qwen3.5/MTP combination acceptance is not finished. Throughput changes from online rounds that failed alignment were not treated as validated acceleration.

## 11. Reproduce This Route Through Git and Experiment Files

Each change needs three things retained together: an implementation that can be disabled, repeatable comparison inputs, and results sufficient to explain the conclusion. The actual entry points, following this article's steps, are:

| Stage | Code / commit | Main evidence |
| --- | --- | --- |
| Read the structure and extend configuration | `models/config.py`, `e8c60a6` | Configuration regression, pinned model revision |
| Implement generic operators | `kernel/qwen35_reference.py`, `4f501ec` | `01-generic/gpu-tests.log`, `01-generic/gdn-hf-smoke.json` |
| Day 0 and model integration | `models/qwen3_5.py`, registration/weights/Engine, `0e3bbf4` | `02-model/smoke/model.json` and failure logs |
| Negative result from the first fusion | `kernel/qwen35_fused.py` | `03-gated-norm/` |
| Early MTP checks | `engine/speculative.py`, `scheduler/mtp.py` | `04-mtp/mtp.json` |
| Generic GDN fusion | `kernel/qwen35_gdn.py`, `ce8acc4` | `05-gdn/`, 23 GPU checks, three-round model comparison |
| Graph extension | `engine/qwen35_graph.py`, integration including `b8da37d` | `06-graph/`, retaining passed and failed scopes separately |
| SM120 parameter comparison | `benchmarks/qwen35/tune_gdn.py`, `a2d3f20` | `07-sm120/sweep.json`, all 54 cases |
| Online and numerical diagnosis | HTTP benchmark, `batch_numerics.py`, `49d8eeb`, `89316b3` | `07-http/`, `09-batch-numerics/` |

Early measurements ran in a worktree whose changes were not all committed, so the base HEAD in a result does not always reconstruct the complete code by itself. Later model comparisons added `source_sha256`, which must also be checked when reproducing results. This article's public [A/B excerpts](/data/my-sglang/qwen35-stages.en.json) include per-round samples, comparison scope, source-file hashes, and code hashes; the complete raw files remain in the project's experiment directory.

For example, in an isolated CUDA venv prepared with the recorded versions, rerun a baseline on the model's ordinary path:

```bash
# Run from the My_Sglang repository root; the model path points to a local snapshot at the pinned revision.
source /path/to/qwen35-venv/bin/activate
export PYTHONPATH="$PWD/python"
export MINISGL_QWEN35_GDN=reference
export MINISGL_QWEN35_GATED_NORM=reference
export MINISGL_QWEN35_LINEAR=reference
export MINISGL_QWEN35_SDPA=auto
export MINISGL_QWEN35_GRAPH=0

python benchmarks/qwen35/model_baseline.py \
  --model /path/to/pinned-Qwen3.5-4B-snapshot \
  --input-tokens 512 --output-tokens 128 --rounds 3 \
  --check-hf --profile --output artifacts/qwen35-reference-rerun
```

To test GDN, change only `MINISGL_QWEN35_GDN` to `triton`, write to another new output directory, and keep everything else the same. Profiling explains execution structure; formal three-round timing takes place in the script's unprofiled region. Do not convert screenshot width into speedup, and do not overwrite the previous round's failure data.

The working order that emerged from this integration is to align configuration, weights, and reference semantics first, then establish an ordinary path with explicit state; for every new execution mode, inspect how it consumes and commits state; for every proposed optimization, use an independent switch, numerical regression, and same-workload comparison to decide whether to retain it. Next, real greedy regression still needs to be resolved before completing end-to-end acceptance for mixed scheduling, overlap, and larger input ranges.
