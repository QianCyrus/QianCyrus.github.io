---
title: "My_Sglang (2): Integrating Qwen3.5, Then Optimizing from the Profile"
description: "A record of bringing ordinary Qwen3.5 inference into My_Sglang: start with configuration, weights, and state semantics; build a general GPU implementation; use Perfetto to locate overhead; evaluate GDN fusion, CUDA Graph, and SM120 tuning one at a time; and investigate numerical differences under batching."
date: 2026-09-24T00:00:00Z
lang: en
translationKey: my-sglang-02-qwen35-model
tags: [LLM Inference, Model Integration, Qwen3.5, My_Sglang]
draft: false
---

Adding a model to an inference framework usually takes one more line in the registry. The difficult questions come afterward: what exactly does one forward pass consume? Which history must survive into the next call? When an individual request joins a batch, do the original computation and state contracts still hold?

This post documents the integration of the Qwen3.5-4B text model into My_Sglang. We started with a general PyTorch CUDA implementation, got ordinary generation working, inspected the profile, and replaced operators with fused implementations one at a time. The clearest improvement came from the GDN recurrence: for a fixed single request with a 512-token input and a 128-token output, complete offline generation fell from **2273.968 ms to 1582.695 ms, a 30.399% reduction in elapsed time**. We also had an experiment where an operator became five times faster while the model became slightly slower.

These results come from measurements on an RTX 5090 on September 23, 2026. After the device recovered on September 25, we added numerical regression tests. The account below follows the implementation dependencies. It stays with ordinary inference, covering the implementation, measurements, and several points where we had to return to correctness checks.

## 1. Read the configuration, weights, and reference forward first

Before writing code, I examined three things: the configuration describes the network structure, the checkpoint specifies the actual parameters, and the reference forward determines the computation order. Each often contains information the others do not.

[The config.json at our pinned revision](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json) has a multimodal wrapper at the top level, with the language model configuration under `text_config`. This round covered only the text path, so we first followed `text_config` to establish the backbone without bringing the vision encoder into the minimal implementation.

| Configuration | What the integration must handle |
| --- | --- |
| 32 layers: 24 GDN and 8 full attention | Construct two types of token mixer according to `layer_types` |
| hidden size=2560, Q heads=16, head dim=256 | Attention has an internal width of 4096; head dim cannot be inferred by dividing hidden size by heads |
| KV heads=4, partial rotary factor=0.25 | Implement GQA; rotate only the first 64 dimensions of each Q/K head |
| GDN Q/K heads=16, V heads=32, all with dimension 128 | Align the head groups and retain recurrent state for each value head |
| causal conv width=4 | Retain each layer's convolution input window across calls |
| tied embedding, vocabulary size 248320 | Reuse the embedding weights for the output projection |

This 4B configuration is a dense model. Both mixers sit inside the same pre-norm, residual, and `2560 → 9216 → 2560` SwiGLU structure. Most integration changes concern the mixers and their history; the existing Linear, MLP, and framework interfaces can remain in use.

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="Qwen3.5 ordinary-inference backbone and request state">
<img src="/images/my-sglang/qwen35-report/en/blocks.svg" alt="The GDN and Full Attention backbone of Qwen3.5, with packed projections followed by per-request KV, convolution windows, and recurrent state." loading="lazy" />
</div>
<figcaption>Figure 1: The text backbone integrated in this work. Batched matrix projections and per-request history have an explicit boundary in the implementation. <a href="/images/my-sglang/qwen35-report/en/blocks.svg" target="_blank" rel="noopener">View full image ↗</a></figcaption>
</figure>

Next, we inspected the safetensors keys, shapes, and dtypes. Text weights live under `model.language_model.*`, which the loader maps to the local `model.*`; `model.visual.*` is skipped because it falls outside the text scope. We initially restricted this hybrid branch to TP=1, avoiding the additional QKV-merging and sharding assumptions of the existing loader during the first integration.

Finally, we traced four details through the reference forward: how Q and the output gate are arranged, whether Q/K norm comes before or after RoPE, whether the convolution cache stores inputs or outputs, and whether GDN reads its result from the state before or after the update. We also recorded floating-point conversions. Two implementations can express the same formula yet produce different outputs if intermediate BF16 rounding happens at different points.

Back in mini's code, the entry points for the changes were clear: `ModelConfig` passes through the hybrid-layer and GDN parameters, the registry locates the local model, `BaseOP` performs strict weight loading, and the Engine supplies the new tokens for the current iteration. We did not introduce a separate execution framework.

The experiments used a separate worktree and branch from upstream `20fcd7f`, plus a dedicated venv, preserving the existing serving environment. The actual stack was PyTorch 2.13.0+cu130, Triton 3.7.1, and Transformers 5.12.1, with model revision pinned to `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. These versions and the weight hashes establish the starting point for subsequent comparisons.

## 2. Start with a general GPU implementation that is easy to inspect

The first version did not rush into hardware specialization. `qwen35_reference.py` expresses causal conv, GDN recurrence, and gated RMSNorm through PyTorch tensor operations. With inputs on CUDA, both the state and computation remain on CUDA. This general path can be checked line by line against the reference formulas, giving later fusion work a clear comparison target.

### GDN: get the recurrence and state layout right first

GDN receives four projections: qkv has shape `[N,8192]`, z is `[N,4096]`, and a/b are each `[N,32]`, where N is the token count for this call. qkv passes through a width-4 depthwise causal conv and SiLU, then splits into Q, K, and V. The other three projections do not pass through the convolution.

The convolution state stores the four most recent **inputs before convolution**. When the next chunk arrives, we concatenate the historical window with the new inputs, take the convolution outputs corresponding to the current chunk, and retain the last four raw inputs. Caching the activated outputs might look fine for the first chunk, but it changes the computation as soon as the second chunk arrives.

Each of the 16 Q/K heads is repeated twice to match the 32 V heads. The recurrent state uses layout `[B,H,K,V]`, with a `128×128` FP32 matrix per head. Q/K undergo L2 normalization, and Q is additionally scaled by `128^(-1/2)`. The two gates are:

```text
β_t = sigmoid(b_t)
g_t = -exp(A_log) · softplus(a_t + dt_bias)
```

Following token order, the recurrence can be written directly as:

```python
state = state * g_t.exp()[..., None, None]
memory = (state * k_t.unsqueeze(-1)).sum(-2)
delta = (v_t - memory) * beta_t.unsqueeze(-1)
state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
output = (state * q_t.unsqueeze(-1)).sum(-2)
```

Three ordering decisions matter here: decay the history before reading memory; correct the state using the memory error for the current key; and have the query read the **updated** matrix. The recurrence and state remain in FP32, while the output returns to the input dtype. Q/K normalization follows the reference implementation's rounding order rather than being casually rewritten to use FP32 throughout.

The operators explicitly accept old state and return new state without overwriting the caller's tensors. Full-input and chunked-input execution can then start from the same initial state, and one test cannot quietly alter state that a later comparison depends on.

### Full Attention: correct shapes still require the correct layout within each head

The Q projection produces both query and gate. The correct split first reshapes by head, then divides each head into two halves:

```python
qg = q_proj(x).view(N, 16, 512)
q, gate = qg.chunk(2, dim=-1)  # Each is [N,16,256]
```

Splitting the 8192 dimensions directly into two halves can still yield valid final shapes while assigning the wrong contents to each head. Q/K are then normalized separately, and only the first 64 dimensions are rotated. Stored KV retains four heads, expanded during computation to match the 16 Q heads. We initially call PyTorch SDPA; its output is multiplied by `sigmoid(gate)` and projected back to 2560 dimensions.

Norm implementations also cannot be reused indiscriminately. The backbone's general RMSNorm multiplies by `1 + weight`, while GDN gated norm multiplies directly by `weight` and converts the normalized result back to the input dtype before multiplying by the weight:

```python
x = hidden.float()
x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
x = weight * x.to(hidden.dtype)
out = (x * F.silu(gate.float())).to(hidden.dtype)
```

These are the conversion boundaries that later fusion must preserve. Keeping every intermediate in FP32 and converting to BF16 only at the end creates a different numerical path.

The first GPU test run for the general operators passed 30/30 checks. Independent HF comparisons at the actual GDN dimensions covered B=1/4, T=1/16, and FP32/BF16, for eight configurations; maximum absolute errors were 0 for both output and final state. During development, CPU BF16 `rsqrt` showed a 1 ULP difference between full-input and chunked shapes. The tests therefore isolated normalization from recurrence continuity when diagnosing that issue, while the GPU checks retained the complete path.

## 3. Day 0: load the backbone, carry state forward, and run ordinary generation

Here, Day 0 means the minimal working loop for ordinary text inference: real backbone weights load strictly, decode can continue after prefill, and a limited set of greedy cases matches an independent reference. It names a milestone, not a claim that all the work took one day.

The model uses `BaseOP` to construct `_TextModel`, `_DecoderLayer`, `_Attention`, `_GatedDeltaNet`, and `_MLP`. Parameter shells are first created on the meta device and then loaded with real tensors, avoiding initialization of a complete GPU model beforehand. The ordinary backbone requires 426 keys: 14 for each of the 24 GDN decoders, 11 for each of the 8 Attention decoders, plus the embedding and final norm.

The first strict load exposed a problem: `A_log` and `linear_attn.norm.weight` are FP32 in the checkpoint, but their initial declarations had followed the default dtype and become BF16. The fix preserved the actual precision of these two parameter types and kept the loader's name, shape, and dtype checks, rather than casting everything uniformly to bypass the failure.

Offline generation started with a simple loop:

```python
state = Qwen35State()
out = model.forward_tokens(prompt_ids, state=state)

for step in range(output_tokens):
    token = out.logits[-1].argmax().view(1)
    generated.append(token)
    if step + 1 < output_tokens:
        out = model.forward_tokens(token, state=state)
```

The distinction between “emitted” and “consumed” matters. When the first token is selected from the final prompt logits, it has not yet entered the model state; the next forward pass consumes it. `state.length` tracks consumption and must not be incremented early just because SSE has emitted a token.

### Keep three kinds of history in one state

| State | Shape per layer | Update |
| --- | --- | --- |
| K, V | Each `[L,4,256]`, BF16; 8 layers | Append as consumed length L grows |
| causal conv | `[1,8192,4]`, BF16; 24 layers | Retain a rolling input window |
| GDN recurrent | `[1,32,128,128]`, FP32; 24 layers | Advance once per consumed token |

From these shapes, the basic backbone state occupies approximately `49.5 MiB + L×32 KiB`, excluding temporary tensors, Graph staging, and allocator overhead. Even a short request needs fixed-size GDN state, so capacity management cannot count only KV tokens.

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="Qwen3.5 state continuation across chunks">
<img src="/images/my-sglang/qwen35-report/en/state-continuation.svg" alt="Prefill chunks and Decode share request state: KV grows with length, the convolution window rolls forward, and the GDN matrix continues its recurrence." loading="lazy" />
</div>
<figcaption>Figure 2: All three states must continue together. The token blocks depict logical positions; this hybrid implementation uses request-private dynamic KV tensors. <a href="/images/my-sglang/qwen35-report/en/state-continuation.svg" target="_blank" rel="noopener">View full image ↗</a></figcaption>
</figure>

With P historical tokens and a current chunk of length T, query i should see keys satisfying `j ≤ P+i`. Continuing with multiple tokens requires explicitly handling this position offset, rather than directly applying a T×T causal mask that starts from zero. Convolution and GDN load their window and matrix respectively, then continue from the previous chunk.

Once connected to the Engine, `forward_packed` flattens request tokens and preserves boundaries in `lengths`. Linear and MLP run across the packed batch; Attention and GDN apply state per request. The model checks ownership using `(table_idx, uid)` and verifies `state.length == req.cached_len`. Completion or cancellation releases all three state types, with the necessary stream dependencies established before slot reuse.

This division also sets the boundary between the model and scheduler: the scheduler decides how many tokens to consume in the current iteration, while the model applies those tokens to the corresponding history. When a request is temporarily unscheduled, all three states stay where they are; the next iteration processes only the new segment. When a new request reuses the same table slot, the UID check prevents it from reading the previous request's matrix. Checking length and ownership here is much easier than trying to infer cross-request contamination from text after many tokens have been generated.

The current hybrid KV path is not yet connected to the ordinary models' paged Attention backend. It is restricted to TP=1 and page_size=1, with hybrid Prefix Cache disabled. KV alone, without its corresponding convolution/GDN state, cannot restore a valid prefix.

The first ordinary generation run used a 32-token input and a 16-token output, reproduced identically across three rounds, and matched HF's first 16 greedy tokens for one fixed chat example. Although the initial script recorded a chunked comparison, chunk size=64 meant that no chunk boundary was crossed. A later test used a 512-token input split into eight 64-token chunks, providing an actual cross-chunk comparison for the first 16 output tokens. We retain this correction because a test's name does not prove that it exercised the intended scenario.

## 4. Open Perfetto and separate GPU computation from CPU launches

With the reference path working, we added `record_function` ranges to each layer, token mixer, and MLP, collected CPU/CUDA traces with `torch.profiler`, and expanded an ordinary decode step in Perfetto.

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="The first layer of the general model implementation in Perfetto">
<img src="/images/my-sglang/qwen35-report/perfetto-baseline.png" alt="Perfetto with qwen35.layer.0 selected, showing the token mixer and dense sequences of small operator calls." loading="lazy" />
</div>
<figcaption>Figure 3: An actual Perfetto page screenshot. The trace was collected on 2026-09-23 and reopened on 2026-09-25. The selected qwen35.layer.0 is a CPU user_annotation; its 997.738 μs includes host execution and launches, not just GDN GPU execution. Only Perfetto page content is retained in the screenshot. <a href="/images/my-sglang/qwen35-report/perfetto-baseline.png" target="_blank" rel="noopener">View full image ↗</a></figcaption>
</figure>

The first single-step decode trace contained 2777 GPU kernels with approximately 9.172 ms of cumulative execution. Two main GEMV families accounted for about 5.892 ms, or 64.2% of that total. Meanwhile, dtype conversions, reductions, and elementwise operations produced many short calls.

When reading this kind of timeline, I first select a layer, follow its CPU range to the GPU work it launches, and then narrow down to an individual kernel. The overview shows call density and gaps; an individual event provides the name, duration, and launch parameters; aggregate statistics show the total contribution of a call family. Using all three views helps avoid mistaking one unusually slow event for a hotspot, or reporting a CPU range's duration as kernel time.

This suggested two directions: matrix projections deserved further attention, and small-operator launch overhead was worth reducing. In particular, reference GDN prefill loops over tokens in Python, so its long-input cost is not fully visible in a single decode-step view.

We chose gated RMSNorm first. It has no state dependency across tokens, and its reduction, weight, and SiLU boundaries are clear. It was a useful place to verify the replacement and measurement process before addressing the more complex GDN recurrence.

Performance measurements and profiling ran separately. Profilers add recording overhead; CUDA events surrounding a sequence of Python launches can also include GPU idle time caused by the host not issuing work quickly enough. The model A/B runs below used identical weights and inputs, warmed up with the same shapes, and disabled profiling during the three measured rounds. Complete offline generation includes prefill, 127 subsequent decode calls, and token selection; it excludes loading, HTTP, queuing, and networking.

## 5. The first fusion: a faster operator without a faster model

`qwen35_fused.py` uses one Triton program per gated-norm row, combining mean square, rsqrt, weight multiplication, and the SiLU gate. The implementation explicitly preserves the reference path's dtype conversions and disables floating-point fusion that would alter the corresponding rounding order. GDN remained unchanged; only `MINISGL_QWEN35_GATED_NORM` was switched.

All 12 GPU tests passed. In the B1/T1 microbenchmark, the median CUDA-event interval fell from 54.606 μs to 9.918 μs, about a 5.5-fold improvement; the trace went from 12 kernels to 1.

Back in the model at B1, with a 512-token input and a 128-token output, the result was:

| Elapsed time, ms | Reference, mean ± standard deviation across three rounds | Fused gated norm |
| --- | ---: | ---: |
| Prefill | 516.042 ± 0.651 | 521.302 ± 0.659 |
| Decode | 1769.918 ± 4.760 | 1796.328 ± 14.073 |
| Complete offline generation | 2285.960 ± 5.274 | 2317.630 ± 14.565 |

The 128 output tokens matched, and the model trace contained 24 calls to the fused kernel, confirming that the replacement was active. Yet complete generation time increased by **1.385%**, providing no model-level benefit.

These groups ran in separate processes without interleaved A/B execution, so the roughly 1.4% regression cannot establish a specific hardware cause. We retained both the implementation and the negative result, leaving reference as the default. Removing a dozen short calls locally may not materially change the model's main costs; the next step still required a new, independent comparison.

## 6. The second fusion: advance GDN through time inside the kernel

For each token, reference GDN launches several multiplications, reductions, and additions, repeatedly materializing intermediate state as tensors. This loop itself was a more promising target.

The general Triton implementation in `qwen35_gdn.py` fixes the actual dimensions at K=V=128, initially using `value_tile=32, num_warps=4`. One program owns a batch/head/value tile and holds a `128×32` FP32 state block inside the kernel, executing decay, memory reduction, delta update, and query readout in token order.

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="Implementation of the two independent fusions">
<img src="/images/my-sglang/qwen35-report/en/fusion-process.svg" alt="Gated norm combines elementwise operations and reductions while preserving BF16 rounding boundaries; GDN tiles along value columns, retains the full key axis, and advances through time within one program." loading="lazy" />
</div>
<figcaption>Figure 4: Fusion scope. Projections, causal convolution, and output projection remain outside the GDN recurrence kernel; box widths do not represent measured durations. <a href="/images/my-sglang/qwen35-report/en/fusion-process.svg" target="_blank" rel="noopener">View full image ↗</a></figcaption>
</figure>

We tile the value axis because each column needs a reduction over the full key axis, while different value columns can be updated independently. Parallelism comes from batch, head, and value tiles; the time axis still follows the recurrence dependencies. Initial state is read-only, and final state is allocated separately. Q/K normalization retains intermediate BF16 conversions. This version uses no SM120-specific instructions, and other shapes fall back to reference.

The 23 GPU checks covered B=1/4/8, T=1/16/65, two dtypes, nonzero initial state, and chunking. Maximum absolute differences were approximately `1.22e-4` for BF16 output and `1.79e-7` for state, both within the predefined tolerances. We then switched only `MINISGL_QWEN35_GDN` and remeasured reference; gated-norm fusion remained disabled.

| Elapsed time, ms | Reference, mean ± standard deviation across three rounds | GDN Triton | Reduction |
| --- | ---: | ---: | ---: |
| Prefill | 512.128 ± 0.858 | 42.210 ± 0.039 | 91.758% |
| Decode | 1761.840 ± 4.195 | 1540.485 ± 22.505 | 12.564% |
| Complete offline generation | 2273.968 ± 5.009 | 1582.695 ± 22.496 | **30.399%** |

The two paths produced identical 128-token outputs across three rounds and matched on the first 16 tokens of the cross-chunk comparison. The fused path also underwent a 16-token comparison against HF on a fixed chat example. Its first round was slower than the next two and remained included in the mean and standard deviation.

The large prefill reduction addresses a specific cost: the general PyTorch path's per-token, repeated launches. The result applies to the measured ordinary single-request 512/128 workload. Its comparison target is our reference implementation, not a mature parallel GDN backend.

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="CPU and GPU timeline for ordinary Decode after GDN fusion">
<img src="/images/my-sglang/qwen35-report/perfetto-decode.png" alt="Perfetto shows CPU calls for native_decode_step above the GPU stream; projections and auxiliary operators remain after fusion." loading="lazy" />
</div>
<figcaption>Figure 5: An actual Perfetto timeline retained from 2026-09-23, with CPU ranges above and GPU streams below. Reference in the same experimental stage had 2785 kernels, versus 2185 after fusion. The 30.399% reduction comes from model measurements without profiling, not from the screenshot's width. <a href="/images/my-sglang/qwen35-report/perfetto-decode.png" target="_blank" rel="noopener">View full image ↗</a></figcaption>
</figure>

Fusion removed 600 kernels and reduced cumulative GPU execution from 9.257 ms to 8.451 ms. Gaps between CPU launches and GPU execution remained, as did the remaining Linear, Attention, norm, and state operations. We continued in two directions: use Graph to study submission overhead, and use parameter sweeps to study the GDN kernel itself.

## 7. CUDA Graph and SM120 tuning: examining the remaining overhead

CUDA Graph requires fixed addresses. We added private KV/GDN staging for ordinary decode: load request state, replay, then commit state back to the request. Timing includes these copies rather than measuring only the short replay call.

The first issue, however, was numerical. Attention with a dynamic valid KV length and Attention with a fixed capacity plus a mask may select different computation paths. Diagnostics showed that static and Graph agreed; the difference already appeared between eager and static. After switching to math SDPA, checks for logits, hidden, KV, conv, and recurrent state agreed over three steps at B1, prefix16, capacity256. B4/B8 and larger capacities did not all pass.

Within the passing scope, the complete forward wrapper fell from `16.176 ± 0.029 ms` to `10.807 ± 0.001 ms`, a 33.192% reduction. Both sides of this A/B used reference GDN, reference gated norm, and math SDPA, so this percentage cannot be added to the previous section's 30.399%. The wrapper also excludes the external test harness's state cloning and token selection, and is not an HTTP latency measurement.

At capacity8192, the maximum absolute logit difference reached 0.1171875, exceeding the predefined 0.05. With Graph disabled, math SDPA was also about 5.44% slower than auto for 512/128 generation. Graph therefore remains disabled by default, with only a short-context experimental entry point retained.

Only then did we formulate a hypothesis about the launch size on the 5090.

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="A selected GDN GPU kernel in Perfetto">
<img src="/images/my-sglang/qwen35-report/perfetto-kernel.png" alt="Perfetto search finds 24 recurrent-kernel calls and shows the selected event's duration and launch association." loading="lazy" />
</div>
<figcaption>Figure 6: An actual GPU event, captured on 2026-09-25 by reopening the trace recorded on 2026-09-23. Search finds 24 recurrence calls, corresponding to the 24 GDN layers. The selected event lasts 3.104 μs, and the interval from its associated CPU launch to GPU start is 2.061 μs. These are two different metrics for one event, not operator averages. <a href="/images/my-sglang/qwen35-report/perfetto-kernel.png" target="_blank" rel="noopener">View full image ↗</a></figcaption>
</figure>

The event's grid is `(32,4,1)`, or 128 CTAs; the 5090 reports 170 SMs. Reducing the value tile can increase CTA count, but also repeats more Q/K reads and normalization. We therefore swept B=1/4/8, T=1/16/128, tile=16/32/64, and warps=4/8, for 54 configurations. Each passed through numerical checks before separate measurements of the Python wrapper and the GPU interval for repeated calls inside a Graph.

All numerical checks passed. For B1/T1, tile16 increased the CTA count to 256 and reduced the GPU interval from about 2.436 μs to 2.308 μs; the wrapper increased from about 13.37 μs to 13.84 μs. Of the nine input shapes, seven still had the lowest time with the general tile32/warp4 configuration. Larger tiles increased register pressure, and more warps offered no consistent advantage.

We therefore added no default SM120-specific dispatch in this round. The model currently calls recurrence per request, so a B8 microbenchmark also cannot be equated directly with serving concurrency 8.

We also tried FlashInfer's SM120 GDN backend. The trace confirmed that the corresponding kernels actually ran, but final state failed this project's strict tolerance for all nine shapes, requiring further checks of normalization, state layout, and computation contracts. Timing in this stage also overlapped with another GPU service, so the performance numbers were invalidated. It did not establish that our implementation outperformed a mature backend.

## 8. Larger batches brought the work back to correctness

After small examples passed, more realistic batching exposed greedy differences within the ordinary path itself. The same request did not always produce identical output when run alone and inside a batch. Before comparing speed further, we needed to locate the earliest difference.

`batch_numerics.py` uses teacher forcing: B1/B4/B8 consume the same fixed token history, and a different prediction is not fed back into the next step. It records activation differences layer by layer and checks byte hashes of all weights before and after execution. This separates operator differences from the effects of token histories that have already diverged.

The earliest diagnostic round pointed to `in_proj_qkv` in the GDN of layer 0. Disabling BF16 reduced-precision reduction moved the first prefill difference to the MLP down projection, while decode still differed at QKV. Changing that precision option alone did not solve the issue.

We therefore added an optional fixed-reduction Linear: fixed `16×64×32` tiles, BF16 inputs, FP32 accumulation, and a fixed K reduction order, without split-K or autotune, so changes in M primarily affect row masks. Operator tests passed for seven real matrix shapes, with the tested identical rows matching bit for bit at different batch positions.

After the device recovered on September 25, we first reran four kernel test files, passing 77 checks, and then examined the full model. In the fixed-Linear diagnostic for one request, argmax matched across 102 position/configuration combinations, but some internal activations still differed, with the earliest differences moving to gated norm or ordinary norm. That covered only one prompt and a fixed continuation; generation still needed to run freely afterward.

We used nine fixed SPEED requests, each producing 32 tokens while ignoring EOS, and ran ordinary generation at planned B=1/4/8. Each backend used its own ordinary B1 output as the reference; B1's 9/9 is therefore a self-comparison, not an independent correctness check. A smaller final batch was retained when the request count did not fill the planned size. Changing only gated norm gave:

| Planned batch | Fixed Linear + reference norm | Fixed Linear + Triton norm |
| --- | ---: | ---: |
| 1 | 9/9 | 9/9 |
| 4 | 8/9 | 9/9 |
| 8 | 8/9 | 8/9 |

The table counts requests whose entire 32-token output matched exactly. With reference norm, case3 diverged at output token 17 for B4/B8, changing token ID from 2014 to 15771. With Triton norm, B4 passed but B8 still failed; moreover, the two norms' own B1 baselines matched on only 8/9 requests. It changed some model outputs and cannot be counted as a fix for batch consistency.

This points the next experiment at a more specific location: for an actual failing request, with the same batch members and row position, save norm inputs and outputs before the first divergence, along with the final logits' top1/top2 margin. We still need to determine how activation changes within tolerance cross an argmax boundary, rather than continuing to loosen the comparison threshold.

At this point, ordinary inference integration, explicit hybrid state, and GDN fusion on a limited workload all have concrete results. Broader batch consistency has not passed. Fixed Linear and Graph remain experimental options; passing individual tests has not automatically made them the default path.

### Code and reproduction entry points

The implementation is concentrated in a few files, which can be read in the order of this post:

| Topic | Repository entry point |
| --- | --- |
| Configuration, native model, and weight mapping | `python/minisgl/models/{config,qwen3_5,weight}.py` |
| General operators, norm fusion, and GDN fusion | `python/minisgl/kernel/qwen35_{reference,fused,gdn}.py` |
| Graph staging and fixed Linear | `python/minisgl/engine/qwen35_graph.py`, `python/minisgl/kernel/qwen35_linear.py` |
| Model comparisons, layerwise diagnostics, and parameter sweeps | `benchmarks/qwen35/{model_baseline,batch_numerics,tune_gdn}.py` |
| Chinese experiment notes and raw records for each round | `docs/experiments/qwen35-sm120/` |

The general operators, model integration, GDN fusion, and fixed Linear can be traced from `4f501ec`, `0e3bbf4`, `ce8acc4`, and `89316b3`, respectively; post-recovery results are archived in `97a0ebc`. Some early measurements ran with uncommitted working-tree changes, so reproduction also needs the results' `source_sha256`, rather than relying only on the base HEAD. The [public A/B extract](/data/my-sglang/qwen35-ordinary-experiments.json) retains per-round samples and source hashes. The [screenshot source manifest](/data/my-sglang/qwen35-perfetto-screenshots.json) records hashes for the three page captures and their original traces.

In a dedicated CUDA venv prepared according to the recorded versions, start by rerunning the ordinary reference path:

```bash
source /path/to/qwen35-venv/bin/activate
export PYTHONPATH="$PWD/python"
export MINISGL_QWEN35_GDN=reference
export MINISGL_QWEN35_GATED_NORM=reference
export MINISGL_QWEN35_LINEAR=reference
export MINISGL_QWEN35_SDPA=auto
export MINISGL_QWEN35_GRAPH=0

python benchmarks/qwen35/model_baseline.py \
  --model /path/to/pinned-Qwen3.5-4B \
  --input-tokens 512 --output-tokens 128 --rounds 3 \
  --check-hf --profile --output artifacts/qwen35-reference-rerun
```

To test GDN, change only `MINISGL_QWEN35_GDN` to `triton` and save to a different new directory. Check tokens and state first, then compare the three rounds timed without profiling; use the trace to explain where time went. The next round still focuses on numerical differences in ordinary batching. Once resolved, we can extend input lengths, scheduling combinations, and the range of online performance tests.
