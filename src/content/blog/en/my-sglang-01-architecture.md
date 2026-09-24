---
title: "My_Sglang (I): Module Boundaries, the Execution Loop, and State Management"
description: "Explore My_Sglang through module contracts and state ownership: the scheduling loop, Paged KV and Radix Cache, CUDA execution dependencies, and how MTP extends generation and state commit."
date: 2026-09-23T17:50:00Z
lang: en
translationKey: my-sglang-01-architecture
tags: [LLM Inference, System Architecture, My_Sglang]
draft: false
---

The architecture of an inference engine ultimately has to answer three interdependent questions: **which tokens run in this iteration, which states execution depends on, and when the results can become inputs to the next iteration.** Continuous batching changes the first; KV Cache management constrains the second; CUDA Graph, overlap, and speculative decoding keep reshaping the boundary of the third.

My_Sglang builds on mini-sglang, retaining its serving, scheduling, execution, and cache-management structure while exploring model adaptation, mixed scheduling, native MTP, and operator optimization. This article uses commit `566bf693` as its reference point to connect these modules. The focus is on data structures, resource ownership, and execution order; model adaptation and performance experiments are left to later chapters.

## 1. System Boundaries: Serving Protocols, Scheduling Decisions, and Device Execution

My_Sglang retains mini-sglang's process organization. The HTTP frontend handles connections, request parameters, and responses; the tokenizer/detokenizer converts between text and token sequences; the scheduler maintains the waiting and running sets and owns an Engine within its process. The Engine, model, and sampler are objects in that same process, while the GPU provides the computation and tensor storage they submit work to.

The number of processes varies with the tokenizer and TP configurations. The diagram below uses a single GPU and a shared tokenizer/detokenizer worker process to illustrate the module relationships; it does not prescribe that deployment topology as the project's only architecture.

<my-sglang-explorer view="architecture">
<p>The HTTP frontend and tokenizer/detokenizer communicate with the scheduler through messages. Within its process, the scheduler calls the Engine, model, and sampler; the Engine drives the GPU. Request slots, the page table, cache indexes, and physical tensors are managed by their respective modules, and results return to the frontend through the detokenizer.</p>
</my-sglang-explorer>

Different information crosses each boundary. Interprocess messages carry UIDs, token IDs, sampling parameters, termination markers, and counts; the scheduler and Engine exchange a `Batch`, sampling arguments, and mappings; the model and Attention backend share metadata and device state for the current execution. Full hidden states and logits stay on the execution side rather than making a round trip through the HTTP message channel.

| Module | State and decisions it owns | Contract provided to the next layer |
| --- | --- | --- |
| Serving and tokenization | Connections, text encoding/decoding, request UIDs | Token requests and cancellation messages; receives results and finish reasons |
| Scheduler | Pending/running sets, admission, batch composition | This iteration's requests, input intervals, positions, and writeback locations |
| Resource management | Request slots, free pages, prefix indexes, and locking relationships | Available capacity and mappings from logical to physical positions |
| Engine | Model, backend, sampler, CUDA stream, Graph runner | GPU tokens, host token copies, and a completion event |
| Model and kernels | Weights, layer computation, model-specific state | Logits; advances state over the agreed input prefix |

HTTP, continuous batching, Chunked Prefill, Paged KV, Radix Cache, ordinary decode Graph, and overlap come from the upstream foundation. This branch mainly works on model-specific state, native MTP, mixed-scheduling adaptation, lifecycle handling, and switchable operator paths. This distinction also determines attribution in the sections that follow: integrating an existing mechanism and adding a new one should be discussed separately.

## 2. Request Identity and Three Lengths: The Execution Loop's Data Model

`Req`, `Batch`, and `Context` in `core.py` form the runtime's common language. A `Req` survives across iterations; a `Batch` describes one scheduling decision; a `Context` exposes the current batch, page table, and backend during a forward pass and clears the active batch when the pass exits. Context is an execution context, not a container for every request's history.

First, distinguish three kinds of indexes. `uid` identifies the request in the message channel; `table_idx` is a reusable request slot used to index the token pool and page table; the values in the page table are the physical KV token-slot addresses. A `table_idx` can be reused after a request finishes, but that must not allow the new request to inherit the old UID's model state or late results.

Second, the lengths in `Req` describe execution progress, rather than three copies of the same token count:

| Field | Execution semantics |
| --- | --- |
| `cached_len` | Length of the prefix with existing computed state before this iteration's input |
| `device_len` | Logical end of the current device input sequence; the difference between the two lengths is `extend_len` |
| `max_device_len` | Input length at the construction of this `Req` plus `output_len`, bounding how far it can advance; intermediate chunks use a temporary `Req` |

Ordinary decode usually starts with `device_len = cached_len + 1`: the last token has been generated but is still waiting to be consumed by the model. Prefill can instead have an `extend_len` greater than one. Once a forward pass is submitted, `complete_one()` advances `cached_len` to the old `device_len` and increments `device_len` by one to reserve a position for the new token.

**These are logical updates on the CPU; they do not mean that GPU execution has completed.** Sampling results are first written to the device token pool, while the host `input_ids` are appended only after the asynchronous copy finishes. With overlap enabled, the next iteration can obtain its inputs from the device token pool while the previous iteration's host records are still being processed. This deliberate difference in progress is what allows scheduling to hide some CPU overhead.

A `Batch` flattens the new intervals from different requests and carries `positions`, `out_loc`, and backend metadata. Flattening lets them share one forward pass, but lengths and mappings still preserve request boundaries; batched computation and state isolation must both hold.

## 3. The Scheduling Loop: Prepare, Submit, Reconcile

To explain the dependencies, one execution iteration can be divided into preparation, submission, and result processing. The labels below describe the flow; they are not three newly introduced interface layers in the code.

**Prepare** selects runnable requests, allocates the necessary resources, and constructs an execution description. The scheduler selects a Batch from the pending and running sets, allocates new pages through CacheManager, generates positions and input/writeback mappings, and then asks the Attention backend to prepare metadata. The input mapping extracts this iteration's tokens from the device token pool, while `out_loc` tells cache-write operations where to put the new K/V.

**Submit** enqueues model execution or Graph replay on the Engine stream, then samples to obtain device tokens and asynchronously copied host counterparts. `ForwardOutput` also returns a completion event so that the caller does not mistake “I have a Python object” for “the data is ready to read.” After device token writeback, unfinished requests remain in the decode set.

**Reconcile** waits for the output copy, updates host sequences, handles EOS, length-based termination, and resource release, then sends results to the detokenizer. Intermediate prefill chunks do not publish generated results to the user; a request enters the usual generation loop only after its final input chunk completes.

<my-sglang-explorer view="flow">
<p>Prepare: admit and select requests, allocate pages, and construct a Batch. Submit: enqueue model computation, sampling, and device token writeback. Reconcile: wait for the copy event, publish output, determine completion, and reclaim resources. Unfinished requests reenter the next iteration, while cancelled requests exit through the same lifecycle.</p>
</my-sglang-explorer>

### A Batch Is Determined by Budgets, Not a Fixed Request List

The key to continuous batching is rebuilding the Batch each iteration. Completed requests leave, and new requests enter when they meet admission conditions; prefill requests can advance across iterations, and decode requests can execute alongside different neighbors.

Admission currently checks both request slots and cache capacity. `PrefillAdder` estimates the uncached input plus the output budget and includes reservations for running decode requests; it checks available capacity again after locking the matched prefix. This second check matters because locking converts previously evictable cache into protected capacity, potentially invalidating the first estimate.

Chunked Prefill limits each input-processing step to a token budget. An unfinished chunk retains its original request slot and cache handle as a `ChunkedReq`, returns to the front of the pending set, and continues from the updated `cached_len` in the next iteration. This bounds the new computation per iteration; capacity admission must still account for the request's later resource needs.

The branch also offers `prefill_first`, `decode_first`, and `mixed`, with `prefill_first` remaining the default. The mixed path schedules decode first and uses the remaining token budget for prefill. If input is waiting, it reserves at least one token for it; if decode exceeds the budget, requests rotate by UID; if prefill cannot currently be admitted, the reserved budget is reclaimed. The goal is to give both kinds of work opportunities to progress, but reserving budget does not replace starvation checks under load: capacity constraints and actual service latency still require separate validation.

The core `Batch.phase` still contains only prefill/decode. A mixed batch describes its internal composition through request intervals and `batch_mix`, then enters a model path that can handle variable-length segments. Adding a scheduling policy therefore requires more than changing queue-selection order: the model and backend must also understand that execution description.

## 4. KV Cache: Who Owns Storage, Mappings, and Reuse?

The ordinary Attention path separates KV management into three layers: `MHAKVCache` owns the physical tensors, the page table maps request-logical positions to device storage positions, and Radix Cache tracks reusable token prefixes. They work together but have different lifecycles.

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="KV cache mapping diagram; scroll horizontally on narrow screens">
<img src="/images/my-sglang/en/kv-ownership.svg" alt="KV ownership in the ordinary Attention path: request slots index the page table, and the Radix prefix index reuses physical token slots; locking and eviction control page reclamation." loading="lazy" />
</div>
<figcaption>Figure 1: Mappings and ownership in the ordinary Attention path. Arrows indicate references or reclamation relationships; addresses and requests are examples, not runtime measurements. The hybrid model's separate state path is discussed at the end of this section. <a href="/images/my-sglang/en/kv-ownership.svg" target="_blank" rel="noopener">View the full SVG</a>; scroll horizontally on narrow screens.</figcaption>
</figure>

### The Physical Pool and Page Table

Engine initializes the physical pool, whose layout includes K/V, layers, pages, tokens within a page, KV heads, and head dimension. CacheManager allocates pages, then expands them into token slots to write into the page table. Accordingly, `page_table[table_idx, position]` in the code stores a physical token position, not an unexpanded page ID.

After the model computes this iteration's K/V, `store_kv` writes them into the physical pool according to `out_loc`, and the Attention backend reads the existing cache using each request's history length and mappings. Model layers do not need to allocate a contiguous region of device memory for the entire history, and the scheduler does not need to manipulate the actual K/V values in each layer.

### The Radix Index and Reference Protection

Radix nodes store token prefixes and their physical indexes; **they do not store another full copy of K/V**. After a prefix hit, the new request's page table can reference existing slots. A node's `ref_count` is incremented or decremented along its ancestor path: paths held by active requests are protected and become evictable only when their counts reach zero. Matching uses at most the first `N−1` input tokens, leaving at least one new token to compute output logits.

When free pages are insufficient, CacheManager requests eviction from the prefix cache. Radix starts with unlocked leaf nodes, selects candidates by timestamp, and returns the physical locations they reference. Admission capacity therefore includes both free and evictable capacity; looking only at the unallocated free list is insufficient.

A request finishing does not mean that all the KV it computed is immediately destroyed. `cache_req` inserts cacheable full-page prefixes into the index, removes the old handle's protection, and frees this request's allocations for duplicate prefixes as well as tails that cannot be retained; the remaining prefixes can wait for future hits. Reclaiming a request slot, unlocking a prefix, and reclaiming physical pages are three different actions. Conflating them can cause premature release or device-memory leaks.

### Model-Specific State Is a Separate Storage Path

The Qwen3.5 adaptation illustrates the limits of these abstractions: a hybrid model owns KV, convolution windows, and recurrent state together. The current reference implementation sets the Engine's `kv_cache=None`, associates independent state with `(table_idx, uid)` in the model, and appends actual KV through dynamic tensors; the scheduling layer still uses the page table and page budget for capacity accounting. This path has not been connected to the ordinary path's physical KV pool.

It also forces the use of a naive prefix cache. Restorable state must cover all the history the model needs to continue execution; matching KV prefixes alone cannot justify cache reuse. Unifying resource management later requires contracts for allocating, restoring, and releasing the complete state. Merely placing everything in a class called CacheManager does not achieve that unification.

## 5. Boundaries Between Engine, Backends, and Asynchronous Execution

Engine combines model computation, sampling, a CUDA stream, and Graph management into a schedulable forward pass. The Attention backend converts a generic Batch into backend-specific metadata and provides computation and Graph-coordination interfaces; the model defines the network structure, and kernels implement individual operations. This lets an operator replacement remain localized to the execution side, but changes to memory layouts or shape constraints still require checking the contracts above it.

For ordinary decode, `GraphRunner` captures the model forward pass at preset batch sizes, using fixed input, position, write-location, and logits buffers. At runtime, the request count is padded up to a captured size, actual inputs are copied in, and replay metadata is prepared; dummy requests use a dedicated slot and dummy page. Model execution is what gets captured here. Sampling, queue selection, result handling, and the entire HTTP lifecycle remain outside the Graph.

Graph removes some repeated submission overhead, while overlap organizes dependencies using two streams and host processing delayed by one iteration. The scheduler stream prepares metadata; the Engine stream waits for that preparation and executes the current Batch; the CPU then processes the previous iteration's output, waiting for its copy event. The two mechanisms can be combined, but they address different overheads and do not imply that two model Batches execute concurrently on the GPU.

This organization makes reclamation part of the execution protocol: when the previous iteration's result signals completion, the next iteration's work may already be queued on the Engine stream. Before returning a request slot, the release path must establish a dependency on the completion of that work; otherwise, a new request could reuse a slot that old work will still write to. The code also uses completion markers to prevent late results from being published again and request-identity checks to prevent model-specific state from crossing requests. Cancellation messages reaching the scheduler must likewise handle in-flight execution and existing state, rather than merely closing the frontend connection.

## 6. MTP: Extending a Generation Step into a Transaction with a Commit Boundary

MTP's architectural impact extends beyond the prediction head itself. Each ordinary decode iteration consumes one pending token and produces a new token; the speculative path first generates several candidates, runs target verification, and then decides which prefix can be committed. The end of a model execution and the commit of a request's state therefore no longer naturally coincide.

The current branch divides the responsibilities between two locations: `MTPBatchHandler` in `scheduler/mtp.py` maps scheduled requests to MTP sessions, reserves the verification interval, and updates the token pool; `GreedyMTPController` in `engine/speculative.py` organizes candidates, verification, and state commit. “Transaction” here describes how state is isolated across the commit boundary. The implementation remains a model-specific experimental greedy path, rather than a speculative plugin abstracted for arbitrary models.

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="MTP state-commit diagram; scroll horizontally on narrow screens">
<img src="/images/my-sglang/en/mtp-transaction.svg" alt="MTP state commit: temporary candidate and verification states branch from committed state; target verification selects an accepted prefix, after which target state, MTP state, and new tokens are committed." loading="lazy" />
</div>
<figcaption>Figure 2: Committed state, temporary branches, and the commit boundary within one MTP iteration. Acceptance lengths and tokens illustrate the mechanism; they are not acceptance-rate or throughput measurements. <a href="/images/my-sglang/en/mtp-transaction.svg" target="_blank" rel="noopener">View the full SVG</a>; scroll horizontally on narrow screens.</figcaption>
</figure>

A session holds target state, MTP state, the last target hidden state, and a `pending` token. The `pending` token has already been emitted to the caller but has not yet been consumed by the target model; this is central to understanding the handoff between iterations. Candidate generation unfolds on a copy of the MTP state, while the target model verifies `[pending, candidates…]` on a copy of the target state. Different requests can have different acceptance lengths.

After verification, the controller accepts only the consecutively matching candidate prefix and appends the next token determined by the target model. For a rejected branch, the current implementation replays the input prefix that must actually be retained from the old target state; the appended token also comes from the logits of this actual committed path. MTP state is advanced again using target hidden states, so hidden states from the candidate rollout cannot be committed directly. This establishes explicit state-recovery semantics, while replay and copying overhead still require later optimization.

Only after all model operations, length checks, and allocations of tensors to be published have completed does the controller update the session references. The adapter then synchronizes request lengths, releases reserved locations that belong only to rejected suffixes, and writes the new pending token to the token pool. Requests in a batch can share a verification execution, but acceptance lengths and state commits still belong to each request individually.

The serving layer must consequently accept token lists rather than assume one token per iteration. EOS and output budgets apply to the sequence actually committed, and usage accumulates by token count; the number of SSE events cannot stand in for the number of output tokens. Resource release must also end both ordinary request state and the MTP session.

MTP decode currently bypasses ordinary `Engine.forward_batch` and Sampler and calls the controller's model path directly; combinations with mixed scheduling and Graph are explicitly rejected, and overlap is disabled. These restrictions reflect execution and state contracts that have not yet been unified. Support in the ordinary path does not imply support in the speculative path.

## 7. Extension Points and Current Boundaries

Adding a capability should begin by identifying the contract it changes. Model adaptation changes state and forward interfaces; scheduling optimization changes Batch membership and input intervals; backend optimization changes execution metadata and storage access; MTP changes how many results an execution can commit. Changes that cross boundaries must include the lifecycle, rather than merely verify that a local call succeeds.

| Path | Implementation scope described here | Capabilities that cannot be inferred automatically |
| --- | --- | --- |
| Ordinary Attention | Upstream physical KV pool, page table, Radix, ordinary decode Graph/overlap | Validation of every model and arbitrary backend combination |
| Hybrid-state adaptation | Independent KV/convolution/recurrent state per request; variable-length packed forward is implemented | Completed integration of actual paged KV or hybrid Prefix Cache |
| Native MTP | Greedy candidates, verification, batching, and transactional state commit | A general model plugin, random sampling, or combinations with Graph/overlap |

This article discusses architecture and where mechanisms are implemented; the presence of a module is not treated as a performance conclusion. Model support matrices, numerical regression, and optimization gains must be tied to specific versions and configurations. Their evidence will be covered in the corresponding articles.

When reading the source, these boundaries provide entry points without requiring you to trace the entire model line by line first:

| Question to investigate | Main entry points, relative to `python/minisgl/` |
| --- | --- |
| Processes, protocols, and messages | `server/launch.py`, `server/`, `tokenizer/`, `message/` |
| Request and batch data contracts | `core.py` |
| Admission, batch selection, and result processing | `scheduler/scheduler.py`, `scheduler/prefill.py`, `scheduler/decode.py` |
| Request slots, page allocation, and prefix reuse | `scheduler/table.py`, `scheduler/cache.py`, `kvcache/` |
| Forward execution, backends, Graph, and sampling | `engine/engine.py`, `engine/graph.py`, `engine/sample.py`, `attention/` |
| MTP scheduling integration and state commit | `scheduler/mtp.py`, `engine/speculative.py` |
| Network structure and operator implementations | `models/`, `layers/`, `kernel/` |

Later chapters will cover the request lifecycle, cache management, scheduling, model adaptation, MTP, and execution optimization individually. Each will locate its changes within the module boundaries established here and explain the data, dependencies, and validation conditions those changes affect.

## Further Reading

The diagrams in this article were redrawn from the project code to explain its mechanisms. Their organization draws on how systems papers distinguish logical mappings, physical storage, and state transitions; the implementation details follow the code version identified in this article.

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180): understanding the separation of logical sequences from physical KV storage.
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104): understanding prefix reuse and runtime organization as discussed in RadixAttention.
- [Inside vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm): moving from a system-wide view into the execution loop, scheduling, and serving layer.
- [Mini-SGLang Project Introduction](https://www.lmsys.org/blog/2025-12-17-minisgl/): the positioning, architecture, and foundational capabilities of this project's upstream.
- [SGLang v0.4](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/): an example connecting scheduling mechanisms to concrete execution overheads.
