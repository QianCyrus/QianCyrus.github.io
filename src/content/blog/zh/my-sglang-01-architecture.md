---
title: "My_Sglang（一）：从一条请求看懂推理引擎的整体架构"
description: "以 Qwen3.5 的单卡文本推理为例，沿一条请求梳理 My_Sglang 的进程划分、调度循环、模型执行与混合状态管理，并说明当前实现和实验结果的边界。"
date: 2026-09-23T17:50:00Z
lang: zh
translationKey: my-sglang-01-architecture
tags: [LLM 推理, 系统架构, My_Sglang]
draft: false
---

客户端发来一句话，模型返回一段文字。这个接口很简单，服务内部却需要持续协调几件事：把文本转成 token，为请求分配资源，决定下一轮运行哪些请求，执行模型，保存生成状态，再把结果及时送回客户端。

当多个请求同时进入系统，有人提交长提示词，有人正在等待下一个 token，还有人中途关闭连接，这些环节便共同决定了服务的正确性和性能。理解推理引擎，可以先跟着一条请求走完整个过程，再看每项优化改变了哪一段。

这是 My_Sglang 系列的第一篇。本文以实验代码提交 `566bf693` 为观察点，介绍整体架构；后续文章再展开模型、状态管理、调度与算子的实现。

## 1. 项目从哪里开始

My_Sglang 是基于 **mini-sglang 的实验分支**。选择这个起点，是因为它保留了现代推理服务的主要组成部分，同时让请求如何流动、批次如何构造、GPU 如何被驱动仍然容易追踪。

HTTP 服务、tokenizer、连续批处理、Chunked Prefill、Radix Cache、普通解码 CUDA Graph 和 CPU/GPU overlap，都是 mini-sglang 已有的基础。本项目围绕 Qwen3.5 的混合模型与原生 MTP 扩展这些执行路径，并增加混合状态管理、调度适配、服务生命周期检查和可选择的 GPU 融合算子。

本轮范围限定为单张 RTX 5090、Qwen3.5-4B BF16、TP=1 和文本输入输出。这样可以集中观察单卡内部的执行成本与状态变化。多卡通信、视觉输入和量化不在本轮验收范围内。

精简架构也意味着有所取舍：先保留容易核对的通用实现，测出瓶颈，再逐项加入优化。一个功能有代码、一个短用例通过、一个负载变快，分别代表不同程度的进展。

## 2. 先分清进程、对象和 GPU

当前混合模型 HTTP 配置使用三个主要 CPU 进程：前端进程接收请求；共享 tokenizer/detokenizer 进程处理文本与 token；scheduler 进程管理队列，并持有执行模型的 Engine 对象。**Engine 与 scheduler 位于同一个进程，GPU 也不是一个额外的 Python 进程。**

前端负责参数检查、请求标识、HTTP 连接与 SSE 输出。共享 tokenizer/detokenizer 进程既编码输入，也解码输出，并维护相应的 token 计数。这就是当前配置要求 `num_tokenizer=0` 的含义：共享同一个工作进程，并非关闭分词。

scheduler 决定每轮处理哪些请求；Engine 负责准备和执行 GPU 工作，持有模型、采样器、CUDA stream 等对象。GPU 上则是模型权重、请求状态和实际执行的矩阵、Attention、GDN 等算子。CPU 进程之间通过 ZeroMQ 交换消息，Engine 通过 CUDA 驱动设备计算。

下面的交互图可以查看各模块的职责与联系。即使不使用交互，也可以沿这条路径阅读：**客户端 → HTTP 前端 → 共享 tokenizer → scheduler 内的 Engine → GPU；结果再经 detokenizer 和前端返回客户端。**

<my-sglang-explorer view="architecture"><p>HTTP 前端 → 共享 tokenizer/detokenizer 进程 → scheduler 进程内的 Engine 与模型 → GPU 计算；生成结果沿 detokenizer 与前端返回客户端。</p></my-sglang-explorer>

这里的分工把服务协议、执行决策和模型计算各自收在明确的边界内。增加一种模型时，HTTP 前端无需了解它的每层计算；调整调度策略时，也不必重写分词逻辑。

## 3. 三个贯穿执行路径的对象

沿源码阅读时，`core.py` 中的 `Req`、`Batch` 和 `Context` 是很好的入口。它们分别描述一条请求、一轮执行和执行所需的上下文。

| 对象 | 它回答的问题 | 主要内容 |
| --- | --- | --- |
| `Req` | 这条请求进行到了哪里？ | 请求 UID、CPU token 序列、请求槽、已计算长度和输出预算 |
| `Batch` | 这一轮准备运行什么？ | 请求集合、展平后的输入 token、位置和执行元数据 |
| `Context` | 模型执行时从哪里取得这些信息？ | 当前 batch、页表、后端及相关资源的引用 |

例如，`cached_len` 描述请求已经计算到的位置，`extend_len` 描述本轮还要推进多少 token。prefill 往往推进一段输入，普通 decode 通常推进一个 token。模型使用同一套长度信息与调度器对齐状态。

Batch 的成员会随执行轮次改变：新请求可以进入，结束的请求会退出。连续批处理正是建立在这样的执行循环上。Context 则在一次前向期间提供当前 batch；它不会代替每个请求持有自己的历史状态。

## 4. 一条普通请求的完整旅程

假设客户端提交一条聊天请求，要求最多生成 128 个 token。前端先检查接口参数并分配 UID，将消息交给 tokenizer。聊天模板应用和编码完成后，scheduler 收到 token IDs、采样参数与请求标识，将其加入等待队列。

进入队列不等于立即执行。scheduler 还需要检查上下文限制、请求槽和容量预算，并在每一轮选择可运行的工作。如果输入较长，Chunked Prefill 会让它分多轮进入模型；只有读完最后一块输入后，才向客户端提交第一个生成 token。

选定请求后，scheduler 构造 Batch，准备位置与映射等元数据，再调用 Engine。Qwen3.5 的原生模型路径执行各层计算，得到 logits，由 GPU 上的 greedy 选择产生下一个 token。这里的 Transformers 模型只用于独立正确性参考，在线执行走的是 mini 的模型和 Engine 路径。

结果准备好后，scheduler 把 token 追加到请求的输出记录，检查 EOS 和输出预算，并把结果发给 detokenizer。前端最终将文字以 JSON 或 SSE 形式返回。只要请求尚未结束，它就会再次参与调度；普通 decode 使用新增 token 和已有状态继续生成。

下面可以逐步查看这条执行路径。文字流程是：**接收与编码 → 排队和准入 → 一轮或多轮 prefill → 首 token → 重复 decode → EOS、长度结束或取消 → 释放资源。**

<my-sglang-explorer view="flow"><p>接收与编码 → 排队和准入 → 分块或整段 prefill → 首 token → 重复 decode → EOS、长度结束或取消 → 后端释放资源。</p></my-sglang-explorer>

生命周期的最后一段同样属于正确性。正常结束时，请求需要退出运行集合，归还请求槽和调度资源，并删除模型状态。取消则沿前端、tokenizer、scheduler 的消息路径传递，后端确认停止后再回收资源；客户端断开连接本身不能证明 GPU 状态已经释放。

状态回收还受 CUDA 执行顺序约束。若上一轮 GPU 工作尚未完成，就把相同请求槽交给新请求，迟到的写入可能污染新任务。因此释放路径必须照顾 stream 之间的依赖，同时识别迟到结果与重复释放。

## 5. 混合模型改变了“缓存”的含义

Qwen3.5 同时包含完整 Attention 和 Gated DeltaNet（GDN）路径。为继续生成，请求除了保存 Attention 的 K/V，还需要保存卷积窗口和循环状态。分块输入、投机验证以及取消回收，都必须让这些状态保持一致。

当前实现中，`Qwen35State` 持有各层的 KV、conv、recurrent 张量和长度信息。模型以 `table_idx` 查找状态，同时核对请求 `uid`：槽位可以复用，但复用后属于另一条请求，不能沿用旧状态。每次前向还检查模型状态长度是否与调度器的 `cached_len` 一致。

这里有一个容易从架构图中误读的细节：**当前 hybrid 路径并没有把真实 KV 存入 mini 原有的 Paged KV 物理池。** Engine 在这条路径上将原有 `kv_cache` 设为 `None`。真正的 K/V 是模型请求状态中的动态 GPU 张量，普通 Attention 通过追加张量延长历史。

调度层仍保留页表、空闲页计数和请求槽，用来做逻辑映射、容量预算及准入控制。因此，“存在页表”不能直接推导为“混合模型已经采用 Paged Attention 存储”。这是当前参考实现的重要边界，也是后续显存优化需要解决的问题。

同样，混合 Prefix Cache 当前未启用。即使两个请求拥有相同文本前缀，只恢复 K/V、遗漏对应的卷积与循环状态，也无法正确继续计算。因此 hybrid 路径使用 naive cache，暂时放弃跨请求前缀复用。

## 6. 后续能力接在架构的什么位置

理解普通执行循环后，MTP、混合调度和 CUDA Graph 就各有落点。

混合调度改变 scheduler 构造 Batch 的方式，让一轮前向容纳等待处理的 prefill token 和正在生成的 decode token。模型的 `forward_packed` 共享投影与 MLP 计算，但 Attention 和循环更新仍按请求切分，维护彼此独立的状态。当前已有这条路径，长短请求的正式 GPU 调度收益仍待验证。

原生 MTP 则在生成循环中增加候选、目标验证和提交。scheduler 的 MTP 适配负责组织请求，`engine/speculative.py` 管理投机流程，模型提供 MTP 层计算。候选只保留验证通过的前缀，并追加目标模型确定的下一个 token；未通过的候选后缀会被丢弃，状态按实际提交路径恢复和推进。因此它同时影响执行、状态管理和服务计数。

CUDA Graph 作用于重复执行路径，overlap 则协调 CPU 调度与 GPU 工作。它们都依赖稳定的执行与状态边界，不能因为上游分别具备这些能力，就宣称新模型的所有组合均可使用。当前 Graph 默认关闭，仅有 B=1、容量 256、math SDPA 的有限实验；MTP+mixed 和 MTP+Graph 显式拒绝，MTP 路径关闭 overlap。

算子融合位于更靠近 GPU 的一层。GDN 与 gated norm 各有独立开关，通用路径作为对照保留。这样可以单独回答一项融合是否改变数值、是否减少热点耗时，以及是否真正缩短模型执行时间。

## 7. 当前进展如何解读

本项目仍在分阶段验收。下面几个结果用于说明进展，完整条件、原始测量和失败记录保存在项目的中文实验文档中。

| 项目 | 当前证据与边界 |
| --- | --- |
| 模型接入 | 441 个文本/MTP 权重键严格加载；普通推理已有有限 HF/greedy 对照，更广回归仍需继续 |
| GDN 融合 | 普通推理 B=1、输入 512、输出 128，三轮平均离线模型生成耗时（prefill + decode）相对通用路径降低 **30.399%**；不代表 HTTP 延迟、所有负载或超过成熟融合后端 |
| gated norm 融合 | 微基准更快，但已测模型耗时反增约 **1.385%**，保留负结果并默认关闭 |
| 在线 MTP | 普通和 MTP 各完成 81 个请求；两种模式只有 **55/81** 原始 token 序列严格一致，尚未通过完整正确性与性能验收 |

表中数据摘自已经归档的历史实验，本文写作期间没有重新运行 GPU 测量。[测量摘录与来源哈希](/data/my-sglang/chapter-01-evidence.json)保留了融合实验的三轮原值，以及 MTP 的严格对齐计数。MTP 的这组在线结果发生在后续重放提交修复之前，修复后仍需重新验收。

这也解释了为什么实验记录需要和代码一起维护。性能数字只有绑定模型、精度、执行开关、负载和正确性条件，才足以支持一次工程决策。协议成功与资源回收通过，也不能代替模型输出的严格对齐。

## 8. 从整体架构继续深入

阅读代码时，可以从 `server/launch.py` 看进程启动，再沿 `tokenizer/`、`scheduler/` 和 `engine/` 跟踪消息与执行。`models/qwen3_5.py` 展示混合状态如何进入前向，`kernel/qwen35_reference.py` 则提供通用算子起点。实验工具放在 `benchmarks/qwen35/`，记录位于 `docs/experiments/qwen35-sm120/`，与服务执行代码分开维护。

本系列计划共十篇。本文建立整体视图，后续依次展开以下主题；尚未发布的文章会随实现与验收进展调整：

2. 请求生命周期与 HTTP 服务；
3. Qwen3.5 模型与通用算子接入；
4. KV、卷积与循环状态管理；
5. Chunked Prefill 与混合调度；
6. 原生 MTP 的验证与提交；
7. 从 profile 到算子融合与 SM120 实验；
8. CUDA Graph 与 CPU/GPU overlap；
9. 跨 batch 数值分歧的诊断；
10. Benchmark、正确性与复现证据。

## 参考阅读

- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)：先建立引擎全貌，再展开调度、执行与服务，适合与本文对照理解不同项目的模块边界。
- [Mini-SGLang: Efficient Inference Engine in a Nutshell](https://www.lmsys.org/blog/2025-12-17-minisgl/)：介绍本项目所基于的精简引擎及其已有能力，有助于区分上游基础与本分支扩展。
- [SGLang v0.4](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/)：展示调度等系统优化如何联系具体开销与测量，后续讨论 overlap 时可继续参考。
