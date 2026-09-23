---
title: "My_Sglang（一）：推理引擎的模块边界、执行循环与状态管理"
description: "从模块契约与状态所有权出发，解析 My_Sglang 的调度循环、Paged KV 与 Radix Cache、CUDA 执行依赖，以及 MTP 如何扩展生成与提交路径。"
date: 2026-09-23T17:50:00Z
lang: zh
translationKey: my-sglang-01-architecture
tags: [LLM 推理, 系统架构, My_Sglang]
draft: false
---

推理引擎的架构，最终要回答三个相互约束的问题：**这一轮执行哪些 token，执行依赖哪些状态，结果何时可以成为下一轮的输入。** 连续批处理改变第一项，KV Cache 管理约束第二项，CUDA Graph、overlap 与投机解码则不断调整第三项的边界。

My_Sglang 基于 mini-sglang，保留其服务、调度、执行与缓存管理骨架，在此基础上探索模型适配、混合调度、原生 MTP 和算子优化。本文以代码提交 `566bf693` 为观察点，建立这些模块之间的联系。重点放在数据结构、资源所有权和执行顺序；模型适配与性能实验留到后续章节。

## 1. 系统边界：服务协议、调度决策与设备执行

My_Sglang 沿用 mini-sglang 的进程组织。HTTP 前端负责连接、请求参数与响应；tokenizer/detokenizer 负责文本和 token 序列的转换；scheduler 维护等待与运行集合，并在自己的进程中持有 Engine。Engine、模型和采样器都是该进程内的对象，GPU 承载由它们提交的计算与张量存储。

进程数量随 tokenizer 配置和 TP 配置变化。下图以单卡、共享 tokenizer/detokenizer 工作进程为例，展示模块关系；它并不把某个部署拓扑固定为项目的唯一架构。

<my-sglang-explorer view="architecture">
<p>HTTP 前端与 tokenizer/detokenizer 通过消息连接 scheduler。scheduler 在进程内调用 Engine、模型与采样器；Engine 驱动 GPU。请求槽、页表、缓存索引和物理张量分别由对应模块管理，结果经 detokenizer 返回前端。</p>
</my-sglang-explorer>

这几个边界传递的信息不同。进程间消息携带 UID、token IDs、采样参数、结束标记和计数；scheduler 与 Engine 之间传递 `Batch`、采样参数和映射；模型与 Attention backend 之间共享当前执行的元数据及设备状态。完整 hidden states 和 logits 留在执行侧，不经 HTTP 消息通道往返。

| 模块 | 拥有的状态与决策 | 向下一层提供的契约 |
| --- | --- | --- |
| 服务与分词 | 连接、文本编解码、请求 UID | token 请求、取消消息；接收结果与结束原因 |
| Scheduler | pending/running 集合、准入、批次组成 | 本轮请求集合、输入区间、位置与写回位置 |
| 资源管理 | 请求槽、空闲页、前缀索引及锁定关系 | 可用容量、逻辑位置到物理位置的映射 |
| Engine | 模型、backend、采样器、CUDA stream、Graph runner | GPU token、主机 token 副本及完成事件 |
| 模型与 kernel | 权重、层计算、模型专属状态 | logits；在约定的输入前缀上推进状态 |

HTTP、连续批处理、Chunked Prefill、Paged KV、Radix Cache、普通 decode Graph 和 overlap 来自上游基础。本分支的工作主要落在模型专属状态、原生 MTP、混合调度适配、生命周期处理和可切换算子路径。这一划分也限定了后文的归属：接入已有机制与新增机制应分别讨论。

## 2. 请求身份与三条长度：执行循环的数据模型

`core.py` 中的 `Req`、`Batch`、`Context` 构成运行时的公共语言。`Req` 跨轮次存活；`Batch` 描述一次调度选择；`Context` 在前向期间暴露当前 batch、页表与 backend，退出前向后清除活动 batch。Context 是执行上下文，不承担全部请求历史。

首先要区分三种索引。`uid` 是请求在消息通道中的身份；`table_idx` 是可复用的请求槽，用来索引 token pool 和页表；页表中的值才是物理 KV 的 token 槽地址。请求结束后可以复用 `table_idx`，却不能因此继承旧 UID 的模型状态或迟到结果。

其次，`Req` 的长度描述的是执行进度，而非三份相同的 token 计数：

| 字段 | 执行语义 |
| --- | --- |
| `cached_len` | 本轮输入之前已有计算状态的前缀长度 |
| `device_len` | 当前设备输入序列的逻辑末端；两者之差是 `extend_len` |
| `max_device_len` | 该 `Req` 构造时的输入长度加 `output_len`，限定可推进的范围；中间 chunk 使用临时 `Req` |

普通 decode 开始时通常有 `device_len = cached_len + 1`：最后一个 token 已生成，尚待模型消费。prefill 则可以让 `extend_len` 大于一。一次前向提交后，`complete_one()` 令 `cached_len` 前移到旧的 `device_len`，并把 `device_len` 加一，为新 token 留出位置。

**这些是 CPU 侧的逻辑更新，不能解释成 GPU 已经执行完成。** 采样结果首先写入设备端 token pool，主机端 `input_ids` 要等异步复制完成后才追加。启用 overlap 时，下一轮可以通过设备 token pool 取得输入，而上一轮的主机记录仍在处理中。这种有意保留的进度差，是调度能够覆盖部分 CPU 开销的前提。

`Batch` 将不同请求的新增区间展平，并携带 `positions`、`out_loc` 和 backend 元数据。展平便于共用一次前向，但请求边界仍由长度和映射保留；计算合批与状态隔离必须同时成立。

## 3. 调度循环：prepare、submit、reconcile

为解释依赖关系，可以把一轮执行分成准备、提交和结果处理三个阶段。下面的英文标签用于描述流程，并不是代码中新增加的三层接口。

**Prepare** 负责选择可运行请求、分配必要资源，并构造执行描述。scheduler 从 pending 和 running 集合中选出 Batch，经 CacheManager 分配新增页，生成位置、输入映射与写回映射，再让 Attention backend 准备元数据。输入映射从设备 token pool 提取本轮 token，`out_loc` 则告诉缓存写入操作将新 K/V 放到哪里。

**Submit** 在 Engine stream 上提交模型或 Graph replay，随后采样，得到设备 token 与异步复制到主机的副本。`ForwardOutput` 同时返回完成事件，避免上层把“已经拿到 Python 对象”误当成“数据已经可读”。设备 token 写回后，未完成请求继续进入 decode 集合。

**Reconcile** 等待输出复制完成，更新主机序列，执行 EOS、长度结束与资源释放，再把结果发送到 detokenizer。中间 prefill chunk 不向用户提交生成结果；只有最后一个输入块完成后，请求才转入通常的生成循环。

<my-sglang-explorer view="flow">
<p>Prepare：准入、选择请求、分配页并构造 Batch。Submit：提交模型计算、采样与设备 token 写回。Reconcile：等待复制事件、提交输出、判断结束并回收资源。未完成请求重新进入下一轮，取消请求沿同一生命周期退出。</p>
</my-sglang-explorer>

### 批次由预算决定，而非固定的请求列表

连续批处理的关键在于每轮重建 Batch。已经完成的请求退出，新请求满足准入条件后进入；prefill 请求可以跨轮次推进，decode 请求也可以与不同的邻居共同执行。

当前准入同时检查请求槽和缓存容量。`PrefillAdder` 估计未缓存输入加输出预算，并计入运行中 decode 请求的预留量；匹配前缀被锁定后，还会再次检查可用容量。后一次检查很必要：锁定会把原本可驱逐的缓存转为受保护容量，第一次估计可能因此失效。

Chunked Prefill 将单次输入工作限制在 token budget 内。未完成的块以 `ChunkedReq` 保留原请求槽和缓存句柄，回到 pending 集合前部，下一轮从新的 `cached_len` 继续。它限制的是每轮新增计算量，容量准入仍需要考虑请求后续的资源需求。

分支还提供 `prefill_first`、`decode_first` 和 `mixed` 选择，默认仍为 `prefill_first`。mixed 路径先安排 decode，再用剩余 token budget 放入 prefill；存在等待输入时为其保留至少一个 token，decode 超出预算时按 UID 轮转，若 prefill 暂时无法准入，再收回预留预算。其目标是让两类工作都有推进机会，但预算预留不能代替负载下的无饥饿验收，容量约束和实际服务时延仍需独立检查。

核心 `Batch.phase` 仍只有 prefill/decode。mixed batch 通过请求区间及 `batch_mix` 描述内部组成，并进入可处理变长片段的模型路径。增加调度策略因而不仅是改选队顺序，还要求模型和 backend 理解这个执行描述。

## 4. KV Cache：存储、映射与复用分别归谁

普通 Attention 路径把 KV 管理拆成三个层次：`MHAKVCache` 持有物理张量，页表维护请求逻辑位置到设备存储位置的映射，Radix Cache 维护可以复用的 token 前缀。它们共同工作，但生命周期不同。

![普通 Attention 路径的 KV 所有权：请求槽映射至页表，Radix 前缀索引复用物理 token 槽；锁定与驱逐控制页的回收。](/images/my-sglang/kv-ownership.svg)

*图 1：普通 Attention 路径中的映射与所有权示意。颜色表示资源角色，箭头表示引用或回收关系；页号与请求均为示例，不是运行测量。混合模型的独立状态路径见本节末尾。*

### 物理池与页表

Engine 初始化物理池，布局包含 K/V、层、页、页内 token、KV heads 和 head dimension。CacheManager 按页分配，随后把页展开为 token 槽写入页表。因此，代码中的 `page_table[table_idx, position]` 保存的是物理 token 位置，并非未经展开的 page ID。

模型计算出本轮 K/V 后，`store_kv` 根据 `out_loc` 写入物理池，Attention backend 利用请求的历史长度和映射读取已有缓存。模型层无需自行分配一段连续显存来容纳整条历史；调度器也无需操作各层 K/V 的具体数值。

### Radix 索引与引用保护

Radix 节点保存 token 前缀及其物理索引，**不会另存一份完整 K/V**。命中前缀后，新的请求页表可以引用已存在的槽位。节点的 `ref_count` 沿祖先路径增减：有活动请求持有的路径受保护，计数归零后才转入可驱逐集合。匹配最多使用输入的前 `N−1` 个 token，保留至少一个新 token 计算输出 logits。

当空闲页不足，CacheManager 向前缀缓存申请驱逐；Radix 从未锁定的叶节点开始，按时间戳选择候选，归还它们引用的物理位置。因而可用于准入的容量包含空闲容量与可驱逐容量，不能只查看尚未分配的 free list。

请求结束也不等于它计算的全部 KV 立即销毁。`cache_req` 将可缓存的整页前缀插入索引，解除旧句柄的保护，并释放重复前缀对应的本请求分配以及不能保留的尾部；其余前缀可以继续等待后续命中。请求槽回收、前缀解锁与物理页回收是三个动作，混为一谈会导致过早释放或显存泄漏。

### 模型专属状态是另一条存储路径

Qwen3.5 适配展示了上述抽象的边界：混合模型同时拥有 KV、卷积窗口和循环状态。当前参考实现令 Engine 的 `kv_cache=None`，由模型以 `(table_idx, uid)` 关联独立状态，真实 KV 使用动态张量追加；调度层仍用页表与页预算做容量核算。它并未接入普通路径的物理 KV 池。

这条路径也强制使用 naive prefix cache。可恢复状态必须覆盖模型继续执行所需的全部历史，不能只因为 KV 前缀相同就复用缓存。后续统一资源管理，需要定义完整状态的分配、恢复与释放契约；仅将它们放进一个名为 CacheManager 的类并不能完成统一。

## 5. Engine、backend 与异步执行的边界

Engine 将模型计算、采样、CUDA stream 和 Graph 管理组合成可调度的一次前向。Attention backend 负责将通用 Batch 转换为后端所需的元数据，并提供计算及 Graph 配合接口；模型定义网络结构，kernel 实现具体操作。这使算子替换可以局限在执行侧，但更改内存布局或形状约束时，仍需检查上层契约。

普通 decode 的 `GraphRunner` 按预设 batch size 捕获模型前向，使用固定输入、位置、写入位置与 logits buffer。运行时将请求数向上填充到已捕获尺寸，复制实际输入并准备 replay 元数据；虚拟请求使用专门的槽与 dummy page。这里捕获的是模型执行，采样、队列选择、结果判定和整个 HTTP 生命周期仍在 Graph 外部。

Graph 消除部分重复提交成本，overlap 则利用两个 stream 与延后一轮的主机处理组织依赖。scheduler stream 准备元数据，Engine stream 等待这些准备工作后执行当前 Batch；CPU 随后处理上一轮输出，等待对应的复制事件。二者可以组合，但解决的是不同的开销，也不意味着两个模型 Batch 在 GPU 上并发执行。

这一组织方式把回收变成执行协议的一部分：上一轮结果宣布结束时，下一轮工作可能已经排入 Engine stream。归还请求槽前，释放路径必须建立对这些工作完成的依赖，否则新请求可能复用仍会被旧工作写入的槽。代码同时用完成标记防止迟到结果被再次提交，用请求身份检查防止模型专属状态串用。取消沿消息通道到达 scheduler 后，也需要处理在途执行和已有状态，而非只关闭前端连接。

## 6. MTP：将“一次生成”扩展为有提交边界的事务

MTP 在架构上的影响超出预测头本身。普通 decode 每轮推进一个待消费 token，再产生一个新 token；投机路径则先生成多个候选，执行目标验证，最后决定能够提交的前缀。因此，一次模型执行结束和一次请求状态提交不再天然重合。

当前分支把职责拆在两处：`scheduler/mtp.py` 的 `MTPBatchHandler` 将已调度请求映射为 MTP 会话、预留验证区间并更新 token pool；`engine/speculative.py` 的 `GreedyMTPController` 组织候选、验证及状态提交。这里的“事务”描述提交前后状态的隔离方式；实现仍是模型专属的 greedy 实验路径，尚未抽象成任意模型可用的投机插件。

![MTP 的状态提交示意：从已提交状态派生候选与验证临时状态，依据目标验证接受前缀，再提交目标状态、MTP 状态及新 token。](/images/my-sglang/mtp-transaction.svg)

*图 2：MTP 一轮中的已提交状态、临时分支和提交边界。接受长度与 token 是机制示例，不是接受率或吞吐测量。*

会话持有 target state、MTP state、最后一个目标 hidden state，以及 `pending` token。`pending` 已向调用方输出，但目标模型尚未消费；这是理解轮次交接的关键。候选生成在 MTP 状态的副本上展开，目标模型则在 target state 副本上验证 `[pending, candidates…]`，不同请求可以得到不同接受长度。

验证完成后，控制器只接受连续匹配的候选前缀，并追加目标模型确定的下一个 token。当前实现对遭拒绝的分支，从旧 target state 重放真正需要保留的输入前缀；追加的 token 也取自这条实际提交路径的 logits。MTP 状态使用目标 hidden states 重新推进，不能直接提交候选展开时的 hidden states。这里选择了明确的状态恢复语义，重放与复制开销仍需要后续优化。

所有模型操作、长度检查和待发布张量分配完成后，控制器才更新各会话引用。随后适配层同步请求长度、释放只属于拒绝后缀的预留位置，并把新的 pending 写入 token pool。一个批次中请求可以共同执行验证，但接受长度和状态提交仍属于各自请求。

服务层因此必须接收 token 列表而非假设每轮只有一个 token。EOS 与输出预算作用于实际提交序列，usage 根据 token 数累加；SSE 事件数量不能代替输出 token 数。资源释放也要同时结束普通请求状态与 MTP session。

MTP decode 当前绕过普通 `Engine.forward_batch` 和 Sampler，直接调用 controller 的模型路径；mixed 和 Graph 组合被显式拒绝，overlap 关闭。这些限制反映了尚未统一的执行与状态契约，不能从普通路径支持某能力推导出投机路径也支持。

## 7. 架构中的扩展点与当前边界

新增能力应先确定改变哪项契约。模型适配改变状态和前向接口；调度优化改变 Batch 的成员与输入区间；backend 优化改变执行元数据与存储访问；MTP 改变一次执行能够提交多少结果。跨越边界的改动，需要把生命周期一并纳入，而不能只验证局部调用成功。

| 路径 | 本文描述的实现范围 | 不能自动推导的能力 |
| --- | --- | --- |
| 普通 Attention | 上游物理 KV 池、页表、Radix、普通 decode Graph/overlap | 所有模型与任意后端组合均已验收 |
| 混合状态适配 | 请求独立 KV/卷积/循环状态；已有变长 packed 前向 | 真实 paged KV、混合 Prefix Cache 已完成 |
| 原生 MTP | greedy 候选、验证、批处理与事务式状态提交 | 通用模型插件、随机采样及 Graph/overlap 组合 |

本文讨论的是架构及实现落点，不将模块存在视为性能结论。模型支持矩阵、数值回归和优化收益需要绑定具体版本与配置；这些证据会在对应专题中展开。

阅读源码时，可以按下面的边界进入，而无需先逐行跟完模型：

| 要理解的问题 | 主要入口（相对 `python/minisgl/`） |
| --- | --- |
| 进程、协议与消息 | `server/launch.py`、`server/`、`tokenizer/`、`message/` |
| 请求与批次的数据契约 | `core.py` |
| 准入、批次选择与结果处理 | `scheduler/scheduler.py`、`scheduler/prefill.py`、`scheduler/decode.py` |
| 请求槽、页分配与前缀复用 | `scheduler/table.py`、`scheduler/cache.py`、`kvcache/` |
| 前向、backend、Graph 与采样 | `engine/engine.py`、`engine/graph.py`、`engine/sample.py`、`attention/` |
| MTP 的调度接入与状态提交 | `scheduler/mtp.py`、`engine/speculative.py` |
| 网络结构与算子实现 | `models/`、`layers/`、`kernel/` |

后续章节将分别进入请求生命周期、缓存管理、调度、模型适配、MTP 与执行优化。每一篇都沿本文的模块边界定位修改，并同时解释它改变的数据、依赖与验收条件。

## 参考阅读

本文图示为依据项目代码重新绘制的机制图。图的组织借鉴系统论文区分逻辑映射、物理存储和状态转换的表达方式，具体实现以本文注明的代码版本为准。

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)：理解逻辑序列与物理 KV 存储的解耦。
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)：理解 RadixAttention 所讨论的前缀复用与运行时组织。
- [Inside vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)：从系统全貌进入执行循环、调度与服务层。
- [Mini-SGLang 项目介绍](https://www.lmsys.org/blog/2025-12-17-minisgl/)：本项目上游的定位、架构及基础能力。
- [SGLang v0.4](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/)：将调度机制与具体执行开销联系起来的案例。
