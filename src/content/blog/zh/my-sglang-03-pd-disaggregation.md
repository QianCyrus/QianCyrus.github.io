---
title: "My_Sglang（三）：在八张 A100 上实践 PD 分离"
description: "沿着单节点 PD 的实现展开：管理请求与 KV 所有权，将整段交接改为逐层异步传输，用真实 profile 和三轮参数对照解释输出稳定性、首 token 等待与 goodput 的权衡。"
date: 2026-09-26T00:00:00Z
lang: zh
translationKey: my-sglang-03-pd-disaggregation
tags: [LLM 推理, PD 分离, NCCL, 性能分析, My_Sglang]
draft: false
---

> My_Sglang 工程实践 · 2026-09-26。本文使用已归档的实测数据；长输出一致性仍未完成验收，性能结果用于分析，尚不构成可默认启用的加速方案。

一条回答已经开始生成，接着系统收到几个长输入，正在输出的文字突然停了一下。几百毫秒后，它又恢复了原来的速度。

在共享同一组 GPU 的推理服务里，这种停顿可能来自 Prefill 与 Decode 的调度干扰：前者一次处理大量输入，后者需要不断推进下一步生成。把输入切成 chunk 可以限制单次计算规模，却不保证 Decode 在相邻两块之间一定获得执行机会。

我们想在 My_Sglang 里回答一个具体问题：把八张 GPU 分成两组，四张负责 Prefill、四张负责 Decode，能否让输出更连续，并提高满足延迟目标的请求数量？

这次实践借鉴了 vLLM 的 [单节点 PD 分离文章](https://vllm.ai/blog/2026-04-07-moriio-kv-connector)：使用相同总卡数比较普通部署与 P4+D4，跟踪首 token 延迟、生成间隔和 goodput，再分析 KV 交接的代价。实现采用 NVIDIA/NCCL；测试模型是 Qwen3-32B BF16。参考文章的 AMD/MORI-IO、FP8 MoE 模型和收益数值不直接迁移到这套环境。

## 先看结果：输出间隔改善了，首 token 等待变长了

先给出最接近参考配方的一轮结果，后面再拆解它是怎样产生的。三组均使用八张 A100 40GB，同一模型、同一份输入和到达序列，每组 100 条请求，输入 2000 tokens、输出 1000 tokens，目标到达率 8 req/s。

| 指标 | 普通：两组 TP4 | PD：整段交接 | PD：逐层交接 |
| --- | ---: | ---: | ---: |
| TTFT P95，秒 ↓ | **0.663** | 6.913 | 5.517 |
| 请求 P99 ITL 的 P95，毫秒 ↓ | 205.04 | **47.34** | 47.74 |
| 请求总延迟 P95，秒 ↓ | **37.244** | 43.383 | 43.396 |
| 输出吞吐，token/s ↑ | **2446.1** | 1838.5 | 1856.9 |
| 同时满足两项 SLO 的请求 ↑ | **42/100** | 8/100 | 9/100 |
| 全窗口 goodput，请求/s ↑ | **1.027** | 0.147 | 0.167 |

表 1：作业 `9722670`，代码 `f23125c`。每种部署仅测一轮；普通 Prefill 预算为 8192，PD 为 4096，属于部署配方对照，不能把全部差异单独归因于 PD。三组都使用 `prefill_first`，没有启用下文讨论的 `mixed` 普通基线。

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="同一轮请求的 SLO 达标情况、TTFT 与 ITL 分布">
<img src="/images/my-sglang/pd-a100/deployment-results.png" alt="同一轮请求的 SLO 达标情况、TTFT 与 ITL 分布" loading="lazy" />
</div>
<figcaption>图 1：来自客户端原始 token 到达记录。每个散点是一条请求，虚线是固定的延迟目标；只有一个到达率点，没有用其他配置的历史结果拼成速率曲线。 <a href="/images/my-sglang/pd-a100/deployment-results.png" target="_blank" rel="noopener">查看原图 ↗</a></figcaption>
</figure>

PD 改善了请求内的高分位生成间隔，但代价很明显：许多请求在开始回答之前，已经等待了数秒。整段 PD 有 92 条请求超过首 token 的 1 秒目标，逐层 PD 有 91 条。D 端正式请求的准入排队 P95 约为 2 ms，说明这轮长等待不能主要归因于 D 没有空闲请求槽位。

这里的“输出更连续”也不等于零停顿。逐层组每请求最大 ITL 的 P95 仍约为 217 ms；表中的 47.74 ms 是另一种、较不敏感于单次尖峰的统计量。

### 为什么同时报告 goodput 和吞吐

输出吞吐回答的是整个测量窗口里产生了多少 token。交互式服务还关心这些 token 是怎样到达用户的。

我们预先固定两项 SLO：`TTFT < 1 s`，并且**每条请求自己的 P99 ITL < 50 ms**。只有两项都满足，这条请求才计入达标数量。跨请求汇总 ITL 时，再对这些请求内 P99 取 P95，得到表 1 的第二行；不能将它简写成全部 token 间隔的 P95。

本文的 goodput = 达标请求数 / 测量窗口秒数，窗口包含最后一条请求的排空时间。8 req/s 是负载的到达速率，不是完成速率。我们没有通过完整速率扫描测出“满足 SLO 的最大可持续到达率”，也不把 `8 × 达标比例` 当成表中的全窗口 goodput。

## 架构：保留原引擎，只增加交接职责

My_Sglang 已有模型执行、TP、分页 KV、请求调度和 HTTP 服务。PD 接入继续使用这些模块，每一组 TP4 都加载完整模型，由已有引擎执行本组的计算。

| 组件 | 本次部署中的职责 |
| --- | --- |
| Proxy | 普通模式轮询两个实例；PD 模式将生成请求转发到 D，并透传响应流 |
| P，GPU 0–3 | 处理输入，产生 prompt KV 和首个输出 token，保留源资源直到交接完成 |
| D，GPU 4–7 | 预留目标资源，接收 KV，提交首 token，随后继续普通 Decode |
| PD 控制器 | 管理准入、等待、提交、取消和资源所有权 |
| KV transport | 负责 staging tensor、配对 NCCL 传输及完成事件 |

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="My_Sglang 的单节点 PD 架构和默认整段交接路径">
<img src="/images/my-sglang/pd-a100/architecture.svg" alt="My_Sglang 的单节点 PD 架构和默认整段交接路径" loading="lazy" />
</div>
<figcaption>图 2：默认整段模式的架构示意。橙色表示交接路径；逐层模式复用同样的进程和 GPU 分组，但会改变中间的传输时序。图中的“等待期间不解码”仅指当前等待 KV 的请求，D 仍可推进其他已就绪请求。 <a href="/images/my-sglang/pd-a100/architecture.svg" target="_blank" rel="noopener">查看原图 ↗</a></figcaption>
</figure>

具体实现集中在三个位置：PD 控制器（`scheduler/pd.py`）、NCCL transport（`distributed/pd_transport.py`） 和 HTTP proxy（`server/pd_proxy.py`）。模型、采样器和 KV allocator 继续复用。

控制消息通过 ZMQ 进入对端，再沿引擎原有的 rank 0 广播分发。KV 数据走额外建立的四个双 rank NCCL group：P rank 0 对 D rank 0，以此类推。每个引擎内部的 TP 通信保持独立。

这里有一个决定时序的实现选择：**HTTP 请求先进入 D，D 完成准入和目标页预留后，才向 P 派发输入。** Proxy 没有并行向 P、D 各发一条 HTTP 请求。我们的逐层模式因此也不能直接等同于参考博客的 Write 路径。

## 从整段交接开始：传过去的不只是 KV

设输入长度为 N。P 执行 Prefill 后，拥有 N 个输入 token 的 KV，并采样出首个输出 token。

这个首 token 还没有自己的 KV。D 接管时，需要保持 `cached_len=N`、`device_len=N+1`；下一步 Decode 在位置 N 上处理首 token，产生后续输出。若将首 token 误计为已缓存输入，就会造成位置或输出计数错位。

另一方面，P 的物理页编号对 D 没有意义。两边分别分配自己的 KV 页，P 按逻辑 token 顺序 gather 成连续 tensor，经 NCCL 发送；D 再 scatter 到本地页表对应的位置。相同 TP 布局使同编号 rank 可以直接交换相应的 KV head 分片，无需在这版实现中重分片。

### Whole：先准备好全部 KV，再交接

整段模式的流程如下：

1. D 预留请求槽位和 prompt KV 页，将输入发送给 P。
2. P 完成 Prefill 和首 token 采样，向 D 发送 `READY`。
3. D 投递接收，再回复 `START`。
4. P 打包并发送完整 prompt KV；各 rank 使用各自的 NCCL 配对通道。
5. D 在全部 rank 就绪后，将 scatter 排入执行流，提交首 token，安排后续 Decode，并回复 `ACK`。
6. P 收到 ACK 后，还要确认本地 send 依赖完成，才能回收源请求和 KV 页。

这样容易建立所有权边界，但整段打包与传输位于 Prefill 之后。控制消息还要由调度循环处理，下一轮较长的计算可能延迟协议推进。早期实现确实遇到了这种情况，后来增加了交接优先处理；早期改动与负载结果保留在项目实验记录中；本文对应测量见[公开摘录](/data/my-sglang/pd-a100-experiments.json)。

## Layered：让已生成的 KV 提前出发

整段模式稳定运行后，我们把关注点移到它的串行尾部：必须等所有层计算结束，才能开始发送吗？

同一层的 KV 一旦写入，就可以建立发送依赖。后续层仍然在计算时，传输流可以打包并发送已经就绪的 KV。因此新增了 `MINISGL_PD_TRANSFER_MODE=layered`，保留 `whole` 作为默认值和对照路径。

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="整段交接与逐层交接的协议时序示意">
<img src="/images/my-sglang/pd-a100/handoff-flow.svg" alt="整段交接与逐层交接的协议时序示意" loading="lazy" />
</div>
<figcaption>图 3：依据代码绘制的时序示意，不按实测时长缩放。逐层模式取消 START 往返，将 KV 传输与后续计算重叠；仍需等待全部层和首 token 就绪，才提交这条请求。图中 Whole 等 ACK 再安排下一批 P，对应本轮 <code>MINISGL_PD_HANDOFF_FIRST=1</code>；关闭该开关的 Whole 允许继续安排 Prefill。 <a href="/images/my-sglang/pd-a100/handoff-flow.svg" target="_blank" rel="noopener">查看原图 ↗</a></figcaption>
</figure>

逐层模式在最终 Prefill chunk 的前向开始之前发送 `PLAN`，D 按模型层序预先投递接收。P 在每层 KV store 后调用发送钩子：传输流等待已有计算流依赖，再执行本层的 gather 和 NCCL send。这个时刻不必等到整个 Transformer block 都结束。

前向结束后，P 单独发送首 token 信息。D 收齐所有层、收到首 token，并在 TP 组内确认各 rank 就绪，才统一安排 scatter 和提交。Scatter 与后续 Decode 依靠流依赖维持顺序；提交客户端首 token 前没有额外插入一次 CPU 端的逐层同步。

Chunked Prefill 在这里还有一个容易忽略的细节：**最后一个 chunk 完成时，要交接整条 prompt 的 KV。** 假设此前已处理 1920 tokens，本轮补上最后 80，最终需要导出的是 2000-token KV，不能只发送刚计算的 80 tokens。中间 chunk 则继续由 P 持有状态。

| 属性 | Whole | Layered |
| --- | --- | --- |
| 传输开始时机 | 整段 Prefill 后，经 READY / START | 最终 chunk 前发 PLAN，各层 KV 就绪后发送 |
| 数据通道 | 配对 NCCL send/recv | 相同的配对 NCCL send/recv |
| staging 范围 | 完整 prompt、全部层 | 完整 prompt、逐层 staging |
| D 开始该请求 Decode 的条件 | 全部 KV 与首 token 就绪 | 全部层 KV 与首 token 就绪 |
| P 源页释放 | ACK 加本地 send 完成 | 相同 |
| 当前在途交接 | 单个交接批次 | 单个前向对应的交接批次 |

这是层级传输流水化。D 不会在收到第一层后就开始该请求的 Decode，也没有把 P 的页当成可直接写入的 D 远端页地址。下一批 P 仍需等待当前交接批完成；多批在途流水线不属于这版实现。

### 等待确实减少了，但端到端只改善了一部分

作业 `9722670` 的阶段审计覆盖两条长预热和 100 条正式请求，共 102 条。整段与逐层处理的输入总量均为 204000 tokens。

| P 阶段指标 | Whole | Layered |
| --- | ---: | ---: |
| 前向累计主机墙钟时间 | 16.632 s | 16.709 s |
| 前向结束后至 ACK 的累计等待 | 2.586 s | 0.903 s |
| 前向批数 / 交接批数 | 52 / 54 | 52 / 52 |

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="真实阶段审计中的前向时间和交接尾部等待">
<img src="/images/my-sglang/pd-a100/handoff-stages.png" alt="真实阶段审计中的前向时间和交接尾部等待" loading="lazy" />
</div>
<figcaption>图 4：累计等待下降约 65.1%。这里是引擎审计数据，不是 profiler 截图；前向墙钟时间包含主机发起、回调与结果同步，不能当作纯 GPU 计算时间。 <a href="/images/my-sglang/pd-a100/handoff-stages.png" target="_blank" rel="noopener">查看原图 ↗</a></figcaption>
</figure>

整段模式的打包上限与逐层按前向批次导出的方式不同，导致两边交接批数也不同。因此这次比较衡量的是整套交接改造，不是仅隔离通信重叠的消融。

对应到正式客户端测量，TTFT P95 从 6.913 s 降至 5.517 s，但吞吐只提高约 1.0%，总延迟 P95 基本不变。局部等待被削减了，模型输入计算的主体成本还在。

## 用 profiler 决定下一步，而不是继续堆传输机制

接下来单独采集了一小段 P0 和 D0 的 Torch Profiler trace，分析方法参考 [BBuf 的 AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS)，固定到提交 `475af5a803db671c75d60a95c5a1c623024b466e`。诊断作业 `9723407` 与正式测速分开，带 profiler 的客户端延迟不放入表 1。

P 窗口只有一次前向，输入组合是 `80 + 2000 + 2000 + 16 = 4096` tokens。其中三个请求完成，合计需要导出 6000-token KV；第四条只计算了第一个 chunk。D 则捕获了 13 次前向，在等待新 KV 时继续生成已有请求的输出。

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="由真实 Torch Profiler 事件绘制的 P/D GPU 分类时间线">
<img src="/images/my-sglang/pd-a100/gpu-overview.png" alt="由真实 Torch Profiler 事件绘制的 P/D GPU 分类时间线" loading="lazy" />
</div>
<figcaption>图 5：真实 trace 的离线重绘，不是 Perfetto UI 截图。两个进程分别归零，横轴不能用于直接推断跨进程的同时发生关系。完整 CPU/GPU 轨道见 <a href="/images/my-sglang/pd-a100/prefill-timeline.png">P 时间线</a> 与 <a href="/images/my-sglang/pd-a100/decode-timeline.png">D 时间线</a>。 <a href="/images/my-sglang/pd-a100/gpu-overview.png" target="_blank" rel="noopener">查看原图 ↗</a></figcaption>
</figure>

P 的采样窗口中，GEMM kernel 累计约 234.8 ms，TP AllReduce 约 50.6 ms，KV send 约 2.27 ms。64 次发送中，按事件区间计算，约 98.6% 的发送执行时间与已关联的非 NCCL 前向计算重叠。它支持“逐层路径实际发生了重叠”，不代表端到端加速 98.6%。

D 的 NCCL recv 累计约 450 ms，看起来远大于发送时间。但接收提前投递后，需要等待 P 逐层生产 KV，这个时间包含等待。其中约 229 ms 还与 D 的非 NCCL 解码计算重叠，不能拿 450 ms 直接换算链路带宽。

这份 trace 使下一步的范围变小了：先研究 P 的计算批量，暂时没有证据要求继续扩大通信协议。它也有边界：只采 rank 0，不能排除 TP rank 间差异；profiler 启动会扰动主机时间线，不能把这个窗口的空隙都解释成稳态瓶颈。

## 独立调整 P：更大的批量为什么没有提高 goodput

PD 的一个便利是两侧可以分别配置。我们只把 P 的 token 预算从 4096 改到 8192，D 保持 4096，其他条件不变。每组模型只加载一次，先做短回归与预热，再运行三轮固定负载；两个配置对应轮次使用相同的到达序列。

| 轮次 | TTFT P95：4096 → 8192 | 输出吞吐：4096 → 8192 | SLO 达标请求：4096 → 8192 |
| --- | ---: | ---: | ---: |
| 1 | 5.532 → 5.166 s | 1856.4 → 1863.3 token/s | 9 → 7 |
| 2 | 4.141 → 3.769 s | 1848.5 → 1865.6 token/s | 9 → 6 |
| 3 | 6.106 → 5.659 s | 1847.4 → 1867.6 token/s | 7 → 7 |

表 2：作业 `9723891`，代码 `6afdd52`。每行每组 100 请求，共 600 请求。先运行 4096，再运行 8192；这是同一部署内的三轮重复，没有三次独立重载，也没有排除固定顺序的影响。

三轮的 TTFT P95 都降低了约 7%–9%，吞吐只增加约 0.4%–1.1%。六轮中没有请求违反 ITL SLO，达标数量的变化全部来自 TTFT。

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="三轮请求在固定首 token 延迟门槛附近的变化">
<img src="/images/my-sglang/pd-a100/budget-slo-boundary.png" alt="三轮请求在固定首 token 延迟门槛附近的变化" loading="lazy" />
</div>
<figcaption>图 6：每轮全部 100 条请求均保留，局部窗口放大早期请求。1 秒门槛在实验前固定。 <a href="/images/my-sglang/pd-a100/budget-slo-boundary.png" target="_blank" rel="noopener">查看原图 ↗</a></figcaption>
</figure>

例如第一轮 `random-5` 的 TTFT 从 0.924 s 增至 1.220 s，`random-6` 从 0.783 s 增至 1.079 s。部分较慢请求少等了几百毫秒，仍然超过 1 秒；原本刚好达标的早期请求，却有几条越过了门槛。

阶段数据也给出了一种解释。没有预热的第二轮中，P 前向批数由 50 减少到 26，但总前向墙时仅从 16.241 s 降至 15.985 s；每批平均墙时则由约 0.325 s 增至 0.615 s。批量增大减少了轮次，却没有让输入处理能力成倍增加，也可能拉长同批请求的等待。由于没有完整的逐请求到 GPU batch 映射，这仍是与证据相符的解释，不是全部因果链已经得到证明。

我们保留了原来的 P4096 配方。P8192 本次没有 OOM，但 P 侧采样最高显存占用从 39153 MiB 增至 40043 MiB；没有联合 SLO 收益，也没有理由消耗更多显存将它设为默认值。

## 还有一个必须补上的对照：普通混批

目前的表 1 使用 `prefill_first`。它会优先执行可运行的 Prefill，即使已经切块，也不保证 Decode 在两个 chunk 之间前进。

项目另有 `mixed` 策略：先给 Decode 分配 token 预算，再用剩余预算安排 Prefill，同一次前向推进两类请求。它和 `decode_first` 不同，后者仍是一轮只选择一种 phase。为了避免饥饿，`mixed` 在有待处理输入时保留最小 Prefill 预算，Decode 请求过多时轮转安排。

混批能够缓解调度层面的等待，但同批 Decode 仍受 Prefill 计算时长影响。当前实现还会让含 Prefill 的混批走 eager 路径，只有纯 Decode 批使用已有 CUDA Graph。因此预算、执行路径和生成间隔需要一起比较。

我们还没有完成调优后的普通 `mixed` 与 PD 的正式对照。表 1 说明的是这套 Prefill 优先基线下的结果，不能推广成“PD 优于普通混批”，更不能据此认为它已经复现参考文章的 goodput 收益。

下一组有意义的实验应复用当前输入和固定 SLO，加入有实际混批计数的普通部署，同时披露预算、Graph、overlap 和容量。现有不同容量的低速率实验不适合拼接进这组对照；没有必要为了凑齐好看的曲线扩大扫描。

## 适用范围与仍未完成的验收

PD 的收益需要两个条件同时成立：普通部署确实受到 Prefill 干扰，而专用 P 组又能在首 token 目标内处理新输入。当前实验实现了前者的缓解，尚未解决后者。

本版范围限定为单节点、两组相同 TP、标准 dense Attention、greedy、naive KV cache 和 page size 1。Qwen3.5 的循环状态、MTP、异构 TP、跨节点传输和多批在途流水线不属于这次实验。

功能状态也需要与性能结果分开：作业 `9722670` 的两种 PD 都通过了 8/8 短 greedy 对照和 2/2 串行长预热，有载 1000-token 输出相对普通基线却分别只有 17/100、19/100 的 hash 完全一致。普通路径此前也出现过随批次变化的长输出差异，但这不足以说明 PD 的差异无害。

因此，请求完成、token 计数准确、KV 页和请求表全部归还，说明测量与正常生命周期完成；它们不能替代长输出一致性验收。逐层路径的 GPU 取消压力测试也没有在这轮追加。当前仍保持普通服务默认，逐层 PD 通过实验开关选择。

## 实验配置与复现

| 项目 | 本文主对照配置 |
| --- | --- |
| 硬件 | Wisteria 单节点，8×A100 40GB；两组 TP4 |
| 模型 | Qwen3-32B BF16，revision `9216db5781bf21249d130ec9da846c4624c16137` |
| 核心软件 | PyTorch 2.9.1+cu128，Transformers 4.57.3，FlashInfer 0.6.12，NCCL 2.27.5 |
| 工作负载 | 固定随机 token 输入；100 请求/组；2000 输入、1000 输出；8 req/s |
| 上下文与容量 | 上下文上限 16384；每实例 104 请求槽位、300032 KV 页 |
| 缓存与执行 | naive cache、page size 1；Graph 上限 104；CPU/GPU overlap scheduling 关闭 |
| 调度与预算 | `prefill_first`；普通 8192、PD 4096；P8192 为后续独立实验 |
| 测量规则 | 长预热后计时；正式测速关闭 profiler；使用引擎原生 token 计数 |

完整启动 argv 和环境保存在项目各次结果目录的 `launch.json`、`packages.json`；[公开数据文件](/data/my-sglang/pd-a100-experiments.json)提供各组统计、三轮预算实验的逐请求延迟摘录、profile 分类及源文件 SHA-256。模型与到达序列固定，但硬件、模型、传输实现和调度语义与参考文章不同；本文不将“沿用负载参数”写成同环境复现。

### 重放已归档的三组配置

以下入口对应表 1 的历史实验，使用项目私有 venv 和 PJM 八卡作业，包含加载与清理的墙时上限为 20 分钟。它会重新申请 GPU；只阅读数据或绘图无需执行。

```bash
# 将路径替换为已准备好依赖、模型和冻结输入的 f23125c 检出。
export PD_REPO=/path/to/My_Sglang-pd-f23125c
cd "$PD_REPO"
pjsub -L rscgrp=debug-a -L elapse=00:20:00 \
  -x "PD_REPO=$PD_REPO,PD_RECIPE=1,PD_RATE=8,PD_ROUNDS=1,PD_INLINE_SHORT_GATE=1,PD_COMPARE_LAYERED=1,PD_PAGES=300032,PD_MAX_RUNNING=104,PD_GRAPH_BS=104,PD_CAMPAIGN_TIMEOUT=1100,MINISGL_PD_HANDOFF_FIRST=1" \
  benchmarks/pd/wisteria/campaign.pjm
```

部署使用项目专用 venv，P/D 两端需使用同一权重和精度，并在同一节点同时启动。普通入口不传 `--pd-role`；在 PD 内部，`MINISGL_PD_TRANSFER_MODE=whole|layered` 选择交接方式。这些是实现开关，不代表所有组合已经通过验收。

### 只读数据，不申请 GPU

可下载[公开测量摘录 JSON](/data/my-sglang/pd-a100-experiments.json)核对表格。完整逐 token 日志和 trace 保留在实验项目中，公开摘录不包含账户路径或服务日志。已有图片可以直接查看。需要重新绘制时，在仓库根目录执行：

```bash
artifacts/pd-a100/local-venv/bin/python benchmarks/pd/plot_results.py \
  docs/experiments/pd-a100/results/campaign-32b-9722670/measurement/campaign.json \
  --rate 8 --output docs/experiments/pd-a100/results/campaign-32b-9722670/plots

artifacts/pd-a100/local-venv/bin/python benchmarks/pd/analyze_budget.py \
  docs/experiments/pd-a100/results/budget-32b-9723891
```

| 材料 | 代码版本 | 公开数据 |
| --- | --- | --- |
| 普通 / Whole / Layered 三组对照 | `f23125c` | [9722670 测量摘录](/data/my-sglang/pd-a100-experiments.json) |
| P0 / D0 短窗口 profile | `2898492` | [9723407 测量摘录](/data/my-sglang/pd-a100-experiments.json) |
| P4096 / P8192 三轮参数对照 | `6afdd52` | [9723891 测量摘录](/data/my-sglang/pd-a100-experiments.json) |

三次作业分别分配 928、358、877 秒，均为八卡，合计约 4.807 GPU·小时。连同此前环境、失败与诊断作业，项目累计约 14.05 GPU·小时，按集群每 GPU·小时 3 个计费 token 计约 42.14。本文写作和示意图生成只使用本地已有材料，没有新增 GPU 作业。

后续最值得补齐的两件事，是长输出差异的定位，以及普通混批基线。它们会决定下一步应该继续优化 P 的计算与调度，还是调整 PD 的资源配比。逐层交接已经缩短了一段等待，整个服务是否因此更好，仍要回到每条请求的延迟目标上判断。
