---
title: "My_Sglang（二）：把 Qwen3.5 接进推理引擎，再沿着 Profile 做优化"
description: "从配置、权重与状态语义出发，记录 Qwen3.5 普通推理的接入过程：建立通用 GPU 实现，用 Perfetto 找到开销，逐项验证 GDN 融合、CUDA Graph 与 SM120 调参，并追查批处理下的数值分歧。"
date: 2026-09-24T00:00:00Z
lang: zh
translationKey: my-sglang-02-qwen35-model
tags: [LLM 推理, 模型适配, Qwen3.5, My_Sglang]
draft: false
---

给推理框架增加一个模型，注册表里通常只需要多一行。真正费力的是后面的问题：这个模型的一次 forward 到底消费了什么？下一轮应该保留哪些历史？把单请求放进 batch 以后，原来的计算和状态约定还成立吗？

这篇记录 Qwen3.5-4B 文本模型在 My_Sglang 中的接入过程。我们从通用 PyTorch CUDA 实现开始，先跑通普通生成，再看 profile，逐项换成融合算子。最明确的一次改善来自 GDN 递推：固定单请求、512-token 输入和 128-token 输出时，完整离线生成由 **2273.968 ms 降到 1582.695 ms，耗时下降 30.399%**。与此同时，也有算子快了五倍、模型却略微变慢的实验。

这些结果来自 2026 年 9 月 23 日的 RTX 5090 测量；9 月 25 日恢复设备后，又补做了数值回归。下面按实际实现的依赖顺序展开。本文沿着普通推理这条路径，记录实现、测量和几次需要退回来检查的地方。

## 1. 先读配置、权重和参考 forward

开始写代码前，我先看三份东西：配置描述网络结构，checkpoint 说明实际参数，参考 forward 决定计算顺序。三者经常藏着不同的信息。

[这份固定 revision 的 config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json)顶层是多模态包装，语言模型配置位于 `text_config`。本轮只接文本路径，因此首先沿着 `text_config` 确认主干，而不把视觉编码器带进最小实现。

| 配置 | 接入时必须处理的事情 |
| --- | --- |
| 32 层，24 层 GDN、8 层 full attention | 按 `layer_types` 构造两种 token mixer |
| hidden size=2560，Q heads=16，head dim=256 | Attention 内部宽度为 4096，不能从 hidden size 除以 heads 推导 head dim |
| KV heads=4，partial rotary factor=0.25 | 实现 GQA；每个 Q/K head 只旋转前 64 维 |
| GDN Q/K heads=16，V heads=32，维度均为 128 | 对齐 head 分组，保存每个 value head 的递推状态 |
| causal conv width=4 | 为每层保留跨调用的卷积输入窗口 |
| tied embedding，词表 248320 | 输出投影复用 embedding 权重 |

这份 4B 配置是 dense 模型。两种 mixer 外面都包着 pre-norm、残差和 `2560 → 9216 → 2560` 的 SwiGLU。适配的主要变化集中在 mixer 和历史状态，已有 Linear、MLP 与框架接口可以继续使用。

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="Qwen3.5 普通推理主干与请求状态">
<img src="/images/my-sglang/qwen35-report/blocks.svg" alt="Qwen3.5 的 GDN 与 Full Attention 主干，以及 packed 投影之后按请求维护 KV、卷积窗口和循环状态的边界。" loading="lazy" />
</div>
<figcaption>图 1：本次接入的文本主干。共批的矩阵投影与按请求隔离的历史状态，在实现中有明确边界。 <a href="/images/my-sglang/qwen35-report/blocks.svg" target="_blank" rel="noopener">查看原图 ↗</a></figcaption>
</figure>

随后检查 safetensors 的键名、shape 和 dtype。文本权重位于 `model.language_model.*`，加载时映射到本地 `model.*`，跳过文本范围之外的 `model.visual.*`。这个混合模型分支先限制 TP=1，避免在首次适配时叠加已有加载器的 QKV 合并与分片假设。

最后沿参考 forward 追四个细节：Q 与输出 gate 如何排列，Q/K norm 位于 RoPE 的哪一侧，卷积缓存保存输入还是输出，GDN 从更新前还是更新后的状态读取结果。浮点转换也一起记下来；两段代码的公式一样，中间 BF16 舍入的位置不同，输出仍可能不同。

回到 mini 的代码，改动入口就清楚了：`ModelConfig` 继续传递混合层和 GDN 参数，注册器找到本地模型，`BaseOP` 完成严格权重加载，Engine 负责提供本轮新增的 tokens。我们没有另起一套执行框架。

实验从上游 `20fcd7f` 建独立 worktree 和分支，使用专用 venv，保留原有服务环境。实际软件为 PyTorch 2.13.0+cu130、Triton 3.7.1、Transformers 5.12.1，模型 revision 固定为 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。这些版本与权重哈希构成后续对照的起点。

## 2. 先把模型写成容易检查的通用 GPU 实现

第一版没有急着写硬件特化。`qwen35_reference.py` 用 PyTorch 张量运算表达 causal conv、GDN recurrence 和 gated RMSNorm；输入在 CUDA，状态和运算也都在 CUDA。通用路径的价值在于能逐行对应参考公式，给后面的融合提供一个明确的比较对象。

### GDN：先把时间递推和状态布局写对

GDN 接收四路投影：qkv 为 `[N,8192]`，z 为 `[N,4096]`，a/b 各为 `[N,32]`，N 是本轮 token 数。qkv 经过宽度为 4 的 depthwise causal conv 与 SiLU，然后拆成 Q、K、V；另外三路不经过卷积。

卷积状态保存最近四个**卷积之前的输入**。下一块到来时，把历史窗口与新输入拼接，取当前块对应的卷积输出，再留下末尾四个原始输入。如果缓存的是激活后的输出，第一块可能看不出问题，第二块就已经换了计算。

Q/K 的 16 个 heads 各重复两份，与 V 的 32 个 heads 对齐。循环状态布局约定为 `[B,H,K,V]`，每个 head 持有一个 `128×128` 的 FP32 矩阵。Q/K 做 L2 normalization，Q 额外乘 `128^(-1/2)`；两个门控为：

```text
β_t = sigmoid(b_t)
g_t = -exp(A_log) · softplus(a_t + dt_bias)
```

沿 token 顺序，递推可以直接写成：

```python
state = state * g_t.exp()[..., None, None]
memory = (state * k_t.unsqueeze(-1)).sum(-2)
delta = (v_t - memory) * beta_t.unsqueeze(-1)
state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
output = (state * q_t.unsqueeze(-1)).sum(-2)
```

这里有三个顺序不能颠倒：先衰减历史，再读取 memory；用当前 key 对 memory 的误差修正状态；query 最后读取**更新后的**矩阵。递推与状态保持 FP32，输出回到输入 dtype。Q/K normalization 则沿用参考实现的舍入顺序，不能顺手改成全程 FP32。

算子显式接收旧状态、返回新状态，不覆盖调用方传入的张量。这样完整输入与分块输入可以从同一份初态开始比较，也不会因为一个测试提前修改 state，让后续结果失去可比性。

### Full Attention：shape 对了，还要核对 head 内布局

Q 投影同时输出 query 与 gate。正确拆法是先按 head reshape，再切每个 head 的两半：

```python
qg = q_proj(x).view(N, 16, 512)
q, gate = qg.chunk(2, dim=-1)  # 每个都是 [N,16,256]
```

若直接把 8192 维切成前后两半，最终 shape 依然可以成立，head 对应关系却错了。随后 Q/K 各自归一化，只旋转前 64 维。真实 KV 保留四个 heads，计算时展开到 16 个 Q heads，先调用 PyTorch SDPA；Attention 输出乘 `sigmoid(gate)`，再投影回 2560 维。

Norm 也不能一概复用。主干一般 RMSNorm 乘 `1 + weight`，GDN gated norm 直接乘 `weight`，并在乘权重之前将归一化结果转回输入 dtype：

```python
x = hidden.float()
x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
x = weight * x.to(hidden.dtype)
out = (x * F.silu(gate.float())).to(hidden.dtype)
```

后续融合保留的正是这些转换边界。把中间结果全部留在 FP32，到最后才转 BF16，是另一条数值路径。

通用算子的首轮 GPU 测试为 30/30 通过；真实 GDN 维度的独立 HF 对照覆盖 B=1/4、T=1/16、FP32/BF16 共八组，输出与最终 state 的最大绝对误差均为 0。开发中 CPU BF16 `rsqrt` 曾因整段与分块形状不同出现 1 ULP 差异，因此测试将归一化和递推连续性分开定位，GPU 仍保留完整路径检查。

## 3. Day 0：加载主干，接续状态，跑通普通生成

这里把 Day 0 定义为普通文本推理的最小闭环：真实主干权重能严格加载，prefill 后能够连续 decode，有限 greedy 用例与独立参考对齐。它是一个阶段名称，不代表所有工作在一天内完成。

模型沿用 `BaseOP` 构造 `_TextModel`、`_DecoderLayer`、`_Attention`、`_GatedDeltaNet` 和 `_MLP`。参数外壳先建在 meta device，再加载真实张量，避免先初始化一份完整 GPU 模型。普通主干需要 426 个键：24 个 GDN decoder 各 14 个，8 个 Attention decoder 各 11 个，加 embedding 与最终 norm。

第一次严格加载就发现了问题：checkpoint 中 `A_log` 和 `linear_attn.norm.weight` 是 FP32，最初声明却跟随默认 dtype 成了 BF16。修复是保留这两类参数的真实精度，让加载器继续检查名称、shape、dtype，而不是统一 cast 后绕过失败。

离线生成先从一个简单循环开始：

```python
state = Qwen35State()
out = model.forward_tokens(prompt_ids, state=state)

for step in range(output_tokens):
    token = out.logits[-1].argmax().view(1)
    generated.append(token)
    if step + 1 < output_tokens:
        out = model.forward_tokens(token, state=state)
```

这里区分“已输出”和“已消费”很重要。首 token 从 prompt 末尾 logits 选出时，还没进入模型状态；下一次 forward 才消费它。`state.length` 记录后者，不能随着 SSE 发出 token 就提前递增。

### 让一个 state 同时承载三种历史

| 状态 | 每层形状 | 更新方式 |
| --- | --- | --- |
| K、V | 各 `[L,4,256]`，BF16；8 层 | 随已消费长度 L 追加 |
| causal conv | `[1,8192,4]`，BF16；24 层 | 滚动保留输入窗口 |
| GDN recurrent | `[1,32,128,128]`，FP32；24 层 | 每消费一个 token 递推一次 |

按这些形状，主干基础状态约为 `49.5 MiB + L×32 KiB`，尚未包含临时张量、Graph staging 和 allocator。请求即使很短，也需要固定大小的 GDN 状态，容量管理不能只数 KV tokens。

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="Qwen3.5 跨块状态接续示意图">
<img src="/images/my-sglang/qwen35-report/state-continuation.svg" alt="Prefill 的不同块与 Decode 共享请求状态：KV 随长度增长，卷积窗口滚动，GDN 矩阵持续递推。" loading="lazy" />
</div>
<figcaption>图 2：三个状态必须一起接续。图中的 token 块表示逻辑位置；本次 hybrid 实现使用请求私有的动态 KV 张量。 <a href="/images/my-sglang/qwen35-report/state-continuation.svg" target="_blank" rel="noopener">查看原图 ↗</a></figcaption>
</figure>

已有 P 个历史 token、当前块长 T 时，第 i 个 query 应看到 `j ≤ P+i` 的 key。多 token 续接必须显式处理这个位置偏移，不能直接套一个从零开始的 T×T causal mask。卷积与 GDN 则分别读回窗口、矩阵，从上一块继续。

接到 Engine 后，`forward_packed` 将请求 tokens 展平，用 `lengths` 保存边界。Linear 和 MLP 共批执行；Attention 与 GDN 按请求应用状态。模型以 `(table_idx, uid)` 核对归属，并检查 `state.length == req.cached_len`。结束或取消时释放三类状态，槽位复用前建立必要的 stream 依赖。

这样的拆分也确定了模型与调度器的边界：调度器负责决定本轮消费多少 token，模型负责把这些 token 作用到对应历史上。请求暂时没有被调度时，三类状态都留在原处；下一轮只处理新增片段。新请求复用相同 table 槽位时，UID 检查又能阻止它误读前一个请求的矩阵。这里多做一次长度与归属检查，比生成了许多 token 后再从文本反推串请求容易得多。

当前 hybrid KV 尚未接入普通模型的 paged Attention 后端，路径限制为 TP=1、page_size=1，也没有启用 hybrid Prefix Cache。只有 KV 而没有对应的卷积/GDN 状态，无法恢复一个有效前缀。

首轮普通生成使用 32-token 输入、16-token 输出，三轮重复一致，一个固定聊天样例与 HF 的前 16 个 greedy token 相同。最初脚本虽记录了分块比较，但 chunk size=64，没有真正跨块；后续改用 512-token 输入，分成八个 64-token 块，前 16 个输出才形成了实际的跨块对照。保留这条勘误，是因为测试名本身并不能证明触发了预想的场景。

## 4. 打开 Perfetto，先分清 GPU 在算什么、CPU 在发什么

参考路径跑通后，我们给每层、token mixer 和 MLP 加上 `record_function`，用 `torch.profiler` 采集 CPU/CUDA trace，再在 Perfetto 中展开一次普通 decode。

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="Perfetto 中的通用模型第一层详情">
<img src="/images/my-sglang/qwen35-report/perfetto-baseline.png" alt="Perfetto 中选中 qwen35.layer.0，展开 token mixer 及密集的小算子调用。" loading="lazy" />
</div>
<figcaption>图 3：真实 Perfetto 页面截图。trace 采集于 2026-09-23，2026-09-25 重新打开查看。选中的 qwen35.layer.0 为 CPU user_annotation，997.738 μs 包含主机执行与发射，不能当作 GDN 的纯 GPU 耗时。截图只保留 Perfetto 页面内容。 <a href="/images/my-sglang/qwen35-report/perfetto-baseline.png" target="_blank" rel="noopener">查看原图 ↗</a></figcaption>
</figure>

首轮单步 decode 有 2777 个 GPU kernels，累计执行约 9.172 ms；两个主要 GEMV 家族约 5.892 ms，占累计时间的 64.2%。与此同时，类型转换、归约和逐元素运算产生了大量短调用。

读这类时间线时，我会先选一层，沿 CPU 范围找到它发出的 GPU 工作，再缩小到具体 kernel。总览负责回答“调用密不密、有没有间隔”，单个事件负责核对名称、持续时间和 launch 参数，聚合统计再回答某类调用一共占了多少。三种视角一起看，才能避免把一个偶然较慢的事件当成热点，或把 CPU 范围的长度误写成 kernel 时间。

这给出两个方向：矩阵投影值得继续看，小算子的发射成本也值得压缩。尤其 GDN 的 reference prefill 在 Python 中逐 token 循环，长输入下的开销未必能从一张单步 decode 图里看完整。

我们先选 gated RMSNorm。它没有跨 token 状态依赖，归约、权重、SiLU 的边界清楚，适合先验证替换方式和测量流程，再处理更复杂的 GDN。

性能测量与 profile 分开跑。Profiler 有额外记录成本；CUDA event 包住一串 Python 发射时，也会包含主机来不及发射造成的 GPU 空闲。下面模型 A/B 使用相同权重和输入、同形状预热，正式三轮不启用 profiler。完整离线生成包括 prefill、127 次后续 decode 和 token 选择，排除加载、HTTP、排队与网络。

## 5. 第一个融合：算子快了，模型却没有变快

`qwen35_fused.py` 用一个 Triton program 处理一行 gated norm，把均方、rsqrt、权重乘法和 SiLU gate 合并。实现显式保留参考路径的 dtype 转换，关闭会改变对应舍入顺序的浮点融合。GDN 保持不动，只切换 `MINISGL_QWEN35_GATED_NORM`。

12 项 GPU 测试通过。B1/T1 微基准中，CUDA event 区间中位数由 54.606 μs 降到 9.918 μs，约 5.5 倍；trace 中 12 个 kernels 变成 1 个。

但放回 B1、输入 512、输出 128 的模型，结果如下：

| 耗时，ms | Reference，三轮均值 ± 标准差 | 融合 gated norm |
| --- | ---: | ---: |
| Prefill | 516.042 ± 0.651 | 521.302 ± 0.659 |
| Decode | 1769.918 ± 4.760 | 1796.328 ± 14.073 |
| 完整离线生成 | 2285.960 ± 5.274 | 2317.630 ± 14.565 |

输出的 128 个 tokens 相同，模型 trace 也能找到 24 次融合 kernel，说明替换确实生效。但完整生成耗时增加了 **1.385%**，没有得到模型收益。

这两组分进程执行，未做交错 A/B，因此不能从约 1.4% 的回退推断具体硬件原因。我们保留了实现和负结果，默认继续走 reference。局部减少十几个短调用，也许不足以改变模型主要开销；下一步仍然需要新的独立对照。

## 6. 第二个融合：让 GDN 在 kernel 内推进时间

GDN reference 每推进一个 token，就发射若干乘法、归约、加法，把中间状态反复物化成张量。更值得改动的是这段循环本身。

`qwen35_gdn.py` 的通用 Triton 实现固定真实 K=V=128，初始配置为 `value_tile=32, num_warps=4`。一个 program 对应一个 batch/head/value tile，在 kernel 内持有 `128×32` 的 FP32 状态块，沿 token 顺序执行衰减、memory 归约、delta 更新与 query 读取。

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="两项独立融合的实现方式">
<img src="/images/my-sglang/qwen35-report/fusion-process.svg" alt="gated norm 合并逐元素和归约操作并保留 BF16 舍入边界；GDN 沿 value 列分块，保留完整 key 轴，在一个 program 内推进时间。" loading="lazy" />
</div>
<figcaption>图 4：融合范围。投影、因果卷积和输出投影仍在 GDN recurrence kernel 之外；方框宽度不表示实际耗时。 <a href="/images/my-sglang/qwen35-report/fusion-process.svg" target="_blank" rel="noopener">查看原图 ↗</a></figcaption>
</figure>

选择 value 轴分块，是因为每一列都需要沿完整 key 轴归约，而不同 value 列可以独立更新。并行性放在 batch、head 和 value tile 上，时间轴仍按递推依赖顺序执行。初始 state 只读，最终 state 另行分配；Q/K normalization 保留 BF16 的中间转换。这个版本没有使用 SM120 专属指令，其他形状回退 reference。

23 项 GPU 检查覆盖 B=1/4/8、T=1/16/65、两种 dtype、非零初态和分块。最大 BF16 输出绝对差约 `1.22e-4`，最大状态绝对差约 `1.79e-7`，均在预定容差内。随后只切 `MINISGL_QWEN35_GDN`，重新测 reference；gated norm 仍关闭。

| 耗时，ms | Reference，三轮均值 ± 标准差 | GDN Triton | 下降 |
| --- | ---: | ---: | ---: |
| Prefill | 512.128 ± 0.858 | 42.210 ± 0.039 | 91.758% |
| Decode | 1761.840 ± 4.195 | 1540.485 ± 22.505 | 12.564% |
| 完整离线生成 | 2273.968 ± 5.009 | 1582.695 ± 22.496 | **30.399%** |

两条路径三轮的 128-token 输出一致，跨块前 16-token 对照一致；融合路径另做了固定聊天与 HF 的 16-token 比较。融合的第一轮比后两轮慢，仍保留在均值和标准差里。

Prefill 的大幅下降对应一个具体问题：通用 PyTorch 逐 token、多次发射的成本被压缩了。这个结果适用于所测普通单请求 512/128 负载；比较对象是我们建立的 reference，不能换成“超过成熟并行 GDN 后端”。

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="GDN 融合后普通 Decode 的 CPU GPU 时间线">
<img src="/images/my-sglang/qwen35-report/perfetto-decode.png" alt="Perfetto 展示 native_decode_step 的 CPU 调用与下方 GPU stream，融合后仍保留投影及辅助算子。" loading="lazy" />
</div>
<figcaption>图 5：2026-09-23 留存的真实 Perfetto 时间线，上方为 CPU 范围，下方为 GPU stream。同阶段 reference 有 2785 个 kernels，融合后为 2185 个；30.399% 来自无 profiler 的模型测量，与截图宽度无关。 <a href="/images/my-sglang/qwen35-report/perfetto-decode.png" target="_blank" rel="noopener">查看原图 ↗</a></figcaption>
</figure>

融合后少了 600 个 kernels，累计 GPU 执行由 9.257 ms 降到 8.451 ms。CPU 发射与 GPU 执行还有间隔，剩下的 Linear、Attention、norm 和状态操作也都还在。这让我们继续分成两个方向：用 Graph 研究提交开销，用参数扫描研究 GDN kernel 本身。

## 7. CUDA Graph 与 SM120 调参：继续拆解剩余开销

CUDA Graph 需要固定地址。我们为普通 decode 增加私有 KV/GDN staging，每轮先载入请求状态，再 replay，最后提交回请求。计时包含这些拷贝，避免只测一个很短的 replay 调用。

首先遇到的却是数值差异。动态有效 KV 长度与固定容量加 mask 的 Attention 可能选择不同计算路径。诊断发现 static 与 Graph 相同，差异已经出现在 eager 与 static 之间。改用 math SDPA 后，B1、prefix16、capacity256 的三步 logits、hidden、KV、conv、recurrent 检查一致；B4/B8 和更大容量没有全部通过。

在通过的局部范围，完整 forward wrapper 从 `16.176 ± 0.029 ms` 降到 `10.807 ± 0.001 ms`，下降 33.192%。这组 A/B 的 GDN、gated norm 都是 reference，双方都用 math SDPA，因而不能与上一节的 30.399% 相加。wrapper 也不包含外部测试框架的 state clone、token 选择，更不是 HTTP 延迟。

capacity8192 的 logits 最大绝对差达到 0.1171875，超过预设的 0.05；Graph 关闭时，math SDPA 在 512/128 生成中又比 auto 慢约 5.44%。因此 Graph 继续默认关闭，只保留短上下文实验入口。

接下来才针对 5090 的调度规模提出假设。

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="Perfetto 中选中的 GDN GPU kernel">
<img src="/images/my-sglang/qwen35-report/perfetto-kernel.png" alt="Perfetto 搜索 recurrent kernel，显示 24 次调用，并查看选中事件的 duration 和 launch 关联。" loading="lazy" />
</div>
<figcaption>图 6：重新打开 2026-09-23 的 trace，于 2026-09-25 截取的真实 GPU 事件。搜索得到 24 次 recurrence，与 24 层 GDN 对应；选中事件为 3.104 μs，关联 CPU launch 到 GPU 开始的间隔为 2.061 μs。这是一次事件的两个不同指标，并非算子均值。 <a href="/images/my-sglang/qwen35-report/perfetto-kernel.png" target="_blank" rel="noopener">查看原图 ↗</a></figcaption>
</figure>

该事件 grid 为 `(32,4,1)`，即 128 个 CTA；5090 实测有 170 个 SM。缩小 value tile 可以增加 CTA，但也会重复更多 Q/K 读取和归一化。我们据此扫描 B=1/4/8、T=1/16/128、tile=16/32/64、warps=4/8，共 54 组，先检查数值，再分开测 Python wrapper 和 Graph 内重复调用的 GPU 间隔。

全部数值检查通过。B1/T1 的 tile16 将 CTA 增至 256，GPU 间隔由约 2.436 μs 降到 2.308 μs；wrapper 却由约 13.37 μs 增至 13.84 μs。九种输入形状中，七种仍以通用 tile32/warp4 最低。更大的 tile 增加寄存器压力，更多 warps 也没有稳定优势。

所以这轮没有新增默认 SM120 特化分派。模型当前按请求调用 recurrence，也不能把 B8 微基准直接等同于服务并发 8。

我们还尝试过 FlashInfer 的 SM120 GDN 后端，trace 确认实际进入对应 kernel，但九种形状的最终 state 都未满足本项目的严格容差，需要继续核对归一化、状态布局与计算契约。该阶段计时又与另一 GPU 服务重叠，性能数字作废。它没有形成“自写实现优于成熟后端”的结论。

## 8. 批次扩大后，工作重新回到正确性

小样例通过以后，更真实的批处理暴露了普通路径自身的 greedy 分歧。同一个请求独立运行和放入 batch，输出不总相同。继续比较速度之前，需要找到差异最早出现的位置。

`batch_numerics.py` 采用 teacher forcing：B1/B4/B8 消费同一段固定 token 历史，即使某条路径预测不同，也不把不同 token 喂回下一步。逐层记录激活差异，前后核对所有权重的字节哈希。这样才能把算子误差与“输入历史已经分叉”区分开。

最早一轮诊断指向第 0 层 GDN 的 `in_proj_qkv`。关闭 BF16 reduced-precision reduction 后，prefill 的首个差异移到 MLP down projection，decode 仍在 QKV 出现差异。只切这一个精度选项并没有解决问题。

因此增加了可选的固定归约 Linear：tile 固定为 `16×64×32`，BF16 输入、FP32 累积，固定 K 归约顺序，不使用 split-K 或 autotune，让 M 的变化主要影响行掩码。七个真实矩阵形状的单算子测试通过，所测相同行在不同 batch 位置逐位一致。

9 月 25 日设备恢复后，我们先复跑四个 kernel 测试文件，77 项通过，再检查整模型。固定 Linear 的单请求诊断中，102 个位置/配置组合的 argmax 相同，但部分内部激活仍有差异，最早位置转向 gated norm 或普通 norm。这只是一个 prompt 和固定 continuation 的结果，随后仍要让生成自由运行。

我们使用九条固定 SPEED 请求，每条输出 32 tokens，忽略 EOS，分别对计划 B=1/4/8 做普通生成。每种后端以自己的普通 B1 输出作为对照；因此 B1 的 9/9 是自比较，并非独立正确性验证。不足整批的末尾保留小 batch。只改变 gated norm，得到：

| 计划 batch | 固定 Linear + reference norm | 固定 Linear + Triton norm |
| --- | ---: | ---: |
| 1 | 9/9 | 9/9 |
| 4 | 8/9 | 9/9 |
| 8 | 8/9 | 8/9 |

这里统计的是整条 32-token 输出完全相同的请求数。Reference norm 下，case3 在 B4/B8 的第 17 个输出出现分歧，token ID 从 2014 变成 15771。切到 Triton norm 后，B4 虽通过，B8 仍失败；而两种 norm 各自的 B1 基线也只有 8/9 条相同。它改变了部分模型输出，不能算作批次一致性的修复。

这一步把下一次实验指向更具体的位置：在实际失败请求、相同批次成员和行位置下，保存首次分歧前的 norm 输入输出，以及最终 logits 的 top1/top2 间隔。还需要判断容差内的激活变化怎样跨越 argmax 边界，而不是继续放宽比较阈值。

到这里，普通推理接入、显式混合状态和有限负载的 GDN 融合已经有实际结果；更广的 batch 一致性仍未过关。固定 Linear 与 Graph 都保留为实验选项，没有因为单项测试通过就自动成为默认路径。

### 代码与复现入口

这条实现路线集中在几个文件中，方便沿本文顺序阅读：

| 内容 | 仓库内入口 |
| --- | --- |
| 配置、原生模型与权重映射 | `python/minisgl/models/{config,qwen3_5,weight}.py` |
| 通用算子、norm 融合、GDN 融合 | `python/minisgl/kernel/qwen35_{reference,fused,gdn}.py` |
| Graph staging 与固定 Linear | `python/minisgl/engine/qwen35_graph.py`、`python/minisgl/kernel/qwen35_linear.py` |
| 模型对照、逐层诊断、参数扫描 | `benchmarks/qwen35/{model_baseline,batch_numerics,tune_gdn}.py` |
| 中文实验笔记和逐轮原始记录 | `docs/experiments/qwen35-sm120/` |

通用算子、模型集成、GDN 融合、固定 Linear 分别可从 `4f501ec`、`0e3bbf4`、`ce8acc4`、`89316b3` 查起；恢复后的结果归档在 `97a0ebc`。早期测量存在未提交工作树，所以复现还应对照结果中的 `source_sha256`，不能只依赖 base HEAD。[公开 A/B 摘录](/data/my-sglang/qwen35-ordinary-experiments.json)保存各轮样本和来源哈希。[截图来源清单](/data/my-sglang/qwen35-perfetto-screenshots.json)记录三个页面截图与原始 trace 的哈希。

在已按版本记录准备的独立 CUDA venv 中，可以先重跑普通 reference：

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

测试 GDN 时只把 `MINISGL_QWEN35_GDN` 改为 `triton`，保存到另一个新目录。先核对 token 和状态，再比较无 profiler 区间的三轮计时；trace 用来解释时间花在了哪里。下一轮的重点仍是普通批处理数值分歧，解决之后再扩展输入长度、调度组合和在线性能范围。
