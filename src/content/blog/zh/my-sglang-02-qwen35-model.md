---
title: "My_Sglang（二）：Qwen3.5 的模型结构与逐步接入"
description: "从 Gated Attention、Gated DeltaNet 与 MTP 的张量计算出发，详细还原配置解析、通用算子、严格权重加载、混合状态、分块推理和引擎接入，并检查真实回归证据的边界。"
date: 2026-09-24T00:00:00Z
lang: zh
translationKey: my-sglang-02-qwen35-model
tags: [LLM 推理, 模型适配, Qwen3.5, My_Sglang]
draft: true
---

[第一篇](/zh/blog/my-sglang-01-architecture/)介绍了 My_Sglang 的模块边界。这一篇进入模型接入：当一个模型同时包含 Attention、卷积和循环状态时，怎样把它变成能被 mini-sglang 调度、续写和回收的执行对象？

我们以实际接入的 **Qwen3.5-4B 文本路径**为例。先拆开网络的张量计算，再沿配置、算子、权重、状态、Engine 和 MTP 的依赖顺序解释实现。文中的「我们做了什么」依据本分支代码与实验记录；有些实现已通过局部测试，有些组合仍未通过完整验收，两者会分别标明。

本文固定模型 revision 为 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，代码观察点为 `566bf693`。后续文档提交没有改变这里分析的模型代码。维度取自[该 revision 的官方配置](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json)，计算细节以仓库中 `models/qwen3_5.py` 与 `kernel/qwen35_reference.py` 为准；下文源码路径均相对 `python/minisgl/`。

## 1. 先确定接入对象：文本主干、视觉编码器与 MTP

官方 checkpoint 的顶层架构名是 `Qwen3_5ForConditionalGeneration`，包含视觉编码器、语言模型和 MTP 参数。此次实现读取文本配置、加载文本与 MTP 权重，将架构名映射到我们自己的 `Qwen3_5ForCausalLM`。前向计算由 My_Sglang 执行，Transformers 模型用于独立正确性对照。

这个范围对应文本输入输出、BF16 主计算路径和 TP=1。视觉输入、多模态位置编码、任意长上下文都需要额外适配；官方模型的能力不能直接算作本引擎已具备的能力。[官方模型说明](https://huggingface.co/Qwen/Qwen3.5-4B)提供了网络概览及原生 MTP 使用说明。

### 1.1 32 层如何排列

4B 的文本主干是 dense 网络。每组包含三个 Gated DeltaNet 层和一个 Gated Attention 层，重复八次。按从零开始的层号，full attention 出现在 `3, 7, 11, …, 31`，其余 24 层是 GDN。每层后面都有 dense FFN，不能把家族介绍中的 MoE 概念套到这个配置上。

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="Qwen3.5-4B 文本主干与 MTP 结构，窄屏可横向滚动">
<img src="/images/my-sglang/qwen35-blocks.svg" alt="32 层文本主干由三个 GDN 和一个 Gated Attention 重复八次构成；每层包含两次 pre-norm residual，MTP 使用独立层并共享 embedding 与输出 head。" loading="lazy" />
</div>
<figcaption>图 1：本次接入的网络范围。蓝色表示主干计算，橙色表示独立 MTP 路径。MTP 使用自己的 decoder 与状态。<a href="/images/my-sglang/qwen35-blocks.svg" target="_blank" rel="noopener">查看完整 SVG</a>。</figcaption>
</figure>

为避免后续公式混淆，令 `N` 表示本次 packed forward 的总 token 数，`D` 表示 hidden size，`L` 表示一个请求已消费的 token 数。

| 参数 | 4B 配置 | 对实现的影响 |
| --- | ---: | --- |
| Hidden size `D` | 2560 | 层间残差与 embedding 的宽度 |
| 词表 | 248320 | embedding 权重 `[248320, 2560]` |
| 主干层数 | 32 | 24 层 GDN、8 层 full attention |
| Attention Q / KV heads | 16 / 4 | 分组查询，需要对应的 head 映射 |
| Attention head dimension | 256 | Q 合并后为 4096 维，再投影回 2560 |
| Rotary dimension | 64 | 每个 Q/K head 仅前四分之一旋转 |
| GDN QK / V heads | 16 / 32 | Q/K 展开到 32 个 heads |
| GDN key / value dimension | 128 / 128 | 每个 head 的循环状态为 `128×128` |
| 卷积宽度 | 4 | 每层保留原始投影的短窗口 |
| FFN intermediate size | 9216 | 两路升维，再经乘法与降维 |
| MTP trained layers | 1 | 反复使用同一预测模块展开候选 |

这里第一个容易继承错误的假设是 `head_dim = hidden_size / num_heads`。`2560 / 16 = 160`，但此模型明确配置了 `head_dim=256`。Attention 内部宽度可以与 residual stream 宽度不同，投影矩阵负责二者之间的转换。

### 1.2 两类 decoder 共用什么

`_DecoderLayer` 共用同一个 pre-norm residual 外壳，仅替换 `Mixer`：

```text
u       = x + Mixer(Norm_in(x))
x_next  = u + MLP(Norm_post(u))

Mixer ∈ {Gated DeltaNet, Gated Attention}
```

这使我们可以分别验收两个 token mixer，同时共享 FFN、残差和层间连接。`_TextModel` 顺序执行 32 层，最后执行 final norm。输出 head 与 token embedding 共享权重，`[N,2560]` 的 hidden 经同一张 `[248320,2560]` 权重表映射成 logits。

## 2. 共享部件也有数值约定：RMSNorm 与 SwiGLU

复用已有层之前，先核实参数语义。这里一般 RMSNorm 的可学习参数表示相对单位缩放的增量，实际乘数为 `1 + weight`：

```python
y = x.float()
y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
out = (y * (1.0 + weight.float())).to(x.dtype)
```

均方、倒平方根、权重加一与乘法都在 FP32 完成，最后转回输入 dtype。这个 `_Norm` 用于 decoder 的两处 norm、Attention 的 Q/K norm、文本 final norm，以及 MTP 的输入 norm。若把 checkpoint 的 `weight` 直接传给通常的 `x_normalized * weight` 实现，即使张量形状完全匹配，结果也已经改变。

FFN 使用 SwiGLU，三个无 bias 投影分别为：

```text
gate = linear(x, W_gate)                W_gate: [9216,2560]
up   = linear(x, W_up)                  W_up:   [9216,2560]
out  = linear(silu(gate) * up, W_down)   W_down: [2560,9216]
```

本文统一按 PyTorch 的 `[out_features, in_features]` 表示权重存储形状。通用路径保留三个投影；合并 gate/up、融合激活或更换 GEMM 是后续可独立比较的优化，而模型接入阶段先固定计算语义。

GDN 输出处还有另一种 gated RMSNorm，它直接乘 `weight`，并且有额外的 BF16 中间舍入。它需要单独实现，后面会具体展开。

## 3. Gated Attention：投影布局、部分 RoPE 与带历史的 mask

### 3.1 Q 与输出 gate 按 head 交织存储

输入 `x` 的形状是 `[N,2560]`。Q 投影同时产生 query 和输出门控，因此其输出宽度翻倍：

```text
q_proj(x): [N,8192] = [N,16,2×256]
k_proj(x): [N,1024] = [N,4,256]
v_proj(x): [N,1024] = [N,4,256]
```

代码先 reshape，再拆分每个 head 内部的 Q 和 gate：

```python
qg = q_proj(x).view(N, 16, 512)
q, gate = qg.chunk(2, dim=-1)  # 二者均为 [N,16,256]
```

这个顺序由 checkpoint 布局决定。直接把 `[N,8192]` 切成前后两半，会把不同 head 的 query 与 gate 混在一起。两种写法最后都能得到相同 shape，只有检查布局和数值才能发现问题。

Q/K 先经过各自的 `_Norm`，再执行 RoPE。gate 不经过 Q norm 或 RoPE，保留到 Attention 输出后使用。

### 3.2 仅旋转前 64 维

配置中的 `partial_rotary_factor=0.25` 与 `head_dim=256` 给出 `rotary_dim=64`，基数为 `10,000,000`。每个 head 拆成旋转部分和保留部分：

```text
x_rot  = x[..., :64]
x_tail = x[..., 64:]          # 后 192 维原样保留
inv_freq[i] = base^(-2i/64)   # i = 0,…,31
angle[p,i]  = position[p] * inv_freq[i]
```

`x_rot` 再按前后两半构造 `rotate_half(x_rot)`，与重复展开的 cos/sin 相乘后拼回 `x_tail`。频率和角度在 FP32 构造，cos/sin 转为输入 dtype 后参与计算。

分块输入时，第二块的位置必须从已消费长度继续。例如第一块消费 `[0,64)`，第二块应使用 `64,65,…`，不能重新从零开始。文本路径采用一维绝对位置；官方配置还包含多模态 RoPE 字段，但我们没有据此实现视觉 token 的空间或时间位置。

### 3.3 GQA 展开与输出门控

通用实现把 4 个 K/V heads 各重复 4 次，与 16 个 Q heads 对齐，然后调用 PyTorch SDPA。真实缓存仍保存 4 个 K/V heads；展开是本轮计算的中间张量。

缩放因子取完整的 `head_dim`：`256^(-1/2)=1/16`，而非 rotary dimension 的平方根。Attention 输出形状为 `[N,16,256]`，先乘 `sigmoid(gate)`，再展平为 `[N,4096]`，经 `o_proj` 返回 residual stream 的 `[N,2560]`。

这层有输出 gate，但不会因此改变历史 K/V 的定义。KV Cache 保存经过 K norm/RoPE 的 K 与投影得到的 V，gate 只作用于本轮 Attention 的输出。

### 3.4 非方形 causal mask 是续写正确性的关键

已有 `P` 个历史 token，新 chunk 有 `T` 个 token。Q 长度为 `T`，拼接后的 K/V 长度为 `P+T`。当前 chunk 的第 `i` 个 query 对应全局位置 `P+i`，允许访问的 key 满足 `j ≤ P+i`：

```python
rows = torch.arange(T, device=device)[:, None] + P
cols = torch.arange(P + T, device=device)[None, :]
mask = cols <= rows
```

例如 `P=3, T=2`，布尔 mask 应为：

```text
                K0 K1 K2 K3 K4
Q at position 3  1  1  1  1  0
Q at position 4  1  1  1  1  1
```

本实现区分三种情况：无历史的 prefill 使用 causal attention；有历史且 `T>1` 时显式构造上述偏移 mask；有历史的普通单 token decode 可以访问现有缓存全部位置。这样同一层才能同时处理首次输入、后续 chunk 与逐 token 续写。

## 4. Gated DeltaNet：从局部卷积到循环记忆

GDN 的主要历史信息保存在一个固定大小矩阵中，而非为每个过去 token 保留 K/V。理解它需要同时追踪短卷积窗口和循环状态；两者丢失任意一个，后续 token 的计算都会改变。

### 4.1 四条输入投影承担不同职责

`_GatedDeltaNet` 从 `[N,2560]` 的输入计算：

| 投影 | 输出形状 | 后续用途 |
| --- | --- | --- |
| `in_proj_qkv` | `[N,8192]` | Q/K/V 的局部卷积输入 |
| `in_proj_z` | `[N,4096]` | 最终输出的 SiLU gate |
| `in_proj_b` | `[N,32]` | 经 sigmoid 得到更新强度 β |
| `in_proj_a` | `[N,32]` | 与 A_log、dt_bias 计算衰减 |

8192 维可以拆为 `2048 + 2048 + 4096`，分别对应 `16×128` 的 Q、`16×128` 的 K，以及 `32×128` 的 V。只有 qkv 路径经过卷积，z、a、b 不经过该卷积。

### 4.2 卷积状态必须保存卷积之前的输入

qkv 转为 `[1,8192,T]`，执行宽度为 4 的 depthwise causal convolution，再应用 SiLU。depthwise 表示每个通道拥有自己的时间卷积核，通道之间不在这一步混合。

`causal_conv1d` 的接口显式接收旧窗口，并返回新窗口：

```python
mixed, next_conv = causal_conv1d(
    qkv_segment.T.unsqueeze(0),     # [1,8192,T]
    conv_weight[:, 0],             # [8192,4]
    initial_state=old_conv,         # [1,8192,4]
)
```

旧窗口保存最近 4 个 **pre-convolution inputs**，从旧到新排列。实现把它与新 qkv 拼接，做卷积后截取最后 `T` 个输出，再保存拼接序列的最后 4 个输入。按照参考缓存约定多保留一个最旧样本，并不意味着每个输出使用了 5 个输入。

若缓存的是卷积结果或 SiLU 结果，下一块就会再次对已经变换过的数值执行卷积。首次 prefill 可能正常，第二个 chunk 才出现误差，这也是单次 forward 测试不足以验收模型支持的原因。

### 4.3 GDN 展开的是 Q/K heads

卷积输出拆分后，Q/K 是 `[1,T,16,128]`，V 是 `[1,T,32,128]`。我们将每个 Q/K head 重复两次，匹配 32 个 value heads。

这里容易与上一节的 GQA 混淆：full attention 展开 KV 去匹配 Q；GDN 展开 Q/K 去匹配 V。两条路径中相同的 `repeat_interleave` 对应不同的 head 关系，不能共用未经检查的 reshape 规则。

### 4.4 先遗忘，再纠正，再读取

每个 head 的状态 `S` 形状为 `[128,128]`，行轴是 key dimension，列轴是 value dimension。门控参数为：

```text
β_t = sigmoid(b_t)
g_t = -exp(A_log) * softplus(a_t + dt_bias)
α_t = exp(g_t)
```

`g` 是 log-decay，真正乘到历史状态上的衰减是 `exp(g)`。代码以 FP32 计算 `A_log.exp()`、`a.float()` 和 softplus 路径；β 的 sigmoid 先使用投影结果的 dtype，进入递推时再转 FP32。因此准确的说法是循环累计使用 FP32，而不是整层 GDN 的所有操作均使用 FP32。

Q/K 先做 L2 normalization，query 额外乘 `128^(-1/2)`。令归一化后的 key 为 `k̂_t`，额外缩放后的 query 为 `q̄_t`，每个 head 的更新为：

```text
S_decay = α_t · S_previous
memory  = k̂_tᵀ · S_decay                  # [128]，对 key 轴归约
delta   = β_t · (v_t - memory)             # [128]
S_next  = S_decay + outer(k̂_t, delta)     # [128,128]
y_t     = q̄_tᵀ · S_next                  # [128]
```

这组顺序给出了 delta rule 的具体含义：先按衰减率保留历史，再查询当前 key 在记忆中对应的 value，用真实 value 与记忆的差修正状态，最后从更新后的状态读取输出。把 memory 改成从未衰减的旧状态计算，或让输出读取更新前的状态，都会改变模型。

通用算子对应的核心代码很短：

```python
state = state * g[:, t].exp()[..., None, None]
memory = (state * k_t.unsqueeze(-1)).sum(-2)
delta = (v_t - memory) * beta[:, t].unsqueeze(-1)
state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
output[:, t] = (state * q_t.unsqueeze(-1)).sum(-2)
```

这里 state、memory、delta 与 output buffer 都是 FP32，返回输出时再转回 query dtype。Q/K 的 L2 normalization 则遵循参考实现，在输入 dtype 中完成相应张量运算后才转 FP32；本文的数学式描述依赖关系，实际数值对齐还必须保留这个转换顺序。

### 4.5 Gated RMSNorm 的一次中间舍入也属于模型语义

递推输出与 z 都整理成 `[N,32,128]`，按每个 value head 的 128 维执行 gated norm：

```python
dtype = y.dtype
normalized = y.float()
normalized *= torch.rsqrt(normalized.square().mean(-1, keepdim=True) + eps)
weighted = weight * normalized.to(dtype)
out = (weighted * F.silu(z.float())).to(dtype)
```

与第二节的一般 norm 相比，这里有两点变化：直接乘 `weight`，且归一化结果在乘权重之前先转回输入 dtype。此 checkpoint 的 gated norm weight 本身为 FP32，乘法后会进入 FP32，再乘 FP32 的 SiLU gate，最后转回 BF16。

把所有操作连续保持在 FP32、仅在出口做一次 BF16 cast，可能在数学上更接近实数结果，却不再复现参考路径的舍入。接入阶段我们先保留这种顺序，后续融合算子也以它作为对照。最后将 `[N,32,128]` 展平成 `[N,4096]`，通过 `out_proj` 回到 2560 维。

## 5. MTP 的网络组成：带一位输入偏移的独立 decoder

官方已经提供 MTP 权重，本次工作不涉及训练预测头。本地 `_MTPModel` 包含两个输入 norm、一个 `5120→2560` 的 FC、一层 full attention decoder、一个 final norm，并复用主干的 embedding 和输出 head。

令 `x[t+1]` 是下一个位置的 token，`h[t]` 是上一个位置的 hidden：

```text
embedding = shared_embedding(x[t+1])
joined    = concat(Norm_embedding(embedding), Norm_hidden(h[t]))
u         = linear(joined, W_fc)             W_fc: [2560,5120]
draft_h   = MTP_final_norm(MTP_decoder(u))
logits    = linear(draft_h, shared_embedding_weight)
```

拼接顺序是 embedding 在前、hidden 在后。MTP 层虽然在其局部列表中编号为 0，但使用独立 `Qwen35State` 保存 Attention KV，不会写入主干第 0 层的状态。

当前控制器传入的是主干 **final norm 之后的 `hidden_states`**。模型也返回 `hidden_states_before_norm`，但这条 MTP 路径没有使用它。这里明确记录实现事实；对官方完整 MTP 数值行为的对齐，仍受后文回归结果约束。

单层 MTP 可以重复调用，第一次用 target hidden，后续候选展开使用前一次的 draft hidden。配置中的“一层训练模块”与运行时“候选生成几步”因而是两个参数。真正把这些候选变成可靠输出，还需要目标验证和状态提交，第十节继续展开。

## 6. 第一步接入：让配置与注册准确描述模型

理解网络之后，第一项代码改动是把结构信息带进引擎。`e8c60a6` 为 `ModelConfig` 增加混合层、GDN 和 MTP 字段，并读取 partial RoPE。这个阶段先核对已有解析，再补齐模型信息：

1. **沿用并核对 `text_config` 展开。** hidden size、层布局等来自嵌套文本配置，已有解析保留顶层 architecture 信息，用于模型注册。
2. **沿用并核对显式 `head_dim`。** 已有逻辑仅在缺失时回退到常见的除法规则，此模型依赖显式值 256。
3. **携带完整 `layer_types`。** 模型按配置逐层构建 GDN 或 full attention，不在前向里猜测层类型。
4. **分开 rotary dimension 与 head dimension。** 读取 `rope_parameters` 的 base 与 partial factor，而非默认旋转整个 head。
5. **携带 GDN 的四个 head 参数与卷积宽度。** 这些参数决定投影、状态形状和后续内存预算。

`tests/core/test_qwen35_config.py` 同时检查混合配置与已有 dense 配置：前者应得到 rotary dimension 64、正确的 MTP/GDN 字段，后者仍保留原有 RoPE 行为。模型注册随后在集成提交中加入，将 checkpoint 的 conditional-generation 名称连接到文本实现。

这一层适配很薄，但它规定了后续每个矩阵和缓存的形状。配置解析错误应当在这里暴露，而不是等加载数 GB 权重后才靠矩阵乘法报错发现。

## 7. 第二步接入：先建立通用 GPU 算子与对照接口

`4f501ec` 引入 `qwen35_reference.py` 和相应微基准、profiler 工具。第一版使用 PyTorch 张量算子，按输入所在设备执行；在 5090 实验中输入和计算都在 CUDA 上。通用路径没有依赖 SM120 特定指令，也没有预先把所有操作融合成一个 kernel。

卷积与递推显式接收旧状态、返回新状态；gated norm 是无状态变换。三个接口分别为：

```text
causal_conv1d(x, weight, initial_state)      -> output, next_conv
recurrent_gated_delta_rule(..., initial_state) -> output, next_recurrent
rms_norm_gated(output, z, weight)            -> gated_output
```

卷积和递推不会原地覆盖调用者传入的旧状态。递推从 `initial_state.float().clone()` 开始；卷积的新窗口也拥有独立存储。这让测试能比较旧状态是否变化，也为后续投机分支的丢弃提供基础。模型层负责把返回状态写回请求；算子本身不认识 UID、调度器或 HTTP 请求。

测试先覆盖 shape、dtype、初始状态、连续执行以及分块等价关系，再与独立参考递推比较。实际留存的 GPU 通用算子测试为 **30/30 通过**；另有真实 GDN 形状的 8 组比较，覆盖 `B∈{1,4}`、`T∈{1,16}`、FP32/BF16，head 数 32、key/value dimension 128，记录中输出与最终 state 的最大绝对误差均为 0。

这些数字只描述该算子集合和这些输入，不意味着整个模型在所有 batch 下 bitwise invariant。开发时 CPU BF16 分块测试曾遇到向量化 `rsqrt` 随调用形状变化产生的舍入差异。为单独检查递推连续性，相应测试对共享输入预先归一化，避免把归一化形状差异与状态递推错误混在一起；完整算子的数值测试仍单独保留。

从这里开始才有清楚的优化基线：输入、旧状态和输出契约固定，profiler 可以把开销定位到投影、卷积、归约、逐元素运算或状态复制。后续 GDN 融合和 SM120 tile/warp 实验属于执行优化，不能回写成第一版模型已经具备的能力。

## 8. 第三步接入：严格加载 441 个文本与 MTP 权重键

配置和局部算子就绪后，`0e3bbf4` 集成了文本模型、MTP 与请求生命周期等改动。这个提交包含多个文件；下文按依赖关系拆解其实现，不把每个讲解步骤虚构成独立 Git 提交。

### 8.1 先建立参数外壳，再接入 checkpoint

模型继承项目已有的 `BaseOP`，通过对象层级形成参数名，例如 `model.layers.3.self_attn.q_proj.weight`。加载脚本先在 meta device 上建立参数外壳，获得预期 shape/dtype，再把真实权重装入，避免先分配一份随机初始化的完整 GPU 模型。

hybrid 分支的权重映射为：

| Checkpoint 名称 | 本地处理 |
| --- | --- |
| `model.language_model.*` | 改为 `model.*` |
| `model.visual.*` | 按文本支持范围跳过 |
| `mtp.*` | 保留，加载到独立 MTP 模块 |
| 独立 `lm_head.weight` | tied 配置不创建该参数 |

该分支要求 TP=1，并绕开原有面向其他模型的 QKV 合并与张量切分流程。这样 Q/gate 的布局、GDN 投影名称与形状能直接对照 checkpoint。`BaseOP.load_state_dict` 消费每个预期键，检查 shape/dtype，并在末尾拒绝未消费的键。

### 8.2 首次失败来自 FP32 参数

最初严格加载遇到 dtype 不匹配：checkpoint 中的 `A_log` 和 `linear_attn.norm.weight` 是 FP32，而最初的参数声明随默认 dtype 成为了 BF16。

修复是让 `_GatedDeltaNet.A_log` 与 `_GatedNorm.weight` 显式声明 FP32，保持它们的数值语义。当前 Engine 加载时按每个预期参数的 dtype 转换；独立基线加载脚本直接用 checkpoint dtype 做严格检查。两条入口都需要保留这些 FP32 声明，不能通过整模型统一转 BF16 来绕过问题。

### 8.3 441 个键如何反推

加载成功不仅可以看一个 count，还可以从结构反算：

| 来源 | 每份键数 | 份数 | 合计 |
| --- | ---: | ---: | ---: |
| GDN decoder | 14 | 24 | 336 |
| Full attention decoder | 11 | 8 | 88 |
| 文本 embedding 与 final norm | — | — | 2 |
| 单层 MTP 模块 | — | — | 15 |
| **总计** | | | **441** |

一个 GDN decoder 包含 qkv/z/a/b 四个投影、卷积、A_log、dt_bias、gated norm、out projection，再加两处 decoder norm 和三个 FFN 投影，共 14 个键。一个 full attention decoder 有 q/k/v/o、q_norm/k_norm，再加两处 decoder norm 与三个 FFN 投影，共 11 个键。MTP 则在一个 11-key decoder 外增加 FC、两个输入 norm 和 final norm，共 15 个键。

这与首轮记录中的 441 相符，但键数本身不能证明数值正确；真正的结构检查仍是每个键的名称、shape、dtype 和共享关系。这里的完整性指所声明的文本与 MTP 范围，视觉权重没有被纳入。

## 9. 第四步接入：让模型跨 chunk、跨轮次、跨请求继续执行

能对完整 prompt 算一次 logits，只完成了无状态接口的一部分。推理引擎会反复调用模型，每次只提供新增 token；因此模型必须明确“当前状态已经消费到哪里”。

### 9.1 一个请求有三类持久状态

`Qwen35State` 的核心字段为：

```python
length: int
position_offset: int
kv: dict[layer_id, tuple[K, V]]
conv: dict[layer_id, conv_window]
recurrent: dict[layer_id, recurrent_matrix]
```

普通文本主干的 `position_offset=0`。单请求、BF16 权重路径下，三类张量分别为：

| 状态 | 每层形状 | 层数与增长方式 |
| --- | --- | --- |
| Attention K、V | 各 `[L,4,256]`，BF16 | 8 层，随已消费长度增长 |
| 卷积窗口 | `[1,8192,4]`，BF16 | 24 层，固定窗口 |
| GDN recurrent | `[1,32,128,128]`，FP32 | 24 层，固定矩阵 |

由此可以计算主干基础状态的理论字节数。每层 recurrent 为 2 MiB，24 层共 48 MiB；卷积窗口共 1.5 MiB；8 层 K/V 每个历史 token 合计 32 KiB：

```text
主干状态 ≈ 49.5 MiB + L × 32 KiB
L = 8192 时：49.5 MiB + 256 MiB = 305.5 MiB / 请求
```

这是按张量形状计算的基础存储量，未计入权重、MTP、验证副本、动态拼接临时张量、Graph staging 与 allocator。它说明并发准入还需要预留每请求的固定 GDN 状态，不能只用每 token KV 字节数推算容量。

### 9.2 Chunked Prefill 只改变本轮消费区间

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="分块输入与逐 token 续写的混合状态连续性，窄屏可横向滚动">
<img src="/images/my-sglang/qwen35-state-continuation.svg" alt="同一请求依次消费两个 prefill chunk 和一个 decode token，KV 追加、卷积窗口滚动、GDN 状态继续递推；槽位复用由请求 UID 与长度检查保护。" loading="lazy" />
</div>
<figcaption>图 2：状态长度等于已经被模型消费的前缀长度。每个阶段都需要延续 KV、卷积与 recurrent 三类状态。区间是机制示例。<a href="/images/my-sglang/qwen35-state-continuation.svg" target="_blank" rel="noopener">查看完整 SVG</a>。</figcaption>
</figure>

第一块输入结束后，Attention 保存这段前缀的 K/V；卷积留下末尾窗口；GDN 保存最后的 recurrent matrix。第二块携带同一个 state，从 `state.length + position_offset` 生成位置：Attention 读取过去的 K/V，卷积读取过去的原始投影窗口，GDN 从过去的矩阵继续更新。所有层完成本轮计算后，`state.length` 增加本轮 token 数。

普通 decode 则是同样接口下长度为 1 的输入。首个输出 token 由 prompt 最后位置的 logits 选出，此时它已生成，但尚未进入模型状态。下一轮消费它之后，状态长度才增加。因此“已经输出多少 token”和“状态已经消费多少 token”不能混为一个计数。

### 9.3 Packed forward 共用计算，但保持请求状态隔离

`forward_packed` 接收一维 token 数组与分段长度。例如一个 3-token prefill chunk 与一个单 token decode 可以表示为：

```text
input_ids = [a0, a1, a2, b0]
lengths   = [3, 1]
states    = [state_A, state_B]
positions = [L_A, L_A+1, L_A+2, L_B]
```

embedding、norm、线性投影与 FFN 在总 token 轴上一起执行；Attention 和 GDN 在各层内部按分段分别读取、推进对应 state，再把输出拼回原顺序。请求 A 的末尾不会成为请求 B 的卷积历史，B 也不会看到 A 的 KV。

入口检查 token 数是否等于 `sum(lengths)`、每段是否非空、state 数量是否一致，并拒绝两个请求共用同一个可变 state 对象。默认输出每段最后位置的 logits，供普通采样使用；MTP 验证可以要求返回所有位置的 logits。

因此当前实现已经具有变长 packed 计算接口，但 Attention/GDN 仍按请求执行。多请求进入一次模型调用，不代表其全部计算已合并成一个 GPU kernel。

### 9.4 接入 mini 的 Request 与 Engine

普通路径由 Scheduler 调用 `Engine.forward_batch`，再进入模型的 `forward()`；模型从当前 `Context.batch` 取得请求及输入。每个请求通过 `_request_states[table_idx] = (uid, state)` 绑定混合状态：

```text
req.cached_len == 0       → 为新请求建立 state
保存的 uid != req.uid    → 拒绝使用旧槽位状态
state.length != cached_len → 报告模型与调度进度不一致
req.extend_len           → 本轮 packed segment 的长度
```

`table_idx` 可以被复用，UID 才能区分新旧请求。请求结束或取消时，释放路径删除对应模型状态，同时处理 MTP session 等附属对象。若存在异步在途执行，归还槽位之前还必须建立 stream 依赖；仅从 Python 字典里删一个对象并不足以保证槽位安全。

Engine 为该模型选择 `TorchHybridBackend`，使用模型内部的 PyTorch SDPA 路径。它的 `kv_cache=None`；真实 KV 由模型动态 `torch.cat` 追加，调度层保留页表与页预算做容量管理。这是明确的首版实现边界：普通模型的物理 Paged KV 池并没有被直接用于这条混合模型路径。

当前接入路径要求 `page_size=1`、TP=1，并使用 naive prefix cache。一个可恢复的混合模型前缀必须同时包含 KV、卷积窗口与 GDN state，只恢复 KV 会改变后续结果。普通 decode Graph 也没有直接继承；后来加入的短上下文 Graph 是独立、默认关闭的实验入口。

## 10. 第五步接入：把 MTP 模块变成可提交的推理流程

MTP 网络具备前向能力后，还要解决输入对齐、候选验证与状态恢复。`engine/speculative.py` 的 `GreedyMTPController` 管理这些计算，`scheduler/mtp.py` 的 `MTPBatchHandler` 将其接回在线调度与 token pool。

### 10.1 Prefill 时先建立一位偏移的 MTP 历史

对长度为 `n` 的 prompt，target 先产生各位置 hidden。MTP 用后移一位的 token 配对前一个位置的 hidden：

```python
mtp_input = input_ids[1:]
previous_hidden = target.hidden_states[:-1]
mtp_state = Qwen35State(position_offset=1)
```

这样主干消费了 `n` 个 token，MTP 消费了 `n−1` 对输入，MTP 的绝对位置从 1 开始。会话保存最后一个 target hidden，以及由 prompt 末尾 logits 选出的 `pending`。pending 已经输出，尚未由 target 消费。

分块 prefill 还存在跨块的配对：新块的第一个 token 需要上一块最后一个 target hidden。调度适配层保留 priming 状态，把这个边界补上，而不能对每块分别执行 `tokens[1:]` 后丢掉边界 token。

### 10.2 一个候选轮次做了哪些 forward

假设已提交 target state 消费了 `c` 个 token，pending 为 `p`，本轮展开三个候选 `d1,d2,d3`：

1. **候选生成。** 克隆 MTP state，用 `(p, last_target_hidden)` 预测 d1，再用前一步 draft hidden 继续预测 d2、d3。原会话的状态保持原样。
2. **目标验证。** 克隆 target state，一次消费 `[p,d1,d2,d3]`，取四个位置的目标 logits。第一个位置的预测与 d1 比较，依次找出连续匹配前缀。
3. **恢复 target。** 若只接受 d1、d2，验证分支还消费了 d3，不能提交。当前实现从旧 target state 重放 `[p,d1,d2]`，得到长度为 `c+3` 的正确前缀状态。
4. **恢复 MTP。** 从旧 MTP state 出发，以真实 target hidden 对 `[p,d1,d2]` 做 teacher forcing，得到对应的新 MTP KV。
5. **发布。** 输出 `[d1,d2,bonus]`，其中 bonus 来自实际提交路径的最后一个 target logits。bonus 成为下一轮 pending。

GDN 只有最后的 recurrent matrix，无法像一段 token 列表那样简单截去末尾。首版采用 clone 与重放，避免假定存在尚未实现的逆向状态恢复算子。它有额外开销，但提供了可检查的提交语义。

MTP 的 free-running draft hidden 与 target hidden 不相同。因此，即使候选全部被接受，代码仍用 target hidden 推进待提交的 MTP 状态，没有直接把候选展开的 cache 作为最终状态。

### 10.3 状态长度不变量与拒绝分支修复

每轮完成后保持：

```text
mtp_state.length = target_state.length - 1
pending          = 已输出、但 target 尚未消费的最后一个 token
last_hidden      = target 已消费前缀的最后一个 hidden
```

一个 batch 中请求的接受长度可以不同。控制器先计算各请求的新状态和返回结果，完成检查及待发布 hidden 的分配，再更新 session 引用，避免处理到一半就提交部分请求。

后续审查发现拒绝重放需要同步携带重放后的 hidden 和 bonus。`49d8eeb` 修复这条内部一致性路径，使下一轮使用实际提交状态对应的 hidden/logits。它解决的是一个具体状态关联问题，不能据此宣称早先的整个 greedy 回归已经恢复通过；修改后的完整 GPU 对照仍需另测。

### 10.4 在线接口也要知道一轮可能输出多个 token

调度适配层为候选验证预留位置，提交后释放只属于拒绝后缀的预留量，并更新 token pool 与请求长度。服务输出从单 token 扩展成 token 列表，EOS 与长度预算需要在列表内部截断。usage 累加实际 token 数，不能按 SSE 消息数计量。

当前 MTP decode 直接调用 controller 的模型路径，绕过普通 `Engine.forward_batch` 与 Sampler。首版为固定步数 greedy；MTP 与 mixed/Graph 组合被拒绝，overlap 关闭。这些约束让模型接入的可运行范围明确，也标出了后续统一执行接口的工作。

## 11. 我们如何验证，以及哪些问题仍然存在

模型适配的验证需要逐层扩大范围。参数加载成功检查结构；单算子对齐检查局部数值；连续执行检查状态；真实在线回归才会暴露 batch 形状、请求到达与提交路径的组合问题。

### 11.1 按证据强度阅读已有结果

以下均为此前 5090 实验留存，本文写作期间没有重新执行 GPU 测试。结果路径相对 `docs/experiments/qwen35-sm120/results/`。

| 阶段 | 实际证据 | 可以支持的判断 |
| --- | --- | --- |
| 通用算子 | `01-generic/gpu-tests.log`：30/30；`01-generic/gdn-hf-smoke.json`：8 组输出/state 最大误差 0 | 指定算子用例通过 |
| 首次模型 smoke | `02-model/smoke/model.json`：441 键；32 输入、16 输出，3 轮相同；固定聊天前 16 token 与 HF 相同 | 该模型路径可加载、可连续生成，单个独立参考用例对齐 |
| 真正跨块的输入 | `03-gated-norm/model-reference/model.json`：512 输入、128 输出、3 轮；64-token chunk 的前 16 输出与完整输入相同 | 该输入的 8 块续算与非分块前缀输出一致 |
| 早期 MTP 小集 | `04-mtp/mtp.json`：B=1/3，普通/k=1/k=3，每配置 3 轮，共 18 条 batch-run，输出 32 token | 其中 12 条 MTP batch-run 与普通输出一致；使用强制长度生成 |
| HTTP 与回收 | `07-http/`：普通/MTP 各 81/81 请求成功；各 6/6 运行中取消有后端释放记录 | 该轮协议、计数与回收用例通过 |
| 在线严格 greedy | `07-http/mtp1/speed/comparison-vs-ordinary.json`：55/81 一致，26 条不同 | **完整普通/MTP 对齐未通过** |

首轮 `32-token` smoke 中也记录了 `chunked_greedy_equal=true`，但脚本 chunk size 为 64，输入没有真正跨块。因此本文没有把它用作 chunk 边界正确性的证据；实际跨块检查采用后来 512-token 输入的记录。验收不仅要读 JSON 的布尔字段，还要检查测试参数是否触发了目标行为。

早期 MTP 18 条记录使用固定输出长度、忽略 EOS。它适合检查执行过程，却不能代替真实在线结束行为或任务质量评估。HTTP 阶段另行检查了 EOS、usage、取消与资源回收；成功返回 HTTP 响应也不等于 token 序列已与普通基线一致。

### 11.2 为什么单请求对齐仍不足够

更大的在线子集中，普通模式自身也出现跨 batch 的 greedy 差异。这让问题不能仅靠“投机路径有没有漏输出”来判断。不同 batch 和验证长度会改变矩阵计算形状，可能触发不同浮点归约路径；微小 logits 差异在接近的候选之间可能改变 argmax，随后自回归轨迹继续分离。

定位这类问题应固定同一 token 前缀，逐层比较 hidden/logits，而不是让两条已分叉的生成序列继续滚动比较。我们增加了 teacher-forced 数值诊断、不同 batch 形状对照，以及默认关闭的固定归约 BF16 linear 路径（`89316b3`）。这些是排查与实验入口；原始失败记录保留，后续受干扰的 GPU timing 没有纳入结论。

因此当前可以说：文本主干、混合状态、packed forward 和原生 MTP 控制流程已经接入，部分算子、模型与服务用例有真实证据；**不能说全范围、全组合的 Qwen3.5/MTP 支持已完成验收，也不能把未对齐轮次中的吞吐变化列为已验收收益。**

### 11.3 优化接在这个基线之后

我们按通用算子、profile、融合、硬件配置比较的顺序推进。GDN recurrence 的可选融合在 `ce8acc4` 加入，SM120 tile/warp 独立比较由 `a2d3f20` 留存，Graph 和数值路径有各自开关与限制。

模型接入阶段建立的显式 state、可切换算子与固定回归，使后续可以回答“哪部分变快、改变了什么数值路径、对端到端有多少贡献”。具体 profiler 时间线、融合方案与性能正负结果留到执行优化专题；这里不把参考模型实现与后续优化成果合并叙述。

## 12. 从源码与实验记录复现这条接入路线

按依赖关系阅读代码，可以把整条路径压缩成下面的索引：

| 要核对的内容 | 主要入口 |
| --- | --- |
| 模型配置与 architecture 分派 | `models/config.py`、`models/register.py` |
| checkpoint 映射与严格加载 | `models/weight.py`、`layers/base.py` |
| Attention/GDN/MLP/MTP 网络 | `models/qwen3_5.py` |
| 三个通用算子的数值顺序 | `kernel/qwen35_reference.py` |
| 模型状态与 packed contract | `Qwen35State`、`Qwen3_5ForCausalLM.forward_packed` |
| backend、容量与普通执行 | `attention/torch_hybrid.py`、`engine/engine.py` |
| MTP 候选与提交 | `engine/speculative.py`、`scheduler/mtp.py` |
| 结束、取消与回收 | `scheduler/scheduler.py`、`scheduler/cache.py`、`tokenizer/server.py` |

Git 中几个关键节点如下，表示实际留存粒度：

```text
e8c60a6  配置：混合层、GDN 参数与 partial RoPE
4f501ec  通用算子、测试与 profiler 微基准
0e3bbf4  文本模型、原生 greedy MTP 与请求生命周期集成
2b554b9  混合模型准入与投机资源释放修复
49d8eeb  拒绝重放后的 target hidden/logits 一致性
89316b3  可选固定归约 BF16 linear 与严格回归入口
566bf69  留存审计及 trace 归档观察点
```

例如，`git show e8c60a6 -- python/minisgl/models/config.py` 可以查看第一步配置适配，`git show 4f501ec --stat` 可以查看通用算子阶段新增了哪些文件。运行实验时还要核对结果中的源码 SHA；早期实验曾在尚未全部提交的工作树中执行，只凭记录的 base HEAD 不能完整重建当时执行代码。

现有模型基线脚本在 `benchmarks/qwen35/model_baseline.py`。在按已记录的软件版本准备的独立 CUDA venv 中，使用固定 revision 的本地模型快照，可以重新生成一组结果：

```bash
# 位于 My_Sglang 仓库根目录；所有 Python 依赖均来自实验 venv。
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
  --check-hf --profile --output artifacts/qwen35-model-rerun
```

脚本对测量形状预热，分别记录 prefill、decode 与整体耗时，另做 64-token 分块检查；`--check-hf` 使用独立 HF 模型验证固定聊天用例，`--profile` 记录一轮 native decode。它不是完整验收套件，固定模型、开关、输入与干净 GPU 环境仍是比较前提；新结果应进入新目录。

完整实验轨迹保存在仓库的 `docs/experiments/qwen35-sm120/实验记录.md`，权重及环境清单、原始 JSON、HTTP 阶段验收和数值分歧诊断与其同目录维护。本文沿这些记录说明模型接入的具体机制；后续工作的起点，是补齐尚未通过的组合回归，再在同一基线上评价性能改动。
