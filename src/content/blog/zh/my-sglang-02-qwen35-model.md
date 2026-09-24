---
title: "My_Sglang（二）：从读懂模型到 Day 0 支持，再到逐步优化"
description: "复盘 Qwen3.5 接入 My_Sglang 的实际过程：先读配置、权重与参考实现，建立通用算子和最小推理闭环，再接入请求状态、MTP，依据 profile 逐项融合并验证收益。"
date: 2026-09-24T00:00:00Z
lang: zh
translationKey: my-sglang-02-qwen35-model
tags: [LLM 推理, 模型适配, Qwen3.5, My_Sglang]
draft: false
---

给一个推理框架增加新模型支持，入口通常很小：在注册表里加一个类名。但真正开始做的时候，很快就会遇到更具体的问题：权重能否直接映射到已有层，模型需要保存哪些历史状态，第二次 forward 从哪里继续，以及框架原来的调度与缓存假设是否仍然成立。

这篇文章复盘我们如何把 **Qwen3.5-4B 的文本路径接入 My_Sglang**。讲解顺序沿着实现依赖推进：先确定差异，写出通用计算，跑通最小生成闭环，再让框架管理请求，最后从测量结果决定下一项优化。模型结构会在影响实现决策时展开。

文中的 **Day 0** 仅指「普通文本推理的最小原生离线闭环」：主干权重能够严格加载，普通 prefill/decode 能连续执行，固定 greedy 用例与独立参考对齐。**MTP 不属于 Day 0 的功能或验收条件**；候选生成、目标验证和状态提交是普通推理基线建立后的独立扩展。Day 0 是本文给一个里程碑取的名字，不表示发布当天完成适配，也不代表全部功能已经验收。实际 Git 中，模型与 Engine、调度、MTP 的初版曾一起提交；提交被合在一起，不等于这些能力属于同一验收阶段。

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="从模型审计到 Day 0 与逐项优化的接入路线，窄屏可横向滚动">
<img src="/images/my-sglang/qwen35-support-roadmap.svg" alt="配置、权重和参考代码形成差异清单，经过通用算子、Day 0 最小推理、请求状态与 MTP 接入，再进入 profile、单项修改、正确性和性能对照的循环。" loading="lazy" />
</div>
<figcaption>图 1：本文沿着这条接入路线展开。箭头表示实现依赖，不表示各步骤的实际耗时。所有机制图由项目内容重画，绘画风格参考 <a href="https://vllm.ai/blog/2025-09-05-anatomy-of-vllm">Inside vLLM</a> 的白底手绘线条、彩色描边与短标签。<a href="/images/my-sglang/qwen35-support-roadmap.svg" target="_blank" rel="noopener">查看完整 SVG</a>。</figcaption>
</figure>

本文基于代码观察点 `566bf693`，固定模型 revision 为 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。下文代码路径相对 `python/minisgl/`，实验文件相对 `docs/experiments/qwen35-sm120/results/`。测量都是此前 5090 实验的留存，本轮写作没有重新执行 GPU benchmark。

## 1. 接到模型以后，先读什么

我先把检查对象分成三份：**配置描述结构，checkpoint 描述实际参数，参考实现描述计算顺序。** 三者结合，才能判断哪些地方可以沿用框架，哪些地方必须补。

### 1.1 先从 config 建立一张差异清单

第一份是[固定 revision 的官方 config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json)。顶层 architecture 是 `Qwen3_5ForConditionalGeneration`，真正的语言模型参数放在 `text_config` 中。看到 `vision_config` 后，我们先明确文本支持范围，避免把视觉编码、图像 token 与多模态位置编码一起卷入首轮适配。

沿着 `text_config` 看下去，最影响实现的并不是参数量，而是以下差异：

| 读到的配置 | 对接入的直接影响 |
| --- | --- |
| `layer_types`：三个 linear attention 后接一个 full attention，重复八次 | 32 层不能全部实例化成同一种 Attention；其中 24 层需要 GDN 状态 |
| `hidden_size=2560`、`head_dim=256`、Q heads=16 | Attention 内部宽度为 4096，不能用 `2560/16=160` 推算 head dimension |
| `partial_rotary_factor=0.25` | Q/K 每个 head 只旋转前 64 维，后 192 维保留 |
| GDN QK heads=16、V heads=32，维度均为 128，卷积宽度=4 | 新增 QK 分组展开、短卷积窗口和 FP32 循环矩阵 |
| `tie_word_embeddings=true` | 输出 head 复用 `[248320,2560]` 的 embedding 权重 |
| `mtp_num_hidden_layers=1` | checkpoint 自带预测模块；后续可以接原生 MTP，无需训练新预测头 |

这个 4B 配置是 dense 网络。每层的 FFN 都是 `2560 → 9216 → 2560` 的 SwiGLU；它不需要专家路由。主干包含 24 个 GDN mixer 与 8 个 gated full-attention mixer，二者共用 pre-norm residual 和 FFN 外壳。

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="本次模型接入需要处理的网络差异，窄屏可横向滚动">
<img src="/images/my-sglang/qwen35-blocks.svg" alt="Qwen3.5-4B 的共享 embedding、32 层混合主干和 tied head；24 层 GDN 保存卷积及循环状态，8 层 gated attention 保存 KV，MTP 使用独立预测层和状态。" loading="lazy" />
</div>
<figcaption>图 2：审计配置后得到的模型地图。这里标出的是接入需要处理的差异，模型能力范围仍以文本路径为限。<a href="/images/my-sglang/qwen35-blocks.svg" target="_blank" rel="noopener">查看完整 SVG</a>。</figcaption>
</figure>

### 1.2 再看 checkpoint，验证配置在权重里长什么样

第二份是 safetensors 中的参数名、shape 与 dtype。此时要核对的是：文本权重位于哪个前缀，Q 投影是否带输出 gate，GDN 的投影和状态参数叫什么，MTP 参数是否真的存在，以及 tied head 是否还有一份独立权重。

我们最终采用的映射是 `model.language_model.* → model.*`，保留 `mtp.*`，按文本支持范围跳过 `model.visual.*`。hybrid 分支限定 TP=1，先绕过已有加载器面向其他模型的 QKV 合并与分片逻辑，让本地对象名称与 checkpoint 直接对应。

这一步还应记录 dtype。后面首次严格加载恰恰因为漏掉两类 FP32 参数而失败；shape 对了，不代表参数声明已经完整。

### 1.3 最后沿参考 forward 追踪状态和舍入

第三份是 Qwen/Transformers 的参考计算。我们重点核对调用顺序，而不是只抄模块类名：

- Attention 的 Q 与 gate 如何排列，Q/K norm 在 RoPE 前还是后。
- causal conv 保存输入还是输出，分块后要保留多宽的窗口。
- GDN 先衰减还是先更新，输出读取旧状态还是新状态。
- 哪些计算转 FP32，哪些位置必须先舍入回 BF16。
- MTP 的 token 与 hidden 是否错开一位，使用哪份 hidden。

对应到 mini-sglang，要回头阅读 `ModelConfig`、模型注册、`BaseOP` 权重加载、`Engine.forward_batch` 和 `Req.cached_len`。此时得到的接入清单是：补混合配置、写两种 mixer、加载真实参数、定义完整请求状态，再连接 Engine。已有的 HTTP 与调度骨架可以沿用，但新的状态生命周期需要明确接上。

### 1.4 在独立环境中固定这个起点

实验从上游 `20fcd7f` 建立新分支和独立 worktree，避开远端已有的未提交修改。GPU 实测为 RTX 5090、SM120。环境使用专用 venv；系统 Python 缺少 ensurepip 时，改用已有 uv 创建隔离环境，关闭 system-site-packages。记录软件版本与模型哈希，后续每次更换算子都使用同一模型快照。

这一阶段还修正了 SM120 与 SM100 的能力判断：同属 Blackwell 不能自动选择同一后端。我们先使用通用 PyTorch CUDA 路径，使第一轮模型支持不依赖某个专用 kernel 恰好能运行。

## 2. 开始写代码：先补配置，再建立可对照的通用计算

### 2.1 配置改动保持薄，先把结构传进来

`e8c60a6` 扩展 `ModelConfig`，增加 `layer_types`、GDN heads/dimensions/conv width、MTP 层数，并读取 partial RoPE。嵌套 `text_config` 展开与显式 `head_dim` 的处理已经存在，这里沿用并核对，新增贡献是把混合模型缺失的信息继续传到构造函数。

配置测试同时覆盖新模型和已有 dense 路径：新模型应得到 64 维 RoPE、正确的层类型与 GDN 参数；已有模型仍保留原来的 RoPE 配置。这样后续 shape 错误可以尽早定位到配置或层构造，避免全部堆到真实权重加载阶段。

### 2.2 先写 reference 算子，并明确谁拥有状态

`4f501ec` 加入三个通用算子。卷积与递推显式接收旧状态并返回新状态，gated norm 则是无状态变换：

```text
causal_conv1d(x, weight, initial_state)        → output, next_conv
recurrent_gated_delta_rule(..., initial_state) → output, next_recurrent
rms_norm_gated(output, z, weight)              → gated_output
```

它们使用 PyTorch 张量运算，在输入所在设备执行。GPU 实验中，输入、状态和计算均在 CUDA 上；初版没有硬件特化和自动调优。

我们让算子不原地覆盖传入的旧状态。GDN 从旧 state 的 clone 开始，卷积返回独立的新窗口。模型层决定何时替换请求持有的状态；算子不需要知道请求 UID。这既便于检查连续性，也让后续 MTP 能丢弃临时分支。

### 2.3 写 GDN 时，先把递推翻译成可以逐行对照的代码

GDN 的输入由四路投影产生：qkv 为 `[N,8192]`，z 为 `[N,4096]`，a/b 各为 `[N,32]`，其中 N 是本轮总 token 数。qkv 经过宽度 4 的 depthwise causal conv 和 SiLU，再拆成 Q、K、V；z 与 a/b 不经过卷积。

卷积缓存保存最近四个**卷积之前的投影输入**。计算下一块时把旧窗口与当前输入拼接，执行卷积并取当前块对应的最后 T 个输出，再保存末尾四个原始输入。若保存成 SiLU 后的结果，第一块可能正常，第二块就会对错误的历史继续卷积。

Q/K 原有 16 个 heads，各重复两份，与 V 的 32 个 heads 对齐。每个 head 的 recurrent state 为 `128×128`，按 key 轴、value 轴组织。对 Q/K 做 L2 normalization，query 再乘 `128^(-1/2)`；衰减和更新门为：

```text
β = sigmoid(b)
g = -exp(A_log) * softplus(a + dt_bias)
```

每个 token 的计算按以下顺序落地：

```python
state = state * g[:, t].exp()[..., None, None]
memory = (state * k_t.unsqueeze(-1)).sum(-2)
delta = (v_t - memory) * beta[:, t].unsqueeze(-1)
state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
output[:, t] = (state * q_t.unsqueeze(-1)).sum(-2)
```

这段代码把容易混淆的语义固定下来：memory 从衰减后的历史读取；delta 修正当前 key 对应的 value；输出从更新后的 state 读取。循环累计与 state 使用 FP32，输出再转回输入 dtype。Q/K normalization 遵循参考路径的输入 dtype 运算顺序，不能把整层简单概括为“全 FP32”。

### 2.4 接 Attention 时，先解决布局，再解决历史访问

Q 投影同时产生 query 与输出 gate。它输出 `[N,8192]`，需要先按 head reshape，再在每个 head 内拆分：

```python
qg = q_proj(x).view(N, 16, 512)
q, gate = qg.chunk(2, dim=-1)   # 均为 [N,16,256]
```

直接对 8192 维切成两半，最终 shape 也可能正确，但每个 head 的 Q/gate 对应关系已经改变。随后 Q/K 各自做 norm，只对前 64 维执行 RoPE，后 192 维保持原值。K/V 的四个 heads 各重复四份去匹配 16 个 Q heads。

我们先调用 PyTorch SDPA，scale 使用完整 head dimension 的 `256^(-1/2)`。输出乘 `sigmoid(gate)`，展平成 4096 维，再经输出投影回到 2560。真实 KV 仍保存四个 heads，展开是计算中间结果。

### 2.5 两类 norm 分开实现，避免错误复用

一般 `_Norm` 在 FP32 归一化后乘 `1 + weight`，最后转回输入 dtype。GDN 的 gated norm 则直接乘 weight，而且中间需要一次 BF16 舍入：

```python
x = hidden.float()
x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
x = weight * x.to(hidden.dtype)
out = (x * F.silu(gate.float())).to(hidden.dtype)
```

这里的 cast 是后续融合必须保留的边界。把全部步骤保持 FP32 到最后，虽然仍然像同一条数学公式，却已经换了浮点执行路径。

共享的 decoder 外壳则很直接：

```text
u      = x + mixer(input_norm(x))
x_next = u + down_proj(silu(gate_proj(post_norm(u))) * up_proj(post_norm(u)))
```

在完整模型加载前，通用 GPU 算子测试已经 **30/30 通过**；独立 HF GDN 对照覆盖 B=1/4、T=1/16、FP32/BF16 共八组，记录中输出与最终 state 最大绝对误差均为 0。这一步确认的是指定算子的计算和状态契约，尚未证明整个模型行为。

开发中还遇到 CPU BF16 `rsqrt` 的形状相关舍入：整段和分块调用出现 1 ULP 差异。我们将“归一化数值”和“后续递推连续性”分开检查，保留失败轨迹；GPU 仍测试完整 normalization 加递推路径。这个区分避免了为了让测试变绿而模糊比较对象。

## 3. 完成 Day 0：严格加载，再跑通 prefill → decode

### 3.1 用 BaseOP 构建原生模型，而不是调用 HF generate

我们沿用 mini 的 `BaseOP` 组织 `_TextModel`、`_DecoderLayer`、`_Attention`、`_GatedDeltaNet`、`_MLP`，在注册表里将 checkpoint 的架构名映射到本地 `Qwen3_5ForCausalLM`。模型先在 meta device 建立参数外壳，再加载真实张量，避免先分配一份随机初始化的完整 GPU 权重。

`BaseOP.load_state_dict` 逐个消费预期键，检查 shape/dtype，最后拒绝剩余键。我们的目标是让结构不匹配在启动时明确失败。

首次加载确实失败了：checkpoint 中 `A_log` 与 `linear_attn.norm.weight` 为 FP32，而最初参数声明随默认 dtype 成了 BF16。修复是显式声明这两类参数为 FP32，并让 Engine 按各参数的预期 dtype 加载；没有把所有参数统一降为 BF16。

### 3.2 先核对普通推理所需的 426 个主干键

Day 0 要核对的是普通文本主干，可以从网络结构反推需要的 426 个键：

| 部分 | 键数推导 |
| --- | ---: |
| 24 个 GDN decoder | 24 × 14 = 336 |
| 8 个 Attention decoder | 8 × 11 = 88 |
| 文本 embedding 与 final norm | 2 |
| 普通文本主干合计 | **426** |

GDN 的 14 个键包含四个输入投影、conv、A_log、dt_bias、gated norm、out projection，再加两处 decoder norm 和三个 FFN 投影。Attention 层的 11 个键则由 q/k/v/o、Q/K norm、两处 decoder norm 和三个 FFN 投影组成。tied embedding 不额外创建 LM head 权重。

键数只是结构交叉检查，完整性仍取决于每个键的名称、shape、dtype。实际集成实现启动时还一并加载了 15 个 MTP 参数，因此历史日志中的总数是 441；这是加载器的实现粒度，不将 MTP 的执行与正确性计入 Day 0。新增的 15 个键及其执行流程放到第 4.4 节说明。

### 3.3 先用一段最小循环隔离模型问题

我先通过 `forward_tokens` 做离线生成，让一个 state 贯穿 prompt 与后续 token。核心逻辑如下，实际测试脚本还包含计时同步、预热和结果保存：

```python
state = Qwen35State()
out = model.forward_tokens(prompt_ids, state=state)  # 消费整个 prompt
generated = []

for step in range(output_tokens):
    token = out.logits[-1].argmax().view(1)
    generated.append(token)
    if step + 1 < output_tokens:
        out = model.forward_tokens(token, state=state)
```

这时需要检查的不只是文本是否像一句话，还包括：logits 是否有限、state 长度是否正确增长、下一轮是否真正消费上一轮选出的 token、重复运行是否一致。首个输出来自 prompt 末尾的 logits；它刚被选出时尚未进入 state，下一次 forward 才消费它。

第二次 smoke 还暴露了测试脚本自身的问题：tokenizer 返回的对象不是脚本预期的 Tensor，调用 `numel` 失败。我们改成先用 chat template 生成文本，再显式取得 `return_tensors="pt"` 的 input IDs，避免将客户端 API 错误误判为模型计算失败。

### 3.4 什么证据允许我们跨过这个里程碑

首轮 `02-model/smoke/model.json` 保存了：真实 441 键加载（426 个主干键及一并加载的 15 个 MTP 键）、32-token 输入/16-token 输出的三轮普通生成重复一致，以及一个固定聊天用例的前 16 个 greedy token 与独立 HF 模型一致。这组普通生成不执行 MTP 候选、验证或提交。到这里，我们把它称为 Day 0 最小闭环通过，继续扩展框架接口。

这里有一条后来的审计勘误：首轮虽记录 `chunked_greedy_equal=true`，但输入只有 32 tokens，脚本 chunk size 是 64，实际仍只有一块。真正的跨块证据来自后续 512-token 输入分成八个 64-token 块的实验。Day 0 smoke 不能提前承担它没有触发的验收。

这些实现与 Engine、scheduler、MTP 初版最后一起保存在 `0e3bbf4`，不是一串人为拆分的独立 Day 0 提交。

## 4. 让框架真正接管请求：状态、分块、批次与 MTP

离线循环明确了模型的输入输出。下一步是把这组约定接到 mini 的请求生命周期，让调度器每轮只提供新增 token，并让模型找到正确的历史。

### 4.1 先把三类历史放进同一个请求状态

`Qwen35State` 包含 `length`、`position_offset`，以及按层保存的 `kv`、`conv`、`recurrent`。一个普通请求的主要张量为：

| 状态 | 每层形状 | 特点 |
| --- | --- | --- |
| K、V | 各 `[L,4,256]`，BF16；8 层 | 随已消费 token 数 L 增长 |
| 卷积窗口 | `[1,8192,4]`，BF16；24 层 | 固定窗口，保存卷积前输入 |
| GDN recurrent | `[1,32,128,128]`，FP32；24 层 | 固定大小，保存递推后的记忆 |

按形状计算，主干基础状态约为 `49.5 MiB + L×32 KiB`。这未计入临时张量、MTP 分支、Graph staging 与 allocator，但足以说明准入时还要给每个请求预留固定 GDN 状态，不能只按 KV token 数估算容量。

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="请求跨块续写与状态归属，窄屏可横向滚动">
<img src="/images/my-sglang/qwen35-state-continuation.svg" alt="两个 prefill 块与后续 decode 依次消费 token，KV 追加、卷积窗口滚动、GDN 矩阵继续递推；已输出 pending 与已消费状态分开计数，槽位复用检查 UID。" loading="lazy" />
</div>
<figcaption>图 3：实现状态接口时需要同时维护的三条历史。token 小块表示逻辑位置，不是物理 KV 页；当前实现使用请求独立的动态 KV。<a href="/images/my-sglang/qwen35-state-continuation.svg" target="_blank" rel="noopener">查看完整 SVG</a>。</figcaption>
</figure>

### 4.2 分块不是重复做几次独立 prefill

第一块结束后，下一块必须使用同一个 state。位置从 `state.length + position_offset` 开始，Attention 读取历史 KV，卷积带上旧窗口，GDN 从旧矩阵继续更新。

Attention 还有一个具体修改：已有 P 个历史 token，当前块有 T 个 token，第 i 行 query 能看到的 key 满足 `j ≤ P+i`。代码显式建立偏移 mask：

```python
rows = torch.arange(T, device=device)[:, None] + P
cols = torch.arange(P + T, device=device)[None, :]
mask = cols <= rows
```

例如 P=3、T=2，两行分别允许看到 key `0…3` 和 `0…4`。直接从左上角画一个没有历史偏移的三角，会把合法历史遮掉。普通单 token decode 已有 past 时，现有缓存中的位置全部合法。

后来 `03-gated-norm/model-reference/model.json` 的 512-token 输入真正跨越八个 64-token 块，分块前 16 个输出与整段路径一致。这是首批实际跨块证据，范围仍然有限。

### 4.3 将多个请求展平，但不混合它们的历史

`forward_packed` 接收 `input_ids`、`lengths`、每请求一个 state 和对应 positions。例如三 token 的输入块与一个 decode token 可以打成：

```text
input_ids = [a0, a1, a2, b0]
lengths   = [3, 1]
states    = [state_A, state_B]
positions = [L_A, L_A+1, L_A+2, L_B]
```

投影、norm 和 FFN 在总 token 轴上共用计算；Attention/GDN 按请求片段读取和更新各自历史。入口拒绝两个请求共享同一个可变 state，默认只返回每段最后位置的 logits，MTP 验证则可以要求所有位置的 logits。

Scheduler 通过 `Engine.forward_batch` 进入模型 `forward()`。模型维护 `_request_states[table_idx]=(uid,state)`：新请求创建空状态，UID 不匹配报错，`state.length` 必须等于 `Req.cached_len`。请求结束或取消时释放对应状态，在存在在途执行时建立必要的 stream 依赖，再让槽位被新请求使用。

我们没有把这一步包装成已完成统一 Paged KV。当前 hybrid 路径的真实 KV 由模型动态追加，Engine 的 `kv_cache=None`，页表与页预算仍服务于调度和容量管理。当前路径限定 TP=1、page_size=1、naive prefix cache；只有 KV 相同而没有对应卷积/GDN state，不能恢复一个正确前缀。

### 4.4 Day 0 之后：单独接入 MTP 候选、验证与提交

普通推理基线建立后，再扩展 MTP。虽然初版代码与模型集成在同一个提交里，这里需要单独实现和验收一条投机执行路径。它额外加载 15 个键：FC、两个输入 norm、一层含 11 个键的 Attention decoder、final norm；连同主干的 426 个键，得到历史加载记录中的 441。

MTP 网络先分别归一化下一位置的 token embedding 与上一位置的 hidden，再拼接，经 `[2560,5120]` 的 FC、一层独立 full-attention decoder、final norm 和共享 head 产生预测。当前控制器传入 target final norm 后的 `hidden_states`；这是一条明确的实现选择，完整数值对齐仍须回归。

Prefill priming 使用 `input_ids[1:]` 配 `target.hidden_states[:-1]`，MTP 的 `position_offset=1`，因此其状态长度比 target 少一。跨块时还要将上一块最后的 hidden 留给新块第一个 token，不能在每块各自做切片后丢掉边界配对。

`GreedyMTPController` 在临时状态上展开候选，再让 target 在另一份 clone 上验证 `[pending,candidates…]`。接受的是连续匹配前缀。若出现拒绝，GDN state 无法像 token 数组一样切掉末尾，我们从旧 target state 重放真正接受的输入前缀，再用真实 target hidden teacher-force 待提交的 MTP state。

所有新状态与待发布 hidden 准备完毕，才统一替换 session 引用；输出为接受前缀加 target bonus，bonus 成为下一轮 pending。在线适配还要更新 token pool、释放拒绝后缀的预留量，并在一个输出块内部处理 EOS 与长度预算。

`04-mtp/mtp.json` 的早期短回归覆盖 B=1/3、普通/k=1/k=3、每配置三轮，共 18 条 batch-run，输出与对应普通路径一致。但它只有三个短聊天提示，固定生成 32 tokens、忽略 EOS。我们把它当作状态提交的早期检查，后面的真实 HTTP 子集仍然发现了严格对齐问题。

## 5. Day 0 以后，先用 profile 找下一项改动

参考路径可运行之后，性能问题才有可以比较的对象。通用算子阶段已经采过局部 trace；模型跑通后，我们给每层、token mixer 与 MLP 加 `record_function` 标记，另外采一轮 native decode 的 CPU/CUDA trace。

这时我同时看两件事：GPU 时间集中在哪里，以及 CPU 在向 GPU 发射多少碎小操作。只看总 token/s，无法判断应该改投影、递推、状态搬运还是提交方式。

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="首轮真实 decode 的 kernel 热点，窄屏可横向滚动">
<img src="/images/my-sglang/qwen35-day0-profile.png" alt="由实际 Chrome trace 重绘的 kernel 热点图，两个 GEMV 家族占据主要累计执行时间，其余包含转换、归约与逐元素运算。" loading="lazy" />
</div>
<figcaption>图 4：此前真实 trace 的统计重绘，保持测量图原貌。首轮记录有 2777 个 GPU kernels；这是 kernel 累计执行时间分布，不是完整生成延迟。<a href="/images/my-sglang/qwen35-day0-profile.png" target="_blank" rel="noopener">查看原图</a>。</figcaption>
</figure>

这份单步 trace 中，GPU kernel 累计约 9.172 ms，两个 GEMV 家族约 5.892 ms，占 64.2%。同时，转换、归约、逐元素操作产生了大量启动。它提示投影很重要，也说明可以先选择一个数值边界清楚的小区域，验证融合是否真的能传导到模型收益。

第一项选择是 gated RMSNorm：它只有明确的归约、权重和 SiLU gate，没有跨 token 状态依赖，适合先验证融合方法。这个选择不是预先承诺它一定改善整体延迟。

正式比较时，profiler 与 benchmark 分开运行。Profiler 有记录与注入开销；CUDA event 包住一串 Python 发射时，也可能包含 stream 等待主机的空洞。模型对照使用同形状预热后的无 profiler 三轮生成，并分别保存 prefill、decode 与完整离线生成时间。

下面两项 A/B 都固定 B=1、输入 512、生成 128 tokens、相同模型/输入哈希和 greedy 路径。完整离线生成包含 prefill、127 次后续 decode 与 token 选择，排除权重加载、HTTP、排队和网络。每项优化重新测自己的 reference；三轮均值和样本标准差见表，原始样本摘录可[直接查看 JSON](/data/my-sglang/qwen35-stages.json)。

## 6. 第一次融合：局部变快以后，模型却没有收益

`qwen35_fused.py` 让一个 Triton program 处理一行 gated norm，将均方、rsqrt、权重乘法与 SiLU gate 合在一次调用中。实现保留 normalization 转回输入 dtype 的位置；对非 FP32 权重，还保留乘权重之后相应的舍入，并设置 `enable_fp_fusion=False`。

我没有同时更换 GDN recurrence。这样回归出问题时只需检查一个替换边界，性能变化也能归属于这一项开关：`MINISGL_QWEN35_GATED_NORM=reference|triton`。

局部测试 12/12 通过。B=1/T=1 的微基准 CUDA event 区间中位数从 54.606 μs 到 9.918 μs；单次 trace 的结构从 12 个 kernels 变为 1 个。接下来仍需重新跑完整模型：

| 指标，ms | Reference：三轮均值 ± 标准差 | 融合 gated norm | 观察 |
| --- | ---: | ---: | --- |
| Prefill | 516.042 ± 0.651 | 521.302 ± 0.659 | 没有下降 |
| Decode | 1769.918 ± 4.760 | 1796.328 ± 14.073 | 没有下降 |
| 完整离线生成 | 2285.960 ± 5.274 | 2317.630 ± 14.565 | 耗时约增加 1.385% |

两条路径的三轮 128-token 输出相同，各自分块输入的前 16-token 输出也相同。另一次模型 trace 确认 24 层确实执行了融合 kernel，排除了「开关没生效」这种简单解释。

因此我们保留实现与测量，默认仍为 reference。两组模型实验分进程运行、没有交错 A/B，约 1.4% 的回退不能凭这组数据归因于某一种缓存或硬件行为。能确定的是：这次没有获得模型收益，微基准的约 5 倍提升不能写成模型加速。

这项负结果决定了下一步仍然要做独立实验，而不是将多个融合一起打开再挑一个更好的总数。

## 7. 第二次融合：把 GDN 的时间递推留在一个 kernel 内

### 7.1 这次改变的是状态读写与发射方式

reference GDN 在 Python 中沿 T 个 token 循环，每个 token 再调用若干逐元素和归约操作。它清楚地表达了公式，却会不断发射 kernel，并把中间 state 物化成张量。T 增大时，这种执行方式的成本迅速累积。

`ce8acc4` 引入通用 Triton recurrence。我们先固定真实形状 K=V=128，使用 `value_tile=32, num_warps=4`。一个 program 对应一个 batch/head/value tile，持有 `128×32` 的 FP32 状态块，沿 token 顺序在 kernel 内执行归一化、衰减、memory 归约、delta、outer-product 更新和 query 读取。

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="gated norm 与 GDN recurrence 的独立融合方法，窄屏可横向滚动">
<img src="/images/my-sglang/qwen35-fusion-process.svg" alt="gated norm 将多个运算融合但保留 BF16 舍入边界；GDN 按 value 列分块，完整 key 轴参与归约，一个 program 沿时间推进状态，通用实现与后续 SM120 参数扫描分开。" loading="lazy" />
</div>
<figcaption>图 5：两项独立融合的实现方法。方框表示计算与数据归属，不表示实测时间比例；投影、卷积及输出投影仍在 recurrence kernel 之外。<a href="/images/my-sglang/qwen35-fusion-process.svg" target="_blank" rel="noopener">查看完整 SVG</a>。</figcaption>
</figure>

为什么沿 value 轴切？每个 value 列的更新都需要对完整 key 轴求 memory/query 归约，但不同 value 列可以独立处理。按 value 列分块能保留完整 K 轴，把并行性放在 batch、head 和 value tile 上。token 之间仍有递推依赖，不能随意并行重排。

初始 state 只读，最终 state 另行分配，保留 reference 的所有权约定。Q/K normalization 也在 kernel 内，但显式保留 BF16 的 square、sum 结果、加 epsilon、rsqrt、multiply 等转换边界；没有把“全 FP32 重写”当作原路径的等价替换。

这仍是一个通用 Triton 实现，没有使用 SM120 专属指令。其他不满足支持条件的形状回退 reference，投影、因果卷积、门控参数准备、输出 norm 与 out projection 仍由原路径执行。

### 7.2 先通过数值检查，再回到完整生成

23 项 GPU 测试覆盖 B=1/4/8、T=1/16/65、FP32/BF16、非零初态、分块和回退，全部通过预设容差。最大 BF16 输出绝对差为 `0.0001220703125`，最大 state 绝对差约 `1.7881393e-7`；这是容差内一致，不是逐 bit 相同。

模型实验只切换 `MINISGL_QWEN35_GDN`，gated norm 保持 reference，并重新测基线。当前代码的 GDN 默认仍为 reference，融合路径需要显式启用，尚未按这一个负载的结果全局打开：

| 指标，ms | Reference：三轮均值 ± 标准差 | GDN Triton | 耗时下降 |
| --- | ---: | ---: | ---: |
| Prefill | 512.128 ± 0.858 | 42.210 ± 0.039 | 91.758% |
| Decode | 1761.840 ± 4.195 | 1540.485 ± 22.505 | 12.564% |
| 完整离线生成 | 2273.968 ± 5.009 | 1582.695 ± 22.496 | **30.399%** |

Reference 三轮完整生成为 `2271.372 / 2270.791 / 2279.743 ms`，融合路径为 `1608.604 / 1571.362 / 1568.119 ms`。融合第一轮较慢，仍计入统计。两路径三轮 128-token 输出一致；分块前 16-token 一致；融合路径另有固定聊天 16-token 与 HF 的比较。

这里 prefill 的大幅下降有明确的比较对象：我们移除了通用 PyTorch 逐 token、多次发射的主要成本。它不能解释为超过成熟项目的并行 GDN prefill，也不能推广成所有请求快 30%。获得证据的是这个普通单请求、512/128 的离线负载，MTP/HTTP/Graph 组合仍需单独测。

### 7.3 再看一次 profile，确认优化发生在哪里

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="GDN 融合后的实际 CPU 与 GPU 时间线，窄屏可横向滚动">
<img src="/images/my-sglang/qwen35-gdn-profile.png" alt="由真实单步 decode trace 重绘的 CPU 发射和 GPU stream 时间线，融合后仍有投影和辅助计算，不能将图中跨度作为完整生成耗时。" loading="lazy" />
</div>
<figcaption>图 6：融合后的真实 CPU/GPU trace 重绘。同阶段 reference 为 2785 个 kernels，融合为 2185 个，其中 24 次为 recurrence kernel；30.399% 来自无 profiler 的三轮生成，不是由这张图的宽度计算。<a href="/images/my-sglang/qwen35-gdn-profile.png" target="_blank" rel="noopener">查看原图</a>。</figcaption>
</figure>

我们还在 Perfetto 中搜索 `_recurrent_kernel`，确认调用次数与 24 层 GDN 对应。这个检查让“代码里开了融合”变成“实际轨迹里执行了融合”。减少约 600 个 kernels 与替换范围相符，但剩余投影、Attention、norm 和状态操作继续占据开销。

## 8. CPU 发射仍有成本，于是单独尝试 CUDA Graph

融合改变 kernel 的计算粒度；CUDA Graph 改变重复提交方式。两者解决的问题不同，因此我们把 Graph 留作独立开关，用固定地址的 KV/GDN staging 捕获普通 decode，再把请求状态载入、replay 并提交回来。

这一步先遇到了数值问题。动态 Attention 的有效长度与静态容量带 mask 的路径可能选择不同数值实现。诊断中 static 与 Graph 一致，而 eager 与 static 不一致，因此问题不应直接归咎于 capture。统一到 math SDPA 后，B=1、prefix=16、capacity=256 的三步 logits、hidden、KV、conv、recurrent 检查一致；更大 batch 和容量并没有全部通过。

这组独立 Graph A/B 将 GDN 与 gated norm 都设为 reference，两侧均使用 math SDPA；它没有在上一节的 GDN 融合配置上累加收益。在这个有限范围内，包含 staging/replay/commit 的 forward wrapper 从 `16.176 ± 0.029 ms` 到 `10.807 ± 0.001 ms`。它排除了外部测试框架的 clone 和 token 选择，capture 另有成本，也不是完整 HTTP 生成。capacity=8192 的 logits 检查仍失败。

我们还把 math SDPA 的额外成本单独测出来：Graph 关闭，512/128 生成交错 A/B 各三轮，math 比 auto 慢约 5.44%。因此当前默认仍是 auto，Graph 关闭；只暴露 B=1、capacity=256、math 的实验入口，没有宣布 8K 或 MTP Graph 已完成支持。

这个实验说明扩展已有框架能力时，需要同时确认它改变的输入布局和数值路径。捕获成功本身不是验收终点。

## 9. 最后才针对 5090 调参数：从 CTA 假设到负结果

通用 GDN 融合稳定后，才开始用实际硬件信息做假设。B=1、H=32、value tile=32 时，launch grid 对应 `32×4=128` 个 CTA，而该 5090 实测有 170 个 SM。由此提出一个可测试的假设：缩小 value tile，增加 CTA 数，是否能提高小 batch 的并行利用？

它不是保证。value tile 越小，Q/K normalization 和读取也可能重复更多次；tile 越大，又可能增加寄存器占用。因此我们独立扫描：

```text
B       = 1, 4, 8
T       = 1, 16, 128
tile    = 16, 32, 64
warps   = 4, 8
总计    = 3 × 3 × 3 × 2 = 54 组
```

每组先与通用 tile32/warp4 对照输出和最终 state，通过后再测三轮。测量分成 Python wrapper 和 Graph 中连续调用的 GPU 间隔，后者用于减少主机发射空洞影响，**不是在宣称完整模型的 Graph 已通过**。

结果中九个输入形状有七个仍以通用 tile32/warp4 最低。仅 B1/T1 与 B8/T1 的 tile16/warp4 得到约 5.27%、5.79% 的局部 GPU 间隔下降。B1/T1 的 CTA 从 128 增至 256，GPU 间隔从约 2.436 到 2.308 μs，但 Python wrapper 均值却从约 13.371 增至 13.844 μs。

我们还看到 T>1 时 tile64/warp4 的寄存器数明显高于 tile32/warp4，增加 warp 通常更慢。没有据此添加默认 SM120 特化分派：完整模型按请求调用 GDN，不能直接套用 B8 微基准；局部改善也没有新的端到端证据。

成熟后端也纳入过比较。FlashInfer SM120 GDN 在九个测试形状上的最终 state 均未满足本项目预设容差，原因仍需进一步核对，不能直接认定为后端 bug。该轮及后续 fixed-linear 微基准还与外部 GPU 负载重叠，计时作废，仅保留数值和调用结构记录。这些都没有计入已验收优化。

## 10. 在线回归又把我们带回正确性问题

把模型放进 HTTP、SSE 和真实批处理以后，我们固定 SPEED 的九条 coding/math/reasoning 请求，对普通与 MTP k=1 分别测并发 1/4/8、各三轮。这组在线对照使用 reference GDN 与 gated norm，关闭 Graph 和 overlap，调度策略为 `prefill_first`。两种模式各 81/81 请求完成；计数、EOS、SSE 拼接与每模式六次运行中取消有实际后端证据。

但原始 token hash 只有 **55/81** 对齐。普通路径自身也出现跨 batch 差异。早期三个短聊天提示通过，不能覆盖真实到达顺序和更长输出；协议成功率也不能代替 greedy 一致性。

下一步我们固定同一 teacher-forced token 历史，逐层比较 B=1/4/8 的激活，并在前后核对完整权重哈希。最早可见分歧出现在第 0 层 GDN 的 `in_proj_qkv`。关闭 BF16 reduced-precision reduction 后，prefill 的首个分歧移到 MLP down projection，并未消失。这个短诊断的 17 个预测位置仍然 argmax 相同，所以它只缩小排查范围，没有复现并解释所有在线分叉。

由此增加了默认关闭的固定归约 BF16 linear 路径（`89316b3`），并修正 rejection replay 后必须同步携带实际重放 hidden 与 bonus 的内部一致性问题（`49d8eeb`）。有限算子/CPU 检查有结果，修复后的完整 GPU 模型与在线回归仍待完成。

这也是本文的当前终点：我们已经建立 Day 0 基线，接上混合状态与 MTP，并得到部分负载的局部和离线收益；完整 Qwen3.5/MTP 组合验收仍未结束。在线未对齐轮次中的吞吐变化没有被当成通过的加速成果。

## 11. 这条路线如何在 Git 和实验文件里复现

每次改动需要同时留下三个东西：能关闭的实现、可重复的比较输入，以及足以解释结论的结果。下面按本文步骤给出实际入口：

| 阶段 | 代码 / 提交 | 主要证据 |
| --- | --- | --- |
| 读结构、补配置 | `models/config.py`，`e8c60a6` | 配置回归，固定模型 revision |
| 写通用算子 | `kernel/qwen35_reference.py`，`4f501ec` | `01-generic/gpu-tests.log`、`01-generic/gdn-hf-smoke.json` |
| Day 0 与模型集成 | `models/qwen3_5.py`、注册/权重/Engine，`0e3bbf4` | `02-model/smoke/model.json` 与失败日志 |
| 首项融合负结果 | `kernel/qwen35_fused.py` | `03-gated-norm/` |
| MTP 早期检查 | `engine/speculative.py`、`scheduler/mtp.py` | `04-mtp/mtp.json` |
| GDN 通用融合 | `kernel/qwen35_gdn.py`，`ce8acc4` | `05-gdn/`，23 项 GPU 检查、三轮模型对照 |
| Graph 扩展 | `engine/qwen35_graph.py`、`b8da37d` 等集成 | `06-graph/`，通过与失败范围分别保留 |
| SM120 参数比较 | `benchmarks/qwen35/tune_gdn.py`，`a2d3f20` | `07-sm120/sweep.json`，54 组全部结果 |
| 在线与数值诊断 | HTTP benchmark、`batch_numerics.py`、`49d8eeb`、`89316b3` | `07-http/`、`09-batch-numerics/` |

早期测量曾在未全部提交的工作树中执行，因此结果中的 base HEAD 不总能单独重建全部代码。后续模型对照加入 `source_sha256`，复现时需要一起核对。本篇公开的 [A/B 摘录](/data/my-sglang/qwen35-stages.json)包含各轮样本、比较范围、来源文件哈希和源码哈希；完整原始文件继续保存在项目实验目录。

例如，在按已记录版本准备的独立 CUDA venv 中，以模型普通路径重跑一组基线：

```bash
# 在 My_Sglang 仓库根目录；模型路径指向固定 revision 的本地快照。
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

测试 GDN 时只将 `MINISGL_QWEN35_GDN` 切到 `triton`，输出到另一个新目录，其他条件保持相同。profile 用于解释结构，正式三轮计时在脚本的无 profiler 区间完成；不要用截图宽度换算加速，也不要覆盖上一轮失败数据。

从这次接入中形成的工作顺序是：先让配置、权重和参考语义一致，再建立显式状态的普通路径；每增加一种执行模式，都检查它如何消费和提交状态；每提出一种优化，都用独立开关、数值回归和同负载对照决定是否保留。接下来仍需先解决真实 greedy 回归，再补齐混合调度、overlap 与更大输入范围的端到端验收。
