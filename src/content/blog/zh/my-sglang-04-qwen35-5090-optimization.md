---
title: "My_Sglang（四）：在 RTX 5090 上逐步优化 Qwen3.5"
description: "从通用实现出发，用真实 profile 追踪 Qwen3.5 的开销：Norm 为什么只在微基准里变快，GDN 融合为什么更有用，以及扩大回归后，哪些漂亮数字需要重新审视。"
date: 2026-09-27T08:00:00Z
lang: zh
translationKey: my-sglang-04-qwen35-5090-optimization
tags: [LLM 推理, Qwen3.5, Triton, 性能分析, My_Sglang]
draft: false
---

Qwen3.5 刚在 My_Sglang 里跑通时，实现很朴素：Linear 和 MLP 调 PyTorch，Attention 用 SDPA，GDN 的状态更新则照着公式逐步写出来。长输入切成几块以后，KV、卷积窗口和循环状态能够接起来，模型也能继续生成。

接下来才轮到速度。

这段优化里，有两组数字很能说明问题。同样是减少 kernel 调用，gated norm 在微基准里快了约四倍，放回模型却几乎没有收益；GDN 递推融合后，固定的 512-token 输入、128-token 输出场景，生成耗时从 2.395 秒降到了 1.663 秒。

问题出在扩大测试之后。换一组输入，或者增大 batch，部分输出就开始不一致。后半段的工作因此从调 tile、看延迟，转回了数值诊断。

这篇沿着这些问题展开。[第二篇](/zh/blog/my-sglang-02-qwen35-model/)记录了模型接入与早期实现；这里把已有的通用路径、融合路径和框架开关放回同一套测量中比较。**下文性能数字全部来自 2026 年 9 月 27 日的单卡 RTX 5090 重测。** 优化实现有先后，本文的整理顺序不代表所有 kernel 都是在这一天才写出来的。

## 起点：先留下一个看得懂的实现

Qwen3.5-4B 的 32 个 decoder 层里，有 24 层 GDN、8 层 Full Attention。两种 mixer 外面仍然是熟悉的 norm、残差和 MLP，复杂的地方主要在历史状态。

Full Attention 留下 K、V；GDN 则还要留下因果卷积的输入窗口，以及每个 value head 对应的 FP32 循环矩阵。请求从 Prefill 进入 Decode 时，这些东西都要接续。只把某一层的公式写对，还不算模型接好了。

最初我让这些边界尽量显式：旧状态作为参数传进来，新状态作为结果交回去。GDN 参考递推大致是下面几行，省略了 shape 展开：

```python
state = exp(g) * previous_state
memory = sum_over_k(state * k)
delta = beta * (v - memory)
state = state + outer(k, delta)
output = sum_over_k(state * q)
```

先衰减历史，再用 key 读取记忆，根据误差修正矩阵，最后由 query 读取更新后的状态。它很适合核对顺序，也方便比较整段输入和分块输入，但每推进一个 token，都要执行一串张量操作。

这些操作在 GPU 上运行，并不是 CPU 模拟。不过，“放在 GPU 上”与“适合 GPU 执行”之间，还隔着操作如何组织、发射多少次、哪些中间结果需要写回等问题。初版先保留这种容易检查的写法，后面用测量决定改哪里。

这个 reference 也有明确的范围：混合模型使用请求私有的动态 KV 张量，尚未接入普通模型的 paged Attention 后端；GDN 状态也没有与 KV 合成一个统一物理池。后面的收益都相对这份实现，不能替换成与成熟推理引擎的比较。

## 先让几组测量可以比较

早期实验分散在不同脚本和进程里，适合验证某个改动能不能跑。要解释每一项收益，还需要一套更统一的对照。

这次让同一个模型进程轮换执行四条路径：reference、只融合 gated norm、只融合 GDN、两者同时开启。每个形状、每条路径都先完成一次完整生成预热，再交错执行三轮。输入固定，输出固定为 128 个 token，不因为遇到 EOS 提前结束。

测试设备是 RTX 5090 32GB，TP=1、BF16，PyTorch 2.13.0+cu130、Triton 3.7.1。模型使用同一份本地权重，文件重新计算 SHA256；普通对照关闭 CUDA Graph，Linear 保持 reference，SDPA 使用 auto。新实验在独立 worktree 和项目 venv 中运行。

先看通用路径的几个单请求结果：

| 输入 / 输出 tokens | Prefill forward | 后续 127 步 Decode | 完整生成 |
| --- | ---: | ---: | ---: |
| 128 / 128 | 145.4 ms | 1891.1 ms | 2036.5 ± 6.8 ms |
| 512 / 128 | 524.3 ms | 1870.5 ms | 2394.9 ± 3.4 ms |
| 1024 / 128 | 1033.7 ms | 1891.9 ms | 2925.6 ± 7.1 ms |
| 4096 / 128 | 4152.3 ms | 1882.9 ms | 6035.2 ± 4.3 ms |

表中的完整生成包括 Prefill、首 token 选择、后续 Decode 和采样，不包括模型加载、tokenize、HTTP、排队与网络。`±` 是三轮的样本标准差。输入由固定短文本的 token 序列重复、截断得到；B1 的几个长度来自同一份 14-token 种子序列，短输入是长输入的前缀。它们用于性能和严格输出对照，并不是几个独立语义任务，也不是模型质量评测集。

还测了 B=8、B=32 的 128-token 输入。四条路径、六个场景、三轮，总共完成 72 次 packed 生成。每次除了时间，还保存 token 序列、状态长度与显存峰值。之后再看到“快了多少”，就能回到同一形状的 reference，而不必从不同日期挑一组有利的基线。

## 打开 profile：Prefill 和 Decode 不是同一个瓶颈

光看延迟表，已经能发现长输入的 Prefill 成本很高。具体花在哪里，还要展开执行过程。

我用 `torch.profiler` 分别采集 Prefill 和 Decode，在模型层、token mixer、MLP 周围加范围标记，再沿 CPU 操作与 CUDA launch 的关联找到 GPU kernel。分析参考了 [AI-Infra 的 profiler 方法](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS)：先确认实际走到哪条路径，再做模块归属，最后讨论融合机会。

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="reference Decode 的真实 Perfetto 时间线">
<img src="/images/my-sglang/qwen35-5090-20260927/perfetto-reference-decode.png" alt="Perfetto 展开 reference Decode 的 CPU 调用栈与 GPU stream，32 层下面排列着大量短 kernel。" width="1800" height="980" loading="lazy" />
</div>
<figcaption>图 1：本轮 reference Decode。上面是 CPU 调用栈，下面是 GPU stream。截图只包含 Perfetto 页面；时间线用于观察执行结构，正式 benchmark 单独关闭 profiler 测量。 <a href="/images/my-sglang/qwen35-5090-20260927/perfetto-reference-decode.png" target="_blank" rel="noopener">查看原图 ↗</a></figcaption>
</figure>

在 B1、prefix128 的 trace 中，reference Prefill 有 33425 个 kernel，Decode 一步有 2785 个。按模块汇总 kernel 的累计执行时间，差别更清楚了：

| 模块 | Prefill 占比 | Decode 占比 |
| --- | ---: | ---: |
| GDN mixer | 89.52% | 35.93% |
| MLP | 6.63% | 38.58% |
| Full Attention mixer | 1.88% | 12.14% |
| 层内 norm / residual | 0.79% | 4.45% |
| 层外操作 | 1.18% | 8.90% |

GDN mixer 包括投影、卷积、门控、递推和 gated norm，89.52% 不能全部记在一个 recurrence kernel 头上。但这个分布已经足以调整关注点：在当前实现中，长 Prefill 应先研究 GDN；Decode 则还必须看 MLP 和投影。

回头看参考递推，就能把时间分布和代码联系起来。输入变长时，Python 循环重复发射的是整串小操作。一次调用也许很短，但它会乘上 token 数，再乘上 24 层。

这里仍然只是一个待验证的解释。我没有运行 Nsight Compute，因此不能从这些彩条断言显存带宽已经打满，或者 Tensor Core 利用率不足。下一步先把真实维度抽出来，测 gated norm、GDN 和 causal conv 的独立调用。

## 第一个融合：四倍的局部加速，没有进入模型

gated norm 的边界比较小：计算均方与 rsqrt，归一化，乘 learned weight，再乘 SiLU gate。没有跨 token 递推依赖，适合先验证“少发射几个 kernel”能带来什么。

Triton 实现用一个 program 处理一行，把这些操作放在一起。融合时还要保留参考代码的舍入位置：归一化结果先转回输入 dtype，再乘权重。把中间结果一律留成 FP32，看起来精度更高，却已经改变了原来的数值路径。

这次微基准还暴露了一个很小、却会影响比较的细节：最初独立测试用 BF16 weight，真实模型这份 norm 的 weight 是 FP32。因此又补了一组与模型一致的对照，原来的结果保留原标签。

B1/T1、FP32 weight 下，三轮 CUDA event 区间的中位数从 **41.394 μs 降到 9.930 μs**，约 4.17 倍。独立 trace 确认，12 个 kernel 变成了 1 个。event 区间仍可能包含 host 发射间隙，它不是单个 kernel 的纯 GPU 时间。

但回到 B1、512/128 的完整模型：

| 路径 | 完整生成，三轮均值 ± 标准差 |
| --- | ---: |
| reference | 2394.873 ± 3.375 ms |
| 仅 gated norm 融合 | 2399.908 ± 2.552 ms |

模型没有得到可验收的改善。1024、4096 输入的变化也只有千分之几。

所以这一步只证明了局部调用可以做得更紧凑，没有证明它值得成为模型的默认加速。它也提醒我，微基准的倍率很容易显得巨大：reference 是几十微秒，融合后是十微秒，除出来很好看；放到一次几秒的生成里，需要看的是累计省下了多少时间。

## 把 GDN 的循环搬进 kernel

GDN 的问题更接近执行组织本身。reference 的 B1/T1 递推会发射 26 个 kernel，T=16 时已经有 176 个。状态衰减、memory 归约、delta 和矩阵更新之间，还有反复产生的中间张量。

这个融合版本让一个 Triton program 对应一个 request、一个 head，以及 32 个 value channels。它保留完整的 K=128 轴，持有一块 `128×32` 的 FP32 state，沿 token 顺序执行更新，最后写回新状态。

为什么沿 value 轴切？每个 value 列都需要对完整 key 轴求和，不同 value 列之间却可以独立更新。因此并行性放在 request、head 和 value tile 上，时间维仍然按递推依赖往前走。

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="Norm 与 GDN 融合范围示意">
<img src="/images/my-sglang/qwen35-report/fusion-process.svg" alt="Norm 将归约和门控合并；GDN 沿 value 轴分块，在单个 program 内推进 token 递推。" width="850" height="1440" loading="lazy" />
</div>
<figcaption>图 2：复用模型接入篇的机制示意。GDN 的输入投影、卷积和输出投影都在 recurrence kernel 之外；方框宽度不表示实测时间。 <a href="/images/my-sglang/qwen35-report/fusion-process.svg" target="_blank" rel="noopener">查看原图 ↗</a></figcaption>
</figure>

这个版本使用可移植 Triton，没有引入 SM120 专有指令，但已经利用了 **K=V=128 的固定 shape**。不满足条件就回退 reference。我没有先写一个覆盖任意 K/V 的通用融合，再从中推导出这个特例；实际实现的范围需要如实说明。

局部调用的变化很明显：

| GDN shape | reference | 融合 | 调用区间倍率 |
| --- | ---: | ---: | ---: |
| B1 / T1 | 99.703 μs | 14.144 μs | 7.05× |
| B1 / T16 | 704.263 μs | 19.108 μs | 36.86× |
| B8 / T16 | 1198.499 μs | 50.234 μs | 23.86× |

这里同样是三轮 CUDA event 区间中位数。profile 中 T1 的 26 个、T16 的 176 个 kernel 都变成一个 `_recurrent_kernel`。这直接验证了 launch 数量变化；中间读写的减少由代码和 tensor 生命周期解释，没有把它写成未经测量的 DRAM 流量数字。

随后回到完整生成。在下面三个固定输入上，融合与 reference 的 128 个输出 token 三轮都一致：

| B1，输出 128 tokens | reference | 仅 GDN 融合 | 耗时下降 |
| --- | ---: | ---: | ---: |
| 输入 512 | 2394.873 ± 3.375 ms | 1662.587 ± 0.433 ms | 30.6% |
| 输入 1024 | 2925.603 ± 7.065 ms | 1713.887 ± 0.676 ms | 41.4% |
| 输入 4096 | 6035.166 ± 4.277 ms | 1974.403 ± 3.653 ms | 67.3% |

拆开 512-token 场景，Prefill 从 524.3 ms 降到 42.6 ms，后续 Decode 从 1870.5 ms 降到 1619.9 ms。长输入收益更大，与 reference 逐 token 展开操作的成本相符；Decode 的改善约 13.4%，远小于 Prefill。

这仍只是三个指定输入上的结果。其他输入和 batch 的输出并没有全部通过，后文会单独展开。也不能用这个倍率推断相对成熟并行 GDN 后端的优势：目前对照的是自己的通用递推。

## 再看时间线：原来的热点缩小了，别的成本露出来了

融合之后重新采同一形状的 trace，可以看到 GDN mixer 的累计 GPU 时间下降，MLP 则基本没动。

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="Prefill 和 Decode 在融合前后的模块时间构成">
<img src="/images/my-sglang/qwen35-5090-20260927/profile-composition.png" alt="四根按模块分色的柱状图展示 reference 与 GDN 的 Prefill、Decode kernel 累计时间。" width="2360" height="825" loading="lazy" />
</div>
<figcaption>图 3：本轮完整 trace 的模块统计。Prefill kernel 数由 33425 降到 2345，Decode 由 2785 降到 2185；柱长是 kernel 累计执行时间，不是端到端延迟。 <a href="/images/my-sglang/qwen35-5090-20260927/profile-composition.png" target="_blank" rel="noopener">查看原图 ↗</a></figcaption>
</figure>

Decode 中 MLP 从 38.58% 变成 42.38%，绝对时间仍在 3.57 ms 左右。占比上升是因为总量变小了，并不是 MLP 被这次修改弄慢了。GDN mixer 也没有消失：投影、卷积、门控和 norm 仍要执行。

放大首层还能看到这一点：

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="reference 首层 GDN mixer 的 Perfetto 详情">
<img src="/images/my-sglang/qwen35-5090-20260927/perfetto-reference-layer.png" alt="reference 首层 token mixer 选中范围为 658.645 微秒，范围内可见密集的小操作。" width="1800" height="980" loading="lazy" />
</div>
<figcaption>图 4：reference 首层 mixer，所选范围 658.645 μs。 <a href="/images/my-sglang/qwen35-5090-20260927/perfetto-reference-layer.png" target="_blank" rel="noopener">查看原图 ↗</a></figcaption>
</figure>

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="融合后首层 GDN mixer 的 Perfetto 详情">
<img src="/images/my-sglang/qwen35-5090-20260927/perfetto-gdn-layer.png" alt="融合后首层 token mixer 选中范围为 489.081 微秒，仍有投影和其他辅助操作。" width="1800" height="980" loading="lazy" />
</div>
<figcaption>图 5：GDN 路径首层 mixer，所选范围 489.081 μs。两图各自缩放；详情类别均为 gpu_user_annotation，包含 mixer 内操作及间隙，不是单个 recurrence kernel 的时长。 <a href="/images/my-sglang/qwen35-5090-20260927/perfetto-gdn-layer.png" target="_blank" rel="noopener">查看原图 ↗</a></figcaption>
</figure>

从这里开始，只盯着 recurrence 内部已经不够。`forward_packed` 虽然让投影、MLP、norm 共批计算，Attention、conv 和 recurrence 仍按请求执行。模型 B=8，通常意味着每层调用八次 B=1 的 GDN，不能直接拿独立算子 B=8 的成绩代替它。

此外，动态 KV 的 `cat`、GQA 的 `repeat_interleave`、多个独立 projection 仍然存在。状态池、布局调整和投影合并都是后续候选，但本轮没有实现它们，也没有把时间线里的所有空隙都归因于 CPU launch。

## SM120 和框架开关，各自解决了多少

### 调 tile：局部更快，还不足以换默认值

在已有固定 shape 的实现上，又扫描了 value tile 16/32/64、4/8 warps，覆盖 B=1/4/8、T=1/16/128，共 54 组。每组先检查输出和 state，再分别测 eager wrapper 和算子 CUDA Graph。

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="SM120 上 GDN tile 和 warp 扫描结果">
<img src="/images/my-sglang/qwen35-5090-20260927/sm120-sweep.png" alt="54 组 GDN 参数扫描的相对耗时热力图，默认 tile32/4 为 1，长序列仍以默认配置最快。" width="1637" height="1064" loading="lazy" />
</div>
<figcaption>图 6：算子 Graph 时间除以默认 tile32/4 的时间，越小越好。这里使用 Graph 是为了减少微基准的 host 发射空隙，不代表整个模型支持相同的 Graph 组合。 <a href="/images/my-sglang/qwen35-5090-20260927/sm120-sweep.png" target="_blank" rel="noopener">查看原图 ↗</a></figcaption>
</figure>

B1/T1 的 tile16/4 在 Graph 中从 2.437 μs 降到 2.309 μs，改善约 5.3%；但完整 wrapper wall 从 13.478 μs 变为 13.791 μs。B8/T1 也是类似情况。到了 T=128，三个 batch 仍然是默认配置最快。

因此没有增加新的默认 dispatch。为少数形状多维护一个分支，需要完整路径的收益来支持，不能只凭热力图里两个颜色更浅的格子。

本轮也重测了已有 FlashInfer GDN 适配。B=1/4/8、T=16/128/512 的九组形状都未通过现有数值门槛，因此没有把它纳入有效加速。失败发生在我们的适配对照中，尚不能据此判断 FlashInfer 本身有错；归一化、gate、layout 与状态语义仍需要逐项核对。

### CUDA Graph：有收益，但范围很窄

Graph 单独比较：两边都使用 reference GDN、reference norm 和 math SDPA，避免把 Attention backend 的切换混进收益。

在 B1、prefix16、capacity256 下，包含状态 staging、replay 和提交的 wrapper，三轮均值从 16.1295 ms 降到 10.8190 ms，约下降 32.9%。计时不含外层状态 clone 和 token 选择，所以它不是 HTTP TPOT，也不能与前面的 GDN 倍率相乘。

capacity 改成 8192 后，第一次 logits 比较就失败了：248320 个元素中有 1426 个超出既定容差。这个范围没有通过验收，没有靠放宽阈值继续报告加速。

### 混批和 overlap：输出间隔改善，首 token 等待变长

在线服务依次测了 prefill-first、decode 优先混批、混批加 overlap、再打开 GDN。每阶段 C=4/8 各三轮，每轮六条短输入、两条长输入，实际 chat 长度为 522/4106 tokens，固定输出 128。

其中 mixed→overlap 的 48 条完整输出全部一致，可以较清楚地看这个开关的影响。C4 的结果是：

| 指标 | mixed | mixed + overlap |
| --- | ---: | ---: |
| TTFT P50 | 1590.1 ± 0.6 ms | 1892.6 ± 0.8 ms |
| TPOT P95 | 66.60 ± 0.09 ms | 61.39 ± 0.07 ms |
| 输出吞吐 | 52.84 ± 0.06 token/s | 53.64 ± 0.05 token/s |

这里的 P50/P95 先在每轮计算，再报告三轮均值与标准差。TPOT P95 改善约 7.8%，吞吐只增加约 1.5%，TTFT 却上升了。CPU/GPU overlap 改变了处理节奏，不会自动让所有指标一起变好。

其他阶段的输出不全一致；HTTP GDN 还有首轮较慢、一个请求跨轮输出变化的问题。数据都保留了，没有删掉首轮或只报告最高吞吐。AIPerf 的固定 SPEED 子集和取消释放检查也重新跑过，但它们各自只证明测到的服务行为，不能替代完整 greedy 对齐。

## 扩大回归后，输出对不上了

如果只测 B1、512/128，最容易写出的结论就是“融合有效”。问题是，同一份实现放到 P128、B8、B32，完整输出就不再全部相同。

| 场景 | 仅 Norm 对齐 | 仅 GDN 对齐 | 两者同时开启 |
| --- | ---: | ---: | ---: |
| B1 / P128 | 0/3 | 0/3 | 0/3 |
| B1 / P512 | 3/3 | 3/3 | 3/3 |
| B1 / P1024 | 3/3 | 3/3 | 3/3 |
| B1 / P4096 | 3/3 | 3/3 | 3/3 |
| B8 / P128 | 9/24 | 6/24 | 6/24 |
| B32 / P128 | 63/96 | 60/96 | 57/96 |

分母包含三轮重复，不代表等量独立语义样本。比较的是同 batch、同输入下的 128-token 完整输出。native 每种配置自身跨轮稳定，因此不能把差异简单当成某一轮偶然出错。

这时，77 项局部检查通过、16-token HF 短对照通过、分块回归通过，都还是真的；只是这些检查的覆盖不够宽。局部容差通过，并不能保证整个网络长期递推后 argmax 不变。

为了避免“已经生成了不同 token，后面当然不同”的干扰，我加了一个 teacher-forced 诊断：四条路径各自维护状态，每一步却喂入完全相同的 reference 历史。

GDN 首次在第 5 个输出出现 argmax 分歧。当时 reference 的前两名 logits 差了 **7.0**，两条路径的 logits 最大绝对差为 **12.125**。这不适合用“两个 token 本来就差不多，浮点抖一下”来解释。

接着再缩小范围：只执行一次 reference Prefill，每一层都把相同的真实 q/k/v/g/beta 分别交给 reference 和 fused GDN，同时始终把 reference 输出送往下一层。24 层的局部 output/state 比较全部通过既定容差，最大输出绝对差约 `2.44e-4`。

两个结果并不矛盾。单层足够接近，与把所有层替换之后仍然生成相同 token，是两种要求。这个诊断把关注点推向真实输入分布和跨层误差传播，但还没有证明唯一根因就是某个归约、舍入或 gate。

另外，reference 自身在相同 prompt、不同 batch 下也有输出差异。这需要单独处理，不能全部归咎于融合；它也不能让同 batch 的融合失败自动过关。

<figure class="sg-static-figure">
<div class="sg-static-scroll" tabindex="0" role="region" aria-label="六场景四路径的生成耗时与正确性状态">
<img src="/images/my-sglang/qwen35-5090-20260927/model-ablation.png" alt="六个场景的四路径消融，柱为三轮均值、点为各轮，斜线明确标出未通过严格输出比较的候选。" width="2360" height="1406" loading="lazy" />
</div>
<figcaption>图 7：把所有结果放回同一张图。斜线柱的时间同样是真实测量，但输出验收失败，只用于诊断，不能作为可默认启用的加速。 <a href="/images/my-sglang/qwen35-5090-20260927/model-ablation.png" target="_blank" rel="noopener">查看原图 ↗</a></figcaption>
</figure>

## 下一步，先解决这几个失败用例

通用实现让我们有一个能够逐步检查的起点。profile 把注意力引向 GDN；Norm 的负结果说明少发射几个 kernel 未必能改变模型；GDN 融合则确实压缩了长 Prefill 的开销。再往后，调 tile 的局部优势没有穿过 wrapper，Graph 和 overlap 也各自有适用范围。

这些数据足够支持几个固定场景的速度事实，还不足以宣布整个支持范围都通过。尤其现在已经有了可重现的失败输入，继续增加融合只会让数值问题更难拆开。接下来应先沿保存的 P128 和较大 batch 用例定位误差，再决定是否推进并行 Prefill、状态池或投影合并。

本轮没有新增运行时 fast path；新增的是交错 A/B、同历史诊断、真实输入逐层对照，以及证据整理。把早期实现放回更严格的测量后，得到的结论比“所有开关打开最快”窄了一些，但下一步要改什么也更明确了。

### 数据与复核

正文使用的逐轮数值、比较条件、输出通过比例和来源 SHA256，可下载 [本轮公开数据摘录](/data/my-sglang/qwen35-5090-20260927.json)。它提取自已归档结果，不是另一次 benchmark；没有公开机器路径或请求文本。完整实验归档保留了 191 份原始文件和 18 份 trace。

Perfetto 截图来自这次的完整原始 trace。官方静态资源直连超时后，使用未修改的 Perfetto v58.3 资源在本地打开文件，截图只保留页面视口。图表由原始事件或逐轮测量生成，与机制示意图分开标注。

测量代码分别绑定 `ccaecc1`（模型 A/B）、`86e556e`（在线检查与数值诊断）、`6218709`（真实输入逐层对照）；模型与 kernel 实现保持本轮起点 `202aabc`。报告归档提交为 `14e7e62`。本轮只讨论普通文本推理，不包含 MTP、PD、多卡或量化收益。
