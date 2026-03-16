我先给你一个总览：这三篇其实是一条非常清晰的技术演进线——**LaViDa**先回答“离散 masked diffusion 能不能做好多模态理解”，**Lavida-O**进一步回答“能不能把理解、生成、编辑、grounding 统一进一个 MDM 里，而且做强”，**Sparse-LaViDa**再回答“这个统一 MDM 能不能在不明显掉点的前提下显著加速”。三篇原文分别是 **LaViDa**、**Lavida-O**、**Sparse-LaViDa**。

**1. LaViDa：把离散扩散真正做成可用的多模态理解模型。**  
这篇论文的背景是：主流 VLM 基本都是 AR（自回归）模型，优点是成熟，但缺点也很明显——生成慢、难并行，而且对结构化输出、填空式生成、受约束生成不友好；而离散 diffusion language model（DLM/MDM）天然有并行解码、双向上下文和可控生成优势，但此前几乎没有被系统地用到视觉-语言理解里。LaViDa要解决的核心问题，就是如何把一个离散扩散语言模型稳妥地改造成 VLM，并把训练效率、推理速度和低步数采样质量这三个现实问题一起解决。它的模型架构很直接：图像先被 resize 成 (768^2)，再切成 4 个 (384^2) 局部视图，加上 1 个全局 (384^2) 视图，共 5 个 view；每个 view 由 **SigLIP-400M** 编成 (27\times 27) 个视觉 embedding，总共 3645 个，再经过 (2\times 2) average pooling 压成每个 view (14\times 14)，总计 980 个视觉 token，随后经 MLP projector 投到 diffusion backbone 的语义空间；文本侧输入是 prompt (P) 和部分被 mask 的答案 (X_t)，视觉 token、prompt token、masked answer 一起送入**非因果 attention 的 Transformer diffusion LM**，输出是对 clean answer (X_0) 的逐 token 预测。训练上，LaViDa把基础目标写成条件扩散形式 (L_{D\text{-}VLM}=-\mathbb E[\frac1t \log p_\theta(X_0\mid I,P,X_t)])，本质上就是从“图像+提示词+部分遮蔽答案”恢复完整答案；其中只有 mask 掉的位置参与 loss，所以作者又提出 **Complementary Masking**：同一条样本造两个互补 mask 版本，保证所有答案 token 最终都被监督到，特别适合 VQA 这类短答案任务。推理时从全 mask 序列开始，走 (K) 个扩散步逐步去 mask；为了快，作者设计了 **Prefix-DLM**，让图像 token 和 prompt token 只彼此注意、答案 token 再去注意全部，从而可以缓存图像和 prompt 的 KV；为了在少步数下不崩，作者又引入 **Schedule Shift**：(t_i'=\frac{\alpha t_i}{1+(\alpha-1)t_i})，用 (\alpha=\frac13) 让前期更快地释放 token，提高低 NFE 采样质量。效果上，LaViDa 已经证明这条路线是成立的：例如 MMMU 43.3，高于 LLaVA-1.6-7B 的 35.1 和 Open-LLaVA-Next-8B 的 37.4；MathVista 44.8 也明显高于同量级 AR 基线；在 COCO caption 上，用 Prefix-DLM 时 100% NFE 可达 117.3 CIDEr、1.93s/sample，而不缓存时是 121.0 CIDEr、7.65s/sample，速度最高可提升到约 3.9x；在受约束诗歌补全里，LaViDa/LaViDa-FIM 约束满足率达到 100%，说明 diffusion 的双向建模确实带来了 AR 模型不擅长的 controllability。

**2. Lavida-O：把 LaViDa 从“理解模型”升级成真正统一的理解-生成-编辑-定位模型。**  
这篇论文的背景是：统一多模态模型已经从“理解”和“生成”分家，走向 GPT-4o、BAGEL 这类统一框架，但大多数强模型要么是 AR+continuous diffusion 的双目标混合，要么是统一 AR；而之前的统一 MDM（比如 MMaDa、Muddit）虽然概念上更整洁，却在任务覆盖面、图像分辨率、编辑能力和总体效果上还不够强。Lavida-O 要解决三件事：第一，怎样在不重新从零训练一个超大生成模型的前提下，把 LaViDa 补上高质量图像生成；第二，怎样让统一 MDM 真正支持 object grounding、instruction-based editing、interleaved generation；第三，怎样显式利用“理解能力”去帮助“生成”，而不是只做简单混训。它的架构核心是 **Elastic-MoT**。数据流上，理解任务仍沿用 LaViDa：图像经 SigLIP 编成语义 embedding (C_i)，与 prompt embedding (C_p) 拼起来；但生成/编辑时，目标图像会先经 **VQ-Encoder** 离散成 VQ token，编辑任务还把输入图像的 VQ token (C_v) 也作为条件拼进去，于是条件变成 (C=\text{Concat}(C_i,C_v,C_p))，而输出序列 (X_0) 则既可能含文本 token，也可能含图像 VQ token。为了降算力，作者加了一个 **token compression**，把 VQ token 数压到原来的 1/4。Elastic-MoT 的细节很重要：理解分支沿用 8B LaViDa，生成分支不是等宽复制，而是做成更小的 2.4B，hidden size 从 4096 缩到 2048；32 层里前 16 层做 joint attention，让文字和图像交互，后 16 层分模态自注意力。这样理解任务只激活 8B，纯生成任务只需前半段理解分支 4B + 全部生成分支 2.4B，共 6.4B，interleaved 任务才激活全 10.4B。另一个关键设计是 **modality-aware masking**：因为 MDM 是并行解码，模型必须事先知道哪些 mask 属于文本、哪些属于图像，所以作者引入特殊 token **[exp]**；在前向扩散（加噪）里，被完全遮蔽的图像块会在 (t_{exp}) 时刻塌缩成一个 [exp] 文本 token，推理时只要模型生成 [exp]，就把它展开成一串图像 mask token，后续路由到生成分支。训练方面共三阶段：Stage 1 在 LaViDa 基础上继续做 grounding 和 image-level understanding；Stage 2 加入 2.4B 生成分支，只训练生成分支，做 text-to-image 预训练，并把分辨率逐步从 256 提到 512 再到 1024；Stage 3 再把 8B+2.4B 全部端到端联合训练，数据同时覆盖 text-to-image、image understanding、grounding、image editing、planning、reflection。数据上作者用了 2 亿级过滤后的图文对，外加理解、区域级理解、编辑和规划/反思数据。推理时，除标准 MDM 逐步去 mask 外，又加入三套增强：**Universal Text Conditioning** 把分辨率、裁剪、质量分数、亮度、对比度等 micro-condition 直接写成文本附到 prompt 末尾；**Stratified Random Sampling** 用分层 2D 采样代替“高置信 token 先出”的策略，避免图像邻近 token 同时释放带来的相关性问题；**Planning/Reflection** 则让模型先出 layout/bbox 再出图，或者先自检图像是否满足 prompt，不满足就继续修正，这其实把理解能力显式变成了生成时的“内部评审器”。公式上它仍然服从标准 MDM 目标 (L_{MDM}=-\mathbb E[\frac1t\log p_\theta(X_0|X_t)])，本质没变，只是 (X_0/X_t) 已扩展为文本+VQ 图像混合序列；grounding 的关键公式/规则是把坐标归一化到 ([0,1])，再量化成 1025 个 bin（(0/1024) 到 (1024/1024)），所以一个框永远只要 4 个 token，而且可以并行解多个框。效果上，这篇是三篇里最“全能”的：理解上 MMMU 45.1、ChartQA 80.0、MathVista 56.9，相比 LaViDa 明显变强；文生图上 GenEval 0.77、DPG 81.8、FID-30k 6.68，已经明显超过 MMaDa/Muddit，开 planning 后 GenEval 到 0.85，开 reflection 到 0.89；grounding 上 RefCOCOg test 可到 90.6；图像编辑总分 3.71，而加 planning 后到 3.80，并且在 Replace/Remove 这类强依赖局部理解的类别上超过 GPT-4o；速度上对 grounding 相比 Qwen2.5-VL-7B 号称有 6.8x 加速，而 Elastic-MoT 训练速度比标准 BAGEL 式 MoT 快 3.17x。

**3. Sparse-LaViDa：不改模型能力边界，专门解决统一 MDM 的“推理太浪费”。**  
这篇论文的背景是：即使 Lavida-O 已经比很多 AR 体系更统一，MDM 仍然有一个根本低效点——每一步都要把大量“其实什么信息都没有”的 masked token 喂进模型，而且全注意力使得 KV cache 很难像 AR 一样直接用起来；已有加速方案如 Block Diffusion 虽然快，但基本靠把 MDM 改成半自回归/块因果形式，会损失双向上下文，尤其不适合图像生成、编辑、inpainting 这类天然没有严格 left-to-right 顺序的任务。Sparse-LaViDa 的核心贡献，是提出一种**稀疏参数化**：它不再把部分遮蔽序列 (X_t) 的所有 mask token 都显式展开，而只保留四类输入——prompt (p)、已经缓存的已解码 token、当前这一步真正要解码的 token 集 (C_k)、以及一组固定数量的 **register tokens**。直觉上，因为标准 MDM 的建模本来就假设 (p_\theta(X_0|X_t)=\prod_i p_\theta(X_0^i|X_t))，如果这一步只需要某些位置的预测，就没必要为所有位置输出 logits；更进一步，mask token 本身不携带内容，它们只表示“这里还没解出来”，因此可以被压缩表示。论文里用“`I have [m] dog [m] [m] [m]`”举例，说明部分 mask 序列可以改写成“非 mask token + 它们的位置 + 序列长度信息”；但直接只用一个长度 token 会损伤模型容量，所以作者引入固定 **64 个 register tokens**，它们共享同一个 `[reg]` 词表 token，但位置 id 分别是 (L+1,\dots,L+64)，相当于给被截断的 mask 区域留了一组可被注意力读写的“压缩占位记忆”。训练上，它不是重做预训练，而是从 LaViDa-O 初始化，做 100k 步 SFT；数据是 LaViDa-O SFT 数据的高质量子集，包括 20M text-image pair、MAmmoth-VL/VisualWebInstruct 理解数据和 GPT-Edit-1.5M 编辑数据，使用 64 张 H100。为了让训练和推理一致，作者设计了 **step-causal attention mask**：prompt 属于 block 0，已知 clean token 被随机分到 block 1..M，masked/register token 被分到 block M+1..M+N；规则是 clean block 只能看当前及以前 block，masked block 只能看 prompt/clean block 和同 block 的 masked/register，不能看其他 masked block。推理时流程是：先把 prompt 预填到 cache；第 (k) 步只输入上一步新解出的 token、当前待解码 token 和 registers；其中上一步 token 的 query 不能看当前待解码 token，这样它们的 KV 就能安全写进缓存；视觉生成的解码顺序沿用 LaViDa-O 的 stratified sampler，文本生成则按 block 做半 AR 采样。公式上，Sparse-LaViDa **没有发明新 loss**，仍然用标准 (L_{MDM})，所以它的理论定位不是“新任务”，而是“对标准 MDM 的等价但更高效的参数化”；关键改动发生在 (p_\theta(X_0^i|X_t)) 的计算图，而不是目标函数。效果上，它非常像一个“工程上很值”的版本升级：文生图 GenEval 从 0.77 到 0.78，同时端到端时延从 21.27s 降到 10.86s，约 1.95x；DPG 从 81.8 到 82.4；图像编辑从 3.71 到 3.79，同时时延从 63.98s 降到 22.55s，约 2.83x；MathVista 56.7，时延从 10.41s 到 3.72s，约 2.80x。它也不是完全零代价——部分理解基准有轻微回落，例如 MME/MMMU/MMB 略降，但 ChartQA、DocVQA、MathVerse 反而有所提升。消融也很说明问题：不用 step-causal mask，GenEval/DPG 会明显掉到 0.71/78.9；不做训练直接套稀疏推理，几乎会崩到 0.24/47.9；register token 数量上，64 个是最稳的。

**一句话抓主线：**  
如果把这三篇压成一句话，就是：**LaViDa 解决“MDM 能不能做多模态理解”，Lavida-O 解决“MDM 能不能统一做理解+生成+编辑并做到 SOTA 级别”，Sparse-LaViDa 解决“这个统一 MDM 能不能在保持双向上下文和生成质量的前提下显著提速”。** 后面你直接按论文名继续问，我就基于这三篇展开。



我看完你的整套 60 页 PPT 后，先给一个直判断：这套内容其实够讲 1 小时，问题主要不在“材料少”，而在“讲述线索还没被整理出来”。现在最大的共性问题是模板残留、空白占位页、章节编号不统一、结果页缺少一句 take-home message。尤其 Lavida-O / Sparse-LaViDa 段里还反复出现 “MMaDA”“PathFinder” 抬头，15 / 40 / 45 / 52 页还是空白，这会很影响组会观感。

三篇论文本身的演进线其实非常清楚：LaViDa 先证明 diffusion-VLM 做多模态理解是可行的；Lavida-O 把能力扩到统一的理解、生成、编辑、grounding；Sparse-LaViDa 再把统一 MDM 做快、做实用。你这场报告最好沿着这条“能力边界扩展 -> 系统统一 -> 推理加速”的主线讲，而不是三篇独立论文笔记式平铺。

题目页的元信息也建议统一口径：LaViDa 目前可确认是 NeurIPS 2025 Spotlight；作者主页把 Lavida-O 和 Sparse-LaViDa 分别列成 ICLR 2026、CVPR 2026。你现在题目页里“会议”和“时间”混写，容易让听众误以为是发表日期。更稳妥的写法是分成“Venue / arXiv date / code”。([加州大学洛杉矶分校计算机科学](https://web.cs.ucla.edu/~kwchang/bibliography/li2025lavida/?utm_source=chatgpt.com "LaViDa: A Large Diffusion Language Model for Multimodal Understanding"))

先改的优先级，我建议按这个顺序：

1. 先把所有错误抬头、错误编号、空白页处理掉。
    
2. 每一页结果页都补一句“这页你想让听众记住什么”。
    
3. 把大段文字改成“问题 -> 方法 -> 收益”的短句式。
    
4. 把整套 60 页压成“有效信息 45-50 页 + 过渡页 5-8 页”的节奏。
    

下面我按页给你改。

## 逐页修改建议

### 1-20：LaViDa

**P1 封面**  
加一条副标题：`From diffusion-VLM to unified MDM to sparse acceleration`。现在只像三篇论文并列罗列，缺一个总主线。你的姓名、组会名称、单位也建议补上。

**P2 LaViDa 题目页**  
把信息改成三行卡片：`一句话定位 / 作者与单位 / venue+code`。再补一句“一句话摘要”：`首个 diffusion-based VLM，核心解决训练效率、KV cache 和低步数采样问题`。时间建议不要只写一个日期，改成 `NeurIPS 2025 Spotlight / arXiv v3 Jun 2025` 更清楚。([加州大学洛杉矶分校计算机科学](https://web.cs.ucla.edu/~kwchang/bibliography/li2025lavida/?utm_source=chatgpt.com "LaViDa: A Large Diffusion Language Model for Multimodal Understanding"))

**P3 背景：AR 路线**  
这一页字偏密，建议改成左右两栏：左边“AR VLM 代表”，右边“3 个痛点”。再补一个具体例子，比如“JSON 抽取/诗歌补全”，这样后面讲 controllability 更自然。

**P4 背景：Diffusion 趋势**  
这页信息太满，最好拆成“为什么想用 diffusion”+“LaViDa 要解决什么”两个层次。还有一个概念问题：如果你这里讲的是“离散 diffusion language model backbone”，更直接的代表建议写 `LLaDA / Dream`；`Show-o / MMaDA`更适合放到“多模态相关工作”里，避免概念混层。

**P5 主要贡献**  
不要用三大段整句，改成“三个标签 + 一句解释”：`首个 diffusion-VLM / 训练与推理tricks / 系统设计分析`。再在页脚补两个最硬数字：`+4.1 CIDEr`、`1.92x speedup`。

**P6 模型架构**  
这页图很好，建议只补两处：一是把 `condition` 和 `diffusion target` 用颜色标出来；二是在图上直接写 “Vision path” 和 “DLM path”。现在图是对的，但你讲的时候容易散。

**P7 pipeline**  
先改拼写，`pipline -> pipeline`。其次，这页数字太多了，适合改成流程箭头，只保留最关键的 4 个数字：`5 views / 3645 -> 980 / MLP projector / only answer is diffused`。其它数字放讲稿里说就够了。

**P8 训练目标**  
公式页需要一个“符号图例”，至少把 `I / P / Xt / X0 / t` 标注一下。页标题也可以更直白：`本质：conditioned masked denoising`。不然听众很难迅速进入状态。

**P9 Complementary Masking**  
建议把“dog”那个例子做成可视化小示意，而不是纯文字解释。你这里最该强调的不是“两个 mask”，而是“关键答案 token 不再漏监督”。

**P10 Complementary Masking 消融**  
不要直接扔表。建议把最好结果高亮，再补一句大字结论：`Complementary masking 在 4 个 benchmark 上全部提升`。这一页目标不是让大家看完表，而是记住“这招有效”。

**P11 Prefix-DLM**  
逻辑很好，但还是偏长。建议重排成三层：`为什么原始 DLM 不能 cache -> Prefix mask 怎么改 -> 带来什么收益`。把 attention mask 图放大，正文缩短。

**P12 Prefix-DLM 消融**  
这页更适合换成条形图：`Latency` 做主轴，`CIDEr` 作为旁边小标。现在表格不利于讲，重点会丢。

**P13 Timestep Shifting**  
这里标题写成“训练 trick”不太准，建议改成“采样 / inference trick”。再补一张 schedule 曲线图，比只写公式更好懂。

**P14 Timestep Shifting 消融**  
这里有明显编号错误，标题还写成了 `2.4.1`，应改成 `2.4.3`。同时建议高亮 `alpha=1/3` 这一行，并加一句结论：`低 NFE 下 convex schedule 更稳`。

**P15 实验细节**  
这一页现在是空白，建议二选一：要么删掉并入 P16；要么补一页真正的 setup：`stage-1 projector only / stage-2 end-to-end / 558K pretrain + 1M SFT`。

**P16 多模态理解评估**  
表太大，建议只保留 4 个最支持你故事的任务：`MMMU / MathVista / ChartQA / ScienceQA`。然后把 OCR 劣势单独做成红框说明，这个点你现在其实总结得不错。

**P17 可控生成 / 双向推理**  
现在信息量不够。建议直接放论文里的 poem completion 例子，并用颜色标出 AR baseline 不满足约束的地方。否则这页说服力不强。

**P18 速度-质量 tradeoff**  
散点图要配一句大字总结：`NFE 50%-75% 时，LaViDa 已经同时做到更快+更好`。并标出你自己推荐的 operating point。

**P19 同类型比较**  
这页目标不清楚。建议改标题，比如：`LaViDa 与 AR / diffusion baselines 的综合对比`。然后删成 3-4 行对比，不要保留全表。

**P20 LaViDa 结论**  
建议重写成三栏：`最重要贡献 / 当前短板 / 对下一篇的启发`。尤其最后一栏写：`LaViDa 证明了“理解可行”，但还没做到统一生成`，这样自然过渡到 Lavida-O。

---

### 21-41：Lavida-O

**P21 Lavida-O 题目页**  
同 P2，补一句一句话定位：`在 LaViDa 上加入 generation branch，统一做 understanding + generation + editing + grounding`。时间信息建议拆成 `ICLR 2026 / arXiv Sep 2025 / code`。 ([Jack Li Homepage](https://homepage.jackli.org/?utm_source=chatgpt.com "Shufan Li"))

**P22 背景三路线**  
章节编号错了，`一.背景介绍` 下不该写 `2.2`。建议改成一个 3 列比较表：`AR+diff / unified AR / unified MDM`，每列只留“优点 1 个 + 问题 1 个”。

**P23 待解决问题**  
顶部抬头居然还是 `MMaDA`，必须改成 `Lavida-O`。这页内容本身是好的，但建议给三条问题分别加“对应解决方法”：`训练太贵 -> Elastic-MoT`、`MIGM 生态弱 -> sampling+conditioning tricks`、`理解生成未联动 -> planning/reflection`。

**P24 主要贡献**  
抬头同样要改。正文建议压成 3 条关键词：`统一能力完整化 / 高效架构 / 理解驱动生成`。再补一句 benchmark 证据：`RefCOCO / GenEval / ImgEdit`。

**P25 模型架构**  
这一页开始顶部变成 `PathFinder（ICCV 2025）...`，这是最影响专业感的问题之一，必须全段统一替换。图本身可用，但建议加颜色图例解释 `Ci / Cv / Cp / Xt`。

**P26 pipeline**  
结构基本对。建议把“输入条件”和“目标序列”分成左右两个框，减少纯文字堆积。并补一句：`editing task 需要 Cv，是因为仅靠语义 embedding 不够保留低层细节`。

**P27 训练目标**  
这一页和 LaViDa 重复度高。建议改成“和 LaViDa 相同，但 X0 现在可同时包含 text token 和 VQ token”，这样节省时间。

**P28 Elastic-MoT 背景**  
建议加一个小表：`dense / vanilla MoT / Elastic-MoT`。这样听众能立刻明白为什么不是简单复制一套 branch。

**P29 Elastic-MoT 设计**  
这是 Lavida-O 最重要的一页之一。建议把 `N=32, M=16, K=16` 画成层级条，并单独列 `understanding: 8B / generation: 2.4B / T2I active: 6.4B`，这比文字更清楚。

**P30 modality-aware masking 提问页**  
这个“为什么 MDM 不知道哪些 mask 未来是图像 token”问题提得很好。建议补一个最小例子序列，为后面 [exp] 做铺垫。

**P31 modality-aware masking 图页**  
这一页图能用，但你得在图上直接加 2-3 个 callout：`collapse to [exp] / expand during inference / routed to gen branch`。否则现场很难看懂。

**P32 modality-aware masking 解释页**  
建议把现在的段落改成“三步走”流程卡片，不要大段写。核心句只保留一句：`先预测哪里需要展开，再把该位置扩成 image-token block`。

**P33 [exp] 对应的新目标**  
这个点很重要，但现在太抽象。建议做成左右对比：`t < t_exp` 和 `t > t_exp`。公式可以保留，但不是主视觉。

**P34 任务特定 trick**  
内容是对的，但建议拆成两个并排 box：`Universal Text Conditioning` 和 `Stratified Sampling`。每个 box 只写“动机 / 做法 / 收益”。

**P35 图像理解结果**  
把表改小，只保留最关键的 4-5 个指标。你现在那句“在 MDM 领域内显著提高”还不够，最好直接写成“vs MMaDa: MMMU xx -> xx，ChartQA xx -> xx”。

**P36 文生图结果**  
目前是全表截图，适合论文笔记，不适合组会讲。建议只高亮 `LaViDa-O / +Planning / +Reflection / BAGEL / Flux-dev / MMaDa` 这几行，并在右侧放一句总结：`planning/reflection 把 GenEval 从 0.77 提到 0.89`。

**P37 Object Grounding**  
很适合讲，建议重点突出 `1-step` 和 `4-step` 两个版本，说明统一 MDM 连 grounding 都能高效做。

**P38 图像编辑**  
表格非常密。建议只拿 `overall / replace / remove` 三列，原因是这三个最能体现“localized understanding really helps editing”。

**P39 速度效率**  
四个小图直观，但要加一句解释：`前三个是 inference latency，最后一个是 training step latency，单位不同`。否则容易被误读。

**P40 消融实验**  
空白页，建议删掉，或者补成“一个总表总结 Elastic-MoT / UTC / stratified / planning-reflection 各自贡献”。

**P41 Lavida-O 结论**  
顶部抬头要改。正文建议做成三列：`推进了什么 / 仍有哪些弱点 / 为什么 Sparse-LaViDa 是自然下一步`。这样能把第三篇顺出来。

---

### 42-60：Sparse-LaViDa

**P42 Sparse-LaViDa 题目页**  
顶部抬头仍是 `MMaDA`，必须改。另一个明显问题是日期写成了 `2026-12-16`，而论文 PDF 首页是 `2025-12-16`；如果你要写 `CVPR 2026`，那就把 arXiv 日期单列，不要混成一个“时间”。 ([Jack Li Homepage](https://homepage.jackli.org/?utm_source=chatgpt.com "Shufan Li"))

**P43 背景**  
这页内容不错，建议再补一个红框：`目标不是再扩能力，而是在保持能力的前提下显著加速`。这能立刻区分它和前两篇。

**P44 大图：Standard MDM vs Block Diffusion vs Sparse-LaViDa**  
图很好，但缺讲解。建议在三列下面分别再写 1 行中文标签：`原始 MDM / 牺牲双向上下文换速度 / 同时保留速度与双向上下文`。

**P45 主要贡献**  
这一页现在基本是空白，必须补。建议直接写 3 条：`sparse parameterization / register tokens / step-causal mask`。

**P46 模型架构 / inference with sparse parameterization**  
顶部 `PathFinder` 要改。底部建议加一句 take-home：`只 materialize 当前 step 真正需要解码的 token 子集`。这页要让听众一眼明白为什么会快。

**P47 sparse parameterization 直觉**  
建议把 toy example 做成图，而不是现在这种偏文字解释。核心是“mask token 只告诉你位置被遮了，并不携带语义”。

**P48 进一步解释**  
现在一句话太少，像半成品。建议补一个复杂度层面的表达，例如“从 dense sequence processing 变成 subset decoding”，否则这页价值不大。要么并到 P47。

**P49 Register tokens**  
这一页要回答两个问题：`为什么需要 register`、`为什么是 64 个`。建议把这两句直接写成标题，并把消融结果高亮。

**P50 data flow**  
内容很重要，但很容易讲乱。建议把四类 token 颜色和编号固定下来：`1 prompt / 2 cached clean / 3 decode-now masks / 4 registers`，并和图一一对应。

**P51 step-causal attention mask**  
建议把页首改成一句结论：`训练时模拟推理 cache 写入路径，消除 train-inference gap`。然后再展开两个作用点。

**P52 实验细节**  
空白页，建议补真正的 setup：`initialize from LaViDa-O / filtered SFT data / 100k steps / 64 H100 / evaluated on generation + editing + reasoning`。这一页不能空，因为这篇是“加速方法”，听众会天然关心它是不是还额外训练了。

**P53 文生图结果**  
建议只留 `LaViDa-O vs Sparse-LaViDa` 两行，并在右侧写一句结论：`GenEval/DPG 基本不掉点，latency 约 1.95x 加速`。

**P54 图像编辑结果**  
不要整表塞进来。改成“overall score + latency speedup + 1 个最能说明问题的类别”就足够了。

**P55 图像理解 / 数学推理**  
这一页是为了证明“不是只加速图像生成”。建议把标题改成：`加速不仅适用于 generation，也覆盖 reasoning tasks`。然后只高亮 2 个任务。

**P56 消融：速度来源拆解**  
很适合讲。建议加分层解释：`只 cache`、`cache + truncation`、`full sparse training adaptation`。这样逻辑更清楚。

**P57 消融：训练策略**  
这页干净、好讲。建议只再补一句：`-No Training 崩得很厉害，说明 sparse inference 不能只靠 test-time hack`。这一句是整篇的关键说服点。

**P58 定性展示 1**  
现在像 appendix 图墙。建议每组图上方加 prompt，底部加 1 句你希望观众看到的点，比如“身份保持”“复杂组合关系”“细节一致性”。

**P59 定性展示 2**  
同 P58。两页里最好一页放文生图，一页放编辑，这样更有组织。现在视觉上略散。

**P60 总结**  
这页思路其实对，但表达方式还可以更强。建议不要用大段横向表格，改成一条演进链：  
`LaViDa = 先证明 diffusion-VLM 能做理解`  
`LaViDa-O = 再把它扩展成 unified MDM`  
`Sparse-LaViDa = 最后把 unified MDM 做到更可用`  
然后收尾用一句你自己的判断：`这三篇工作的真正价值，是把 diffusion 多模态从“可行”推进到“可用”。`


---


我按“**backbone / encoder / 训练 / 推理 / 实验设置**”来整理，只写原文明确出现的内容；原文**没有单独定义 text encoder**的地方，我会直接说明“没有单独模块”。对应 PDF 分别是：LaViDa 、Lavida-O 、Sparse-LaViDa 。

## 1. LaViDa

**1）Backbone 与 encoder（见 p.4）**

- 整体结构是 **vision encoder + diffusion language model + MLP projector**。
    
- **图像 encoder** 用的是 **SigLIP-400M**。图像先被 resize 到 (768^2)，再切成 4 个不重叠的 (384^2) 视图，并额外保留 1 个整图 (384^2) 视图，一共 5 个 views。每个 view 输出 (27^2) 个 embedding，总计 3645 个 embedding；之后做 (2\times2) average pooling，变成每个 view (14^2)，总共 980 个视觉 token，再 flatten + concat 后送进 projector。
    
- **语言 backbone / diffusion backbone** 用的是 **LLaDA-8B（默认）**，也实验了 **Dream-7B**。
    
- **文本侧**：原文没有单独再放一个独立 text encoder；文本 prompt (P) 和部分 mask 的回答 (X_t) 是直接作为 diffusion language model 的输入。
    

**2）训练步骤（见 p.4–6，附录 p.18–19）**

- 训练样本由 **图像 (I) + 文本 prompt (P) + 干净答案 (X_0)** 组成。多轮对话时，作者会抽一轮作为 answer，把历史当作 prompt。
    
- 先从 ([0,1]) 采样 timestep (t)，再把答案 (X_0) mask 成 (X_t)，模型学习条件反向过程 (p_\theta(X_0\mid I,P,X_t))。**loss 只算在被 mask 的 token 上**。
    
- 作者专门加了 **Complementary Masking**：同一个样本造两份互补 mask 的 (X_t) 和 (X_t^C)，让所有 token 最终都能参与训练；视觉 embedding 在这两份样本间直接 copy，以提高训练效率（p.5）。
    
- 主体训练是 **两阶段**：
    
    - **Stage-1 / pretraining**：**只更新 projector**，把视觉 embedding 对齐到 diffusion LM 的 latent space（p.6）。
        
    - **Stage-2 / finetuning**：**所有组件联合端到端训练**，做 multimodal instruction following（p.6）。
        
- 数据规模：
    
    - **Stage-1**：**558K** image-text pairs（正文 p.6；附录写明是 **LCS-558K**，p.18）
        
    - **Stage-2**：**1M** visual instruction-following examples（p.6）
        
- 附录 B.1 还给了训练细节：**AdamW**、学习率 **5e-3**、cosine decay；Stage-1 训 **1 epoch**，Stage-2 训 **2 epochs**（p.18）。
    
- 另外还有两个 **Stage-3 专门模型**：
    
    - **LaViDa-Reason**：用 **19.2K** long CoT 数据继续训练，teacher 是 **VLRethinker-7B**，数据来自 **ViRL-39K** 过滤后结果（p.7，附录 p.19）。
        
    - **LaViDa-FIM**：用 **20% 的 stage-2 数据**继续训练，训练时在文本中间插入随机长度的 `[S]...[S][FIM]` 片段，用来做可变长度 infilling（p.8）。
        

**3）推理步骤（见 p.5–6，p.8）**

- 标准推理流程是：先造一个长度为 (L) 的全 mask 序列 (X_1)，再沿着 (K) 个离散时间步逐步去 mask，直到得到无 mask 的 (X_0)。
    
- 每一步都先通过 (p_\theta(X_0\mid X_{t_i})) 采一个“全展开”的序列，再按下一时刻 (t_{i+1}) 重新 mask 一部分 token，继续往下走（p.5）。
    
- 默认评测设定是 **(K=L)**，也就是 **NFE=100%**（p.6）。
    
- 论文提出两个关键推理改动：
    
    - **Prefix-DLM**：视觉 token 和 prompt token 只彼此注意，answer token 才看全体，这样就能给多模态 prefix 做 **KV cache**（p.5）。
        
    - **Schedule Shift / timestep shifting**：用  
        [  
        t'_i=\frac{\alpha t_i}{1+(\alpha-1)t_i},\quad \alpha=\frac13  
        ]  
        让前期更早地解出更多 token；并且保证每一步至少解 1 个 token（p.6）。
        
- 做 **text infilling** 时，给一个带 (L_M) 个 mask 的草稿，直接跳到 (t=L_M/L) 再跑标准推理；**LaViDa-FIM**则在 masked segment 后面加 `[FIM]`，允许生成长度可变的补全文本（p.8）。
    

**4）实验细节（见 p.6–9，附录 p.19）**

- 默认主结果用的是 **stage-2 + LLaDA-8B**（p.6）。
    
- 评测工具是 **lmms-eval**（p.6, p.19）。
    
- 评测任务覆盖：
    
    - 通用理解：**MME-P, VQAv2, MMBench, MMMU**
        
    - 推理：**MME-C, MathVista, MathVerse, MathVision**
        
    - 科学：**ScienceQA, AI2D**
        
    - OCR：**TextVQA, DocVQA, ChartQA, InfoVQA**
        
- speed-quality 实验在 **COCO 2017 val 的 500 张图**上做，最大生成长度 **32**，测试 (K\in{32,24,16,8})，单卡 **A5000** 上统计 latency 和 CIDEr（p.8–9）。
    

---

## 2. Lavida-O

**1）Backbone 与 encoder（见 p.4–7，附录 p.25–26）**

- 这篇是建立在 **LaViDa** 之上的 unified masked diffusion model。
    
- **理解分支**沿用 LaViDa：用 **SigLIP** 把输入图像变成连续语义 embedding (C_i)（p.4）。
    
- 为了支持图像生成，作者再引入 **VQ-Encoder / VQ-VAE**，把目标图像转成离散 VQ token（p.4–5）。附录明确写了：**VQ encoder 采用 Meissonic 的 encoder**（p.25）。
    
- 文本侧用的是 **prompt embedding (C_p)**；原文同样**没有把 text encoder 单独拆成一个独立模块**，而是把 prompt embedding 与视觉条件拼接后送入统一 diffusion model（p.4–5）。
    
- 对图像编辑和 interleaved generation，条件输入是  
    [  
    C=\text{Concat}(C_i, C_v, C_p)  
    ]  
    因为只用语义 embedding 不足以保留编辑所需的低层细节（p.5）。
    
- 生成/编辑分支用了 **token compression**，把 VQ token 数量压到原来的 **1/4**（p.5）。
    
- 核心 backbone 设计是 **Elastic-MoT**：
    
    - 理解分支来自 LaViDa，是 **8B**
        
    - 新增生成分支是 **2.4B**
        
    - 一共 **32 层**，前 **16 层**做 joint attention，后 **16 层**文本和图像各自 self-attention（p.5–6）
        
    - 任务激活参数量：
        
        - text-to-image：**6.4B**
            
        - understanding：**8B**
            
        - interleaved task：**10.4B**
            

**2）训练步骤（见 p.7，附录 p.25–26）**

- 训练是 **三阶段**：
    
    - **Stage 1**：继续训练 base model，让它覆盖 **object grounding + image-level understanding**（p.7）
        
    - **Stage 2**：加入 **2.4B image generation branch**，做 **text-to-image pretraining**；训练时分辨率从 **256 → 512 → 1024** 逐步提高（p.7）
        
    - **Stage 3**：把整个 **2.4B + 8B** 模型端到端联合训练在 **image understanding / T2I / image editing / interleaved generation（planning 与 self-reflection）** 上（p.7）
        
- 附录表 7 给了更细的 stage 配置（p.26）：
    
    - **Stage 1**：LR **5e-6**，**80k** steps，数据用 **B,C**
        
    - **Stage 2**：LR **1e-4**，**400k** steps，数据用 **A**
        
    - **Stage 3**：LR **2e-5**，**100k** steps，数据用 **A,B,C,D,E**
        
    - 三阶段优化器都用 **AdamW**，(\beta_1=0.99,\beta_2=0.999)
        
- 数据按附录分成五类（p.25–26）：
    
    - **A: Text-to-Image**：LAION-2B、COYO-700M、SA-1B、JourneyDB、BLIP3o-60k、ShareGPT4o-Image，强过滤后得到 **200M** 图像
        
    - **B: Image-level understanding**
        
    - **C: Region-level understanding**：GranD、RefCOCO
        
    - **D: Image editing**
        
    - **E: Planning / reflection** 数据，作者用 **GroundingDINO-L** 等工具构造
        
- Stage 1 还有一个 **dataset mix scheduler**：新能力（grounding）与旧能力（image-level understanding）的采样比会从 **3:1** 逐渐衰减到 **1:3**（p.26, p.28）。
    

**3）推理步骤（见 p.4–7，附录 p.25）**

- 通用层面上，它仍然沿用 MDM 的标准推理：从全 mask 序列开始，逐步 unmask（p.4）。
    
- 这篇新增的关键推理机制有四个：
    
    1. **Modality-aware masking**（p.6–7）  
        图像 VQ token 在某个特殊时刻 (t_{exp}) 会先折叠成一个特殊文本 token `[exp]`；推理开始时默认所有 mask 都当作文本 token，等模型生成出 `[exp]` 后，再把它展开成 (L_{img}) 个图像 mask token，并在后续步骤里交给 generation branch。
        
    2. **Universal text conditioning**（p.6）  
        不额外做专门 embedding，而是把分辨率、裁剪坐标、score、亮度、对比度等条件直接写成文本，拼到 prompt 后面。
        
    3. **Stratified random sampling**（p.6–7）  
        先从 (2\times2) 网格开始，每个区域先解一个 token，再递归细分成更小子区域继续解；这样避免高置信 token 过度聚集在局部区域。
        
    4. **Planning + Reflection**（p.7，附录 p.25）
        
        - **Planning**：先生成布局/框，再据此生成图像；做编辑时先定位要编辑的区域。
            
        - **Reflection**：先生成一张图，再利用模型自身的理解能力做 self-critique；不符合用户要求就继续生成修正版。附录写到由于上下文长度是 **8192**，历史最多保留 **3 轮**（p.25）。
            
- 对 **object grounding**，作者采用 **coordinate quantization**：把坐标归一化到 ([0,1])，再量化成 **1025 个离散 token**；每个 bbox 固定用 **4 个 token** 表示，推理时可并行解多个框（p.7）。
    

**4）实验细节（见 p.8–9，附录 p.30）**

- 主要实验分四类：
    
    - **Image understanding**：MMMU、MME-P/C、MMB、ChartQA、DocVQA、InfoVQA、ScienceQA、AI2D、MathVista、MathVerse（p.8）
        
    - **Text-to-image**：GenEval、DPG、MJHQ-30k/FID（p.8）
        
    - **Object grounding**：RefCOCO / RefCOCO+ / RefCOCOg（p.9）
        
    - **Image editing**：Image-Edit benchmark（p.8–9）
        
- 速度实验比较 **text-to-image / grounding / math reasoning** 三类推理 latency，同时还比较了 Elastic-MoT 相对 Bagel-style MoT 的训练速度（p.9）。
    
- 原文报告：
    
    - grounding 相比 Qwen2.5-VL-7B 有 **6.8×** speedup（p.9, p.30）
        
    - Elastic-MoT 训练速度相对标准 MoT 有 **3.17×** speedup（p.9）
        
- 计算成本：附录写总训练在 **8 台节点 × 每节点 8 张 A100** 上完成，总 wall-clock **34.2 天**，约 **53k GPU hours**（p.30）。
    

---

## 3. Sparse-LaViDa

**1）Backbone 与 encoder（见 p.6，附录 p.16–17）**

- 这篇不是从零做新 backbone，而是**直接用预训练好的 LaViDa-O 权重初始化**，也就是一个 **10.4B unified diffusion model**（p.6）。
    
- 它真正新增的不是新的 dense backbone，而是三项“稀疏化/加速”设计：
    
    - **Sparse parameterization**
        
    - **Register tokens**
        
    - **Step-causal attention mask**（p.2–6）
        
- **encoder 方面**，这篇论文本身**没有重新设计新的文本或图像 encoder**；它是对 LaViDa-O 做 SFT/post-training（p.6, p.16）。
    
- 所以如果严格按 Sparse-LaViDa 本文的说法，只能说：**encoder 配置继承自 LaViDa-O**。而在 LaViDa-O 原文里，这套配置就是：
    
    - 语义图像 encoder：**SigLIP semantic encoder**
        
    - 图像离散 tokenizer / VQ encoder：**Meissonic 的 VQ encoder**
        
    - 文本侧：**prompt token / prompt embedding**，并没有单独拆出来一个独立 text encoder。
        

**2）训练步骤（见 p.6，附录 p.16–17）**

- 训练目标仍然是**标准 MDM objective**，作者特别说明**没有额外 distillation stage**；变化主要在于如何实现 (p_\theta(X_0|X_t))——改成 sparse parameterization + step-causal mask（p.6）。
    
- 训练方式是 **SFT / post-training**：在 LaViDa-O 的 dense parameterization 基础上，继续微调成 sparse 版本（p.6）。
    
- 正文 4.1 写的训练数据是（p.6）：
    
    - understanding：**MAmmoth-VL、VisualWebInstruct**
        
    - T2I：从 **LAION-2B、COYO-700M、SA-1B、JourneyDB、BLIP3o-60k、ShareGPT4o-Image** 里抽 **20M** text-image pairs
        
    - editing：**GPT-Edit-1.5M**
        
    - 训练 **100k steps**，用 **64 NVIDIA H100**
        
- 但附录 8.1/8.2 又写了另一版（p.16）：
    
    - 20M T2I 数据来自 **LAION-2B、COYO-700M、BLIP3o-60k、ShareGPT4o-Image**
        
    - 并且明确说 **不像 LaViDa-O，那样的 SA-1B 和 JourneyDB 这次没有用**
        
    - 主实验硬件写成 **64 A100（8 nodes）**
        
- 这两处**原文自己不一致**，所以我不替作者统一；这里只把冲突原样指出来。
    
- 附录表 9 给出的统一训练超参是（p.17）：
    
    - **AdamW**
        
    - LR **2e-5**
        
    - **100k** steps
        
    - (\beta_1=0.99,\beta_2=0.999)
        
    - loaded / trainable parameters 都是 **10.4B**
        
    - understanding 分辨率沿用 LaViDa-O 的多视角设定
        
    - generation 分辨率是 **1024**
        
- 附录还说整个训练耗时 **5 天**，大约是 LaViDa-O 从头训练预算的 **15%**（p.16）。
    

**3）推理步骤（见 p.3–5）**

- **Sparse parameterization**：部分 mask 的序列不再把所有 `[M]` 都显式保留下来，而是只保留
    
    1. 非 mask token
        
    2. 它们的位置
        
    3. 原始总长度  
        这样就能唯一恢复 mask 的位置（p.4）。
        
- **Register tokens**：作者发现只用一个特殊 token 会掉图像质量，所以最终用了 **64 个 register tokens**，位置固定放在序列尾部，而且数目不会随着被截断的 mask 数量增长（p.4）。
    
- **具体 sampling 流程**（p.3–4）：
    
    - 先把 prompt (p) 预填进 KV cache
        
    - 第 (k) 步输入由四部分组成：
        
        1. prompt
            
        2. 之前已经解出的 token
            
        3. 当前这一步要解的 token 集合 (C_k)
            
        4. register tokens
            
    - 其中 prompt 和更早一步解出的 token 已经在 cache 里；上一步新解出的 token 会在这一轮被处理并加入 cache
        
    - 模型**只对当前要解的 (C_k)** 输出 logits，然后采样、继续下一步，直到全序列解完
        
- **推理时的注意力规则**（p.3–4）：
    
    - 来自 (C_{k-1}) 的 query 只能看 ({p, C_1,\dots,C_{k-1}})，不能看当前待解的 (C_k)
        
    - register tokens 能看所有 token，但只有 (C_k) 和 register 本身会去看它们
        
- 不同任务的 unmask 顺序不同（p.4–5）：
    
    - **Text-to-image / image editing**：用从 LaViDa-O 继承来的 **stratified random sampler**
        
    - **文本生成 / image understanding**：用 **semi-autoregressive block sampling**；先把长序列切成大小为 (S) 的 block，再在 block 内用 confidence 决定哪些 token 解出来
        
- 作者特别强调：虽然默认 block 是左到右采样，但它**并没有退化成 block-causal**，仍然保留 **bidirectional context**，因此还支持 infilling / constrained generation（p.5）。
    

**4）训练时怎样匹配推理（见 p.5, p.16）**

- 为了让训练和推理一致，作者设计了 **Step-Causal Attention Mask**：
    
    - prompt 分到 **block 0**
        
    - clean token 随机分到 **1..M**
        
    - masked token 随机分到 **M+1..M+N**
        
- 注意力规则是：
    
    - clean block (i) 只能看自己和之前的 block
        
    - masked block (i) 只能看 prompt / clean block，以及**自己这个 masked block**
        
    - 每个 masked block 还会挂对应的 register tokens，规则跟 masked token 相同（p.5）
        
- 这相当于在一次训练里模拟多条实际推理路径，减少 train–inference gap（p.5）。
    

**5）实验细节（见 p.6–8，附录 p.16–17）**

- 评测任务包括：
    
    - **T2I**：GenEval、DPG、MJHQ-30k（p.6）
        
    - **Image editing**：ImgEdit（p.7）
        
    - **Understanding / reasoning**：MathVista，以及 MME-C、MMMU、ChartQA、DocVQA、MathVerse（p.7）
        
    - **Grounding**：RefCOCO / RefCOCO+ / RefCOCOg（附录 p.16–17）
        
- 除非特别说明，图像生成实验都是 **1024 分辨率，单张 A100** 上测试（p.6）。
    
- 论文报告的代表性 speedup：
    
    - **text-to-image**：**1.95×**
        
    - **image editing**：**2.83×**
        
    - **MathVista reasoning**：**2.80×**（p.6–7）
        
- 消融还专门看了三种提速来源：**prompt cache、response-token cache、token truncation**；三者一起开效果最好（p.8）。
    
- register 数量消融表明 **64 个 registers** 是文中最优设置（p.8）。
    

---

### 最后补一句

有两处我觉得你后面读论文时值得留意：

- **LaViDa**：附录里 stage-2 的 global batch size 描述有出入。
    
- **Sparse-LaViDa**：正文和附录对训练 GPU（H100/A100）以及是否用了 SA-1B/JourneyDB 的说法不一致。
    

除此之外，上面这份整理基本就是三篇原文里关于**实验设置、backbone、encoder、训练流程、推理流程**的核心信息。