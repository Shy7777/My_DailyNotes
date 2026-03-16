先澄清一下：你上传的三篇分别是 **LLaDA、LLaDA-V、MMaDA**；你后面提到的 **LaViDa / Lavida-O** 不在上传文件里，所以我另外查了它们的原始 arXiv 页面。下面我按“**三篇论文总结** + **LaViDa vs. Lavida-O 严谨对比**”来写。 ([arXiv](https://arxiv.org/abs/2505.16839 "[2505.16839] LaViDa: A Large Diffusion Language Model for Multimodal Understanding"))

## 总体脉络

这三篇上传论文其实构成了一条很清晰的扩散式大模型演进线：**LLaDA** 先回答“语言大模型的核心能力是否一定依赖自回归 next-token prediction”这个根问题，证明纯离散扩散语言模型也能在 8B 规模逼近主流 AR LLM；**LLaDA-V** 再把这个 diffusion language model 接到视觉指令调优框架中，重点解决“纯扩散 MLLM 能不能做强多模态理解”；**MMaDA** 则进一步把文本推理、多模态理解、文本到图像生成统一到同一个离散扩散框架里，并引入 Mixed Long-CoT 与 UniGRPO，把 post-training 也系统化。换句话说，研究重心从“扩散能否做 LLM”推进到“扩散能否做 MLLM 理解”，再推进到“扩散能否做统一多模态基础模型”。

## 1. LLaDA：把“LLM=AR”这个默认前提拆掉

LLaDA 的背景非常明确：在它之前，社区几乎默认认为大语言模型的可扩展性、in-context learning、instruction following 都是由自回归分解带来的；LLaDA 要解决的是，这种判断到底是“生成建模原则”带来的，还是“左到右 next-token factorization”带来的。它的核心贡献不是做一个 BERT 式掩码模型，而是把 **masked diffusion model** 作为一个**有似然下界支撑的生成模型**扩展到 8B 语言规模：前向过程把文本逐步 mask，反向过程用一个无 causal mask 的 Transformer 同时预测所有被 mask 的 token。其 data flow 很清楚：训练时是 `x0 -> 采样 t∈[0,1] -> 以概率 t 独立 mask 得到 xt -> mask predictor 预测被遮蔽 token`；SFT 时则是 `prompt 保持可见，response 被 mask`；推理时是 `prompt + 全 mask response -> 迭代预测 -> 低置信度 remask -> 直到得到完整 response`。从架构细节看，LLaDA-8B 基本沿 LLaMA3 风格搭建，但把因果注意力换成双向注意力，因而不能做普通 KV cache；它使用 32 层、4096 hidden size、32 attention heads、126,464 词表，FFN 维度 12,288，参数量约 8.02B，而对应的 LLaMA3-8B 用的是 grouped-query attention、FFN 14,336、词表 128,000。

LLaDA 最关键的公式是其训练目标  
[  
L(\theta)=-\mathbb E_{t,x_0,x_t}\left[\frac{1}{t}\sum_i \mathbf 1[x_t^i=M]\log p_\theta(x_0^i|x_t)\right],  
]  
直觉上它做了三件事：第一，只在 masked positions 上算交叉熵；第二，mask ratio 不是固定 15%，而是对 (t\sim U[0,1]) 全范围采样，因此模型学到的是“从各种损坏程度恢复文本”；第三，前面的 (1/t) 不是装饰，而是把目标和似然联系起来，使作者可以给出  
[  
-\mathbb E[\log p_\theta(x_0)]\le L(\theta),  
]  
也就是“训练目标是负对数似然上界”的理论保证。SFT 版公式把无条件建模换成条件建模 (p_\theta(r_0|p_0))，而 Eq.(6) 又把 response 中“恰好 mask 掉 (l) 个 token”的写法拿来做更稳定的条件似然估计。训练上，LLaDA-8B 从头预训练 2.3T tokens，序列长度 4096，训练成本约 0.13M H800 GPU hours；SFT 使用 4.5M instruction pairs。效果上，它的 8B Base 在 MMLU 上 65.9，和 LLaMA3-8B Base 的 65.4 基本持平，在 GSM8K、Math、CMMLU、C-Eval 上显著更强；SFT 后虽然在很多指标上仍落后带 RL 对齐的强 AR 指令模型，但在 ARC-C、Math、GPQA 等指标上仍有竞争力。更重要的是，它在“反向诗句补全”上显著缓解 reversal curse，reversal 得分 45.6，高于 GPT-4o 的 34.3，这说明它的双向生成偏置不是纸面优势，而是会落到真实任务上。

## 2. LLaDA-V：把 LLaDA 接到视觉指令调优，但仍然只做“理解型” MLLM

LLaDA-V 的背景是：2025 年前大多数 MLLM 仍然是 AR，少数扩散路线要么依赖 AR 语言塔，要么语言能力太弱；作者要解决的是**纯扩散训练、纯扩散采样的 MLLM，能不能在多模态理解上站住脚**。它的贡献不是把图像“塞给”LLaDA 那么简单，而是系统补齐了视觉对齐、对话建模、注意力设计、推理过程和训练日程。其模型架构是典型的 visual instruction tuning 三段式，但语言塔换成 LLaDA：`image -> SigLIP2-so400m-patch14-384 -> 729 visual tokens/image -> 两层 MLP projector -> LLaDA-8B-Instruct embedding space -> 与文本 prompt、masked response 拼接 -> 双向 diffusion language tower -> 同时预测所有 masked response token`。这里的关键架构决策是**选择 bidirectional attention，而不是 dialogue-causal attention**；作者做了消融后认为双向注意力在更多 benchmark 上更优。训练目标本质上是 LLaDA 条件目标的多模态扩展：图像特征和 prompt 始终可见，只 mask response，并只在 response 的 masked positions 上算 loss，因此它学到的是 (p_\theta(r_0|v,p_0)) 的反向扩散恢复。

训练流程上，LLaDA-V 采用三阶段：Stage 1 只训 projector，用 LLaVA-Pretrain 的 558K 样本做 language-image alignment；Stage 2 训全模型，先在 MAmmoTH-VL 的 10M 单图指令数据上学单图理解，再在约 2M OneVision 数据上扩展到单图/多图/视频；Stage 3 用 900K VisualWebInstruct 做 reasoning enhancement，并再用约 3M 的 OneVision+reasoning 混合数据做“/think”“/no_think”式平衡训练。推理时它和 LLaDA 一样，从一个全 mask 的 response 开始，反复“预测全部 mask -> 低置信度 remask -> 继续去噪”，因此不是左到右 next-token generation。效果上，LLaDA-V 最值得看的不是绝对分，而是**在同一训练管线下对 LLaMA3-V 的数据可扩展性与 controlled comparison**：它在 MMMU、MMMU-Pro、MMStar、MMBench 等知识/推理类 benchmark 上扩展性更强，甚至在 MMMU-Pro 上 1M 样本的 LLaDA-V 就能超过 9M 样本的 LLaMA3-V；完整评测里它相对同管线 LLaMA3-V 在多数学科知识与数学推理、多图和部分视频任务上占优，但在 AI2D、DocVQA、RealWorldQA 这类图表/文档/OCR 与真实场景理解上仍偏弱。这一点很重要：**LLaDA-V 证明了扩散式语言塔对“结构化推理与多模态上下文整合”有优势，但它还不是统一生成模型，也还没把 OCR/文档理解做到最优。**

## 3. MMaDA：不再只是“图像进、文本出”，而是统一文本推理、多模态理解、文生图

MMaDA 的背景与 LLaDA-V 不同：它关注的不是“扩散 MLLM 能否理解图片”，而是**能否用一个统一的离散扩散基础模型同时覆盖文本推理、多模态理解和文本到图像生成，并把 post-training 也统一起来**。因此它要解决的问题更大：既要统一预训练目标，又要统一 CoT，又要让 RL 能适配非自回归 diffusion。架构上，MMaDA 的范式也明显不同于 LLaDA-V：它不是“视觉编码器 + 语言塔”的 late fusion 理解模型，而是先把**文本和图像都离散化到统一 token 空间**。文本端用 LLaDA tokenizer，图像端用基于 MAGVIT-v2 的 quantizer；在 512×512 下，图像被编码为 (32\times32=1024) 个离散 token，码本大小 8192。于是 data flow 变成 `text/image -> 离散 token 序列 -> 统一 mask diffusion transformer -> 输出 text token 或 image token`。这意味着 MMaDA 是真正的“单一概率形式、单一掩码预测器、跨模态统一目标”，而不是把视觉特征投到语言空间后只生成文本。

MMaDA 的重要公式有三层。第一层是统一预训练目标  
[  
L_{\text{unify}}(\theta)=-\mathbb E\left[\frac{1}{t}\sum_i \mathbf 1[x_t^i=[MASK]]\log p_\theta(x_0^i|x_t)\right],  
]  
它和 LLaDA 同型，但对象从文本扩展到图像/文本混合 token；第二层是 Mixed Long-CoT 的 SFT 目标  
[  
L_{\text{Mixed-SFT}}=-\mathbb E\left[\frac{1}{t}\sum_i \mathbf 1[r_t^i=[MASK]]\log p_\theta(r_0^i|p_0,r_t)\right],  
]  
核心思想是把 `<reasoning_process>` 和 `<result>` 用统一格式组织起来，让文本推理、多模态推理、文生图三种任务共享 CoT 结构；第三层是 **UniGRPO**，它解决了 diffusion 里“没有 AR chain rule、sequence likelihood 不好算、mask ratio 会影响 policy 估计”的强化学习难点。作者通过对 response 随机采样 mask ratio、只在 masked positions 上估计 token log-prob，再把这些量拼成近似 sequence-level policy，构造出带 clipping 和 KL 的 diffusion 版 policy gradient 目标。直白地说，UniGRPO 的贡献不是“把 PPO 名字换一下”，而是把 RL 真正改造成适用于 masked diffusion 的形式。

训练与推理上，MMaDA 也是三阶段：Stage 1 用 RefinedWeb、ImageNet-1k 和图文数据做基础联合预训练；Stage 2 用 Alpaca、LLaVA-1.5 和多类 reasoning 数据做 Mixed Long-CoT finetuning；Stage 3 再用 UniGRPO 在数学/逻辑/多模态/文生图任务上做统一 RL。实现上，它从 LLaDA-8B-Instruct 初始化文本骨干，从 Show-o 初始化图像 tokenizer；整体训练在 64 张 A100 上进行。推理时，文本生成采用 **semi-autoregressive block denoising**：总长 1024、512 个去噪步、block size 64，每步在当前块里放出 2 个低置信度 token；图像生成则是**并行非自回归采样**，1024 个 token 对应 512×512 图像，50 个 timestep，配 cosine remask 和 CFG=3.5。效果上，MMaDA 在文本推理上达到 MMLU 68.4、GSM8K 73.4、MATH 36.0，整体强于 LLaDA-8B；在多模态理解上优于 Show-o、SEED-X 等统一模型；在图像生成上取得 WISE(Cultural) 0.67、ImageReward 1.15、CLIP Score 32.46、GenEval overall 0.63，优于文中对比的 SDXL、Janus 等多类基线。严格地说，它的真正亮点不是某一单项理解 benchmark 的最高分，而是**统一性与 post-training 完整性**。不过也要实事求是：若只看“纯多模态理解”这一窄任务，LLaDA-V 这类专用理解模型在部分 benchmark 上仍更强，MMaDA 的优势主要体现在“任务广度 + reasoning + generation 一体化”。

## 4. LaViDa vs. Lavida-O：这是从“扩散式视觉理解模型”走向“统一理解-生成扩散基础模型”的一次升级

如果你这里说的就是 **LaViDa** 和 **Lavida-O**，那么它们的时间节点非常清楚：LaViDa 首版提交于 **2025-05-22**，定位是**扩散式多模态理解模型**；Lavida-O 首版提交于 **2025-09-23**，定位已经升级为**统一多模态理解与生成的掩码扩散模型**。所以二者不是简单的“小改版”，而是研究目标发生了层级跃迁：LaViDa 主要回答“diffusion VLM 能否做强理解、能否更快更可控”，Lavida-O 回答的是“能否把理解、grounding、editing、1024px 文生图统一到一个 masked diffusion 系统里，并让理解反过来增强生成”。([arXiv](https://arxiv.org/abs/2505.16839 "[2505.16839] LaViDa: A Large Diffusion Language Model for Multimodal Understanding"))

从模型范式看，**LaViDa 更像 diffusion 版 LLaVA**。它的架构是 `SigLIP-400M 五视图编码 -> 3645 visual embeddings -> average pooling 到 980 tokens -> MLP projector -> diffusion language model(LLaDA-8B 或 Dream-7B) -> 文本回答`，本质上还是“视觉编码器 + 文本生成骨干”，只是把 AR LLM 换成了非因果的 diffusion LM。它的关键方法论是三点：**complementary masking** 提高训练样本效率，确保关键答案 token 最终都参与 loss；**Prefix-DLM** 通过特殊注意力掩码缓存图像和 prompt 的 KV，缓解 diffusion 因双向建模而难以缓存的问题；**timestep shifting / schedule shift** 则在低步数采样时改善质量，从而把 diffusion 的 speed-quality tradeoff 真正变成可用能力。它用两阶段训练：Stage 1 只训练 projector，对齐 558K 图文数据；Stage 2 用 1M visual instruction 数据端到端训练，之后再做 reasoning 和 FIM 的 stage-3 专门化。效果上，LaViDa-L 在其对比设置中 MMMU 43.3，高于同类 open-data AR 基线；COCO captioning 相对 Open-LLaVA-Next-Llama3-8B 提升 +4.1 CIDEr、速度 1.92x；Prefix-DLM 在 COCO 上可带来最高 3.9x 加速；在 constrained poem completion 上，相比 AR 基线相对提升 59%，且约束满足率可以做到 100%。([ar5iv](https://ar5iv.org/html/2505.16839v3 "[2505.16839] LaViDa: A Large Diffusion Language Model for Multimodal Understanding"))

**Lavida-O 则不是“把 LaViDa 再调一调”，而是架构层面跨到 unified diffusion foundation model。** 它明确写出自己 built on LaViDa，但在理解任务上保留 LaViDa 的 `semantic image embedding + text prompt -> text answer` 设置之外，又把目标图像表示成 VQ tokens，并在 image editing / interleaved generation 时把输入图像的 VQ token 也并入条件端。最核心的新架构是 **Elastic-MoT**：一个更大的 understanding branch 配一个更小的 generation branch，且只让模态在前几层做 joint attention，后面按模态分开 self-attention，这样理解任务、生成任务和交错任务可以动态加载不同参数子集。文中给的例子是：生成分支新增 2.4B 参数，文本到图像预训练时只训练这 2.4B；图像生成时激活 6.4B 参数，理解任务激活 8B，交错任务激活 10.4B。围绕这个统一模型，它又补上了 **modality-aware masking**、**token compression**（把用于编辑的 VQ 条件 token 压缩 4 倍）、**universal text conditioning**（把亮度、对比度、分数等条件直接写成文本）、**stratified sampling**（避免高置信 token 空间聚集导致图像质量下降），以及**planning + iterative self-reflection**（利用理解能力对生成进行规划和自批评）。这套方法论明显比 LaViDa 更“系统工程化”，因为它要解决的不只是理解，而是生成质量、编辑保真、grounding 和速度。([ar5iv](https://ar5iv.org/html/2509.19244v2 "[2509.19244] Lavida-O: Elastic Large Masked Diffusion Models for Unified Multimodal Understanding and Generation"))

效果层面，LaViDa 的强项是**多模态理解、可控文本输出、速度-质量折中**；Lavida-O 的强项则是**统一能力边界更大**。公开结果里，Lavida-O 在 RefCOCO/RefCOCO+/RefCOCOg 上超过 Qwen2.5-VL-7B 与部分 grounding specialist，在 text-to-image generation 与 image editing 上声称达到 SOTA，并给出“相对 Qwen2.5-VL 与 FluxKontext-dev 等基线，推理最高可到 6.8x speedup”；其消融还显示 planning 与 reflection 能把 GenEval overall 从 0.77 提到 0.89。用一句话概括：**LaViDa 解决的是“扩散 VLM 如何高效、可控地做好理解型任务”，Lavida-O 解决的是“如何把这种扩散理解能力扩展成统一理解-生成-编辑-规划系统”**。([arXiv](https://arxiv.org/abs/2509.19244 "[2509.19244] Lavida-O: Elastic Large Masked Diffusion Models for Unified Multimodal Understanding and Generation"))

## 5. 一个总判断

最严谨的结论是：**LLaDA、LLaDA-V、MMaDA 代表的是同一研究纲领的三步走**——先证明扩散可以做语言，再证明扩散可以做多模态理解，最后证明扩散可以做统一多模态基础模型；而 **LaViDa 与 Lavida-O** 则是另一条并行路线，且同样从“理解型 diffusion VLM”升级到了“统一理解-生成 diffusion model”。如果只看方法学纯度，LLaDA/LLaDA-V 的叙事最干净，因为它们都在回答“纯 diffusion 能否替代 AR”；如果只看系统完整度，MMaDA 和 Lavida-O 更进一步，因为它们把 reasoning、generation、RL 或 planning/reflection 都纳入了同一框架。再具体一点：**LLaDA-V / LaViDa 更像“diffusion 版 LLaVA”，而 MMaDA / Lavida-O 更像“diffusion 版 unified multimodal foundation model”。** 这也是为什么前者在理解 benchmark 上常更聚焦，后者在任务宽度上更强。 ([ar5iv](https://ar5iv.org/html/2505.16839v3 "[2505.16839] LaViDa: A Large Diffusion Language Model for Multimodal Understanding"))

如果你后面要写综述或组会汇报，这版内容已经可以直接当“技术路线综述稿”的骨架；其中若你说的“LaViDa-O”其实不是 **Lavida-O**，而是别的模型名，我建议按你目标模型名再校一次。