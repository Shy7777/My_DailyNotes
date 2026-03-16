先校正一下：你传的这篇更准确地说，不是“单独讲离散扩散模型”的综述，而是**“统一多模态理解与生成模型”**的综述；离散扩散/离散流匹配只是其中一条非常核心的主线。作者真正关心的是：为什么**多模态理解**长期被 **AR/LLM** 主导，而**图像生成**长期被 **Diffusion** 主导；以及怎样把这两条路线合成一个既会“看懂”、又会“生成”的统一模型。第2页图1、第6页表1、第7页图5基本把全景图给出来了：统一模型大体分成 **纯扩散式**、**纯自回归式**、**AR+Diffusion 融合式**，再往下又按图像 token 的来源分成 pixel / semantic / learnable query / hybrid 几条支线。原文在这里：[PDF](sandbox:/mnt/data/73cd986b-6395-4107-9fef-41bf9be78781.pdf)。

我先给你一句总纲：**这篇综述的核心矛盾，是“文本天然离散、适合 AR；图像天然高维连续、适合 diffusion”，而统一模型就是在 tokenization、attention mask、decoder 设计这三件事上做权衡。** 你把 Show-o、Transfusion、LLaDA、LaViDa、MMaDA 放到这三个坐标轴里，关系就会非常清楚。

## 1. 这篇综述想解决什么问题

作者认为，近年的多模态理解模型本质上继承了 LLM 的范式：输入文本 token，接上视觉编码器和 connector，把视觉特征送进 decoder-only LLM，然后用 next-token prediction 做理解、问答、推理。相反，图像生成这边从 GAN 转到 diffusion 之后，U-Net、LDM、DiT、CLIP/T5 条件编码器形成了另一套完全不同的技术栈。于是统一模型的难点不是“把图和字放一起训练”这么简单，而是：**一个系统到底应该沿着 AR 走、沿着 diffusion 走，还是两边折中？图像到底该编码成重建导向的像素 token，还是语义导向的 CLIP/SigLIP token，还是两者都要？** 这就是全文的主问题。

## 2. 预备知识：作者先铺了三块地基

第一块地基是**多模态理解模型**。第3页图2把它抽象成“多模态 encoder + connector + LLM”。视觉/音频/视频先被 encoder 变成特征，再通过 connector 接入语言模型。connector 又分三类：projection-based、query-based、fusion-based。这个视角很重要，因为后面很多统一模型，本质上都是在问：**我到底是把图像变成‘像文本一样能被 LLM 接受的 token’，还是保留图像自己的表示再单独解码？**

第二块地基是**扩散生成**。第3-4页图3和公式部分回顾了 diffusion 的前向加噪、反向去噪：前向过程逐步把数据扰乱，反向过程学习如何一步步还原。作者回顾了 pixel-space diffusion、latent diffusion、再到 DiT 的演进，强调 diffusion 的优势是图像质量高、模式覆盖更好、条件控制灵活；但代价是推理多步、慢，而且传统上更适合连续表征。

第三块地基是**AR 图像生成**。第4-5页图4把 AR 图像建模分成三种：next-pixel、next-token、next-multiple-tokens。也就是可以直接逐像素预测，也可以先用 VQ-VAE/VQGAN 把图变成离散视觉 token，再像语言一样自回归生成，还可以一次预测多个 patch/block/scale 来加速。这里的关键信息是：**AR 想统一文本和图像，必须把图像“语言化”；Diffusion 想统一文本和图像，往往要把文本也拉进一个可去噪的统一空间。**

## 3. 这篇综述最值钱的地方：把统一模型压缩成三维坐标系

第6页表1、第7页图5其实给了一个非常好的“读图方法”。

第一维是**主干范式**：  
纯 diffusion、纯 AR、AR+Diffusion 融合。

第二维是**图像编码方式**：  
pixel encoding（重建优先）、semantic encoding（语义对齐优先）、learnable query encoding（可学习查询压缩视觉信息）、hybrid encoding（像素细节和语义表示一起上）。

第三维是**mask / attention 结构**：  
AR 大多是 causal mask；diffusion 大多是 bidirectional / full attention；混合模型会做任务相关的 mask 设计。这个维度很关键，因为它本质上决定了模型是在做“顺序生成”，还是在做“联合去噪/联合修复”。

## 4. 你最关心的主线：离散扩散这一支到底怎么发展

这条线可以粗略看成：**Dual Diffusion → UniDisc → MMaDA → FUDOKI / Muddit → Lavida-O / UniModel**。它的共同目标是：别再只让图像扩散、文本自回归了，而是尽量让两种模态进入统一的扩散式生成框架。

**Dual Diffusion** 是比较早期也比较“折中”的做法：文本走 T5 式离散表示，图像走 Stable Diffusion 的 VAE 连续 latent，两边各自加噪、各自去噪，但在去噪过程中做 cross-modal conditioning。你可以把它理解成“双链扩散”：文本和图像都在扩散，但链条还没完全合并。它说明了一件事：统一扩散不是不行，但一开始往往还是“文字一套、图像一套、再做交互”。

**UniDisc** 则更进一步，直接把文本和图像都放进**全离散 diffusion** 框架。文本用 LLaMA2 tokenizer，图像用 MAGVIT-v2 tokenizer，于是二者都变成统一的离散 token 序列，再用 Diffusion Transformer 做联合去噪。这是一个非常关键的跃迁：它告诉你**离散 diffusion** 在统一多模态里之所以重要，是因为它天然更接近“把所有模态都 token 化之后统一建模”。

**MMaDA** 是这条线上的代表性里程碑。它采用 **LLaDA-8B-Instruct** 作为语言 backbone，再配 **MAGVIT-v2** 图像 tokenizer，把文本和图像都拉进统一离散空间。更重要的是，它不是只做“能生成”，而是开始把**推理范式**也并进来：它做了 mixed CoT 微调，让文字推理和视觉推理格式更统一；又引入 **UniGRPO** 这种面向 diffusion 的统一策略梯度 RL，奖励不仅看文本事实正确性，还看视觉-文本对齐和用户偏好。也就是说，**MMaDA 的意义不是单纯把 diffusion 做大，而是证明 diffusion-backbone 也能吸收原本属于 LLM/AR 世界的 CoT 和 RL。** 这正是它和 LLaDA 关系最紧密的地方：LLaDA 提供“语言扩散模型”的底座，MMaDA 把它真正推到统一多模态。

**FUDOKI** 很有意思，因为它不是传统离散 diffusion，而是走向**discrete flow matching**。它基于 Janus-1.5B，但做了几个关键改造：把 causal mask 改成 full attention；不显式依赖 diffusion 的 time-step embedding，而是从输入直接推断腐蚀状态；理解和生成两条路径解耦，理解用 SigLIP，生成用 VQGAN tokenizer。你可以把它看成“**从离散 diffusion 往离散 flow 演化**”的标志：目标仍然是统一 vision-language 生成，但路径从简单 mask corruption 变成更连续、自纠错更强的概率轨迹建模。

**Muddit** 也是纯离散路线，不过它更强调“一个统一的 MM-DiT 直接处理文本和图像”，架构上接近 FLUX 风格，并从 Meissonic 初始化以借高分辨图像先验。图像用 VQ-VAE codebook，文本用 CLIP text embeddings，训练时用 cosine schedule 做 mask，再预测 clean token。它的味道是：**尽量用单一 diffusion transformer 吞掉两种模态**。

**LaViDa / Lavida-O** 这一支更值得和 MMaDA 并着看。参考文献里，作者把 **LaViDa** 定义成一个面向多模态理解的 large diffusion language model；而 **Lavida-O** 则是在这个 backbone 上往“统一理解+生成”再迈一步。综述里说它用 **Elastic Mixture-of-Transformers** 做更资源高效的扩展，用 progressive upscaling 和 token compression 解决 masked generative training 的规模化问题，还引入 planning 和 self-reflection，让模型能用理解能力反过来修正生成。这说明 **LaViDa 系列**和 **LLaDA→MMaDA** 很像，都是“把 diffusion 不只当图像 decoder，而是当 backbone 级语言/多模态建模范式”。区别在于 MMaDA 更突出 CoT+RL，Lavida-O 更突出 scale-up、规划和自反。

**UniModel** 则更激进：它借鉴 DeepSeek-OCR 的想法，把文本也当视觉信号，把字渲染成图，再用 diffusion 同时做理解和生成。这个方向很新，但综述也明确指出挑战：文本渲染保真、多语言生成、长上下文建模都还很难。

## 5. 把 Show-o、Transfusion、LLaDA、LaViDa、MMaDA 串成一条线

最容易记的串法是：

**Transfusion / Show-o** 是“过渡派”，  
**LLaDA → MMaDA、LaViDa → Lavida-O** 是“diffusion 原教旨派”，  
**Janus-flow / BAGEL / Mogao / EMMA** 是“后续再融合派”。

为什么这么说？因为 **Transfusion** 和 **Show-o** 都属于第6页表1里的 **Fused AR and Diffusion Model**。它们承认一个现实：文本推理、指令跟随、对话组织，AR/LLM 还是最强；但图像生成质量上 diffusion 仍占优。所以它们做的不是“彻底统一”，而是**让文本继续自回归，让图像走 diffusion**。其中 **Transfusion** 用的是 **SD-VAE 连续 latent**，属于连续潜空间路线；**Show-o** 用的是 **MAGVIT-v2 离散图像 token**，属于离散符号路线。于是这两者代表了融合派里的两种不同口味：Transfusion 更连续，Show-o 更离散。

接着，社区开始不满足于“文本还是 AR、图像才 diffusion”的折中方案，于是出现 **UniDisc、MMaDA、Lavida-O** 这种更彻底的方向：让 backbone 自身也更 diffusion-native。这里 **LLaDA** 和 **LaViDa** 就变成关键底座。**MMaDA = LLaDA + MAGVIT-v2 + mixed CoT + UniGRPO**，强调 diffusion 也能做强推理和后训练；**Lavida-O = LaViDa + Elastic-MoT + progressive upscaling/token compression + planning/self-reflection**，强调 diffusion backbone 也能把理解能力反哺生成。换句话说，**Show-o / Transfusion 还在做“LLM + image diffusion”的工程折中；MMaDA / Lavida-O 则在尝试“diffusion 直接成为统一生成-理解的通用 backbone”。**

再往后看，**Show-o2** 又被综述归到 AR 的 hybrid encoding 一类，说明 Show-o 系列并没有一路走向“纯 diffusion”，而是在重新吸收 hybrid tokenizer 设计；同时 **Janus-flow、Mogao、BAGEL、EMMA、HBridge** 这些模型继续沿着“AR 主干 + diffusion/flow 生成专家 + 双编码器”的方向演化。也就是说，这个领域并没有收敛到单一路线，而是在三角拉扯：**AR 的 reasoning、diffusion 的 visual quality、hybrid tokenization 的统一表达**。

## 6. 自回归路线：作者分成四种图像编码法

这部分是全文最系统的地方。作者不是简单按模型年份讲，而是按“图像怎样进入 LLM”来拆。

**第一类，pixel-based encoding。** 代表有 LWM、Chameleon、ANOLE、Emu3、Liquid、UGen、TokLIP、Selftok、OneCat 等。核心思路是：用 VQGAN、VQ-IMG、MoVQGAN、VAE 之类先把图像压成重建导向的 token，再和文本 token 一起丢进 AR 主干。优点是保留细节，生成直接；缺点是 token 密、序列长、语义抽象弱，跨模态对齐难。综述还点出 MMAR、Orthus、Harmon 这类连续 latent 方案，用 diffusion MLP 头来回到像素空间，避免离散化的信息损失。

**第二类，semantic encoding。** 代表有 Emu、LaVIT、DreamLLM、Emu2、VL-GPT、MM-Interleaved、Mini-Gemini、VILA-U、PUMA、MetaMorph、ILLUME、UniTok、QLIP、DualToken、UniFork、UniCode2、UniWorld、Pisces、Tar、OmniGen2、Ovis-U1、Qwen-Image、X-Omni、Bifrost-1、Ming-UniVision、MammothModa2。这里的视觉编码器更多是 EVA-CLIP、OpenAI-CLIP、SigLIP、UNIT 这类**与文本对齐的语义编码器**。这条线的优点是对语言更友好、理解强；缺点是缺少局部像素控制，常常还要接一个单独训练的 diffusion decoder 来补足图像细节。LaVIT 在这一类里很关键，它用动态视觉 tokenization 减冗余；UNIT / VILA-U / UniTok 这类则试图兼顾语义对齐和重建能力。

**第三类，learnable query encoding。** 代表是 SEED、SEED-LLaMA、SEED-X、MetaQueries、Nexus-Gen、Ming-Lite-Uni、BLIP3-o、OpenUni、UniLIP、TBAC-UniImage、UniPic 2.0。核心不是把整张图密集切 token，而是用可学习 query 去“抽取最有用的视觉信息”。这类方法压缩效率高、适合作为扩散 decoder 的条件；但 query 太少时会丢细节，query 太多时算力又上来。SEED 是经典起点，MetaQueries 是更轻量的一条，后面很多方法都沿这条路接强 diffusion decoder。

**第四类，hybrid encoding。** 作者再分成 pseudo hybrid 和 joint hybrid。前者如 Janus、Janus-Pro、OmniMamba、UniFluid、MindOmni、Skywork UniPic，虽然训练时同时有语义编码器和像素编码器，但理解时只开语义分支，生成时只开像素分支，本质上还是“分工式双编码器”；后者如 MUSE-VL、Tokenflow、VARGPT、ILLUME+、UniToken、Show-o2，则把 semantic token 和 pixel token 真正一起送进模型。混合编码的优点是希望同时拿到语义对齐和细节保真；难点则是计算贵、对齐难、容易冗余。

## 7. AR+Diffusion 融合路线：Show-o 和 Transfusion 所在的位置

这一类是最像“工程上马上能用”的统一路线。作者在 3.3 节说得很清楚：文本 token 继续 AR 生成，以保留 LLM 的组合推理能力；图像 token 用 diffusion/denoising 生成，以保留视觉质量和全局一致性。**Transfusion、Show-o、MonoFormer、LMFusion** 属于 pixel-based 融合；**Janus-flow、Mogao、BAGEL、LightFusion、HBridge、EMMA** 属于 hybrid encoding 融合。

这里你可以把 **Transfusion** 看成“统一 transformer + 连续视觉 latent + 模态特定层”的设计；把 **Show-o** 看成“离散视觉 token + transformer 兼容性更强”的设计。前者更接近 Stable Diffusion 潜空间世界，后者更接近视觉 token 世界。综述也明确指出这种融合路线的代价：推理多步，计算重；跨模态 latent 不一定天然对齐；离散 token 又会遭遇 codebook collapse 和微妙细节不足。

## 8. 从 text-image 走向 any-to-any

第7页表2和 3.4 节把视野拉宽：统一模型不只想做 text↔image，还想做 audio、video、speech 等任意到任意。代表包括 Next-GPT、Unified-IO 2、Video-LaVIT、AnyGPT、X-VILA、MIO、Spider、OmniFlow、M2-omni。作者的判断是，这些模型大多走模块化路线：每种模态一个 encoder/decoder，中间共享 backbone；问题在于模态不平衡、系统复杂度高、语义一致性更难。

## 9. 数据、评测、挑战：这篇综述不是只列模型，也把“训练生态”讲清楚了

数据部分，作者把统一模型需要的数据分成五类：**多模态理解数据、文本到图像数据、图像编辑数据、交错图文数据、以及 text+image 到 image 的条件生成数据**。代表性的理解数据有 RedCaps、Wukong、LAION、COYO、DataComp、ShareGPT4V、ALLaVA、Infinity-MM；T2I 数据有 CC-12M、LAION-Aesthetics、JourneyDB、DOCCI、PD12M，以及大量文字渲染/合成数据；编辑数据从 InstructPix2Pix、MagicBrush 一路扩展到 HQ-Edit、UltraEdit、AnyEdit、ImgEdit、ByteMorph-6M、GPT-Image-Edit-1.5M、X2Edit；交错图文有 MMC4、OBELICS、CoMM、OmniCorpus；主体驱动和多条件生成则有 MultiGen-20M、Subjects200K、SynCD、Graph200K 等。作者还强调，很多统一模型其实先在大规模文本语料上初始化，再进入这些多模态数据阶段。

评测部分，作者分成四层：**理解评测、图像生成评测、交错生成评测、真正的“统一性”评测**。理解这边是 VQA/VQAv2、CLEVR、GQA、OK-VQA、VCR、MMBench、MMMU、MM-Vet、SEED-Bench、MathVista、General-Bench；生成这边是 DrawBench、PartiPrompts、PaintSkills、GenEval、TIFA、T2I-CompBench、GenAI-Bench，以及编辑方向的 MagicBrush、I2EBench、HQ-Edit、AnyEdit、KRIS-Bench 等；交错生成有 InterleavedBench、OpenLEAF、ISG、MMIE、OpenING、UniBench。最关键的是 5.4 节的 **RealUnify**：它不再把“理解”和“生成”分开测，而是测 **Understanding-Enhanced Generation** 和 **Generation-Enhanced Understanding**，也就是看模型能不能真的让理解帮助生成、让生成反过来帮助理解。这个 benchmark 很能体现作者的立场：统一不是“一个模型同时会两件事”，而是“两种能力开始互相增益”。

挑战部分，作者总结得很扎实。第一是**tokenization/compression**：图文联合后序列太长，必须更高效地压缩；第二是**cross-modal attention 瓶颈**：分辨率和上下文一上去，注意力爆炸；第三是**数据噪声与偏差**：网页图文对、复杂编辑数据、交错图文都很脏；第四是**评测碎片化**：很多 benchmark 只测单任务，测不出“统一性”；第五是**CoT 与 RL** 的并入空间很大，MMaDA 已经给了一个方向；第六是**社会偏见与公平性**；第七是**个性化知识驱动生成**。最后作者还专门点明：现在大多数统一模型其实还是以“图像理解 + text-to-image”为主，真正难的功能——比如图像编辑、空间可控生成、主体驱动生成、交错图文生成——很多还停留在后微调或局部探索阶段。

## 10. 你可以把整篇综述记成这四句话

第一，**统一多模态不是单一技术问题，而是 AR 与 diffusion 两个范式的碰撞。**

第二，**Show-o / Transfusion 代表“先融合”，MMaDA / Lavida-O 代表“让 diffusion backbone 真正接管统一建模”。** 其中 **LLaDA → MMaDA**、**LaViDa → Lavida-O** 是两条最值得跟的 diffusion-language-model 主线。

第三，**tokenizer 决定模型气质**：pixel token 更像生成器，semantic token 更像理解器，learnable query 更像压缩器，hybrid tokenizer 才是试图真正兼顾两者。

第四，**这个方向还远没收敛**。作者给出的时间线（第2页图1）本身就在说明：2023 到 2025 新模型密集爆发，说明大家还在快速试错，还没有出现像 GPT/Stable Diffusion 那样“一锤定音”的统一范式。

你要是把这篇综述当研究地图来读，最值得重点盯住的就是两件事：**一是 MMaDA / Lavida-O 这种“diffusion-language-model 化”的统一主干；二是 Show-o / Transfusion / Janus-flow / BAGEL 这种“AR reasoning + diffusion visual quality”的融合路线。** 前者更激进，后者更实用。

我也可以接着把这篇综述再压缩成一版“研究脉络图”，专门只讲 **Show-o / Transfusion / MMaDA / Lavida-O / Janus-flow / BAGEL** 之间的演化关系。