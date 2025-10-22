#  Multi-Modal Foundation Models for Computational Pathology (MMFM4CPath) — 笔记

## 1. 背景与动机
- **H&E染色图像**
  - 最常用的病理染色方式
  - 捕捉组织形态学特征
- **Whole Slide Images (WSIs)**
  - 超高分辨率，信息丰富
  - 计算与存储成本高 → 常切分为 tile
- **单模态模型**
  - 仅依赖图像
  - 在分类、分割、预后预测上成功
  - 局限：缺乏语义、分子层面信息
- **多模态模型优势**
  - 融合文本（报告、caption）
  - 融合知识图谱（病理学本体）
  - 融合基因表达（RNA-seq, omics）

---

## 2. 模型范式分类
### 2.1 Vision-Language
- **非LLM模型**
  - CLIP-based: PLIP, PathCLIP, QuiltNet, PathAlign-R
  - CoCa-based: PRISM, TITAN, MUSK
- **LLM模型**
  - BLIP-2系列: PathAlign-G
  - LLaVA系列: Quilt-LLaVA, PathGen-LLaVA, WSI-LLaVA
  - GPT系列: Quilt-LLaVA (GPT-4), PathChat (Llama2-13B)
  - 专用模型: HistoGPT, CLOVER, SlideChat

### 2.2 Vision-Knowledge Graph
- KEP (Knowledge-Enhanced Pathology)
- KEEP (Knowledge-Enhanced Embedding for Pathology)

### 2.3 Vision-Gene Expression
- THREADS
- TANGLE
- mSTAR

---

## 3. 预训练目标与策略
- **自监督对比学习 (SSCL)**
  - CLIP: 图像-文本对齐
  - CoCa: 增加caption生成
- **跨模态对齐 (CMA)**
  - 构建统一语义空间
- **生成式任务**
  - Captioning (tile描述)
  - Report Generation (WSI报告)
- **指令微调 (Instruction Tuning, IT)**
  - 大规模指令数据集训练
  - 支持问答、对话、任务执行
- **其他**
  - Next Word Prediction (NWP)
  - Weakly Supervised Learning (MIL, WSL)
  - Reinforcement Learning (RLHF)

---

## 4. 多模态数据集
- **图像-文本对**
  - Tile-Caption: PLIP, PathCLIP, QuiltNet
  - WSI-Report: PathAlign, SlideChat
- **指令数据集**
  - PATHINSTRUCT
  - CLOVER Instruction
  - SlideChat VQA
- **图像-其他模态**
  - 图像-KG: KEP, KEEP
  - 图像-基因表达: THREADS, mSTAR

---

## 5. 下游任务与评估
- **分类**
  - Zero-shot / Few-shot
  - MIL (Multiple Instance Learning)
- **检索**
  - Tile ↔ Caption
  - WSI ↔ Report / Gene Expression
- **生成**
  - Tile Captioning
  - WSI Report Generation
- **分割与预测**
  - 肿瘤区域分割
  - 生存预测、肿瘤厚度
- **VQA与对话**
  - 单轮问答
  - 多轮对话（临床辅助）

---

## 6. 未来方向
- 融合空间组学数据 (Spatial Omics)
- H&E预测MxIF标记 (虚拟染色)
- 标准化评估基准 (Benchmark)
- 临床可解释性与可用性
- 大规模开放数据共享

---

##  总结
- **单模态 → 多模态** 是计算病理学发展的必然趋势。
- **三大范式**：Vision-Language、Vision-Knowledge Graph、Vision-Gene Expression。
- **核心突破点**：对比学习、跨模态对齐、指令微调。
- **未来挑战**：数据规模、跨模态对齐难度、临床可解释性、标准化评估。

---

#  多模态基础模型综述对比表

## 1. 模型对比表

| 模型 | 年份 | 范式 | 核心架构 | 预训练目标 | 输入模态 | 数据规模/来源 | 特点 |
|------|------|------|----------|------------|----------|---------------|------|
| PLIP | 2023 | Vision-Language (CLIP) | ViT-B/32 + Transformer | CLIP对比学习 | Tile + Caption | 208K tile-caption 对 | 早期CLIP迁移到病理 |
| QuiltNet | 2023 | Vision-Language (CLIP) | ViT-B/32 | CLIP | Tile + Caption | 438K tiles, 802K captions | 大规模tile-caption |
| PathCLIP | 2024 | Vision-Language (CLIP) | ViT-B/32 | CLIP | Tile + Caption | 207K pairs | 病理专用CLIP |
| PRISM | 2024 | Vision-Language (CoCa) | ViT-H/14 + BioGPT | CoCa | WSI + Report | 587K WSIs, 195K specimens | 融合BioGPT |
| PathAlign-R | 2024 | Vision-Language | ViT-S/16 + Q-Former | CLIP + MSN | WSI + Report | 354K WSIs, 434K reports | 从零训练 |
| PathAlign-G | 2024 | Vision-Language (LLM) | ViT-S/16 + PaLM-2 | BLIP-2 + CMA | WSI + Report | 同上 | 融合LLM |
| CHIEF | 2024 | Vision-Language | Swin-T + Aggregator | CLIP弱监督 | WSI + Label | 60K WSIs | 聚合网络 |
| CONCH | 2024 | Vision-Language | ViT-B/16 | iBOT + NWP + CoCa | Tile + Text | 16M tiles, 950K text | 多目标SSL |
| TITAN | 2024 | Vision-Language | ViT-L + ViT-S | iBOT + CoCa | WSI + Caption/Report | 336K WSIs, 423K ROI-caption, 183K reports | 多粒度 |
| MUSK | 2025 | Vision-Language | V-FFN + L-FFN | BEIT3 + CoCa | Tile + Caption | 50M tiles, 1B tokens | 超大规模 |
| PathGen-CLIP | 2025 | Vision-Language | ViT-B/32 | CLIP | Tile + Caption | 1.6M pairs | 生成式增强数据 |
| MLLM4PUE | 2025 | Vision-Language (LLM) | SigLIP + Qwen1.5 | CLIP | Tile + Caption | 594K pairs | 通用多模态嵌入 |
| PathAsst | 2024 | Vision-Language (LLM) | ViT-B/32 + Vicuna-13B | CMA + IT | Tile + Instruction | PATHINSTRUCT 35K | 指令调优 |
| Dr-LLaVA | 2024 | Vision-Language (LLM) | ViT-L/14 + Vicuna | IT + RL | Tile + Dialogue | 16K tiles, 多轮对话 | 医学对话 |
| Quilt-LLaVA | 2024 | Vision-Language (LLM) | ViT-B/32 + GPT-4 | CMA + IT | Tile + Instruction | 723K captions, 107K instructions | GPT-4驱动 |
| PathChat | 2024 | Vision-Language (LLM) | ViT-L/16 + Llama2-13B | CoCa + CMA + IT | Tile + Caption + Instruction | 1.18M captions, 457K instructions | 多任务 |
| HistoGPT | 2024 | Vision-Language (LLM) | Swin-T/ViT-L + BioGPT | MIL + NWP | WSI + Report | 15.1K WSIs, 6.7K labels | 专用GPT |
| CLOVER | 2024 | Vision-Language (LLM) | EVA-ViT-G/14 + Vicuna/FlanT5 | BLIP-2 + IT | Tile + Instruction | 438K captions, 45K VQA | 多模态问答 |
| PathInsight | 2024 | Vision-Language (LLM) | LLaVA/InternLM | IT | Tile + Instruction | 45K instances | 多任务覆盖 |
| SlideChat | 2024 | Vision-Language (LLM) | ViT-L + LongNet + Qwen2.5 | CMA + IT | WSI + Report + Instruction | 4.2K reports, 176K VQA | 长上下文 |
| W2T | 2024 | Vision-Language | ViT-S/ResNet/HIPT + PubMedBERT | NWP | WSI + VQA | 804 WSIs, 7.14K QA | 小规模VQA |
| PA-LLaVA | 2024 | Vision-Language (LLM) | ViT-B/32 + Llama3 | CLIP + CMA + IT | Tile + Caption + VQA | 827K captions, 35.5K QA | LoRA微调 |
| WSI-LLaVA | 2024 | Vision-Language (LLM) | ViT-G/14 + LongNet + Vicuna | CLIP + CMA + IT | WSI + Report + VQA | 9.85K reports, 175K QA | 长上下文WSI |
| CPath-Omni | 2024 | Vision-Language (LLM) | ViT-H/14 + Qwen2.5-14B | CMA + IT | Tile + WSI + Instruction | 700K captions, 352K instructions | 全面覆盖 |
| PathGen-LLaVA | 2025 | Vision-Language (LLM) | ViT-B/32 + Transformer | CLIP + CMA | Tile + Caption | 700K pairs | 生成式增强 |
| KEP | 2023 | Vision-Knowledge Graph | ViT + KG encoder | CMA | WSI + KG | 专用KG数据 | 知识增强 |
| KEEP | 2023 | Vision-Knowledge Graph | ViT + KG embedding | CMA | WSI + KG | 专用KG数据 | 知识引导 |
| THREADS | 2024 | Vision-Gene Expression | ViT + RNA encoder | CMA | WSI + RNA-seq | 多中心RNA数据 | 基因表达对齐 |
| TANGLE | 2024 | Vision-Gene Expression | ViT + Omics encoder | CMA | WSI + Omics | 多模态数据 | 融合omics |
| mSTAR | 2025 | Vision-Gene Expression | ViT + RNA encoder | CMA | WSI + RNA-seq | 大规模RNA数据 | 精准医学 |

---

## 2. 数据集对比表

| 数据集 | 类型 | 内容 | 规模 | 用途 |
|--------|------|------|------|------|
| QUILT | 图像-文本对 | Tile-Caption | 438K tiles, 802K captions | CLIP训练 |
| PLIP数据集 | 图像-文本对 | Tile-Caption | 208K pairs | CLIP微调 |
| PathCLIP数据集 | 图像-文本对 | Tile-Caption | 207K pairs | CLIP微调 |
| PathGen数据集 | 图像-文本对 | Tile-Caption | 1.6M pairs | 生成式增强 |
| OPENPATH | 图像-文本对 | Tile-Caption | ~百万级 | 通用训练 |
| PathAlign数据集 | 图像-文本对 | WSI-Report | 434K pairs | 报告生成 |
| SlideChat数据集 | 图像-文本对+指令 | WSI-Report + VQA | 4.2K reports, 176K QA | 长上下文对话 |
| PATHINSTRUCT | 指令 | Tile-level instruction | 35K samples | 指令微调 |
| CLOVER Instruction | 指令 | VQA instruction | 45K | 多模态问答 |
| PathChat数据集 | 指令 | Caption + Instruction | 1.18M captions, 457K instructions | 多任务 |
| W2T数据集 | 图像-文本对 | WSI-VQA | 804 WSIs, 7.14K QA | 小规模VQA |
| KEEP/KEP数据集 | 图像-KG | WSI + KG | 专用知识图谱 | 知识增强 |
| THREADS数据集 | 图像-基因表达 | WSI + RNA-seq | 多中心RNA | 基因对齐 |
| mSTAR数据集 | 图像-基因表达 | WSI + RNA-seq | 大规模RNA | 精准医学 |
