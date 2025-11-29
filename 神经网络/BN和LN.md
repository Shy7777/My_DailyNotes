**快速总结：**  
批次归一化（Batch Normalization, BN）是在 **batch维度上** 对每个特征进行归一化，依赖于批量大小；层归一化（Layer Normalization, LN）是在 **特征维度上** 对单个样本进行归一化，不依赖批量大小。BN常用于卷积网络，LN常用于序列模型（如Transformer）。

---

## 🔑 批次归一化（Batch Normalization, BN）

- **原理**：对一个 mini-batch 中每个特征维度计算均值和方差，然后归一化为均值0、方差1。  
    公式：  
![[Pasted image 20251129113505.png]]
- **优点**：
    - 加快收敛速度
    - 缓解梯度消失/爆炸
    - 有一定正则化效果
- **局限性**：
    - 依赖 batch size，batch 太小效果差
    - 在序列模型或在线推理中不稳定 [知乎专栏](https://zhuanlan.zhihu.com/p/696062068) [CSDN博客](https://blog.csdn.net/ThomasCai001/article/details/146392457) [博客园](https://www.cnblogs.com/RubySIU/p/18206144)

---

## 📚 层归一化（Layer Normalization, LN）

- **原理**：对单个样本的所有特征维度计算均值和方差，然后归一化。  
    公式：  
    ![[Pasted image 20251129113518.png]]
- **优点**：
    - 不依赖 batch size，适合小批量或单样本推理
    - 在序列模型（如RNN、Transformer）中效果好
- **局限性**：
    - 在卷积网络中效果不如BN
    - 正则化效果较弱

---

## ⚖️ BN vs LN 对比

|特性|批次归一化 (BN)|层归一化 (LN)|
|---|---|---|
|归一化维度|batch维度（跨样本）|特征维度（单样本）|
|依赖 batch size|是|否|
|常用场景|CNN、图像任务|RNN、Transformer、NLP|
|优点|收敛快，正则化强|稳定，适合小batch和序列|
|缺点|小batch效果差|CNN中效果有限|

---

## 🎯 总结

- **BN**：适合大批量训练的图像任务，能显著加快训练。
- **LN**：适合序列模型和小批量场景，是Transformer的标配。  
    👉 可以理解为：**BN依赖“跨样本统计”，LN依赖“单样本内部统计”。**

---
