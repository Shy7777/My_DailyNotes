[[Prompt Template]]：比如来一张 ImageNet-1K 验证集的图片，作者把它喂入 CLIP 预训练好的 Image Encoder，得到特征  ，接下来把所有类别的词汇 "cat", "dog" 等，做成一个 prompt："A photo of a {object}"，并将这个 prompt 喂入 CLIP 预训练好的 Text Encoder，依次得到特征  ，最后看**哪个的余弦相似度和  最高**，就代表**该图片是哪个类别的**。

