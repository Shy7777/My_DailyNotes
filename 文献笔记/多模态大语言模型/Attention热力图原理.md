# 一、ViT的注意力可视化在做一件什么事情？

实际上ViT的attention map可视化本质是提取 `[CLS]` token在最后一个Transformer block中对所有image patch的权重整合。

这并不是像Grad-CAM那样通过梯度回传计算出来的saliency map，在机制上两者之间是有明显区别的，ViT的注意力可视化和attention的计算过程密切相关。

在ViT的架构中，`[CLS]` token负责汇聚所有的image patches信息，用于最终分类并输出class label对应的logits，因此，它在最后一层的attention weights直接反映了：模型为了做classification关注哪些image patches更多。

而可视化这一过程的逻辑在于：我们只需要拿到这个权重向量，把它从一维序列还原回二维的grid结构，再插值上采样放大覆盖在原图上，就能得到直观的heatmap——这也是我们最为常见的ViT注意力的可视化形式。

数学表达上，假设输入图像被切分为  个patch，加上 `[CLS]` token一共  个token。在第  层，attention matrix  的形状是 ，其中  是head的数量。

我们要找的是  中对应 `[CLS]` token作为query的那一行，通常是索引0。于是，我们取出 ，这表示所有head中 `[CLS]` 对个图像patch的关注度。这步操作后我们得到一个形状为  的tensor。

为了便于观察，通常会对  个head取平均，得到长度为  的向量。最后一步是根据patch grid的几何排列（比如 ），将这个向量reshape成二维矩阵，这就heatmap中「红色」和「蓝色」区域的根据（假设「红色」代表权值越接近1，「蓝色」代表权值越接近0）。

# 二、代码层面上，注意力可视化过程是如何体现的？

下面是一段基于Hugging Face transformers库里的标准化可视化代码，展示了从加载模型、预处理图片、提取attention weights到最终可视化的完整流程。

```python
import torch  
import numpy as np  
import matplotlib.pyplot as plt  
from PIL import Image  
from transformers import ViTImageProcessor, ViTModel  
   
# 1. Prepare model and image  
model_name = "google/vit-base-patch16-224"  
processor = ViTImageProcessor.from_pretrained(model_name)  
model = ViTModel.from_pretrained(model_name)  
image = Image.open("test.jpg") # Replace with your image path  
   
# 2. Inference and get attention  
inputs = processor(images=image, return_tensors="pt")  
with torch.no_grad():  
    # Key point: set output_attentions=True to retrieve intermediate states  
    outputs = model(**inputs, output_attentions=True)  
   
# 3. Extract Attention weights  
last_layer_attentions = outputs.attentions[-1][0]   
   
# 4. Process weights  
attentions_mean = torch.mean(last_layer_attentions, dim=0)  
patch_attentions = attentions_mean[0, 1:]   
   
# 5. Reshape back to 2D spatial grid  
grid_size = int(np.sqrt(patch_attentions.shape[0]))  
attentions_grid = patch_attentions.reshape(grid_size, grid_size)  
   
# 6. Visualization  
attentions_grid = torch.nn.functional.interpolate(  
    attentions_grid.unsqueeze(0).unsqueeze(0),   
    size=image.size[::-1],   
    mode="bilinear"  
).squeeze().numpy()  
   
plt.imshow(image)  
plt.imshow(attentions_grid, cmap='jet', alpha=0.5)  
plt.axis('off')  
plt.show()
```


我们可以逐行来看一下：

- 首先`output_attentions=True`代表了Transformer前传的过程中会保存所有的attention matrix，否则为了节省显存，这些中间变量会被默认丢弃掉；`outputs.attentions[-1]` 则代表取用的是Transformer最后一层的特征；`torch.mean(..., dim=0)` 是将多个attention head的信息做平均。
    
- 最关键的步骤是`attentions_mean[0, 1:]`和reshape。这里代表：ViT的第0个位置永远是 `[CLS]`，后面的才是image patches；reshape操作实际上是将1D sequence重新变成2D图的形式，相当于flatten的逆过程；最后的`nn.interpolate`是为了将的attention matrix上采样到与原图分辨率一样的heatmap。
    
- 最后再调用`plt`包将attention权重渲染到原图`image`上，至此，ViT注意力可视化的过程结束。