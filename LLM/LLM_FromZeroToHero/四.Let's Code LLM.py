import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import requests
import math

# Stage1:获取数据集
if not os.path.exists('sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt?download=true'
    with open('sales_textbook.txt', 'wb') as f:
        f.write(requests.get(url).content)

with open('sales_textbook.txt', 'r') as f:
    text = f.read()

# Stage2:文本预处理，Token化
import tiktoken

encodings = tiktoken.get_encoding("cl100k_base")

tokenized_text = encodings.encode(text)
tokenized_text = torch.tensor(tokenized_text).long()
max_token_value = tokenized_text.max().item()

# Stage3:split,将数据集切分为训练数据集和测试数据集
train_idex = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_idex]
valid_data = tokenized_text[train_idex:]

# Stage4:设置超参数hyperparameters,每批次多少同时进行，获得矩阵4*16*64
batch_size = 4
# 每个截取token长度
context_length = 16
# 维度
d_model = 64

num_layers = 8  # Number of transformer blocks
num_heads = 4  # Number of heads in Multi-head attention # 我们的代码中通过 d_model / num_heads = 来获取 head_size
learning_rate = 1e-3  # 0.001
dropout = 0.1  # Dropout rate
max_iters = 5000  # Total of training iterations
eval_interval = 50  # How often to evaluate the model
eval_iters = 20  # How many iterations to average the loss over when evaluating the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Instead of using the cpu, we'll use the GPU if it's available.

TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

# x_batch相当于训练数据 y_batch相当于标签
data = train_data
idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
x_batch = torch.stack([data[idx:idx + context_length] for idx in idxs])
y_batch = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs])
# print(x_batch)
# print(y_batch)

# 数据可视化
# import pandas as pd
#
# pd.DataFrame(x_batch[0].numpy())
# word = encodings.decode([872])
# print(word)

# define input embeddings table
input_embedding_lookup_table = nn.Embedding(max_token_value + 1, d_model)

# 把每个词对应的列找出来
x_batch_embedding = input_embedding_lookup_table(x_batch)
y_batch_embedding = input_embedding_lookup_table(y_batch)
# print(x_batch_embedding.shape)
# print(y_batch_embedding.shape)

# 在前向传播算法中传播的权重（weight），通过训练自动更新 Feed-Forward
# print(input_embedding_lookup_table.weight.data)

# 获得位置编码 positional encoding
position_encoding_lookup_table = torch.zeros(context_length,
                                             d_model)  # initial with zeros with shape (context_length, d_model)
position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
# 使用正弦和余弦
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
position_encoding_lookup_table = position_encoding_lookup_table.unsqueeze(0).expand(batch_size, -1,
                                                                                    -1)  # add batch to the first dimension
# print("Position Encoding Look-up Table: ", position_encoding_lookup_table.shape)

# 添加位置编码到embedding中
x = x_batch_embedding + position_encoding_lookup_table
y = y_batch_embedding + position_encoding_lookup_table
# print(x.shape)
# print(y.shape)
# print(pd.DataFrame(x[0].detach().numpy()))

# 多头注意力机制，得到QKV
Wq = nn.Linear(d_model, d_model)
Wk = nn.Linear(d_model, d_model)
Wv = nn.Linear(d_model, d_model)

Q = Wq(x)
K = Wk(x)
V = Wv(x)
# print(Q.shape)
# print(K.shape)
# print(V.shape)

# 应用多头注意力
Q = Q.view(batch_size, context_length, num_heads, d_model // num_heads).permute(0, 2, 1, 3)
K = K.view(batch_size, context_length, num_heads, d_model // num_heads).permute(0, 2, 1, 3)
V = V.view(batch_size, context_length, num_heads, d_model // num_heads).permute(0, 2, 1, 3)

# 应用mask，将上三角元素统一设置
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
output = output.masked_fill(mask, float('-inf'))

# 进行softmax
attention_score = F.softmax(output, dim=-1)

# 将多头注意力机制的QKV中的V合并
A = attention_score @ V
print(A.shape)

# 进行concatenate，合并
A = A.transpose(1, 2).reshape(batch_size, context_length, d_model)
Wo = nn.Linear(d_model, d_model)
output = Wo(A)

# 进行残差连接和标准归一化 residual connection and layer normalization
output = output + x
layer_norm = nn.LayerNorm(d_model)
layer_norm_output = layer_norm(output)

# 前馈神经网络
output = nn.Linear(d_model, d_model*4)(layer_norm_output)
output = output+layer_norm_output
output = layer_norm(output)

# 最终的线性变换层 final linear layer
output = nn.Linear(d_model, max_token_value+1)(output)

logits = F.softmax(output, dim=-1)
predictions = torch.argmax(logits[0,0]).item()