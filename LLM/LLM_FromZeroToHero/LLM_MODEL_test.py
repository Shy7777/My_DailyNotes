import os
import requests
import math
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 4  # 训练步长批次
context_length = 16  # Length of the token chunk each batch
d_model = 64  # The size of our model token embeddings
num_blocks = 8  # Number of transformer blocks
num_heads = 4  # Number of heads in Multi-head attention
learning_rate = 1e-3  # 0.001
dropout = 0.1  # Dropout rate 把已经训练好的数据随机丢百分之十出去重新训练，防止过拟合
max_iters = 5000  #
eval_interval = 50  # 评估间隔
eval_iters = 20  # Number of iterations to average for evaluation
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if it's available.
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

# Stage1:获取数据集
with open('book.txt', 'r', encoding='utf-8') as f:
    text = f.read()
if not os.path.exists('sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
    with open('sales_textbook.txt', 'w') as f:
        f.write(requests.get(url).text)

with open('sales_textbook.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# # 设置输出文件名
# output_file = "book.txt"
# target_folder = 'D:\\shy_study\\My_DailyNotes\\LLM\\LLM_FromZeroToHero\\data\\外国短篇'
#
# # 打开输出文件准备写入
# with open(output_file, 'w', encoding='utf-8') as outfile:
#     # 遍历当前目录下的所有文件
#     for filename in os.listdir(target_folder):
#         # 只处理 .txt 文件，排除输出文件本身
#         if filename.endswith('.txt') and filename != output_file:
#                 with open(filename, 'r', encoding='utf-8') as infile:
#                     # # 写入文件名作为分隔（可选）
#                     # outfile.write(f"\n--- 来自文件：{filename} ---\n")
#                     # 写入文件内容
#                     outfile.write(infile.read())
#                     outfile.write("\n")  # 添加换行分隔
#

# Stage2:序列化 Tokenize the text
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)
max_token_value = max(tokenized_text) + 1  # the maximum value of the tokenized numbers
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)  # 文本张量化

# Stage3:split,将数据集切分为训练数据集和测试数据集
split_idx = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:split_idx]
val_data = tokenized_text[split_idx:]


# 前馈神经网络的包装类 两层线性层+ReLU+Dropout
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.ffn = nn.Sequential(
            # 全连接层线性变换，先放大细节，再回归原维度,对输入 x 计算线性变换 y = xW^T + b。
            nn.Linear(in_features=self.d_model, out_features=self.d_model * 4),
            nn.ReLU(),
            nn.Linear(in_features=self.d_model * 4, out_features=self.d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)


# 多头注意力机制包装类，要做Wq矩阵与Q相乘,单头计算
class Attention(nn.Module):
    def __init__(self, head_size: int): # head_size是维度/多头数，即每头的维度
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.context_length = context_length
        self.dropout = dropout

        self.key_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.query_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.value_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones((self.context_length, self.context_length))))  # 下三角掩码
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        B, T, C = x.shape  # 批次数，文本长度，维度
        # 条件为假则会报错
        assert T <= self.context_length
        assert C == self.d_model
        q = self.query_layer(x)
        k = self.key_layer(x)
        v = self.value_layer(x)

        # 计算公式Q @ K^T / sqrt(d_k),矩阵乘法，张量乘积
        weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # masked attention
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(input=weights, dim=-1)
        weights = self.dropout_layer(weights)

        # weights @ V
        out = weights @ v
        return out


# 多头注意力机制，设置参数4,4个头循环进行
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.d_model = d_model
        self.context_length = context_length
        self.dropout = dropout

        self.heads = nn.ModuleList([Attention(head_size=self.head_size) for _ in range(self.num_heads)])
        self.projection_layer = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # 拼接多个头的输出
        out = self.projection_layer(out)  # 将拼接后的输出映射成原始维度
        out = self.dropout_layer(out)
        return out


class TransformerBlock(nn.Module):

    def __init__(self, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.head_size = d_model // num_heads  # head size should be divisible by d_model
        self.num_heads = num_heads
        self.dropout = dropout

        self.multi_head_attention_layer = MultiHeadAttention(head_size=self.head_size)
        self.feed_forward_layer = FeedForward()
        # 进行归一化处理，防止梯度爆炸和梯度消失
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=self.d_model)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=self.d_model)

    def forward(self, x):
        # LayerNorm -> Multi-head attention -> LayerNorm -> Feed forward
        x = x + self.multi_head_attention_layer(self.layer_norm_1(x))  # Residual connection
        x = x + self.feed_forward_layer(self.layer_norm_2(x))  # Residual connection
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.max_token_value = max_token_value
        # token embedding总词token表
        self.token_embedding_lookup_table = nn.Embedding(num_embeddings=self.max_token_value + 1,
                                                         embedding_dim=self.d_model)

        # 运行所有的Transformer块
        self.transformer_blocks = nn.Sequential(*(
                [TransformerBlock(num_heads=self.num_heads) for _ in range(self.num_blocks)] +
                [nn.LayerNorm(self.d_model)]
        ))
        self.language_model_out_linear_layer = nn.Linear(in_features=self.d_model, out_features=self.max_token_value)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # 将位置编码设为0，为[16,64]
        position_encoding_lookup_table = torch.zeros(self.context_length, self.d_model)
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        # 频率缩放因子，让位置编码奇偶不同，又有正弦余弦函数的性质,连续易求导
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
        # 将位置编码由[16,64]变为[T,64]
        position_embedding = position_encoding_lookup_table[:T, :].to(device)
        x = self.token_embedding_lookup_table(idx) + position_embedding
        x = self.transformer_blocks(x)
        # logits是最终全连接层后的输出，但没有经过softmax函数变换，得到的是所有token的概率
        logits = self.language_model_out_linear_layer(x)

        if targets is not None:
            # B为批次大小，T为序列长度，每个样本Token数，C是词表大小
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx 是当前上下文中的 (B,T) 索引数组
        for _ in range(max_new_tokens):
            # 将索引裁剪到位置嵌入表的最大大小
            idx_crop = idx[:, -self.context_length:]
            # 得到预测值
            logits, loss = self(idx_crop)
            # 从logits中获取最后一个时间步，logits的维度为(B,T,C)
            logits_last_timestep = logits[:, -1, :]
            # 使用softmax来得到概率
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            # 来自概率分布的样本。
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # 将采样的索引 idx_next 附加到 idx
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# 模型初始化
model = TransformerLanguageModel()
model = model.to(device)

# Get input embedding batch
def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    # 对每个起始索引取一个长度为 context_length 的片段作为输入 x。
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    # 对应目标 y 为每个输入片段的下一个 token 序列（右移一位），形状也为 [batch_size, context_length]。
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)
    return x, y

# 计算损失函数，调用TransformerLanguageModel
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 这段代码实现模型的训练流程：每步从训练数据采样一个批次，前向计算损失，反向传播梯度并用优化器更新模型参数；每隔一段步数运行一次评估并记录损失。
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
tracked_losses = list()
for step in range(max_iters):
    # 只输出每20批次，直到最后
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
              round(losses['valid'].item(), 3))

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 保存模型字典
torch.save(model.state_dict(), 'model-ckpt.pt')

# 评估模型
model.eval()
start = 'The salesperson'
start_ids = encoding.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokens=100)
print('---------------')
print(encoding.decode(y[0].tolist()))
print('---------------')
