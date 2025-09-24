import os
import requests
import torch
import math
import tiktoken
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 4
context_length = 16
d_model = 64
num_layers = 8  # Number of transformer blocks
num_heads = 4  # Number of heads in Multi-head attention # 代码中通过 d_model / num_heads = 来获取 head_size
learning_rate = 1e-3  # 0.001
dropout = 0.1  # Dropout rate 把已经训练好的数据随机丢百分之十出去重新训练，防止过拟合
max_iters = 5000  # Total of training iterations
eval_interval = 50  # How often to evaluate the model
eval_iters = 20  # How many iterations to average the loss over when evaluating the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Instead of using the cpu, we'll use the GPU if it's available.
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

# Stage1:获取数据集
if not os.path.exists('sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt?download=true'
    # url = 'https://huggingface.co/datasets/wzy816/scifi/resolve/main/data.zip?download=true'
    with open('sales_textbook.txt', 'wb') as f:
        f.write(requests.get(url).content)

with open('sales_textbook.txt', 'r') as f:
    text = f.read()

# Stage2:序列化 Tokenize the text
encodings = tiktoken.get_encoding("cl100k_base")
tokenized_text = encodings.encode(text)
tokenized_text = torch.tensor(tokenized_text).long()
max_token_value = tokenized_text.max().item()

# Stage3:split,将数据集切分为训练数据集和测试数据集
train_idex = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:train_idex]
valid_data = tokenized_text[train_idex:]


# 前馈神经网络的包装类
class FeedforwardNetwork(nn.Module):
    def __init__(self):
        super(FeedforwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# 多头注意力机制包装类，要做Wq矩阵与Q相乘,单头计算
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        # super(self).__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        # 应用mask
        self.register_buffer('mask', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        Q = x * self.Wq(x)
        K = x * self.Wk(x)
        V = x * self.Wv(x)

        attention = Q * K.transpose(-2, -1) / math.sqrt(d_model // num_heads)
        attention = attention.masked_fill(self.mask == 0, float('-inf'))
        attention = attention * V


# 多头注意力机制，设置参数4,4个头循环进行
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([ScaleDotProductAttention() for _ in range(num_heads)])
        self.projection_layer = nn.Linear(d_model, d_model)

    def forward(self, x):
        self.heads = [head(x) for head in self.heads]
        out = torch.cat(self.heads, dim=-1)
        out = self.projection_layer(out)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self):
        super(TransformerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.multi_head_attention = MultiHeadAttention()
        self.feedforward_network = FeedforwardNetwork()

    def forward(self, x):
        x = x + self.multi_head_attention(self.layer_norm1(x))
        x = x + self.multi_head_attention(self.layer_norm2(x))
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.token_embedding_lookup_table = nn.Embedding(max_token_value, d_model)
        self.transformer_blocks = nn.ModuleList([TransformerBlock() for _ in range(num_layers)])
        self.model_out_linear_layer = nn.Linear(d_model, max_token_value)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        position_encoding_lookup_table = torch.zeros(context_length, d_model, device=device)
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)

        position_embedding = position_encoding_lookup_table[:T, :].to(device)
        x = self.token_embedding_lookup_table(idx) + position_embedding
        x = self.transformer_blocks(x)
        logits = self.model_out_linear_layer(x)

        if targets is not None:
            B, T, C = logits.shape
            logits.reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=targets_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss


# 创建模型，把他丢进gpu内
model = Model().to(device)


# get batch
def get_batch(split: str):
    data = train_data if split == 'train' else valid_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)
    return x, y


# 上下文禁用自动梯度求导，从而减少内存消耗
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model.get_batch(split)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out


# create the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
tracked_losses = list()
for step in range(max_iters):
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
              round(losses['valid'].item(), 3))

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# save the model
torch.save(model.state_dict(), 'model.pt')

# 评估模型
model.eval()
start = 'The product is'
start_ids = encodings.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokends=100)
print('---------------')
print(encodings.decode(y[0].tolist()))
print('---------------')
