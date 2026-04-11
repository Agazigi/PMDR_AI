import os
import torch
import sys
sys.path.append('../utils')
from utils import *
import matplotlib.pyplot as plt


raw_text = read_data_nmt()
print(raw_text[:75])

text = preprocess_nmt(raw_text)
print(text[:80])

source, target = tokenize_nmt(text)
print(source[:6], target[:6])

show_list_len_pair_hist(['Source', 'Target'], '# Tokens per Sample', 'Count', source, target)
plt.show()

# 分别对英语 和 发育建立两个词表
src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
# 出现频率小于 2 次的令牌，将被忽略 <unk>: 未知令牌
# <pad>: 填充令牌，用于将样本长度统一到最大长度
# <bos>: 开始令牌，用于表示句子的开始
# <eos>: 结束令牌，用于表示句子的结束
print(len(src_vocab))
print(src_vocab['<pad>']) # 1
print(src_vocab['<bos>']) # 2
print(src_vocab['<eos>']) # 3


# 填充与截断序列
print(f'【第 0 个源语言样本】{source[0]}')
print(f'【第 0 个源语言样本的词表】{src_vocab[source[0]]}')
print(f'【第 0 个源语言样本的词表截断或填充之后】{truncate_pad(src_vocab[source[0]], 10, src_vocab["<pad>"])}')


train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=10)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print(f'【 X 的形状】{X.shape}')
    print(f'【 X 的内容】{X.type(torch.int32)}')
    print('X的有效长度:', X_valid_len)
    print(f'【 X 的第 0 个样本的词表】{src_vocab.to_tokens(X[0].cpu().numpy().tolist())}')

    print('Y:', Y.type(torch.int32))
    print('Y的有效长度:', Y_valid_len)
    print(f'【 Y 的第 0 个样本的词表】{tgt_vocab.to_tokens(Y[0].cpu().numpy().tolist())}')
    break


