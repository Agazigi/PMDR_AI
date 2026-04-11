import random
import sys
import torch
sys.path.append('../utils')
from utils import tokenize, Vocab, read_time_machine, plot, seq_data_iter_random, seq_data_iter_sequential
import matplotlib.pyplot as plt

tokens = tokenize(read_time_machine())
corpus = [token for line in tokens for token in line] # 按照 token = 'word; = ' 分词
vocab = Vocab(corpus)
print(f'【总字符数】：{len(corpus)}')
print(f'【总词数】：{len(vocab)}')
print(f'【前 20 个词】：{vocab.token_freqs[:20]}')


freqs = [freq for token, freq in vocab.token_freqs]
plot(freqs, xlabel='token: x', ylabel='frequency: n(x)', xscale='log', yscale='log')
plt.show()

# 二元模型
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])

# 三元模型
trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])

bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
plot(
    [freqs, bigram_freqs, trigram_freqs], 
    xlabel='token: x', ylabel='frequency: n(x)',
    xscale='log', yscale='log',
    legend=['unigram', 'bigram', 'trigram']
)
plt.show()


# 总之，传统语言模型使用 n-gram ，本质就是统计语言中单词出现的频率，
# 但是自然语言中的 单词的频率 恰好满足齐普夫定律 -> 大量的低频词、大量的 0 概率 -> 模型很差
# 使用 拉普拉斯平滑 可以缓解这个问题，但是非常粗糙，不能解决。
# 所以要使用 深度学习 的方法。



# 长序列的读取
# 偏移量：窗口大小为 3 时，偏移量为 1 时，窗口为 [1, 2, 3]，偏移量为 2 时，窗口为 [2, 3, 4]

my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=5, num_steps=6):
    print('Batch1:\nX: ', X, '\nY:', Y)
    
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('Batch2:\nX: ', X, '\nY:', Y)
