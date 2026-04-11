import sys
sys.path.append('../utils')
from utils import read_time_machine, tokenize, Vocab, load_corpus_time_machine
import os

# Download the dataset
lines = read_time_machine()
print(f'Total lines of the dataset: {len(lines)}')
print(lines[0])
print(lines[10])
if not os.path.exists(os.path.join('..', 'data', 'timemachine_cleaned.txt')):
    with open(os.path.join('..', 'data', 'timemachine_cleaned.txt'), 'w') as f:
        f.write('\n'.join(lines))
    print(f'Total lines of the cleaned dataset: {len(lines)}')


# 分词
tokens = tokenize(lines, token='word')
for i in range(10):
    print(tokens[i])
    
# 词元表
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:20])

example_1 = tokens[0]
example_2 = tokens[10]
print(f'【原句子】：{example_1}')
print(f'【索引】：{vocab[example_1]}')
print(f'【词元】：{vocab.to_tokens(vocab[example_1])}')
print(f'【原句子】：{example_2}')
print(f'【索引】：{vocab[example_2]}')
print(f'【词元】：{vocab.to_tokens(vocab[example_2])}')


corpus, vocab = load_corpus_time_machine()
print(f'【总字符数】：{len(corpus)}')
print(f'【总词数】：{len(vocab)}') # 28 = 26 个字母 + 1 个空格 + 1 个换行符

print(f'【前 30 个字符】：{corpus[:30]}')
print(f'【前 30 个词】：{vocab.to_tokens(corpus[:30])}')

# 这里的 corpus 就是将整篇文章，按照 token='char' 分词后，每个元素都是一个字符的索引的结果
# vocab 可以通过 vocab.to_tokens() 方法，将索引转换为对应的词元
# 同时 vocab[vocab.to_tokens(corpus[:30])] 也可以将词元转换为对应的索引，但是这就有点套娃了
print(f'【整篇文章】：{"".join(vocab.to_tokens(corpus))}')