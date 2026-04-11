import re
import os
import time
import math
import torch
import random
import hashlib
import zipfile
import tarfile
import requests
import numpy as np
import torchvision
import collections
import torch.nn as nn
from IPython import display
from torch.utils import data
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from matplotlib_inline import backend_inline

DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB ={
    'time_machine': (DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a'),
    'fra-eng': (DATA_URL + 'fra-eng.zip', '94646ad1522d915e7b0f9296181140edcf86a4f5')
}



def train_seq2seq(model, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear: # 如果是一个全连接层
            nn.init.xavier_uniform_(m.weight) # 初始化权重
        if type(m) == nn.GRU:
            for param in m._flat_weights_names: # 遍历所有权重参数
                if "weight" in param: # 如果是权重参数
                    nn.init.xavier_uniform_(m._parameters[param])
                
    animator = Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])
    
    model.apply(xavier_init_weights)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = MaskedSoftmaxCrossEntropyLoss()
    model.train()

    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)
        
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1) # [batch_size, 1]
            
            # 这里的 Y 同时向后移动一个时间步，作为解码器的输入序列s
            # 也就是有监督学习、强制教学，可以联想外推法、内插法
            # 这里都是正确的答案，而不是输入上一步的预测结果
            # 这样训练更加稳定
            dec_input = torch.cat([bos, Y[:, :-1]], 1) # [batch_size, num_steps]
            Y_hat, _ = model(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            # l.mean().backward()
            grad_clipping(model, 1)
            optimizer.step()
            
            num_tokens = Y_valid_len.sum()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1], ))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')


def predict_seq2seq(model, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    '''根据源语言句子预测目标语言句子'''
    model.eval()
    
    # 处理源语言句子
    # 原句子转换成索引序列 + <eos> 令牌
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']] # 源语言句子的索引序列
    enc_valid_len = torch.tensor([len(src_tokens)], device=device) # 源语言句子的有效长度
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>']) # 截断或填充 src_tokens，使长度为 num_steps
    
    # 编码器的输入是源语言句子的索引序列
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0) # [1, num_steps]
    
    # 输入到 encoder 得到上下文状态
    enc_outputs = model.encoder(enc_X, enc_valid_len)
    
    # 初始化解码器的输入隐状态
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_len)
    # 构造解码器的输入序列
    # 这里就是一个 <bos> 令牌
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0) # [1, 1] 只有一个 <bos> 令牌
    
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps): # 自回归生成
        # dec_X: [1, 1]
        # dec_state: [num_layers, 1, num_hiddens]
        Y, dec_state = model.decoder(dec_X, dec_state)
        # dec_X 作为 [1, 1] 输入进去后
        # 会先进行 Embedding 变成 [1, 1, embed_size]
        # 最后经过 GRU + 线性层 变成 [1, 1, vocab_size]
        
        
        # Y: [1, 1, vocab_size]
        dec_X = Y.argmax(dim=2) # [1, 1]
        pred = dec_X.squeeze(dim=0).type(torch.int32).item() # 取出预测的词索引
        if save_attention_weights:
            attention_weight_seq.append(model.decoder.attention_weights)
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
        
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
    

def bleu(pred_seq, label_seq, k):
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def sequence_mask(X, valid_len, value=0):
    """为序列添加掩码"""
    maxlen = X.size(1) # 最大序列长度 num_steps
    mask = torch.arange((maxlen), dtype=torch.int32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    """为序列添加掩码的交叉熵损失"""
    def forward(self, pred_X, label, valid_len):
        # pred_X: [batch_size, num_steps, vocab_size]
        # label: [batch_size, num_steps]
        # valid_len: [batch_size]
        mask = torch.ones_like(label) # 创建全 1 的张量
        mask = sequence_mask(mask, valid_len) # 对 <pad> 令牌进行掩码
        self.reduction = 'none' # 关闭 torch 的自动平均操作
        loss = super(MaskedSoftmaxCrossEntropyLoss, self).forward(
            pred_X.permute(0, 2, 1), # [batch_size, vocab_size, num_steps]
            label # [batch_size, num_steps]
        )
        loss = (loss * mask).mean(dim=1)
        return loss


class Encoder(nn.Module):
    """编码器"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, X, *args):
        raise NotImplementedError
    
class Decoder(nn.Module):
    """解码器"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError
    
    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """编码解码器"""
    def __init__(self, encoder, decoder, **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, enc_X, dec_X, *args):
        # 编码器的输入
        # 解码器的输入
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


def build_array_nmt(lines, vocab, num_steps):
    lines = [vocab[l] + [vocab['<eos>']] for l in lines] # 将所有的文本转换为词表中的索引。这是一个二维列表
    # 例如： go . <eos> -> [47, 4, 3]
    array = torch.tensor(
        [truncate_pad(l, num_steps, vocab['<pad>']) for l in lines]
        # 截断或填充到统一长度 num_steps
    )
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1) # 计算每个样本的有效长度
    # 这个就是计算不为 <pad> 的令牌数量就是有效长度
    # 是一个一维张量，每个元素表示对应样本的有效长度
    
    # 最终的 array 是一个二维张量，每个元素表示一个样本的词表索引
    # [len(lines), num_steps]
    # 每个样本的长度为 num_steps
    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """加载机器翻译数据集"""
    text = preprocess_nmt(read_data_nmt()) # 预处理之后的文本
    source, target = tokenize_nmt(text, num_examples) # 将文本 tokenize 为源语言和目标语言，其中 num_examples 是样本数量
    
    # 建立源语言和目标语言的词表
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    # 这个 valid_len 的效果和 mask 差不多，都是计算 loss 时的掩码用于忽略填充令牌

    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    
    data_iter = load_array(data_arrays, batch_size)
    # 返回数据迭代器、源语言词表和目标语言词表
    return data_iter, src_vocab, tgt_vocab

def truncate_pad(line, num_steps, padding_token): # 这里的 padding_token 是填充令牌
    """截断或填充序列"""
    if len(line) > num_steps: # 如果我的序列长度大于 num_steps，这个类似于 max_seq_len
        return line[:num_steps] # 截断序列
    return line + [padding_token] * (num_steps - len(line)) # 填充序列


def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制源语言和目标语言样本长度直方图"""
    set_figsize()
    _, _, patches = plt.hist([[len(l) for l in xlist], [len(l) for l in ylist]])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    for patch in patches[1].patches: # 遍历目标语言直方图的每个柱子
        patch.set_hatch('/') # 设置目标语言直方图为斜线
    



def tokenize_nmt(text, num_examples=None):
    """将机器翻译数据集 tokenize 为源语言和目标语言"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')): # 每一行
        if num_examples and i > num_examples: # 如果超过 num_examples 行，则停止
            break
        parts = line.strip().split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

def preprocess_nmt(text):
    """预处理机器翻译数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' ' # 如果当前字符是标点符号，且前一个字符不是空格，则添加空格

    text = text.replace('\u202f', ' ') # 替换窄空格字符为普通空格
    text = text.replace('\xa0', ' ').lower() # 替换不间断空格字符为普通空格，转换为小写
    out = [' ' + char if i > 0 and no_space(char, text[i-1]) else char for i, char in enumerate(text)]
    return ''.join(out) # 返回处理后的字符串



def download_extract(name, folder=None):
    """下载并解压数据集"""
    fname = download(name)
    base_dir = os.path.dirname(fname) # 提取数据集目录名
    data_dir, ext = os.path.splitext(fname) # 提取数据集目录名和扩展名
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, f'File format {ext} not supported'
    fp.extractall(base_dir) # 解压数据集
    return os.path.join(base_dir, data_dir) if folder else data_dir # 返回数据集目录名

def read_data_nmt():
    """读取机器翻译数据集"""
    data_dir = download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read() # 返回一个包含所有数据的字符串，每个行是一个样本

def train_rnn_epoch(model, train_iter, loss, updater, device, use_random_iter):
    """训练 RNN 模型"""
    state, timer = None, Timer()
    metrics = Accumulator(2)
    
    for X, Y in train_iter:
        
        # 这部分是对隐状态梯度的处理：截断反向传播
        # 防止梯度爆炸问题，只计算当前 batch 的梯度
        if state is None or use_random_iter: # 如果没有隐状态，则初始化
            state = model.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 如果有隐状态，则断开与前一个 batch 的联系，防止梯度累加
            if isinstance(model, nn.Module) and not isinstance(state, tuple): # 对 GRU 来说是个元组
                state.detach_()
            else:
                for s in state:
                    s.detach_()
            # 总之，模型能够学会从已有的文本中提取隐状态
        
        y = Y.T.reshape(-1) # Y: [batch_size, num_steps] -> [num_steps, batch_size] -> [batch_size * num_steps]
        # 将 y 展平是为了方便计算损失
        X, y = X.to(device), y.to(device)
        y_hat, state = model(X, state) # 此时的 y_hat 是 [num_steps * batch_size, num_outputs]
        l = loss(y_hat, y.long()).mean() # y 是 [batch_size * num_steps]
        # 
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad() # 清空梯度
            l.backward() # 计算梯度
            grad_clipping(model, 1) # 梯度裁剪
            updater.step() # 更新参数
        else:
            l.backward()
            grad_clipping(model, 1)
            updater(batch_size=1) # 更新参数，1 是不用在除以 batch_size，因为已经调用了 mean()
        metrics.add(l * y.numel(), y.numel()) # 累加损失和词元数量
        # print(f'【训练损失】: {l:.3f}')
    return math.exp(metrics[0] / metrics[1]), metrics[1] / timer.stop() # 返回困惑度、词元/秒



def train_rnn(model, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """训练 RNN 模型"""
    loss = nn.CrossEntropyLoss()
    if isinstance(model, nn.Module):
        updater = torch.optim.SGD(model.parameters(), lr) # 使用 torch
    else:
        updater = lambda batch_size: sgd(model.params, lr, batch_size) # 使用自定义的优化函数

    animator = Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])
    predict = lambda prefix: predict_rnn(prefix, 50, model, vocab, device)
    
    for epoch in range(num_epochs):
        ppl, speed = train_rnn_epoch(model, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller '))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.2f}, {speed:.2f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

def grad_clipping(model, theta):
    """梯度裁剪"""
    if isinstance(model, nn.Module): # Pytorch 封装好了
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = model.params
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params)) # 求梯度的范数
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm # 梯度裁剪

def predict_rnn(prefix, num_preds, model, vocab, device):
    """使用 RNN 预测序列"""
    # prefix: 输入的序列 例如 'time machine'
    # num_preds: 预测的词元数量
    # model: RNN 模型
    # vocab: 词表
    # device: 设备
    state = model.begin_state(batch_size=1, device=device) # [batch_size, num_hiddens] -> [1, 512]
    # 这里只有一个子序列
    
    outputs = [vocab[prefix[0]]] # 输入的序列的第一个词元，这里就是第一个字符
    
    # 预热期，RNN 是有记忆的，需要先吃一遍前缀，形成正确的隐状态
    # 这和内插法有点类似，用真实的数据进行预测
    # 为的是形成正确的隐状态，确保模型在预测时能够正确地利用前缀的信息
    for token in prefix[1:]: # 跳过 第一个词元，预测 下一个时间步
        input = torch.tensor([outputs[-1]], device=device).reshape(1, 1) # [1, 1]
        _, state = model(input, state)
        outputs.append(vocab[token]) # 这里存放的就是真实的词元
    
    # 这时候才是正式的预测期
    for _ in range(num_preds):
        input = torch.tensor([outputs[-1]], device=device).reshape(1, 1) # [1, 1] 这里实现的就是单时间步的预测
        y, state = model(input, state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """顺序采样的序列数据迭代器"""
    # corpus: 这是整个文本的索引列表
    # batch_size: 每一个批次的大小，包含几个子序列
    # num_steps: 窗口大小，也就是每个子序列的长度，每个子序列的预测目标是下一个词元
    offset = random.randint(0, num_steps) # 初始偏移量
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size # 这个计算了满足 batch_size 个子序列的总词元数， 这里 -1 是为了最后一个标签
    Xs = torch.tensor(corpus[offset: offset + num_tokens]) # num_tokens 可以被batch_size 整除
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens]) # +1 是预测下一个词元
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1) # [batch_size, seq_len]
    num_subseqs = Xs.shape[1] // num_steps # 每个 Batch 的子序列数量
    for i in range(0, num_steps * num_subseqs, num_steps): # 每个 Batch 的子序列的第一个元素的索引 列表
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

def seq_data_iter_random(corpus, batch_size, num_steps):
    """随机采样的序列数据迭代器"""
    # corpus: 这是序列数据的列表
    # batch_size: 批量大小
    # num_steps: 窗口大小
    corpus = corpus[random.randint(0, num_steps - 1):] # 从随机位置开始，取 之后的所有词元。这就是随机偏移量
    num_subseqs = (len(corpus) - 1) // num_steps # 计算可以形成多少个子序列， 这里 -1 是为了最后一个标签
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps)) # 生成子序列的第一个元素的索引 列表
    random.shuffle(initial_indices) # 随机打乱索引列表
    
    def data(pos):
        return corpus[pos: pos + num_steps] # 按照首索引，返回一个子序列
    
    num_batches = num_subseqs // batch_size # 子序列再组装成 Batch 的数量
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size] # 每个 Batch 的子序列的第一个元素的索引 列表
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y) # 返回 X 和 Y
        # 也就是说：
        # 输入是一个 Batch， 其中每一个都是一个 长度为 num_steps 的序列
        # 输出是一个 Batch， 其中每一个都是平移一个 词元 位置之后，长度为 num_steps 的序列
    
class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size = batch_size
        self.num_steps = num_steps
    
    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
        
def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None: # 这个是要保留的词元，比如 <unk>, <pad>, <bos>, <eos> 等
            reserved_tokens = []
        counter = self._count_corpus(tokens) # 统计词元的出现频率
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True) # 按照出现频率从大到小排序
        self.idx_to_token = ['<unk>'] + reserved_tokens # 词元到索引的映射表，这里先保留了 <unk>
        self.token_to_idx = {
            token: idx for idx, token in enumerate(self.idx_to_token)
        } # 索引到词元的映射表，这里目前也是只保留了 <unk> 和 reserved_tokens 中的词元
        
        for token, freq in self._token_freqs: # 处理其他的词元
            if freq < min_freq:
                break # 出现频率小于 min_freq 的词元，不加入到词元表中
            if token not in self.token_to_idx: # 词元不在 词元到索引 的映射表中，则加入。这里也可以保证对 _token_freqs 进行去重来避免重复
                self.idx_to_token.append(token) # 添加词元，使用索引即可访问。 这就是个列表
                self.token_to_idx[token] = len(self.idx_to_token) - 1 # 词元到索引，使用字典存储
                
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)): # 如果不是列表或元组，也就是 单个 token
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices): # 索引到词元
        if not isinstance(indices, (list, tuple)): # 如果不是列表或元组，也就是 单个索引
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property # 这个注解是为了在类的外部调用时，能够像调用属性一样调用方法
    def unk(self):
        return 0 # unk 索引为0
    
    @property
    def token_freqs(self): # 返回词元的出现频率列表
        return self._token_freqs
    
    def _count_corpus(self, tokens):
        """统计词元的出现频率"""
        if len(tokens) == 0 or isinstance(tokens[0], list): # 将二维列表转换为一维列表
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens) # 返回统计词元的出现频率的字典
    
def tokenize(lines, token='word'):
    """将文本行拆分成单词元或字符元 总之叫词元 token"""
    if token == 'word':
        return [line.split() for line in lines] # 二维列表
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print(f'未知的 token 类型 {token}')

def download(name, cache_dir=os.path.join('..', 'data')):
    """下载数据集"""
    assert name in DATA_HUB, f'{name} 不存在于 {DATA_HUB}'
    url, hash_val = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1]) # 文件名
    if os.path.exists(fname): # 文件已下载
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
            if sha1.hexdigest() == hash_val:
                return fname # 文件已下载，且校验通过
    print(f'正在从 {url} 下载 {fname} ...')
    r = requests.get(url, stream=True, verify=True) # 下载文件
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname 

def read_time_machine():
    """下载和读取时间机器数据集"""
    with open(download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line.strip().lower()) for line in lines]


def load_corpus_time_machine(max_token=-1):
    lines = read_time_machine() # 读取数据，这时是一个一维列表
    tokens = tokenize(lines, token='char') # 分词，按照 字符 分词，分词之后是一个二维列表，每个元素是一个列表
    vocab = Vocab(tokens) # 词元表
    corpus = [vocab[token] for line in tokens for token in line] # 这是一个一维列表，将原文中的每个字符转换为对应的索引
    if max_token > 0:
        corpus = corpus[:max_token]
    return corpus, vocab
    
    

def train_gpu(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型"""
    # 初始化模型参数
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    
    # GPU 训练
    print('training on', device)
    net.to(device)
    
    # 定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    
    # 定义损失函数
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        print(f'epoch {epoch + 1}, loss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}')
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device # 获取模型参数所在设备
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def corr2d(X, K):
    """计算二维卷积/互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = torch.sum(X[i:i+h, j:j+w] * K) # 子矩阵和卷积核元素乘
            
    return Y

def get_dataloader_workers():
    """使用0个进程来读取数据"""
    return 0


def use_svg_display():
    """使用svg格式在Jupyter中显示绘图"""
    backend_inline.set_matplotlib_formats('svg')
    
def set_figsize(figsize=(3.5, 2.5)):
    """设置matplotlib的图表大小"""
    try:
        use_svg_display()
    except:
        plt.rcParams['figure.figsize'] = figsize
    plt.figure(figsize=figsize)
    
    

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
    

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes or plt.gca() # gca() 获取当前轴

    def has_one_axis(X):
        # 检查 X 是否只有一个轴
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    
    
class Timer:
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()
    
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w))) # 从 N(0, 1) 正态分布采取 [num_examples, len(w)]
    Y = torch.matmul(X, w) + b # [num_examples, 2] * [2, 1] + [1, 1] -> [num_examples, 1]
    Y += torch.normal(0, 0.01, Y.shape) # 噪声
    return X, torch.reshape(Y, (-1, 1))


def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad(): # 禁用梯度计算
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
            
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays) # 创建数据集
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def get_fashion_mnist_labels(labels):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten() # 将 axes 变成 1D
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False) # 隐藏坐标轴
        ax.axes.get_yaxis().set_visible(False) # 隐藏坐标轴
        if titles:
            ax.set_title(titles[i])
    return axes

def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
    
    
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1) # 最大概率的索引
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)] # 新的值 加到已有的值

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward() # 这个求和
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
        
        
def train(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f'训练 epoch {epoch} / {num_epochs}, 训练损失 {train_metrics[0]:.3f}, 训练精度 {train_metrics[1]:.3f}, 测试精度 {test_acc:.3f}')
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
    
def predict(net, test_iter, n=6):
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    
    
    
    
def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失"""
    metric = Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]