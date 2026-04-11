import torch
import torch.nn as nn
import sys
sys.path.append('../utils')
import matplotlib.pyplot as plt
from utils import *


class Seq2SeqEncoder(Encoder):
    """序列到序列编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size) # 词嵌入层
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)
    
    def forward(self, X, *args):
        """前向传播"""
        X = self.embedding(X) # [batch_size, num_steps, embed_size]
        X = X.permute(1, 0, 2) # [num_steps, batch_size, embed_size]
        output, state = self.rnn(X)
        # output: [num_steps, batch_size, num_hiddens]
        # state: [num_layers, batch_size, num_hiddens]
        return output, state

encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
# 为什么 embed_size = 8 而 num_hiddens = 16？
# 
encoder.eval()
X = torch.zeros((4, 7), dtype=torch.long) # [batch_size, num_steps]
output, state = encoder(X)
print(output.shape) # [7, 4, 16] # [num_steps, batch_size, num_hiddens]
print(state.shape) # [2, 4, 16] # [num_layers, batch_size, num_hiddens]


class Seq2SeqDecoder(Decoder):
    """序列到序列解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size) # 词嵌入层
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size) # 最后一层全连接层，输出词表大小
        
    def init_state(self, enc_outputs, *args):
        """Encoder 的最后一个时间步的输出状态作为解码器的输入隐状态"""
        # 取出 state
        # return enc_outputs[1] # [num_layers, batch_size, num_hiddens]
        return (enc_outputs[1], enc_outputs[1][-1]) # 第一个是编码器的输出状态，第二个是编码器的最后一个时间步的输出状态
        
    def forward(self, X, state):
        # X: [batch_size, num_steps]
        """前向传播"""
        X = self.embedding(X).permute(1, 0, 2) # [num_steps, batch_size, embed_size]
        
        context = state[-1] # 最后一个时间步的输出状态
        # state: [batch_size, num_hiddens]
        context = context.repeat(X.shape[0], 1, 1) # [num_steps, batch_size, num_hiddens]
        
        # [NOTE] 这里编码器的输出状态 encode 是一个元组，第一个元素是编码器的输出状态，第二个元素是编码器的最后一个时间步的输出状态
        # 这样保证了 context 是一个恒定的量，一直是编码器的最后一个时间步的输出状态，固定不变
        # 从而没有忘记原文的上下文信息
        encode = state[1] # 这里只有两层，所以 encode 就是 state[1] 的最后一个时间步的输出状态
        state = state[0]
        
        
        dec_input = torch.cat((X, context), dim=2) # [num_steps, batch_size, embed_size + num_hiddens]
        
        
        # output: [num_steps, batch_size, num_hiddens]
        # state: [num_layers, batch_size, num_hiddens]
        output, state = self.rnn(dec_input, state)
        
        
        # 这里又是一个全连接层，输出词表大小。后续做 softmax 函数求概率
        output = self.dense(output).permute(1, 0, 2) # [batch_size, num_steps, vocab_size]
        # return output, state
        return output, (state, encode)
    
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X)) # encoder 输入 X，然后获得上下文状态作为解码器的输入隐状态
output, state = decoder(X, state)
print(output.shape) # [4, 7, 10] # [batch_size, num_steps, vocab_size]
# print(state.shape) # [2, 4, 16] # [num_layers, batch_size, num_hiddens]


# 带掩码的交叉熵损失
# 编码器的输入是英文序列，目的是学到英文序列的上下文信息
# 解码器的输入是法语序列，输出是预测下一个词
loss = MaskedSoftmaxCrossEntropyLoss()


embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, torch.device('cuda')
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
model = EncoderDecoder(encoder, decoder)
train_seq2seq(model, train_iter, lr, num_epochs, tgt_vocab, device)
plt.show()


engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        model, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')