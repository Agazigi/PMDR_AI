import torch
import torch.nn as nn
import sys
sys.path.append('../utils')
from utils import *
import matplotlib.pyplot as plt



class Seq2SeqAttentionDecoder(nn.Module):
    """序列到序列注意力解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        # 这里多了一个 Attention 层
        self.attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        
    def forward(self, X, state):
        # enc_outputs: [batch_size, seq_len, num_hiddens]
        # hidden_state: [num_layers, batch_size, num_hiddens]
        # enc_valid_lens: [batch_size]
        enc_outputs, hidden_state, enc_valid_lens = state
        
        X = self.embedding(X).permute(1, 0, 2) # [seq_len, batch_size, embed_size]
        
        outputs, self._attention_weights = [], []
        
        for x in X: # 按时间步解码
            # x: [batch_size, embed_size]
            
            # 取出 encoder 中最后一层的隐藏状态
            # hidden_state: [num_layers, batch_size, num_hiddens] 选出最后一层的隐藏状态
            # query: [batch_size, 1, num_hiddens]
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            
            # 这里的 Attention 是 encoder 自己的输出和硬状态做 Attention
            # context: [batch_size, 1, num_hiddens]
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            
            X = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # X: [batch_size, 1, embed_size + num_hiddens]
            out, hidden_state = self.rnn(X.permute(1, 0, 2), hidden_state)
            # out: [1, batch_size, num_hiddens]
            # hidden_state: [num_layers, batch_size, num_hiddens]
            outputs.append(out)
            # attention_weights: [batch_size, num_steps, num_hiddens]
            self._attention_weights.append(self.attention.attention_weights)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]
            
    def init_state(self, enc_outputs, enc_valid_lens, *args):
        outputs, hidden_state = enc_outputs
        # outputs: [batch_size, seq_len, num_hiddens]
        # hidden_state: [num_layers, batch_size, num_hiddens]
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)
    
    @property
    def attention_weights(self):
        return self._attention_weights # 


encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
encoder.eval()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
decoder.eval()
X = torch.zeros((4, 7), dtype=torch.long)  # (batch_size,num_steps)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
# output: [batch_size, num_steps, vocab_size]
# enc_outputs: [batch_size, seq_len, num_hiddens]
# hidden_state: [num_layers, batch_size, num_hiddens] [2, 4, 16]
# enc_valid_lens: [batch_size]
print(output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape)


embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 250, torch.device('cuda')

train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    
    
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
    print(f'dec_attention_weight_seq: {dec_attention_weight_seq}')
    # [att_0, att_1, ..., att_num_steps-1]
    # att_0: [1, 1, 1, 10]
    attention_weights = torch.cat([step[0][0][0] for step in dec_attention_weight_seq], 0).reshape((1, 1, -1, num_steps))
    # 加上一个包含序列结束词元
    show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(), xlabel='Key positions', ylabel='Query positions') # 这个展示的就是英语和法语之间的注意力权重
    plt.show()

