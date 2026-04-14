from turtle import forward
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, num_hiddens):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.conv = nn.Conv2d(in_channels=3, out_channels=num_hiddens, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, X):
        # X: [batch_size, num_channels, img_size, img_size]
        # conv(x): [batch_size, num_hiddens, img_size//patch_size, img_size//patch_size]
        X = self.conv(X).flatten(2).transpose(1, 2) # [batch_size, (img_size//patch_size)**2, num_hiddens]
        return X

img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = torch.zeros(batch_size, 3, img_size, img_size) # [batch_size, num_channels, img_size, img_size]
print(patch_emb(X).shape, (batch_size, (img_size//patch_size)**2, num_hiddens))

class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, X):
        X = self.dropout1(self.gelu(self.dense1(X)))
        X = self.dropout2(self.dense2(X))
        return X
       
class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens, num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens)
    
    def forward(self, X, valid_lens=None):
        # X: 
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        X = X + self.mlp(self.ln2(X))
        return X
    
        
X = torch.ones((2, 100, 24))
encoder_block = ViTBlock(24, 24, 48, 8, 0.5)
encoder_block.eval()
print(encoder_block(X).shape, X.shape)


class ViT(nn.Module):
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads, 
                 num_blocks, emb_dropout, block_dropout, lr=0.1, use_bias=False, num_classes=10):
        super().__init__()
        self.save_hyperparameters() # 保存超参数
        self.patch_embedding = PatchEmbedding(img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        seq_len = self.patch_embedding.num_patches + 1
        self.positional_embedding = nn.Parameter(torch.zeros(1, seq_len, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.Sequential()
        for i in range(num_blocks):
            self.blocks.add_module(
                f"block_{i}",
                ViTBlock(
                    num_hiddens, num_hiddens, mlp_num_hiddens, num_heads, block_dropout, use_bias
                )
            )
        self.head = nn.Sequential(
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, num_classes)
        )
        
    def forward(self, X):
        X = self.patch_embedding(X)
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), dim=1)
        X = self.dropout(X + self.positional_embedding)
        for block in self.blocks:
            X = block(X)
        X = self.head(X[:, 0])
        return X
    
if __name__ == '__main__':
    img_size, patch_size = 96, 16
    num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
    emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
    batch_size = 32
    model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
                num_blks, emb_dropout, blk_dropout, lr)
    
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size, resize=(img_size, img_size))
    
    max_epochs = 10
    device = torch.device('cuda')
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    
    loss_list = []
    acc_list = []
    
    for epoch in range(max_epochs):
        model.train()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            l = loss(y_pred, y)
            l.backward()
            trainer.step()
        loss_list.append(l.mean().item())
        print(f'Epoch {epoch}, Loss: {l.mean():.4f}')
        
        model.eval()
        acc = evaluate_accuracy(model, test_iter, device)
        acc_list.append(acc)
        print(f'Epoch {epoch}, Test acc: {acc:.4f}')