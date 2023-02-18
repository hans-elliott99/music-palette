import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    """
    Batch First Multi-Head Attention
    https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L30
    """
    def __init__(self, block_size, n_embed, n_head, attn_dropout=0., resid_dropout=0.):
        super().__init__()
        assert n_embed % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embed, 3 * n_embed)
        # output projection
        self.c_proj = nn.Linear(n_embed, n_embed)
        # regularization
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))
        self.n_head = n_head
        self.n_embed = n_embed

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embed)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class LinearProjection(nn.Module):
    def __init__(self, in_feats, out_feats, dropout=0., bias=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, out_feats, bias=bias),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class TrfmrFeedForward(nn.Module):
    def __init__(self, in_feats, hidden_ratio=4, dropout=0.):
        #default in_feats*4, as in Attention is All You Need & AN IMAGE IS WORTH 16X16 WORDS (ViT)
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, hidden_ratio*in_feats), 
            nn.GELU(),
            # project back into residual pathway
            nn.Linear(hidden_ratio*in_feats, in_feats),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class TfmrBlock(nn.Module):
    """Transformer Block
    LayerNorm
    MultiHeadAttention
    LayerNorm
    FeedForward
    """
    def __init__(self, block_size, in_feats, n_heads, attn_dropout, resid_dropout, mlp_dropout):
        super().__init__()
        # head_size = in_feats // n_heads
        mlp_ratio = 4
        self.ln1 = nn.LayerNorm(in_feats)
        self.attn = CausalSelfAttention(block_size, n_embed=in_feats, n_head=n_heads, attn_dropout=attn_dropout, resid_dropout=resid_dropout)
        self.ln2 = nn.LayerNorm(in_feats)
        self.mlp = TrfmrFeedForward(in_feats, mlp_ratio, mlp_dropout)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x)) #skip connections
        x = x + self.mlp(self.ln2(x))
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, conv_filters, kernel_size, pool_size, dropout, conv_activ=nn.GELU) -> None:
        """ConvolutionalBlock
        Conv_kxk
        BatchNorm
        MaxPool_pxp
        Skip-Connection Layer
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=conv_filters,
                              kernel_size=kernel_size,
                              padding="same")
        self.activ = conv_activ()
        self.bn = nn.BatchNorm2d(num_features=conv_filters)
        self.dropout = nn.Dropout(dropout)

        if pool_size == 1 or None:
            self.maxpool = None
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=pool_size,
                                        stride=pool_size)
            
        if self.maxpool is None and in_channels==conv_filters:
            self.skip = False
        else:        
            self.skip = nn.Conv2d(in_channels=in_channels, out_channels=conv_filters,
                                  kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        """inspo: https://github.com/pytorch/vision/blob/a9a8220e0bcb4ce66a733f8c03a1c2f6c68d22cb/torchvision/models/resnet.py#L56-L72
        """
        identity = x
        x = self.conv(x)
        x = self.bn(x)
        if self.maxpool:
            x = self.maxpool(x)
            identity = self.maxpool(identity)
        if self.skip:
            x += self.skip(identity)

        x = self.activ(x)
        return self.dropout(x)
