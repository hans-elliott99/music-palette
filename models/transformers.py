import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class AttHead(nn.Module):
    def __init__(self, block_size, head_size, in_feats, dropout) -> None:
        super().__init__()
        self.key = nn.Linear(in_feats, head_size, bias=False)
        self.query = nn.Linear(in_feats, head_size, bias=False)
        self.value = nn.Linear(in_feats, head_size, bias=False)
        # buffer: since tril is not a module, assign using register buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 #scaled, out:(B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B,T,T)
        wei = F.softmax(wei, dim=1)
        wei = self.dropout(wei)
        # weigted aggregation
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, block_size, num_heads, head_size, in_feats, dropout) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            AttHead(block_size, head_size, in_feats, dropout) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(in_feats, in_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate over channel/feature dimension (last dim)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, in_feats, dropout) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, 4*in_feats), #times 4, as in Attention is All You Need
            nn.ReLU(),
            # project back into residual pathway
            nn.Linear(4*in_feats, in_feats),
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
    def __init__(self, block_size, in_feats, n_heads, dropout) -> None:
        super().__init__()
        head_size = in_feats // n_heads
        self.ln1 = nn.LayerNorm(in_feats)
        self.sa = MultiHeadAttention(block_size, n_heads, head_size, in_feats, dropout)
        self.ln2 = nn.LayerNorm(in_feats)
        self.ffwd = FeedForward(in_feats, dropout)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) #skep connections
        x = x + self.ffwd(self.ln2(x))
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, conv_filters, kernel_size, pool_size, dropout, conv_activ=nn.GELU) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, #(1 channel)
                              out_channels=conv_filters,
                              kernel_size=kernel_size,
                              padding="same")
        self.activ = conv_activ()
        self.maxpool = nn.MaxPool2d(kernel_size=pool_size,
                                 stride=pool_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.activ(self.conv(x))
        x = self.maxpool(x)
        return self.dropout(x)


class CNNLayerNorm(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_feats)
    def forward(self, x):
        #x shape: (b, Channels, H[features], W[time])
        x = x.transpose(dim0=2, dim1=3) 
        x = self.layer_norm(x) #normalize features
        return x.transpose(2,3)



class ConvTransformer(nn.modules.Module):
    """
    """
    def __init__(self, 
                 X_shape,
                 max_seq_len,
                 n_colors=5,
                 # multi-head attention
                 #n_embed=32,
                 n_heads=4,
                 n_layers=4,
                 dropout=0.0
                 ) -> None:
        super().__init__()
        self.n_colors = n_colors
        self.max_seq_len = max_seq_len
        
        conv_filters = [64, 128, 128]  # filter sizes
        kernel_size = (3, 3)  # convolution kernel size
        pool_size = [(4, 4), (8, 8), (4, 4)]  # size of pooling area
        n_conv_layers = len(conv_filters)    #n of conv layers
        self.conv_activ = nn.GELU  # activation function to use after each layer
        
        # Image input shape = (1, 128, 157)
        if len(X_shape) == 5: #batch, seq_length, C, H, W
            self.input_shape = X_shape[1:] #seq, C, H, W
        else:
            assert len(X_shape) == 4
            self.input_shape = X_shape

        seq_len, Channels, H_freq, W_time = self.input_shape

        # Initialize layers
        # Normalize along frequency axis
        self.ln_1 = CNNLayerNorm(H_freq)


        # First conv layer
        self.conv_1 = ConvBlock(in_channels=Channels,
                                conv_filters=conv_filters[0],
                                kernel_size=kernel_size,
                                pool_size=pool_size[0],
                                conv_activ=self.conv_activ,
                                dropout=dropout
                                )

        # More conv layers
        self.conv_layers = nn.Sequential(*[
            ConvBlock(in_channels=conv_filters[i],
                        conv_filters=conv_filters[i+1],
                        kernel_size=kernel_size,
                        pool_size=pool_size[i],
                        conv_activ=self.conv_activ,
                        dropout=dropout) \
            for i in range(n_conv_layers -1)     
        ])
        
        
        # get conv output shape
        self.num_encoded_feats = self.calc_encoder_output_shape()[-1]

        # positional embeddings
        self.pos_embed = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=self.num_encoded_feats)

        # Transformers
        self.blocks = nn.Sequential(
            *[TfmrBlock(block_size=max_seq_len, 
                    in_feats=self.num_encoded_feats, 
                    n_heads=n_heads, 
                    dropout=dropout) for _ in range(n_layers)]
            )

        # Classifier / Head
        self.head_ln = nn.LayerNorm(self.num_encoded_feats)
        self.head_dense0 = nn.Linear(in_features=self.num_encoded_feats,
                                   out_features=n_colors*3) #rgb
        self.head_sig  = nn.Sigmoid()


        # --apply special weight initializations--
        self._special_init_weights()

        # --gen model stats--
        self.n_params = sum([p.numel() for p in self.parameters()])
        self.n_trainable_params = sum([p.numel() for p in self.parameters() if p.requires_grad])
    

    @torch.no_grad()
    def calc_encoder_output_shape(self):
        """For ex, if we added/removed conv layers
        """
        X = torch.zeros(1, 1, 1, 128, 157) #b, T, C, H[mel], W[time]
        batch_size, seq_len, C, H, W = X.shape

        # EMBEDDINGS
        # Conv Embeddings
        x = X.reshape(batch_size*seq_len, C, H, W)
        x = self.conv_1(x)
        x = self.conv_layers(x)
        x = x.reshape(batch_size, seq_len, -1)
        return x.shape


    def _special_init_weights(self):
        """
        Prep specific params with certain weight initialization stategies for better convergence.
        """
        for child in self.children():
            if isinstance(child, (nn.Linear)):
                nn.init.xavier_normal_(child.weight)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)

            if isinstance(child, (nn.Conv2d)):
                if isinstance(self.conv_activ(), (nn.ReLU, nn.GELU, nn.ELU)):
                    nn.init.kaiming_normal_(child.weight)
                
                elif isinstance(self.conv_activ(), (nn.Sigmoid)):
                    nn.init.xavier_normal_(child.weight)

                if child.bias is not None:
                    nn.init.zeros_(child.bias)


    def forward(self, X:torch.tensor, device=torch.device("cuda")):
        # Input: (b, T, 1, 128, 157)
        #      (batch, seq_time, channels, height[n_mels], width[time])
        batch_size, seq_len, C, H, W = X.shape

        # EMBEDDINGS
        # Conv Embeddings
        x = X.reshape(batch_size*seq_len, C, H, W)
        x = self.conv_1(x)
        x = self.conv_layers(x)

        x = x.reshape(batch_size, seq_len, -1)   #(b, T, features)
        
        # Positional Embeddings
        pos = torch.tensor([i for i in range(seq_len)]).to(device) #ie, torch.arange(seq_len)
        pos = self.pos_embed(pos)
        x += pos
        # Shape: (b, T, features])

        # Transformer
        x = self.blocks(x)

        # Head / Classifier
        x = self.head_ln(x)
        x = self.head_dense0(x)
        logits = self.head_sig(x)

        return logits

    # Manual conv "embeddings" produce same output as the reshape method
    # out = None
    # for i in range(batch_size):
    #     x = X[i, :]                               #out: (seq, C, H, W)
    #     x = self.conv_1(x)
    #     x = self.conv_layers(x)                     #out: (seq, C, W, H)

    #     if out is None:
    #         out = x
    #     else:
    #         out = torch.stack((out, x))

