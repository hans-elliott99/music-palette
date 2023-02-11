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
        wei = F.softmax(wei, dim=-1) #(B, T)
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
            nn.GELU(),
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
        self.attn = MultiHeadAttention(block_size, n_heads, head_size, in_feats, dropout)
        self.ln2 = nn.LayerNorm(in_feats)
        self.mlp = FeedForward(in_feats, dropout)
    
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


class ConvTransformer(nn.modules.Module):
    """
    ConvTransformer:
        - Convolution layers extract spectrogram features where Embedding tables
          are typically used (for example, in NLP).
        - Transformer blocks, consisting of MultiHead Attention and Feed-Forward
          layers, are used to "decode" the embeddings into color palettes.
        - A single spectrogram can be thought of as a "token" in a character-level
          language model. Together with a string of spectrograms it "spells" a
          clip from a song.
    """
    def __init__(self, 
                 X_shape,
                 max_seq_len=None, #max sequence length, aka, block_size
                 n_colors=5,
                 n_heads=4,   #heads per layer
                 n_trfmr_layers=4,
                 embed_dropout=0.0,
                 trfmr_dropout=0.0,
                 conv_dropout=0.0,
                 conv_filters=[64, 256, 128],
                 kernel_size=3,
                 pool_sizes=[(4, 4), (8, 8), (4, 4)],
                 conv_activation = "gelu",
                 sigmoid_logits = False,
                 config=None
                 ) -> None:
        super().__init__()

        if config is not None:
            self.max_seq_len = config["max_seq_len"]
            self.n_colors = config["n_colors"]
            self.n_heads = config["n_heads"]
            self.n_trfmr_layers = config["n_trfmr_layers"]
            self.embed_dropout = config["embed_dropout"]
            self.trfmr_dropout = config["trfmr_dropout"]
            self.conv_dropout = config["conv_dropout"]
            self.conv_filters = config["conv_filters"]
            self.kernel_size = config["kernel_size"]
            self.pool_sizes = config["pool_sizes"]
            self.conv_activation = config["conv_activation"]
            self.sigmoid_logits = config["sigmoid_logits"] #Apply sigmoid to logits to keep in range(0, 1). Or, let the model learn it.

        else:
            self.max_seq_len = max_seq_len
            self.n_colors = n_colors
            self.n_heads = n_heads
            self.n_trfmr_layers = n_trfmr_layers
            self.embed_dropout = embed_dropout
            self.trfmr_dropout = trfmr_dropout
            self.conv_dropout = conv_dropout
            self.conv_filters = conv_filters
            self.kernel_size = kernel_size
            self.pool_sizes = pool_sizes
            self.conv_activation = conv_activation
            self.sigmoid_logits = sigmoid_logits #Apply sigmoid to logits to keep in range(0, 1). Or, let the model learn it.
        
        assert len(self.conv_filters) == len(self.pool_sizes)
        self.n_conv_layers = len(self.conv_filters)

        if self.conv_activation.startswith("g"):
            conv_activ = nn.GELU
        elif self.conv_activation.startswith("r"):
            conv_activ = nn.ReLU
        else:
            raise NotImplementedError

        # Conv specifications
        # conv_filters = [64, 256, 128]  # filter sizes, last one will == embed_dim if pools are approriately sized
        # pool_size = [(4, 4), (8, 8), (4, 4)]  # size of pooling area (4*4*8 = 128, which means C,H,W beomes 128,1,1)
        
        # Image input shape = (1, 128, 157)
        if len(X_shape) == 5: #batch, seq_length, C, H, W
            self.input_shape = X_shape[1:] #seq, C, H, W
        else:
            assert len(X_shape) == 4
            self.input_shape = X_shape

        seq_len, Channels, H_freq, W_time = self.input_shape #spectrogram input shape

        # LAYERS
        self.ln_0 = nn.LayerNorm([Channels, H_freq, W_time])

        # Conv Layers for Feature "Embeddings"
        self.conv_1 = ConvBlock(in_channels=Channels,
                                conv_filters=self.conv_filters[0],
                                kernel_size=self.kernel_size,
                                pool_size=self.pool_sizes[0],
                                conv_activ=conv_activ,
                                dropout=conv_dropout
                                )

        self.conv_layers = nn.ModuleList([
            ConvBlock(in_channels=self.conv_filters[i], 
                      conv_filters=self.conv_filters[i+1],
                      kernel_size=self.kernel_size,
                      pool_size=self.pool_sizes[i],
                      conv_activ=conv_activ,
                      dropout=self.conv_dropout
                    ) for i in range(self.n_conv_layers -1)     
        ])
        
        # get conv output shape
        _, _c, _h, _w = self.get_conv_output_shape()
        self.num_encoded_feats = _c*_h*_w
        # Standard Positional Embeddings
        self.pos_embed = nn.Embedding(num_embeddings=self.max_seq_len, embedding_dim=self.num_encoded_feats)
        self.embed_drop = nn.Dropout(self.embed_dropout)

        # Transformer Blocks
        self.transformer = nn.ModuleList([
            TfmrBlock(block_size=self.max_seq_len, 
                      in_feats=self.num_encoded_feats, 
                      n_heads=self.n_heads, 
                      dropout=self.trfmr_dropout
                      ) for _ in range(self.n_trfmr_layers)
            ])

        # Classifier / Head
        self.head_ln = nn.LayerNorm(self.num_encoded_feats)
        self.head_dense0 = nn.Linear(in_features=self.num_encoded_feats,
                                   out_features=self.n_colors*3) #rgb
        self.head_sig  = nn.Sigmoid()

        # --apply special weight initializations--
        self._special_init_weights()

        # --gen model stats--
        self.n_params = sum(p.numel() for p in self.parameters())
        self.n_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    

    @torch.no_grad()
    def get_conv_output_shape(self):
        """So we can add/remove/modify conv layers when specifying the model.
        """
        X = torch.zeros(1, 1, 1, 128, 157) #b, T, C, H[mel], W[time]
        batch_size, seq_len, C, H, W = X.shape

        # EMBEDDINGS
        # Conv Embeddings
        x = X.reshape(batch_size*seq_len, C, H, W)
        x = self.conv_1(x)
        for layer in self.conv_layers:
            x = layer(x)
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
        #      (batch, seq_length/time, img_channels[1], img_height[n_mels], img_width[time])
        if len(X.shape) == 5:
            batch_size, seq_len, C, H, W = X.shape
        elif len(X.shape) == 4:
            batch_size = 1
            seq_len, C, H, W = X.shape
        else:
            raise RuntimeError("X.shape is incorrect")

        # EMBEDDINGS
        # Conv Layers for Feature Embeddings/Extraction
        x = self.ln_0(X)
        x = x.reshape(batch_size*seq_len, C, H, W) #treat batch & time as the batch-dim
        x = self.conv_1(x)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.reshape(batch_size, seq_len, -1)  #(b, T, extracted_features)
        
        # Positional Embeddings
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0) #shape (1, seq_len)
        pos = self.pos_embed(pos) #(1, seq_len, embed_features)
        x += pos
        x = self.embed_drop(x)
        # Shape: (b, T, features)

        # Transformer
        for block in self.transformer:
            x = block(x)

        # Head / Classifier
        x = self.head_ln(x) #final layer norm
        logits = self.head_dense0(x)

        if self.sigmoid_logits:
            # apply sigmoid to keep logits in range 0 1
            logits = self.head_sig(logits)

        return logits

    def get_config(self):
        return dict(
            max_seq_len = self.max_seq_len,
            n_colors = self.n_colors,
            n_heads = self.n_heads,
            n_trfmr_layers = self.n_trfmr_layers,
            embed_dropout = self.embed_dropout,
            trfmr_dropout = self.trfmr_dropout,
            conv_dropout = self.conv_dropout,
            conv_filters = self.conv_filters,
            kernel_size = self.kernel_size,
            pool_sizes = self.pool_sizes,
            conv_activation = self.conv_activation,
            sigmoid_logits = self.sigmoid_logits,
            n_conv_layers = self.n_conv_layers
        )
    
    @staticmethod
    def get_empty_config():
        return dict(
            max_seq_len = None,
            n_colors = None,
            n_heads = None,
            n_trfmr_layers = None,
            embed_dropout = None,
            trfmr_dropout = None,
            conv_dropout = None,
            conv_filters = [None],
            kernel_size = None,
            pool_sizes = [None],
            conv_activation = None,
            sigmoid_logits = None
        )


# Manual conv "embeddings" applied to each time-step produce same output as the reshape method
# out = None
# for i in range(batch_size):
#     x = X[i, :]                               #out: (seq, C, H, W)
#     x = self.conv_1(x)
#     x = self.conv_layers(x)                     #out: (seq, C, W, H)

#     if out is None:
#         out = x
#     else:
#         out = torch.stack((out, x))




class CausalSelfAttention(nn.Module):
    """
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
        self.n_embd = n_embed

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
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
