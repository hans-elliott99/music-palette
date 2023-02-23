import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from src.models.modules import TfmrBlock, ConvBlock, LinearProjection


#TODO: Time embeddings are rudimentary, need to better index which pixels are from which time bin
#TODO: Determine best pooling method/implement others for transformer encoders. https://github.com/lucidrains/vit-pytorch/tree/main/vit_pytorch
#TODO: Implement "patch-out": https://github.com/kkoutini/PaSST
#TODO: Add parameter groups for easier fine-tuning/lr-groups
# --------------------------------------------------------------------------- #
#                      GENERATIVE PATCH TRANSFORMER                           #
# --------------------------------------------------------------------------- #

class GPTPatchTransformer(nn.modules.Module):
    """
    Inputs are shape (n_patches, C, H, W_per_patch).
    C: Image Channels, ie 1 for a Mel-Spectrogram
    H: Image Height / Number of Mel-Features
    W_per_patch: The width of each Mel-Spectrogram patch

    Inspired by ViT: https://arxiv.org/pdf/2010.11929.pdf
    """
    def __init__(self, 
                 X_shape,
                 n_colors=5,
                 n_patches=20,
                 n_freq_bins = None,
                 n_time_bins = None,
                 linear_layers=None,
                 n_heads=4,   #heads per layer
                 n_trfmr_layers=4,
                 embed_dropout=0.0,
                 attn_dropout=0.0,
                 resid_dropout=0.0,
                 mlp_dropout=0.0,
                 config=None
                 ) -> None:
        super().__init__()

        if config is not None:
            self.n_colors = config.n_colors
            self.n_patches = config.n_patches
            self.n_freq_bins = config.n_freq_bins
            self.n_time_bins = config.n_time_bins
            self.linear_layers = config.linear_layers
            self.n_heads = config.n_heads
            self.n_trfmr_layers = config.n_trfmr_layers
            self.embed_dropout = config.embed_dropout
            self.attn_dropout = config.attn_dropout
            self.resid_dropout = config.resid_dropout
            self.mlp_dropout = config.mlp_dropout

        else:
            self.n_colors = n_colors
            self.n_patches = n_patches
            self.n_freq_bins = n_freq_bins
            self.n_time_bins = n_time_bins
            self.linear_layers = linear_layers
            self.n_heads = n_heads
            self.n_trfmr_layers = n_trfmr_layers
            self.embed_dropout = embed_dropout
            self.attn_dropout = attn_dropout
            self.resid_dropout = resid_dropout
            self.mlp_dropout = mlp_dropout

        # Spectrogram input shape = (n_patches, n_features)
        if len(X_shape) == 3: #batch, patches, features
            self.input_shape = X_shape[1:]
        else:
            assert len(X_shape) == 2
            self.input_shape = X_shape
        N_patches, N_features = self.input_shape #spectrogram input shape


        # EMBEDDINGS
        embed_layers = []
        # Patch Embeddings
        # Linear Layers to Project to Constant Latent Vector Size
        if self.linear_layers:
            if not isinstance(self.linear_layers, list):
                self.linear_layers = [self.linear_layers]
            lin_layers = [N_features] + self.linear_layers
            self.linear_proj = nn.ModuleList([
                LinearProjection(in_feats=lin_layers[i],
                                 out_feats=lin_layers[i+1],
                                 dropout=0.,
                                 bias=False) for i in range(len(lin_layers)-1)
            ])
            n_feats = lin_layers[-1]
            embed_layers.append(self.linear_proj)
        else:
            self.linear_proj = None
            n_feats = N_features

        # Frequency Embeddings
        if self.n_freq_bins is not None:
            # freqs normalized to -1, 1.. add small value to avoid boundaries not being included 
            self.freq_bins  = torch.linspace(-1.01, 1.01, self.n_freq_bins)
            self.freq_embed = nn.Embedding(num_embeddings=self.n_freq_bins, embedding_dim=n_feats)
            embed_layers.append(self.freq_embed)
        else:
            self.freq_bins  = None
            self.freq_embed = None
        
        # Time Embeddings
        self.time_idx_tensor = None
        if self.n_time_bins is not None:
            self.time_bins  = torch.arange(0, self.n_time_bins)
            self.time_embed = nn.Embedding(num_embeddings=self.n_time_bins, embedding_dim=n_feats)
            embed_layers.append(self.time_embed)
        else:
            self.time_bins  = None
            self.time_embed = None

        # Standard Positional/Patch Embeddings
        self.ln_0 = nn.LayerNorm(N_features) #applied before projection
        self.pos_embed = nn.Embedding(num_embeddings=N_patches, embedding_dim=n_feats)
        self.embed_drop = nn.Dropout(self.embed_dropout)
        embed_layers += [self.ln_0, self.pos_embed, self.embed_drop]

        # Transformer Blocks
        self.transformer = nn.ModuleList([
            TfmrBlock(block_size=N_patches, 
                      in_feats=n_feats, 
                      n_heads=self.n_heads, 
                      attn_dropout=self.attn_dropout,
                      resid_dropout=self.resid_dropout,
                      mlp_dropout=self.mlp_dropout) for _ in range(self.n_trfmr_layers)
            ])

        # Generative Pretrain Head
        self.head_ln_pt = nn.LayerNorm(n_feats)
        self.head_dense_pt = nn.Linear(in_features=n_feats,
                                     out_features=N_features) #rgb
        
        # Classifier Head
        self.head_ln_clf = nn.LayerNorm(n_feats)
        self.head_dense_clf = nn.Linear(in_features=n_feats,
                                     out_features=self.n_colors*3)

        # Optimization Groups
        self.groups = nn.ModuleDict({
            "embeddings" : nn.ModuleList(embed_layers),
            "transformers" : nn.ModuleList(self.transformer),
            "heads" : nn.ModuleList([self.head_ln_pt, self.head_dense_pt, self.head_ln_clf, self.head_dense_clf])
        })

        # apply weight init
        self.init_weights()
        self.n_params = sum(p.numel() for p in self.parameters())    

    def init_weights(self):
        """
        Prep specific params with certain weight initialization stategies for better convergence.
        """
        for child in self.children():
            if isinstance(child, (nn.Linear)):
                nn.init.normal_(child.weight, mean=0.0, std=0.02)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)

            elif isinstance(child, (nn.Embedding)):
                nn.init.normal_(child.weight, mean=0.0, std=0.02)
            
            elif isinstance(child, (nn.LayerNorm)):
                nn.init.zeros_(child.bias)
                nn.init.ones_(child.weight)

    
    def bin_frequencies(self, x):
        f = torch.bucketize(x, self.freq_bins.to(x.device)) - 1 ##make 0 indexed
        return f.to(x.device)

    def bin_time(self, x):
        _, n_patches, feats = x.shape
        if self.time_idx_tensor is None:
            time_ix = torch.cumsum(
                torch.ones((1, n_patches, feats), dtype=torch.long), 
                dim=2)
            self.time_idx_tensor = torch.bucketize(time_ix, self.time_bins) - 1
        return self.time_idx_tensor.to(x.device)

    def forward(self, X:torch.tensor, pretraining:bool=False):
        assert len(X.shape) == 3, "X must be of shape (batch_size, num_patches, features)"
        # Input: (b, n_patches, n_features)
        n_patches = self.input_shape[0]
        device = X.device

        x = X.clone()
        # Frequency Embeddings
        # Get frequency embeddings, pool per-patch, add to x
        if self.freq_embed:
            freq = self.bin_frequencies(X)    #shape (b, n_patches, feats) -> now an index tensor
            freq = self.freq_embed(freq)      #shape (b, n_patches, feats, embed_feats)
            freq = freq.mean(dim=3)           #shape (b, n_patches, feats)
            x += freq

        # Time Embeddings
        # Get time embeddings, pool per-patch, add to x
        if self.time_embed:
            time = self.bin_time(X)
            time = self.time_embed(time)
            time = time.mean(dim=3)         #shape (b, n_patches, feats)
            x += time

        # Linear Projection
        x = self.ln_0(x)
        if self.linear_proj:
            for layer in self.linear_proj:
                x = layer(x)

        # Positional Embeddings
        pos = torch.arange(0, n_patches, dtype=torch.long, device=device).unsqueeze(0) #shape (1, patches)
        pos = self.pos_embed(pos) #(1, patches, feats)

        x += pos
        x = self.embed_drop(x) # shape (b, patches, feats)

        # Transformer
        for block in self.transformer:
            x = block(x) # shape (b, patches, feats)
        
        # Head / Classifier
        if pretraining:
            x = self.head_ln_pt(x)
            logits = self.head_dense_pt(x)
        else:
            # Pool Hidden States
            x = x.mean(dim=1)  #shape (b, feats)
            x = self.head_ln_clf(x)
            logits = self.head_dense_clf(x) #shape (b, n_colors*3)

        return logits
    
    @staticmethod
    def get_empty_config():
        return dict(
            n_colors = None,
            n_patches = None,
            linear_layers = None,
            n_time_bins = None,
            n_freq_bins = None,
            n_heads = None,
            n_trfmr_layers = None,
            embed_dropout = None,
            attn_dropout = None,
            resid_dropout = None,
        )



# --------------------------------------------------------------------------- #
#                        PATCHED VISION TRANSFORMER                           #
# --------------------------------------------------------------------------- #
class PatchTransformer(nn.modules.Module):
    """
    Inputs are shape (n_patches, C, H, W_per_patch).
    Inspired by ViT: https://arxiv.org/pdf/2010.11929.pdf
    """
    def __init__(self, 
                 X_shape,
                 n_colors=5,
                 n_patches=20,
                 n_freq_bins = None,
                 n_time_bins = None,
                 linear_layers=None,
                 n_heads=4,   #heads per layer
                 n_trfmr_layers=4,
                 embed_dropout=0.0,
                 attn_dropout=0.0,
                 resid_dropout=0.0,
                 mlp_dropout=0.0,
                 sigmoid_logits = False,
                 config=None
                 ) -> None:
        super().__init__()

        if config is not None:
            self.n_colors = config.n_colors
            self.n_patches = config.n_patches
            self.n_freq_bins = config.n_freq_bins
            self.n_time_bins = config.n_time_bins
            self.linear_layers = config.linear_layers
            self.n_heads = config.n_heads
            self.n_trfmr_layers = config.n_trfmr_layers
            self.embed_dropout = config.embed_dropout
            self.attn_dropout = config.attn_dropout
            self.resid_dropout = config.resid_dropout
            self.mlp_dropout = config.mlp_dropout
            self.sigmoid_logits = config.sigmoid_logits #Apply sigmoid to logits to keep in range(0, 1). Or, let the model learn it.

        else:
            self.n_colors = n_colors
            self.n_patches = n_patches
            self.n_freq_bins = n_freq_bins
            self.n_time_bins = n_time_bins
            self.linear_layers = linear_layers
            self.n_heads = n_heads
            self.n_trfmr_layers = n_trfmr_layers
            self.embed_dropout = embed_dropout
            self.attn_dropout = attn_dropout
            self.resid_dropout = resid_dropout
            self.mlp_dropout = mlp_dropout
            self.sigmoid_logits = sigmoid_logits #Apply sigmoid to logits to keep in range(0, 1). Or, let the model learn it.

        # Spectrogram input shape = (n_patches, n_features)
        if len(X_shape) == 3: #batch, patches, features
            self.input_shape = X_shape[1:]
        else:
            assert len(X_shape) == 2
            self.input_shape = X_shape
        N_patches, N_features = self.input_shape #spectrogram input shape

        # LAYERS
        self.ln_0 = nn.LayerNorm([N_features])

        # Patch Embeddings
        # Linear Layers to Project to Constant Latent Vector Size
        if self.linear_layers:
            if not isinstance(self.linear_layers, list):
                self.linear_layers = [self.linear_layers]
            lin_layers = [N_features] + self.linear_layers
            self.linear_proj = nn.ModuleList([
                LinearProjection(in_feats=lin_layers,
                                 out_feats=lin_layers,
                                 dropout=0.,
                                 bias=False) for i in range(len(lin_layers)-1)
            ])
            n_feats = self.linear_layers[-1]
        else:
            self.linear_proj = None
            n_feats = N_features

        # Frequency Embeddings
        if self.n_freq_bins is not None:
            # freqs normalized to -1, 1.. add small value to avoid boundaries not being included 
            self.freq_bins  = torch.linspace(-1.01, 1.01, self.n_freq_bins)
            self.freq_embed = nn.Embedding(num_embeddings=self.n_freq_bins, embedding_dim=n_feats)
        else:
            self.freq_bins  = None
            self.freq_embed = None
        
        # Time Embeddings
        self.time_idx_tensor = None
        if self.n_time_bins is not None:
            self.time_bins  = torch.arange(0, self.n_time_bins)
            self.time_embed = nn.Embedding(num_embeddings=self.n_time_bins, embedding_dim=n_feats)
        else:
            self.time_bins  = None
            self.time_embed = None

        # Standard Positional/Patch Embeddings
        self.pos_embed = nn.Embedding(num_embeddings=N_patches, embedding_dim=n_feats)

        # Embedding dropout
        self.embed_drop = nn.Dropout(self.embed_dropout)

        # Transformer Blocks
        self.transformer = nn.ModuleList([
            TfmrBlock(block_size=N_patches, 
                      in_feats=n_feats, 
                      n_heads=self.n_heads, 
                      attn_dropout=self.attn_dropout,
                      resid_dropout=self.resid_dropout,
                      mlp_dropout=self.mlp_dropout) for _ in range(self.n_trfmr_layers)
            ])

        # Classifier Head
        self.head_ln = nn.LayerNorm(n_feats)
        self.head_dense0 = nn.Linear(in_features=n_feats,
                                   out_features=self.n_colors*3) #rgb
        self.head_sig  = nn.Sigmoid()

        # apply weight init
        self.init_weights()
        self.n_params = sum(p.numel() for p in self.parameters())    

    def init_weights(self):
        """
        Prep specific params with certain weight initialization stategies for better convergence.
        """
        for child in self.children():
            if isinstance(child, (nn.Linear)):
                nn.init.normal_(child.weight, mean=0.0, std=0.02)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)

            elif isinstance(child, (nn.Embedding)):
                nn.init.normal_(child.weight, mean=0.0, std=0.02)
            
            elif isinstance(child, (nn.LayerNorm)):
                nn.init.zeros_(child.bias)
                nn.init.ones_(child.weight)

    
    def bin_frequencies(self, x):
        f = torch.bucketize(x, self.freq_bins.to(x.device)) - 1 ##make 0 indexed
        return f.to(x.device)

    def bin_time(self, x):
        _, n_patches, feats = x.shape
        if self.time_idx_tensor is None:
            time_ix = torch.cumsum(
                torch.ones((1, n_patches, feats), dtype=torch.long), 
                dim=2)
            self.time_idx_tensor = torch.bucketize(time_ix, self.time_bins) - 1
        return self.time_idx_tensor.to(x.device)

    def forward(self, X:torch.tensor):
        assert len(X.shape) == 3, "X must be of shape (batch_size, num_patches, features)"
        # Input: (b, n_patches, n_features)
        n_patches = self.input_shape[0]
        device = X.device

        x = X.clone()
        # Frequency Embeddings
        # Get frequency embeddings, pool per-patch, add to x
        if self.freq_embed:
            freq = self.bin_frequencies(X)    #shape (b, n_patches, feats) -> now an index tensor
            freq = self.freq_embed(freq)      #shape (b, n_patches, feats, embed_feats)
            freq = freq.mean(dim=3)           #shape (b, n_patches, feats)
            x += freq

        # Time Embeddings
        # Get time embeddings, pool per-patch, add to x
        if self.time_embed:
            time = self.bin_time(X)
            time = self.time_embed(time)
            time = time.mean(dim=3)         #shape (b, n_patches, feats)
            x += time

        # Linear Projection
        x = self.ln_0(x)
        if self.linear_proj:
            for layer in self.linear_proj:
                x = layer(x)

        # Positional Embeddings
        pos = torch.arange(0, n_patches, dtype=torch.long, device=device).unsqueeze(0) #shape (1, patches)
        pos = self.pos_embed(pos) #(1, patches, feats)

        x += pos
        x = self.embed_drop(x) # shape (b, patches, feats)

        # Transformer
        for block in self.transformer:
            x = block(x) # shape (b, patches, feats)
        
        # Pool Hidden States
        # TODO: Determine best pooling method/implement others. https://github.com/lucidrains/vit-pytorch/tree/main/vit_pytorch
        # (average pool for now)
        x = x.mean(dim=1)  #shape (b, feats)

        # Head / Classifier
        x = self.head_ln(x)
        logits = self.head_dense0(x) #shape (b, n_colors*3)

        if self.sigmoid_logits:
            # apply sigmoid to keep logits in range [0 1]
            logits = self.head_sig(logits)
        return logits
    
    @staticmethod
    def get_empty_config():
        return dict(
            n_colors = None,
            n_patches = None,
            linear_layers = None,
            n_time_bins = None,
            n_freq_bins = None,
            n_heads = None,
            n_trfmr_layers = None,
            embed_dropout = None,
            attn_dropout = None,
            resid_dropout = None,
            sigmoid_logits = None
        )


# --------------------------------------------------------------------------- #
#                           CONVOLUTION->TRANSFORMER                          #
# --------------------------------------------------------------------------- #

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
                 attn_dropout=0.0,
                 resid_dropout=0.0,
                 mlp_dropout=0.0,
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
            self.max_seq_len = config.max_seq_len
            self.n_colors = config.n_colors
            self.n_heads = config.n_heads
            self.n_trfmr_layers = config.n_trfmr_layers
            self.embed_dropout = config.embed_dropout
            self.attn_dropout = config.attn_dropout
            self.resid_dropout = config.resid_dropout
            self.mlp_dropout = config.mlp_dropout
            self.conv_dropout = config.conv_dropout
            self.conv_filters = config.conv_filters
            self.kernel_size = config.kernel_size
            self.pool_sizes = config.pool_sizes
            self.conv_activation = config.conv_activation
            self.sigmoid_logits = config.sigmoid_logits #Apply sigmoid to logits to keep in range(0, 1). Or, let the model learn it.

        else:
            self.max_seq_len = max_seq_len
            self.n_colors = n_colors
            self.n_heads = n_heads
            self.n_trfmr_layers = n_trfmr_layers
            self.embed_dropout = embed_dropout
            self.attn_dropout = attn_dropout
            self.resid_dropout = resid_dropout
            self.mlp_dropout = mlp_dropout
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
                      attn_dropout=self.attn_dropout,
                      resid_dropout=self.resid_dropout,
                      mlp_dropout=self.mlp_dropout,
                      ) for _ in range(self.n_trfmr_layers)
            ])

        # Classifier / Head
        self.head_ln = nn.LayerNorm(self.num_encoded_feats)
        self.head_dense0 = nn.Linear(in_features=self.num_encoded_feats,
                                   out_features=self.n_colors*3) #rgb
        self.head_sig  = nn.Sigmoid()

        # --apply special weight initializations--
        self._init_weights()

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

    def _init_weights(self):
        """
        Prep specific params with certain weight initialization stategies for better convergence.
        """
        for child in self.children():
            if isinstance(child, nn.Linear):
                nn.init.normal_(child.weight, mean=0., std=0.02)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
            
            elif isinstance(child, nn.Embedding):
                nn.init.normal_(child.weight, mean=0., std=0.02)
            
            elif isinstance(child, nn.LayerNorm):
                nn.init.zeros_(child.bias)
                nn.init.ones_(child.weight)
            
            elif isinstance(child, nn.Conv2d):
                nn.init.kaiming_normal_(child.weight)                
                if child.bias is not None:
                    nn.init.zeros_(child.bias)


    def forward(self, X:torch.tensor):
        # Input: (b, T, 1, 128, 157)
        #      (batch, seq_length, img_channels[1], img_height[n_mels], img_width[time])
        device = X.device
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

    @staticmethod
    def get_empty_config():
        return dict(
            max_seq_len = None,
            n_colors = None,
            n_heads = None,
            n_trfmr_layers = None,
            embed_dropout = None,
            attn_dropout = None,
            resid_dropout = None,
            mlp_dropout = None,
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




