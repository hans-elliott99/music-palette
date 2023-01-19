import torch
import torch.nn as nn
import numpy as np
import math
from collections import deque



class CNNLayerNorm(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_feats)
    def forward(self, x):
        #x shape: (b, Channels, H[features], W[time])
        x = x.transpose(dim0=2, dim1=3) 
        x = self.layer_norm(x) #normalize features
        return x.transpose(2,3)


class SeqRNNConv(nn.modules.Module):
    """
    Each spectogram is like a token, together they spell a sentence (song).
    Process a batch of N length (for ex, 10) spectrograms, which are each (1, 128, 157) in shape...
    thus input is (b, seq_length, 1[channels], 128[features/mels], 157[time-steps]).
    Then, loop through each token in seq_length, 
        extracting and passing into the model arrays of shape (b, 1, 128, 157):
    Encoder:
        - could reshape to (b, 128, 157) and pass into a (batch-first) GRU to encode the
          time-factor in the spectograms (157 cols ~ time-component, so maybe transpose to (b,157,128)?)
        - apply convolutions to extract features, leaving shape (b, 128, 1, 2)
        - reshape to (b, 256).
        - create array H to store hidden states, size (b, seq_length, hidden_size)
        - pass into GRUCell (along with hidden state of shape hidden_size),
          producing hidden state shape: (b, hidden_size) -> add to hidden states array
          and continue loop
    Apply attention to hidden state array H to get attention scores. 
        (Idea, also create a seq-position embedding which is indexed into and concatted with the hidden
        states before the are added to H? So model can learn to update attention based on where in the sequence the
        clip is coming from)
    Take output of GRU (b, hidden_size) and concat with attention scores.
    
    Pass output through some head layers.
    """
    def __init__(self, 
                 X_shape,
                 n_colors=5,
                 gru_hidden_size=32,
                 dropout=0.0
                 ) -> None:
        super().__init__()
        self.n_colors = n_colors

        n_layers = 4    #n of conv layers
        conv_filters = [64, 128, 128, 128]  # filter sizes
        kernel_size = (3, 3)  # convolution kernel size
        pool_size = [(2, 2), (4, 2), (4, 2), (4, 8)]  # size of pooling area
        self.conv_activ = nn.GELU  # activation function to use after each layer
        self.gru_hidden_size  = gru_hidden_size
        
        # Image input shape = (1, 128, 157)
        if len(X_shape) == 4: #seq_length, C, H, W
            self.input_shape = X_shape[1:] #1, 128, 157
        else:
            assert len(X_shape) == 3
            self.input_shape = X_shape
        chanl_ax = 0
        freq_ax  = 1
        time_ax  = 2

        # Initialize layers
        # Normalize along frequency axis
        self.ln_1 = CNNLayerNorm(self.input_shape[freq_ax])

        # First GRU layer
        self.gru_layer0 = nn.GRU(input_size=self.input_shape[freq_ax],
                                 hidden_size=self.gru_hidden_size,
                                 bidirectional=True,
                                 num_layers=1, 
                                 batch_first=True, 
                                 dropout=0) #if num_layers > 1, set to dropout

        # First conv layer
        self.conv_1 = nn.Conv2d(in_channels=self.input_shape[chanl_ax], #(1 channel)
                                out_channels=conv_filters[0],
                                kernel_size=kernel_size,
                                padding="same")
        self.act_1 = self.conv_activ()
        self.bn_1 = nn.BatchNorm2d(num_features=conv_filters[0])
        self.mp_1 = nn.MaxPool2d(kernel_size=pool_size[0],
                                 stride=pool_size[0])
        self.drop_1 = nn.Dropout(dropout)

        # More conv layers
        self.hiddenLayers = nn.ModuleList()
        for i in range(n_layers-1):
            self.hiddenLayers.append(
                        nn.Conv2d(in_channels=conv_filters[i], 
                                out_channels=conv_filters[i+1],
                                kernel_size=kernel_size,
                                padding="same")
            )
            self.hiddenLayers.append( self.conv_activ() )
            self.hiddenLayers.append( nn.BatchNorm2d(conv_filters[i+1]) )
            self.hiddenLayers.append( 
                        nn.MaxPool2d(kernel_size=pool_size[i+1],
                                    stride=pool_size[i+1])
            )
            self.hiddenLayers.append( nn.Dropout(dropout) )
        

        # get conv output shape
        self.encoder_out_shape = self.calc_encoder_output_shape()

        # Recurrent layers
        gru_input_size = self.encoder_out_shape[1]
        self.gru_cell  = nn.GRUCell(input_size=gru_input_size,
                                    hidden_size=self.gru_hidden_size,
                                    bias=True)


        self.head_dense0 = nn.Linear(in_features=self.gru_hidden_size,
                                   out_features=n_colors*3)

        self.head_sig  = nn.Sigmoid()

        # apply weight initializations
        self._special_init_weights()

        # gen model stats
        self.n_params = sum([np.prod(p.size()) for p in self.parameters()])
        self.n_trainable_params = sum([np.prod(p.size()) for p in self.parameters() if p.requires_grad])
        


    @torch.no_grad()
    def calc_encoder_output_shape(self):
        """For ex, if we added/removed hiddenLayers, 
        we'd need to re-determine the gru's input shape, so let's do it automatically
        """
        X = torch.zeros(1, 1, 128, 157) #b, C, H[mel], W[time]

        x = self.ln_1( X )                          #out: (b, 1,  128, 157)
        x = x.squeeze(dim=1)                        #out: (b, 128, 157)
        x = x.transpose(dim0=1, dim1=2)             #out: (b, 157, 128) ie {batch, time, mels}

        h_inner = self._init_grulayer_hidden(x.shape[0],
                                            n_layers=1, bidirectional=True) #(D[2]*layers[1], b, hidden[32])     
        x, h_inner = self.gru_layer0(x, h_inner)
        x = x.unsqueeze(1)                          #out: (b, 1, 157, 64)

        x = self.act_1( self.conv_1(x) )            #out: (b, 64, 157, 64)

        x = self.bn_1(x)                            #out: (b, 64, 157, 64)
        x = self.mp_1(x)                            #out: (b, 64, 78, 32)
        x = self.drop_1(x)                          #out: (b, 64, 78, 32)

        for layer in self.hiddenLayers:
            x = layer(x)                            #out: (b, 128, 1, 1)

        # reshape x for GRUCell input
        x = x.reshape(( x.shape[0], math.prod(x.shape[1:]) )) #out: (b, 128) [128*1*1]
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

    def get_silence_array(self, batch_size):
        return torch.zeros(batch_size, 1, self.input_shape[-2], self.input_shape[-1])

    def _init_grulayer_hidden(self,
                          batch_size:int,
                          n_layers=1,
                          bidirectional=False,
                          hidden_size=None):
        if not hidden_size:
            hidden_size=self.gru_hidden_size
        D = 1
        if bidirectional:
            D = 2
        #shape: (D*n_layers, batch_size, hidden_size)
        # D=1 (or 2 if bidirectional)
        return torch.zeros(D*n_layers, batch_size, hidden_size)


    def init_hidden(self, batch_size:int):
        return torch.zeros(batch_size, self.gru_hidden_size)
        

    def forward(self, X:torch.tensor, h_0:torch.tensor):
        # Input: (b, 1, 128, 157)
        #      (batch, channels, height[n_mels], width[time])
        batch_size = X.shape[0]
        seq_length = X.shape[1]

        # ENCODER

        # # To store gru hidden states, include in outer loop:
        # gru_seq_h = torch.zeros( 
        #         (batch_size, seq_length, self.gru_hidden_size ) 
        #     ) #shape: (b, seq_length, hidden_size[32] )
        # gru_seq_h[:, i, :] = h_0      ##insert the hidden state for the current seq-step into array

        # normalize
        x = self.ln_1( X )                          #out: (b, 1,  128, 157)
        # gru layer to encode the time dimension (could use a transformer here!)
        x = x.squeeze(dim=1)                        #out: (b, 128, 157)
        x = x.transpose(dim0=1, dim1=2)             #out: (b, 157, 128) ie {batch, time, mels}

        h_inner = self._init_grulayer_hidden(x.shape[0], #(D[2]*layers[1], b, hidden[32])  
                                            n_layers=1, 
                                            bidirectional=True
                                            ).to(X.device)    
        x, h_inner = self.gru_layer0(x, h_inner)
        # x:       (b, time_steps[157], D*hidden[32]) => (b, 157, 64)
        # h_inner: (D*layers[1], b, hidden[32])       => (2, b,   32)
        x = x.unsqueeze(1)                          #out: (b, 1, 157, 64)

        x = self.act_1( self.conv_1(x) )            #out: (b, 64, 157, 64)

        x = self.bn_1(x)                            #out: (b, 64, 157, 64)
        x = self.mp_1(x)                            #out: (b, 64, 78, 32)
        x = self.drop_1(x)                          #out: (b, 64, 78, 32)

        for layer in self.hiddenLayers:
            x = layer(x)                            #out: (b, 128, 1, 1)

        # reshape x for GRUCell input
        x = x.reshape(( x.shape[0], math.prod(x.shape[1:]) )) #out: (b, 128) [128*1*1]

        h_0 = self.gru_cell(x, h_0)                           #out: (batch_size, hidden_size[32])
        
        # DECODER

        # Want to add Attention here...
        # Head
        x = self.head_dense0(h_0)               #in: (b, 32) out: (b, n_colors*3)
        logits = self.head_sig(x)

        return logits, h_0







# ----------------------------------------------------
class ConvRNNSequential(nn.Module):
    """
    Now, the conv layers encode the features prior to the RNN processing the time sequences.
    """
    def __init__(self, X_shape, n_colors=5) -> None:
        super().__init__()
        self.n_colors = n_colors

        n_layers = 4    #n of conv layers
        conv_filters = [64, 128, 128, 128]  # filter sizes
        kernel_size = 3  # convolution kernel size
        pool_size = [(2, 1), (2, 1), (2, 1), (4, 1)]  # size of pooling area
        dropout = 0.0
        self.conv_activ = nn.GELU  # activation function to use after each layer
        self.gru_hidden_size  = 32
        
        # Image input shape = (b, 1, 128, 157)
        if len(X_shape) == 4:
            self.input_shape = X_shape[1:] #1, 128, 157
        else:
            assert len(X_shape) == 3
            self.input_shape = X_shape
        chanl_ax = 0
        freq_ax  = 1
        time_ax  = 2

        # Initialize layers
        # Normalize along frequency axis
        self.ln_1 = CNNLayerNorm(self.input_shape[freq_ax])


        # First conv layer
        self.conv_1 = nn.Conv2d(in_channels=self.input_shape[chanl_ax], #(1 channel)
                                out_channels=conv_filters[0],
                                kernel_size=kernel_size,
                                padding=kernel_size//2)        
        self.act_1 = self.conv_activ()
        self.mp_1 = nn.MaxPool2d(kernel_size=pool_size[0],
                                 stride=pool_size[0])
        self.drop_1 = nn.Dropout(dropout)


        # More conv layers
        self.hiddenLayers = nn.ModuleList()
        n_feats = self.input_shape[freq_ax]
        for i in range(n_layers-1):
            n_feats = int(n_feats/2)
            self.hiddenLayers.append( CNNLayerNorm( n_feats=n_feats) ) #nn.BatchNorm2d(conv_filters[i+1])
            self.hiddenLayers.append(
                        nn.Conv2d(in_channels=conv_filters[i], 
                                  out_channels=conv_filters[i+1],
                                  kernel_size=kernel_size,
                                  padding="same")
            )
            self.hiddenLayers.append( self.conv_activ() )
            self.hiddenLayers.append( 
                        nn.MaxPool2d(kernel_size=pool_size[i+1],
                                     stride=pool_size[i+1])
            )
            self.hiddenLayers.append( nn.Dropout(dropout) )
        
        # reshape conv outputs and tack on prev-hidden state (which essentially == hidden since GRU)....
        self.encoder_out_shape = self.calc_encoder_output_shape()
        gru_input_size = self.encoder_out_shape[-1]

        # Recurrent Layer
        self.gru_layer0 = nn.GRU(input_size=gru_input_size,
                                 hidden_size=self.gru_hidden_size,
                                 bidirectional=True,
                                 num_layers=1, 
                                 batch_first=True, 
                                 dropout=dropout)
        # Recurrent to Gru-Cell Dense
        self.dense_g2c = nn.Linear(in_features=self.gru_hidden_size*2, #bidirectional
                                   out_features=self.gru_hidden_size//4)

        self.gru_cell  = nn.GRUCell(input_size= (self.gru_hidden_size//4)*self.input_shape[time_ax],
                                    hidden_size=self.gru_hidden_size,
                                    bias=True)


        self.head_dense0 = nn.Linear(in_features=self.gru_hidden_size,
                                   out_features=n_colors*3)

        self.head_sig  = nn.Sigmoid()

        # init inner GRU layer
        self.initgru_dense = nn.Linear(in_features=self.input_shape[time_ax],
                                       out_features=self.gru_hidden_size)
        # apply weight initializations
        self._special_init_weights()

        # gen model stats
        self.n_params = sum([np.prod(p.size()) for p in self.parameters()])
        self.n_trainable_params = sum([np.prod(p.size()) for p in self.parameters() if p.requires_grad])

    @torch.no_grad()
    def calc_encoder_output_shape(self):
        X = torch.zeros(1, 1, 128, 157) #b, C, H[mels], W[time]

        x = self.ln_1( X )                          #out: (b, 1, 128, 157)

        # encode features using Conv layers
        x = self.act_1(x)            
        x = self.conv_1(x)                          #out: (b, 64, 128, 157)
        x = self.mp_1(x)                            #out: (b, 64, 64, 157)
        x = self.drop_1(x)                          #out: (b, 64, 64, 157)

        for layer in self.hiddenLayers:
            x = layer(x)                            #out: (b, 128, ...[4], 157)

        # reshape for gru
        shape = x.shape
        x = x.view(shape[0], shape[1]*shape[2], shape[3]) #out: (b, n_feature_maps*feats, time[157])
        x = x.transpose(dim0=1,dim1=2)                    #out: (b, time, feats)

        return x.shape


    def _special_init_weights(self):
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
    

    def get_silence_array(self, batch_size):
        return torch.zeros(batch_size, 1, self.input_shape[-2], self.input_shape[-1])

    def _init_grulayer_hidden(self,
                              X,
                              n_layers=1,
                              bidirectional=False,
                              hidden_size=None):
        if not hidden_size:
            hidden_size=self.gru_hidden_size
        D = 1
        if bidirectional:
            D = 2
        
        x = X.mean(dim=2)         #out: (b, C, time) (b, 1, 157)
        x = self.initgru_dense(x) #out: (b, C, hidden_size[32])
        
        shape = x.shape
        x = x.reshape(shape[1], shape[0], shape[2])
        #shape: (D*n_layers, batch_size, hidden_size)
        # D=1 (or 2 if bidirectional)

        D = 2 if bidirectional else 1
        x = torch.vstack([x for _ in range(D*n_layers)])
        return x


    def init_hidden(self, batch_size:int, hidden_size=None):
        if not hidden_size:
            hidden_size=self.gru_hidden_size
        return torch.zeros(batch_size, hidden_size)


    def forward(self):
        # May want to try to implement the gru-cell loop as a general 'forward' case
        pass


    def step(self, X:torch.tensor, h_0:torch.tensor):
        # Input: (b, 1, 128, 157)  ~ (batch, channels, height[n_mels], width[time])
        # batch_size = X.shape[0]
        # seq_length = X.shape[1]

        # ENCODER
        # layer norm
        x = self.ln_1( X )                          #out: (b, 1, 128, 157)

        # encode features using Conv layers
        x = self.act_1(x)            
        x = self.conv_1(x)                          #out: (b, 64, 128, 157)
        x = self.mp_1(x)                            #out: (b, 64, 64, 157)
        x = self.drop_1(x)                          #out: (b, 64, 64, 157)

        for layer in self.hiddenLayers:
            x = layer(x)                            #out: (b, 128, ...[4], 157)

        # reshape for gru
        shape = x.shape
        x = x.view(shape[0], shape[1]*shape[2], shape[3]) #out: (b, n_feature_maps*feats, time[157])
        x = x.transpose(dim0=1,dim1=2)                    #out: (b, time, feats)

        # gru layer to encode time dimension
        h_inner = self._init_grulayer_hidden(X,
                                             n_layers=1,
                                             bidirectional=True) #(D[2]*layers[1], b, hidden[32])
        
        x, h_inner = self.gru_layer0(x, h_inner) 
        # x:       (b, time_steps[157], D*hidden[32]) => (b, 157, 64)
        # h_inner: (D*layers[1], b, hidden[32])       => (2, b,   32)

        # pass through gru cell to update hidden state
        x = self.dense_g2c(x)                           #out: (b, 157, 8)

        # reshape x for GRUCell input
        x = x.reshape(( x.shape[0], math.prod(x.shape[1:]) )) #out: (b, 157*...)

        # update hidden state
        h_0 = self.gru_cell(x, h_0)                           #out: (batch_size, hidden_size[32])
        

        # DECODER
        # Head
        # x = self.head_conv( x.unsqueeze(-1) ) #in: (b, seq_length, hidden[32], 1) #out: (b, n_colors, gru_hidden_size[32], 1)
        x = self.head_dense0(h_0)               #in: (b, 32) out: (b, n_colors*3)
        # x = self.head_dense1(x)                 #out: (b, 5, 3)
        logits = self.head_sig(x)

        return logits, h_0


# https://arxiv.org/pdf/1901.04555.pdf
# https://github.com/ZainNasrullah/music-artist-classification-crnn/blob/master/src/models.py
# https://yerevann.github.io/2016/06/26/combining-cnn-and-rnn-for-spoken-language-identification/

# an alternative:
# paper: https://paperswithcode.com/paper/efficient-large-scale-audio-tagging-via
# https://github.com/fschmid56/EfficientAT

# speech-recognition example:
# https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/


# ------ old 
class ConvRNN(nn.Module):
    def __init__(self, X_shape, n_colors=5) -> None:
        super().__init__()
        self.n_colors = n_colors

        n_layers = 4    #n of conv layers
        conv_filters = [64, 128, 128, 128]  # filter sizes
        kernel_size = (3, 3)  # convolution kernel size
        pool_size = [(2, 2), (4, 2), (4, 2), (4, 2), (4, 2)]  # size of pooling area
        self.conv_activ = nn.GELU  # activation function to use after each layer
        self.gru_hidden_size  = 32
        
        # Image input shape = (?, 1, 128, 157)
        if len(X_shape) == 4:
            self.input_shape = X_shape[1:] #1, 128, 157
        else:
            assert X_shape == 3
            self.input_shape = X_shape
        chanl_ax = 0
        freq_ax  = 1
        time_ax  = 2

        # Initialize layers
        # Normalize along frequency axis
        self.ln_1 = nn.LayerNorm(self.input_shape)

        # First conv layer
        self.conv_1 = nn.Conv2d(in_channels=self.input_shape[chanl_ax],
                                out_channels=conv_filters[0],
                                kernel_size=kernel_size,
                                padding="same")
        self.act_1 = self.conv_activ()
        self.bn_1 = nn.BatchNorm2d(num_features=conv_filters[0])
        self.mp_1 = nn.MaxPool2d(kernel_size=pool_size[0],
                                 stride=pool_size[0])
        self.drop_1 = nn.Dropout(0.0)

        # More conv layers
        self.hiddenLayers = nn.ModuleList()
        for i in range(n_layers-1):
            self.hiddenLayers.append(
                        nn.Conv2d(in_channels=conv_filters[i], 
                                out_channels=conv_filters[i+1],
                                kernel_size=kernel_size,
                                padding="same")
            )
            self.hiddenLayers.append( self.conv_activ() )
            self.hiddenLayers.append( nn.BatchNorm2d(conv_filters[i+1]) )
            self.hiddenLayers.append( 
                        nn.MaxPool2d(kernel_size=pool_size[i+1],
                                    stride=pool_size[i+1])
            )
            self.hiddenLayers.append( nn.Dropout(0.0) )
        
        # reshape conv outputs and tack on prev-hidden state (which essentially == hidden since GRU)....
        encoder_out_shape = self.calc_encoder_output_shape()

        # Recurrent layers
        gru_input_size = encoder_out_shape[-2]*encoder_out_shape[-1]
        self.gru_1 = nn.GRU(input_size=gru_input_size,
                            hidden_size=self.gru_hidden_size,
                            num_layers=1, 
                            batch_first=True,
                            dropout=0.0)
        self.gru_2 = nn.GRU(input_size=self.gru_hidden_size,
                            hidden_size=self.gru_hidden_size,
                            num_layers=1, 
                            batch_first=True, 
                            dropout=0.0)

        # Output layer
        self.head_conv = nn.Conv2d(in_channels=128, out_channels=n_colors, kernel_size=kernel_size, padding="same")
        self.head      = nn.Linear(in_features=self.gru_hidden_size, out_features=3)
        self.head_sig  = nn.Sigmoid()

        # apply weight initializations
        self._special_init_weights()

        # stats
        self.n_params = sum([np.prod(p.size()) for p in self.parameters()])
        self.n_trainable_params = sum([np.prod(p.size()) for p in self.parameters() if p.requires_grad])
        

    @torch.no_grad()
    def calc_encoder_output_shape(self):
        """For ex, if we added/removed hiddenLayers
        """
        X = torch.zeros(20, 1, 128, 157)
        x = self.ln_1(X)                 #in:  (b, 1, 128, 157) out:(b, 1, 128, 157)
        x = self.act_1( self.conv_1(X) ) #out: (b, 64, 128, 157)
        x = self.bn_1(x)                 #out: (b, 64, 128, 157)
        x = self.mp_1(x)                 #out: (b, 64, 64, 78)
        x = self.drop_1(x)               #out: (b, 64, 64, 78)

        for layer in self.hiddenLayers:
            x = layer(x)                 #out: (b, 128, 1, 9)

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
                if isinstance(self.conv_activ(), (nn.ReLU)):
                    nn.init.kaiming_normal_(child.weight)
                
                elif isinstance(self.conv_activ(), (nn.Sigmoid)):
                    nn.init.xavier_normal_(child.weight)

                if child.bias is not None:
                    nn.init.zeros_(child.bias)

    def init_hidden(self, batch_size, hidden_size=None):
        if not hidden_size:
            hidden_size=self.gru_hidden_size
        
        #shape: (D*n_layers, batch_size, hidden_size)
        # D=1 (or 2 if bidirectional)
        return torch.zeros(1, batch_size, hidden_size)
    

    def forward(self, X:torch.tensor):
        seq_length = X.shape[2]

        x = self.ln_1(X)                 #in:  (b, 1, 128, 157) out:(20, 1, 128, 157)
        x = self.act_1( self.conv_1(x) ) #out: (b, 64, 128, 157)
        x = self.bn_1(x)                 #out: (b, 64, 128, 157)
        x = self.mp_1(x)                 #out: (b, 64, 64, 78)
        x = self.drop_1(x)               #out: (b, 64, 64, 78)

        for layer in self.hiddenLayers:
            x = layer(x)                 #out: (b, 128, 1, 9)

        # Reshape to (b, 128, 9) for (batch-first) GRU
        # 128 = sequence length
        # 9   = size of input
        x = x.reshape(*x.shape[:2], -1)
        
        # First GRU
        h_in = self.init_hidden(batch_size=x.shape[0])  #shape: (1, b, 32)
        x, h_out = self.gru_1(x, h_in)
        #x     shape: (b, 128, gru_hidden_size[32]) 
        #h_out shape: (num_layers, b, gru_hidden_size[32])

        # Second GRU
        x, h_out = self.gru_2(x, h_out)
        #h_out shape: (num_layers, b, gru_hidden_size[32])
        #x     shape: (b, 128, gru_hidden_size[32]) 
        
        # Head
        x = self.head_conv(x.unsqueeze(-1)) #out: (b, n_colors, gru_hidden_size[32], 1)
        x = self.head(x.squeeze(-1))   #out: (b, n_colors, 3)
        logits = self.head_sig(x)

        return logits
