import torch
import torch.nn as nn



class ConvRNN(nn.modules.Module):
    def __init__(self, X_shape, n_colors=5) -> None:
        super().__init__()

        n_layers = 4 #n of conv layers
        conv_filters = [64, 128, 128, 128]  # filter sizes
        kernel_size = (3, 3)  # convolution kernel size
        activation = 'relu'  # activation function to use after each layer
        pool_size = [(2, 2), (4, 2), (4, 2), (4, 2), (4, 2)]  # size of pooling area
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
        self.act_1 = nn.ReLU()
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
            self.hiddenLayers.append( nn.ReLU() )
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
        self.head_conv = nn.Conv2d(in_channels=128, out_channels=5, kernel_size=kernel_size, padding="same")
        self.head   = nn.Linear(in_features=self.gru_hidden_size, out_features=3)

        # apply weight initializations
        self._special_init_weights()

        self.prev_hidden = None

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
                nn.init.kaiming_normal_(child.weight)
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
        x = self.act_1( self.conv_1(X) ) #out: (b, 64, 128, 157)
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
        self.prev_hidden = h_out
        #h_out shape: (num_layers, b, gru_hidden_size[32])
        #x     shape: (b, 128, gru_hidden_size[32]) 
        
        # Head
        x = self.head_conv(x.unsqueeze(-1)) #out: (b, n_colors, gru_hidden_size[32], 1)
        logits = self.head(x.squeeze(-1))   #out: (b, 5, 3)

        return logits


# https://arxiv.org/pdf/1901.04555.pdf
# https://github.com/ZainNasrullah/music-artist-classification-crnn/blob/master/src/models.py
# https://yerevann.github.io/2016/06/26/combining-cnn-and-rnn-for-spoken-language-identification/

# an alternative:
# paper: https://paperswithcode.com/paper/efficient-large-scale-audio-tagging-via
# https://github.com/fschmid56/EfficientAT



