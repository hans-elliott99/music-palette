from abc import ABC, abstractmethod
from src.models.transformers import ConvTransformer, PatchTransformer, GPTPatchTransformer

class Config(ABC):

    @abstractmethod
    def __init__(self) -> None:
        # derived classes must define an init method
        pass

    def check_required(self, required_keys) -> None:
        required_keys = set(required_keys)
        current_keys  = set(self.to_dict())
        assert len(required_keys.intersection(current_keys)) == len(required_keys), \
            f"Missing keys in config. Missing={required_keys.difference(current_keys)}"

    def __iter__(self): #-> generator
        for attr, value in self.__dict__.items():
            yield attr, value

    def to_dict(self):
        return {attr: value for attr, value in self}


"""
General Params
    n_colors       : The number of RGB colors which the model will be predicting.
    embed_dropout  : Dropout percentage after embedding layers.
    sigmoid_logits : Bool. Whether to apply sigmoid to logits to keep predictions between [0, 1]. (Not recommended, as the model can learn this).

Transformer Params
    n_heads        : Number of Attention Heads in each instance of Multihead Attention.
    n_trfmr_layers : Number of transformer-block layers.
    attn_dropout   : Dropout percentage applied to attention mask.
    resid_dropout  : Dropout percentage applied to the projection from MultiHead Attention back into residual pathway.
    mlp_dropout    : Dropout percentage applied to MLP layers in transformer-blocks.

Patch-Model Params
    n_patches    : The number of image patches being passed into the patch based models / number of patches each spectrogram is broken into.
    linear_layers: List or Tuple. Hidden dimensions of the linear projection layers which project the flattened patch features
                    into the desired dimension D. The last element of linear_layers is D, the latent space dimension / hidden size for the
                    transformer network. It must be divisible by n_heads.
    n_freq_bins  : Int or None. The number of bins to use when creating frequency embeddings. If None, frequency embeddings are not used.
    n_time_bins  : Int or None. The number of bins to use when creating time embeddings. If None, frequency embeddings are not used.
                   (Note: time embeddings embed across the features of each individual patch. They differ from position embeddings, which encode 
                    the relative positions of items in a sequence, such as a sequence of patches.) 
"""


# ---------------------- GPT Patch Transformer Configs ---------------------- #
class GPTPatchTrfmrBasicConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.n_colors       = 1
        self.n_patches      = 20
        self.n_freq_bins    = 128
        self.n_time_bins    = None
        self.linear_layers  = [8*32]
        self.n_heads        = 8
        self.n_trfmr_layers = 8 
        self.embed_dropout  = 0.0
        self.attn_dropout   = 0.0
        self.resid_dropout  = 0.0
        self.mlp_dropout    = 0.0
        self.check_required(GPTPatchTransformer.get_empty_config())

        assert self.linear_layers[-1] % self.n_heads == 0, f"D={self.linear_layers[-1]} is not divisible by {self.n_heads} heads."


class GPTPatchTrfmrRegConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.n_colors       = 1
        self.n_patches      = 20
        self.n_freq_bins    = 128
        self.n_time_bins    = None
        self.linear_layers  = [8*64, 8*32]
        self.n_heads        = 8
        self.n_trfmr_layers = 12 
        self.embed_dropout  = 0.25
        self.attn_dropout   = 0.0
        self.resid_dropout  = 0.1
        self.mlp_dropout    = 0.25
        self.check_required(GPTPatchTransformer.get_empty_config())

        assert self.linear_layers[-1] % self.n_heads == 0, f"D={self.linear_layers[-1]} is not divisible by {self.n_heads} heads."



class GPTPatchTrfmrFineTuneConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.n_colors       = 1
        self.n_patches      = 20
        self.n_freq_bins    = 128
        self.n_time_bins    = None
        self.linear_layers  = [8*32]
        self.n_heads        = 8
        self.n_trfmr_layers = 8 
        self.embed_dropout  = 0.5
        self.attn_dropout   = 0.5
        self.resid_dropout  = 0.5
        self.mlp_dropout    = 0.5
        self.check_required(GPTPatchTransformer.get_empty_config())

        assert self.linear_layers[-1] % self.n_heads == 0, f"D={self.linear_layers[-1]} is not divisible by {self.n_heads} heads."

# ---------------------- Patch Transformer Configs -------------------------- #
class ViTDefaultForPatchTrfmr(Config):
    def __init__(self) -> None:
        super().__init__()
        self.n_colors       = 1
        self.n_patches      = 20
        self.n_freq_bins    = None
        self.n_time_bins    = None
        self.linear_layers  = [12*22]  
        self.n_heads        = 12        
        self.n_trfmr_layers = 12        
        self.embed_dropout  = 0.1
        self.attn_dropout   = 0.0
        self.resid_dropout  = 0.0
        self.mlp_dropout    = 0.1
        self.sigmoid_logits = False 
        self.check_required(PatchTransformer.get_empty_config())

        assert self.linear_layers[-1] % self.n_heads == 0, f"D={self.linear_layers[-1]} is not divisible by {self.n_heads} heads."

        
class PatchTrfmrBasicConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.n_colors       = 1
        self.n_patches      = 20
        self.n_freq_bins    = 128
        self.n_time_bins    = 128
        self.linear_layers  = [12*22]  #[2**10, 2**9, 2**8]
        self.n_heads        = 12       #vit_base=12
        self.n_trfmr_layers = 9        #vit_base=12
        self.embed_dropout  = 0.1
        self.attn_dropout   = 0.0
        self.resid_dropout  = 0.0
        self.mlp_dropout    = 0.1
        self.sigmoid_logits = False
        self.check_required(PatchTransformer.get_empty_config())

        assert self.linear_layers[-1] % self.n_heads == 0, f"D={self.linear_layers[-1]} is not divisible by {self.n_heads} heads."

class PatchTrfmrRegularizeConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.n_colors       = 1
        self.n_patches      = 20
        self.n_freq_bins    = 128
        self.n_time_bins    = 128
        self.linear_layers  = [12*22]  
        self.n_heads        = 12        #vit_base=12
        self.n_trfmr_layers = 12        #vit_base=12
        self.embed_dropout  = 0.3
        self.attn_dropout   = 0.1
        self.resid_dropout  = 0.1
        self.mlp_dropout    = 0.5
        self.sigmoid_logits = False #Apply sigmoid to logits to keep in range(0, 1). Or, let the model learn it.
        self.check_required(PatchTransformer.get_empty_config())

        assert self.linear_layers[-1] % self.n_heads == 0, f"D={self.linear_layers[-1]} is not divisible by {self.n_heads} heads."

# ----------------------- Conv Transformer Configs -------------------------- #
# -------------------------------***OLD***----------------------------------- #
class ConvTrfmrBasicConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.n_colors        = 1
        self.max_seq_len     = 5
        self.n_heads         = 4
        self.n_trfmr_layers  = 4
        self.embed_dropout   = 0.0
        self.attn_dropout    = 0.0
        self.resid_dropout   = 0.0
        self.mlp_dropout     = 0.0
        self.conv_dropout    = 0.0
        self.conv_filters    = [64, 128, 256, 128, 128]
        self.kernel_size     = 3
        self.pool_sizes      = [(2,2), (2,2), (4,4), (4,4), (4,4)]
        self.conv_activation = "gelu"
        self.sigmoid_logits  = False
        self.check_required(ConvTransformer.get_empty_config())

class ConvTrfmrRegularizeConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.n_colors        = 1
        self.max_seq_len     = 5
        self.n_heads         = 4
        self.n_trfmr_layers  = 4
        self.embed_dropout   = 0.0
        self.attn_dropout    = 0.0
        self.resid_dropout   = 0.0
        self.mlp_dropout     = 0.5
        self.conv_dropout    = 0.5 #conv dropout doesnt slow things down too much. others ^ do
        self.conv_filters    = [128, 256, 512, 256, 128]
        self.kernel_size     = 3
        self.pool_sizes      = [(2,2), (2,2), (4,4), (4,4), (4,4)]
        self.conv_activation = "gelu"
        self.sigmoid_logits  = False
        self.check_required(ConvTransformer.get_empty_config())