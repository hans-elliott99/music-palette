from abc import ABC, abstractmethod
from src.models.transformers import ConvTransformer, PatchTransformer

class Config(ABC):

    @abstractmethod
    def __init__(self) -> None:
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
        out = {}
        for attr, value in self:
            out[attr] = value
        return out


class ViTDefaultForPatchTrfmr(Config):
    def __init__(self) -> None:
        super().__init__()
        self.n_colors       = 1
        self.n_patches      = 20
        self.n_freq_bins    = None
        self.n_time_bins    = None
        # last element of linear_layers is D, the latent space dim / hidden size
        # must be divisible by n_heads
        self.linear_layers  = [12*22]  
        self.n_heads        = 12        #vit_base=12
        self.n_trfmr_layers = 12        #vit_base=12
        self.embed_dropout  = 0.1
        self.attn_dropout   = 0.0
        self.resid_dropout  = 0.0
        self.mlp_dropout    = 0.1
        self.sigmoid_logits = False #Apply sigmoid to logits to keep in range(0, 1). Or, let the model learn it.
        self.check_required(PatchTransformer.get_empty_config())

        assert self.linear_layers[-1] % self.n_heads == 0, f"D={self.linear_layers[-1]} is not divisible by {self.n_heads} heads."

        
class PatchTrfmrBasicConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.n_colors       = 1
        self.n_patches      = 20
        self.n_freq_bins    = 128
        self.n_time_bins    = 128
        # last element of linear_layers is D, the latent space dim / hidden size
        # must be divisible by n_heads
        self.linear_layers  = [12*22]  #[2**10, 2**9, 2**8]
        self.n_heads        = 12       #vit_base=12
        self.n_trfmr_layers = 9        #vit_base=12
        self.embed_dropout  = 0.1
        self.attn_dropout   = 0.0
        self.resid_dropout  = 0.0
        self.mlp_dropout    = 0.1
        self.sigmoid_logits = False #Apply sigmoid to logits to keep in range(0, 1). Or, let the model learn it.
        self.check_required(PatchTransformer.get_empty_config())

        assert self.linear_layers[-1] % self.n_heads == 0, f"D={self.linear_layers[-1]} is not divisible by {self.n_heads} heads."

class PatchTrfmrRegularizeConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.n_colors       = 1
        self.n_patches      = 20
        self.n_freq_bins    = 128
        self.n_time_bins    = 128
        # last element of linear_layers is D, the latent space dim / hidden size
        # must be divisible by n_heads
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