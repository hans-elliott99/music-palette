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



class PatchTrfmrBasicConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.n_colors       = 1
        self.linear_layers  = [2**10, 2**8, 2**7]
        self.n_heads        = 4
        self.n_trfmr_layers = 4
        self.embed_dropout  = 0.0
        self.attn_dropout   = 0.0
        self.resid_dropout  = 0.0
        self.mlp_dropout    = 0.0
        self.sigmoid_logits = False #Apply sigmoid to logits to keep in range(0, 1). Or, let the model learn it.
        self.check_required(PatchTransformer.get_empty_config())


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