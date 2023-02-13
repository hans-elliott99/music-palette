from abc import ABC, abstractmethod
from models.transformers import ConvTransformer, SimpleTransformer

class Config(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def check_required(self) -> None:
        pass

    def __iter__(self): #-> generator
        for attr, value in self.__dict__.items():
            yield attr, value

    def to_dict(self):
        out = {}
        for attr, value in self:
            out[attr] = value
        return out



class SimpleTransformerConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.n_colors = 5
        self.n_heads = 4
        self.n_trfmr_layers = 4
        self.embed_dropout = 0.0
        self.attn_dropout  = 0.0
        self.resid_dropout = 0.0
        self.mlp_dropout   = 0.0
        self.sigmoid_logits = False #Apply sigmoid to logits to keep in range(0, 1). Or, let the model learn it.

    def check_required(self) -> None:
        required_keys = set(SimpleTransformer.get_empty_config())
        current_keys  = set(self.to_dict())
        assert len(required_keys.intersection(current_keys)) == len(required_keys), \
            f"Missing keys in config. Missing={required_keys.difference(current_keys)}"


class ConvTransformerConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.max_seq_len     = 5
        self.n_colors        = 3
        self.n_heads         = 8
        self.n_trfmr_layers  = 8
        self.embed_dropout   = 0.0
        self.trfmr_dropout   = 0.0
        self.conv_dropout    = 0.0
        self.conv_filters    = [64, 128, 256, 128, 128]
        self.kernel_size     = 3
        self.pool_sizes      = [(2,2), (2,2), (4,4), (4,4), (4,4)]
        self.conv_activation = "gelu"
        self.sigmoid_logits  = False
        self.check_required()

    def check_required(self) -> None:
        required_keys = set(ConvTransformer.get_empty_config())
        current_keys  = set(self.to_dict())
        assert len(required_keys.intersection(current_keys)) == len(required_keys), \
            f"Missing keys in config. Missing={required_keys.difference(current_keys)}"