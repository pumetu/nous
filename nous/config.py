from dataclasses import dataclass

@dataclass
class ModelConfig:
    bias: bool = False # linear layer bias
