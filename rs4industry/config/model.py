from dataclasses import dataclass

@dataclass
class ModelArguments:
    model_name: str = None
    embedding_dim: int = 10

    @staticmethod
    def from_dict(d):
        arg = ModelArguments()
        for k, v in d.items():
            setattr(arg, k, v)
        return arg
    
    def to_dict(self):
        return self.__dict__