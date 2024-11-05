from dataclasses import dataclass, field

@dataclass
class TrainingArguments:
    device: str = "cuda"    # "cuda" or "cpu" or "0,1,2"
    # accelerate
    epochs: int = 10
    optimizer: str = "adam"
    gradient_accumulation_steps: int = 1
    lr_scheduler: str = None
    train_batch_size: int = 512
    item_batch_size: int = 2048 # only used for retriever training
    learning_rate: float = 0.001
    weight_decay: float = 0.0000
    gradient_accumulation_steps: int = 1
    logging_dir: str = None
    logging_steps: int = 10
    evaluation_strategy: str = "epoch"  # epoch or step
    eval_interval: int = 1    # interval between evaluations, epochs or steps
    eval_batch_size: int = 256
    cutoffs: list = field(default_factory=lambda : [1, 5, 10])
    metrics: list = field(default_factory=lambda : ["ndcg", "recall"])
    earlystop_strategy: str = "epoch"   # epoch or step
    earlystop_patience: int = 5     # number of epochs or steps
    earlystop_metric: str = "ndcg@5"
    earlystop_metric_mode: str = "max"
    # checkpoint rule:
    # 1. default: save model per epoch
    # 2. optional: save best model, implemented by earlystop callback
    # 3. optional: save model by steps, implemented by checkpoint callback
    checkpoint_best_ckpt: bool = True   # if true, save best model in earystop callback
    checkpoint_dir: str = None  # required
    checkpoint_steps: int = 1000    # if none, save model per epoch; else save model by steps
    
    max_grad_norm: float = 1.0

    # TODO: use below
    learning_rate_schedule: str = "cosine"
    warmup_steps: int = 1000

    @staticmethod
    def from_dict(d):
        args = TrainingArguments()
        for k, v in d.items():
            if hasattr(args, k):
                setattr(args, k, v)
            else:
                raise ValueError(f"Invalid argument: {k}")
        return args

    def to_dict(self):
        return self.__dict__