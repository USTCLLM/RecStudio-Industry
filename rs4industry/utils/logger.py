from accelerate import Accelerator
from loguru import logger as loguru_logger
from rs4industry.config import TrainingArguments


def get_logger(config: TrainingArguments):
    accelerator = Accelerator()
    logger = loguru_logger
    if accelerator.is_local_main_process:
        if config.logging_dir is not None:
            logger.add(f"{config.logging_dir}/train.log", level='INFO')
        elif config.checkpoint_dir is not None:
            logger.add(f"{config.checkpoint_dir}/train.log", level='INFO')
    return logger