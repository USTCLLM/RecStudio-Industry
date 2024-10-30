import logging
from loguru import logger as loguru_logger
from rs4industry.config import TrainingArguments


def get_logger(config: TrainingArguments):
    logger = loguru_logger
    if config.logging_dir is not None:
        logger.add(f"{config.logging_dir}/train.log", level='INFO')
    return logger