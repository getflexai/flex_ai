from flex_ai.common import enums
from flex_ai.utils.collators import ensure_spaces_between_chat_completions_separators
from flex_ai.utils.datasets import convert_instruction_dataset_to_chat_dataset
from flex_ai.data_loaders.files import read_jsonl
from datasets import Dataset
import pandas as pd
from flex_ai.common.logger import get_logger
from transformers import PreTrainedTokenizerBase
from typing import Union

logger = get_logger(__name__)

def _validate_dataset_type(json_row):
    # Check for Instruction type
    if isinstance(json_row, dict) and 'instruction' in json_row and 'output' in json_row:
        return enums.DatasetType.INSTRUCTION
    
    # Check for Chat type
    if isinstance(json_row, list) and all(isinstance(item, dict) and 'role' in item and 'content' in item for item in json_row):
        return enums.DatasetType.CHAT
    
    # Check for Text type
    if isinstance(json_row, dict) and 'text' in json_row:
        return enums.DatasetType.TEXT
    
    return None

def validate_dataset(train_path:str, eval_path:Union[str, None], tokenizer: PreTrainedTokenizerBase):
    train_data = read_jsonl(train_path)
    eval_data = read_jsonl(eval_path)

    train_dataset_type = _validate_dataset_type(train_data[0])
    eval_dataset_type = _validate_dataset_type(eval_data[0])

    if eval_path is not None and train_dataset_type != eval_dataset_type:
        raise ValueError(f"Train dataset type is {train_dataset_type} while eval dataset type is {eval_dataset_type}. They need to be equal")
    
    logger.info(f"Dataset type is {train_dataset_type}")

    if train_dataset_type == enums.DatasetType.CHAT:
        train_data = ensure_spaces_between_chat_completions_separators(train_data)
        eval_data = ensure_spaces_between_chat_completions_separators(eval_data)

        transformed_train_data = [
            {"text": tokenizer.apply_chat_template(chat, tokenize=False)}
            for chat in train_data
        ]
        transformed_eval_data = [
            {"text": tokenizer.apply_chat_template(chat, tokenize=False)}
            for chat in eval_data
        ]
        train_dataset = Dataset.from_pandas(pd.DataFrame(transformed_train_data)).shuffle(seed=42)
        eval_dataset = Dataset.from_pandas(pd.DataFrame(transformed_eval_data)).shuffle(seed=42)
    elif train_dataset_type == enums.DatasetType.INSTRUCTION:
        train_data, eval_data = convert_instruction_dataset_to_chat_dataset(train_data, eval_data)
        train_data = ensure_spaces_between_chat_completions_separators(train_data)
        eval_data = ensure_spaces_between_chat_completions_separators(eval_data)

        transformed_train_data = [
            {"text": tokenizer.apply_chat_template(chat, tokenize=False)}
            for chat in train_data
        ]
        transformed_eval_data = [
            {"text": tokenizer.apply_chat_template(chat, tokenize=False)}
            for chat in eval_data
        ]
        train_dataset = Dataset.from_pandas(pd.DataFrame(transformed_train_data)).shuffle(seed=42)
        eval_dataset = Dataset.from_pandas(pd.DataFrame(transformed_eval_data)).shuffle(seed=42)
    elif train_dataset_type == enums.DatasetType.TEXT:
        train_dataset = Dataset.from_pandas(pd.DataFrame(train_data)).shuffle(seed=42)
        eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_data)).shuffle(seed=42)

    return train_dataset, eval_dataset, train_dataset_type

def _log_transformed_dpo_examples(train_data, eval_data):
        logger.info("Train dataset chat example after model template for DPO:")
        logger.info("prompt:")
        logger.info(train_data[0]["prompt"])
        logger.info("")
        logger.info("chosen:")
        logger.info(train_data[0]["chosen"])
        logger.info("")
        logger.info("rejected:")
        logger.info(train_data[0]["rejected"])
        logger.info("")
        logger.info("Eval dataset chat example after model template for DPO:")
        logger.info("prompt:")
        logger.info(eval_data[0]["prompt"])
        logger.info("")
        logger.info("chosen:")
        logger.info(eval_data[0]["chosen"])
        logger.info("")
        logger.info("rejected:")
        logger.info(eval_data[0]["rejected"])
        logger.info("")

def _log_transformed_sft_examples(train_data, eval_data):
        logger.info("Train dataset chat example after model template:")
        logger.info(train_data[0]["text"])
        logger.info("")
        logger.info("Eval dataset chat example after model template:")
        logger.info(eval_data[0]["text"])
        logger.info("")