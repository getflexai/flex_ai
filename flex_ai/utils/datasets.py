from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

def _instruction_row_to_chat_format(example):
    if "system" in example and example["system"] is not None:
        converted_data = [{ "role": "system",  "content": example["system"]}, { "role": "user",  "content": example["instruction"] },{"role": "assistant", "content": example["output"]}]
    else:
        converted_data = [{ "role": "user",  "content": example["instruction"]},{"role": "assistant", "content": example["output"]}]
    return converted_data

def convert_instruction_dataset_to_chat_dataset(train_data: list[any], eval_data: list[any] ):
    converted_train_data = []
    converted_eval_data = []

    for item in train_data:
        converted_data = _instruction_row_to_chat_format(item)
        converted_train_data.append(converted_data)
    
    for item in eval_data:
        converted_data = _instruction_row_to_chat_format(item)
        converted_eval_data.append(converted_data)
    
    return converted_train_data, converted_eval_data

def convert_instruction_dpo_dataset_to_chat_template(train_data: list[any], eval_data: list[any], tokenizer: PreTrainedTokenizerBase):
    converted_train_data = []
    converted_eval_data = []

    for item in train_data:
        converted_data = _dpo_instruction_row_to_chat_format(item, tokenizer)
        converted_train_data.append(converted_data)
    5
    for item in eval_data:
        converted_data = _dpo_instruction_row_to_chat_format(item, tokenizer)
        converted_eval_data.append(converted_data)
    
    return converted_train_data, converted_eval_data

def convert_chat_dpo_dataset_to_chat_template(train_data: list[any], eval_data: list[any], tokenizer: PreTrainedTokenizerBase):
    converted_train_data = []
    converted_eval_data = []

    for item in train_data:
        converted_data = _dpo_chat_row_to_chat_format(item, tokenizer)
        converted_train_data.append(converted_data)
    5
    for item in eval_data:
        converted_data = _dpo_chat_row_to_chat_format(item, tokenizer)
        converted_eval_data.append(converted_data)
    
    return converted_train_data, converted_eval_data

def _dpo_chat_row_to_chat_format(example, tokenizer: PreTrainedTokenizerBase):
    prompt = tokenizer.apply_chat_template(example["chat"], tokenize=False, add_generation_prompt=True)

    # Format chosen answer
    chosen = example['chosen'] + tokenizer.special_tokens_map['eos_token']

    # Format rejected answer
    rejected = example['rejected'] + tokenizer.special_tokens_map['eos_token']

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

def _dpo_instruction_row_to_chat_format(example, tokenizer: PreTrainedTokenizerBase):
    # Format system
    if 'system' in example and example['system'] != "":
        message = {"role": "system", "content": example['system']}
        system = tokenizer.apply_chat_template([message], tokenize=False)
    else:
        system = ""

    # Format instruction
    message = {"role": "user", "content": example['instruction']}
    prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

    # Format chosen answer
    chosen = example['chosen'] + tokenizer.special_tokens_map['eos_token']

    # Format rejected answer
    rejected = example['rejected'] + tokenizer.special_tokens_map['eos_token']

    return {
        "prompt": system + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


class TextToJsonDataset(Dataset):
    def __init__(self, df, *, tokenizer, in_col='clean_text', out_col='output'):
        self.df = df
        self.in_col = in_col
        self.out_col = out_col
        self.tokenizer = tokenizer
        self.num_rows = len(df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {"input_ids":self.tokenizer(self.df.iloc[idx][self.in_col])["input_ids"], "labels":self.tokenizer(self.df.iloc[idx][self.out_col])["input_ids"]}
