from transformers import AutoTokenizer

def load_default_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated")
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    return tokenizer