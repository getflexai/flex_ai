from flex_ai.common.logger import get_logger

logger = get_logger(__name__)

# This special function is activated only when train on completion_only is enabled.
# When you train only on completions , you have the make sure the separators tokens [/INST] and [INST],
# have space from both sides , so that we can detect when the completion starts and that the tokenizer wont combine
# their tokens and other tokens. In huggingface prompt default templates - they dont put a space before the beginning of the completions,
# so I do it here manually 
def ensure_spaces_between_chat_completions_separators(chat_data):
    # Iterate through each item in the array
    for conversation in chat_data:
        for answer in conversation:
            # Check if the role is "assistant"
            if answer["role"] == "assistant":
                # Ensure the content starts with a space
                if not answer["content"].startswith(" "):
                    answer["content"] = " " + answer["content"]
    return chat_data

def debug_completion_only_collator(cfg, dataset, tokenizer, collator):
    text_sample = dataset[0]["text"]
    tokenized_text_sample = tokenizer(
            text_sample,
            truncation=True,
            padding=False,
            max_length=cfg.max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
            add_special_tokens=True
        )
    
    output = collator([tokenized_text_sample])
    input_ids = output["input_ids"]
    labels = output["labels"]

    # For each example in your batch
    for i in range(input_ids.size(0)):
        segments = []
        train_segment_ids = []
        not_train_segment_ids = []

        # Iterate through each token in the example
        for token_id, label in zip(input_ids[i], labels[i]):
            if label.item() == -100:
                if train_segment_ids:
                    train_segment = tokenizer.decode(train_segment_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    segments.append(f"***{train_segment}***")
                    train_segment_ids = []
                not_train_segment_ids.append(token_id)
            else:
                if not_train_segment_ids:
                    not_train_segment = tokenizer.decode(not_train_segment_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    segments.append(not_train_segment)
                    not_train_segment_ids = []
                train_segment_ids.append(token_id)

        # Add any remaining segments
        if not_train_segment_ids:
            not_train_segment = tokenizer.decode(not_train_segment_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            segments.append(not_train_segment)
        if train_segment_ids:
            train_segment = tokenizer.decode(train_segment_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            segments.append(f"***{train_segment}***")

        # Combine and print the full text with markers
        full_text_with_markers = "".join(segments)
        logger.info("DataCollatorForCompletionOnlyLM will train only text inside the ***    ***:")
        logger.info(full_text_with_markers)
        logger.info("")
