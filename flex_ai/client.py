import json
import time
from typing import List, Literal, Optional, Union
import requests
import os
from flex_ai.api.datasets import create_dataset, download_checkpoint, download_checkpoint_gguf, generate_dataset_upload_urls, get_datasets
from flex_ai.api.endpoints import create_multi_lora_endpoint, get_endpoint
from flex_ai.api.models import get_models
from flex_ai.api.tasks import get_task, get_task_checkpoints
from flex_ai.api.fine_tunes import create_finetune
from flex_ai.api.checkpoints import get_checkpoint
from flex_ai.common import enums
from flex_ai.common.classes import EarlyStoppingConfig, LoraCheckpoint, LoraConfig
from flex_ai.data_loaders.loaders import validate_dataset
from flex_ai.common.logger import get_logger
import uuid
from flex_ai.utils.conversions import download_and_extract_tar_zst
from tqdm.auto import tqdm

logger = get_logger(__name__)

from flex_ai.utils.tokenizers import load_default_tokenizer
from flex_ai.utils.visualize import generate_report

class FlexAI:
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = os.environ.get("FLEX_AI_API_KEY")
        if not api_key:
            raise ValueError("API key must be provided")
        self.api_key = api_key

    def validate_dataset(self, train_path:str, eval_path:Union[str, None]):
        tokenizer = load_default_tokenizer()
        train_dataset, eval_dataset = validate_dataset(train_path, eval_path, tokenizer)

        def tokenize_text(examples):
            return {"num_tokens": [len(tokens) for tokens in tokenizer(examples["text"])["input_ids"]]}
    
        # Apply tokenization to the dataset and compute the maximum token size
        train_dataset_with_tokens = train_dataset.map(tokenize_text, batched=True, num_proc=1)
        max_seq_len_train = max(train_dataset_with_tokens["num_tokens"])

        if eval_dataset:
            eval_dataset_with_tokens = eval_dataset.map(tokenize_text, batched=True, num_proc=1)
            max_seq_len_eval = max(eval_dataset_with_tokens["num_tokens"])
        else:
            max_seq_len_eval = None

        # Print a report
        report = generate_report(max_seq_len_train, max_seq_len_eval)
        logger.info(report)
        
        return f"Using API key: {self.api_key}"
    

    def create_dataset(self, name:str, train_path:str, eval_path:Union[str, None]):
        tokenizer = load_default_tokenizer()
        train_dataset, eval_dataset, dataset_type = validate_dataset(train_path, eval_path, tokenizer)

        def tokenize_text(examples):
            return {"num_tokens": [len(tokens) for tokens in tokenizer(examples["text"])["input_ids"]]}
    
        # Apply tokenization to the dataset and compute the maximum token size
        train_dataset_with_tokens = train_dataset.map(tokenize_text, batched=True, num_proc=1)
        max_seq_len_train = max(train_dataset_with_tokens["num_tokens"])
        total_train_tokens = sum(train_dataset_with_tokens["num_tokens"])

        if eval_dataset:
            eval_dataset_with_tokens = eval_dataset.map(tokenize_text, batched=True, num_proc=1)
            max_seq_len_eval = max(eval_dataset_with_tokens["num_tokens"])
        else:
            max_seq_len_eval = None

        dataset_id = str(uuid.uuid4())
        # upload the train_path and eval_path to the server
        train_upload_url, eval_upload_url, storage_type = generate_dataset_upload_urls(self.api_key, dataset_id)
        
        # Upload the train dataset file to the server using the pre-signed URL
        with open(train_path, 'rb') as f:
            response = requests.put(train_upload_url, data=f)
            if response.status_code == 200:
                print("Train dataset uploaded successfully.")
            else:
                upload_success = False
                print(f"Failed to upload train dataset. Status code: {response.status_code}")
                return

        if eval_path:
            with open(eval_path, 'rb') as f:
                response = requests.put(eval_upload_url, data=f)
                if response.status_code == 200:
                    print("Eval dataset uploaded successfully.")
                else:
                    print(f"Failed to upload eval dataset. Status code: {response.status_code}")
                    return

        new_dataset = create_dataset(self.api_key, dataset_id, name, len(train_dataset), len(eval_dataset) if eval_dataset else None, max_seq_len_train, total_train_tokens , dataset_type, storage_type)
        print("New Dataset created successfully.")
        print(json.dumps(new_dataset, indent=4, sort_keys=True))
        
        return new_dataset
    

    def get_datasets(self):
        my_datasets = get_datasets(self.api_key)
        print("Datasets:")
        print(json.dumps(my_datasets, indent=4, sort_keys=True))
        
        return my_datasets
    
    def download_checkpoint_gguf(self, checkpoint_id:str):
        checkpoint = get_checkpoint(self.api_key, checkpoint_id)
        step = checkpoint["step"]
        url = download_checkpoint_gguf(self.api_key, checkpoint_id)
        download_and_extract_tar_zst(f"{checkpoint_id}-checkpoint-gguf-step-{step}", url)
    
    def download_checkpoint(self, checkpoint_id:str):
        checkpoint = get_checkpoint(self.api_key, checkpoint_id)
        step = checkpoint["step"]
        url = download_checkpoint(self.api_key, checkpoint_id)
        download_and_extract_tar_zst(f"{checkpoint_id}-checkpoint-step-{step}", url)
    
    def get_models(self):
        available_models = get_models(self.api_key)
        print("Available Models:")
        print(json.dumps(available_models, indent=4, sort_keys=True))
        
        return available_models
    
    def get_task(self, id:str):
        task = get_task(self.api_key, id)
        print("Tasks:")
        print(json.dumps(task, indent=4, sort_keys=True))
        
        return task
    
    def get_task_checkpoints(self, task_id:str):
        checkpoints = get_task_checkpoints(self.api_key, task_id)
        
        return checkpoints
    
    def get_checkpoint(self, id:str):
        checkpoint = get_checkpoint(self.api_key, id)
        print("Tasks:")
        print(json.dumps(checkpoint, indent=4, sort_keys=True))
        
        return checkpoint
    

    def create_finetune(self, 
                        name:str, dataset_id: str, 
                        model: str, n_epochs: int,
                        train_with_lora: bool,
                        batch_size: Optional[int] = None, learning_rate: Optional[float] = None,
                        wandb_key: Optional[str] = None,
                        n_checkpoints_and_evaluations_per_epoch: Optional[int] = None,
                        save_only_best_checkpoint: bool = False,
                        lora_config: Union[Optional[LoraConfig], None] = None,
                        early_stopping_config: Union[Optional[EarlyStoppingConfig], None] = None):

        new_task = create_finetune(api_key=self.api_key, name=name, dataset_id=dataset_id, model=model, n_epochs=n_epochs, batch_size=batch_size, wandb_key=wandb_key,
                        learning_rate=learning_rate,n_checkpoints_and_evaluations_per_epoch=n_checkpoints_and_evaluations_per_epoch,
                        save_only_best_checkpoint=save_only_best_checkpoint, train_with_lora=train_with_lora, lora_config=lora_config, early_stopping_config=early_stopping_config)
        
        print("New Task created successfully.")
        print(json.dumps(new_task, indent=4, sort_keys=True))
        
        return new_task

    def wait_for_task_completion(self, task_id: str, interval: int = 1):
        """
        Periodically checks the status of a task until it reaches a final state.
        This method will block the main thread until the task is completed.
        
        :param task_id: The ID of the task to check.
        :param interval: The number of seconds to wait between checks (default is 10).
        :return: The final task status.
        """
        final_statuses = ["COMPLETED", "ERRORED", "CANCELED"]

        task = get_task(self.api_key, task_id)
        model_name = task["models"]["name"]
        dataset_name = task["datasets"]["name"]
        task_name = task["name"]
        
        print(f"Monitoring task {task_name}...")
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset_name}")
        print(f"")

        step_progress = tqdm(total=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}', desc="Steps", ncols=100)
        epoch_info = tqdm(total=0, bar_format='{desc}', desc="Epoch: 0/0", leave=False)
        
        last_step = 0
        last_epoch = 0
        
        try:
            while True:
                task = get_task(self.api_key, task_id)
                status = task.get("stage")
                current_epoch = 0 if task.get("current_epoch") is None else task.get("current_epoch")
                total_steps = 100 if task.get("total_steps") is None else task.get("total_steps")
                current_step = 0 if task.get("current_step") is None else task.get("current_step")
                total_epochs = 0 if task.get("total_epochs") is None else task.get("total_epochs")
                
                # Update step progress
                step_progress.total = total_steps
                step_progress.update(current_step - last_step)
                last_step = current_step
                
                # Update epoch information
                if current_epoch != last_epoch or total_epochs != epoch_info.total:
                    epoch_info.total = total_epochs
                    epoch_info.set_description_str(f"Epoch: {current_epoch}/{total_epochs}")
                    last_epoch = current_epoch
                
                if status in final_statuses:
                    step_progress.close()
                    epoch_info.close()
                    return status
                
                time.sleep(interval)
        except KeyboardInterrupt:
            step_progress.close()
            epoch_info.close()
            print("\nTask monitoring interrupted by user.")
            return None
    
    def create_multi_lora_endpoint(self, name:str, lora_checkpoints: List[LoraCheckpoint], compute: Literal["T4", "A100-40GB", "A100-80GB", "A10G", "A100-80GB", "L4"] = "A100-40GB", idle_timeout_seconds: Optional[int] = 60) -> str:
        data = create_multi_lora_endpoint(self.api_key, name, lora_checkpoints, compute, idle_timeout_seconds)
        print("New Endpoint created successfully.")
        
        return data["endpoint_id"]

    def wait_for_endpoint_ready(self, endpoint_id: str, interval: int = 1):
        final_statuses = ["LIVE"]

        endpoint = get_endpoint(self.api_key, endpoint_id)
        endpoint_name = endpoint["name"]

        print(f"Initializing endpoint {endpoint_name}...")

        try:
            while True:
                endpoint = get_endpoint(self.api_key, endpoint_id)
                status = endpoint.get("stage")
                
                if status in final_statuses:
                    print(f"Endpoint {endpoint_name} is ready.")
                    return endpoint
                
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nEndpoint monitoring interrupted by user.")
            return None
