from typing import Any, Dict, Optional, Union
import requests

from flex_ai.common.classes import EarlyStoppingConfig, LoraConfig
from flex_ai.settings import BASE_URL

# send api key in the header
def generate_dataset_upload_urls(api_key:str, dataset_id:str):
    url = f"{BASE_URL}/v1/datasets/generate_upload_urls"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"id": dataset_id}

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise Exception(response.json()["detail"])
    data = response.json()
    return data["train_upload_url"], data["eval_upload_url"]

def create_finetune(api_key:str, name:str, dataset_id: str, 
                        model: str, n_epochs: int,
                        train_with_lora: bool,
                        batch_size: Optional[int] = None, learning_rate: Optional[float] = None,
                        n_checkpoints_and_evaluations_per_epoch: Optional[int] = None,
                        save_only_best_checkpoint: bool = False,
                        lora_config: Union[Optional[LoraConfig], None] = None,
                        wandb_key: Optional[str] = None,
                        early_stopping_config: Union[Optional[EarlyStoppingConfig], None] = None):
    
    url = f"{BASE_URL}/v1/fine_tunes/create_finetune"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # Construct the request data dynamically
    payload: Dict[str, Any] = {
        "name": name,
        "dataset_id": dataset_id,
        "model": model,
        "train_with_lora": train_with_lora
    }
    
    if n_epochs is not None:
        payload["n_epochs"] = n_epochs
    if batch_size is not None:
        payload["batch_size"] = batch_size
    if wandb_key is not None:
        payload["wandb_key"] = wandb_key
    if learning_rate is not None:
        payload["learning_rate"] = learning_rate
    if n_checkpoints_and_evaluations_per_epoch is not None:
        payload["n_checkpoints_and_evaluations_per_epoch"] = n_checkpoints_and_evaluations_per_epoch
    if save_only_best_checkpoint is not None:
        payload["save_only_best_checkpoint"] = save_only_best_checkpoint
    if lora_config is not None:
        payload["lora_config"] = lora_config  # Convert Pydantic model to dictionary
    if early_stopping_config is not None:
        payload["early_stopping_config"] = early_stopping_config  # Convert Pydantic model to dictionary

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise Exception(response.json()["detail"])
    data = response.json()
    return data[0]