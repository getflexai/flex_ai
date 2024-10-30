import requests
from flex_ai.common.enums import DatasetType
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
    return data["train_upload_url"], data["eval_upload_url"], data["storage_type"]

def create_dataset(api_key:str, dataset_id: str, name:str, train_rows_count:int, eval_rows_count:int, max_tokens:int, total_tokens:int, type: DatasetType, storage_type: str):
    url = f"{BASE_URL}/v1/datasets/create_dataset"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"id": dataset_id, "name": name, "train_rows_count": train_rows_count, "eval_rows_count": eval_rows_count, "max_tokens": max_tokens, "total_tokens": total_tokens, "type": type.value, "storage_type": storage_type}

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise Exception(response.json()["detail"])
    
    data = response.json()
    return data[0]

def get_datasets(api_key:str):
    url = f"{BASE_URL}/v1/datasets"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(response.json()["detail"])
    
    data = response.json()
    return data

def download_checkpoint_gguf(api_key:str, checkpoint_id:str):
    url = f"{BASE_URL}/v1/checkpoints/download_gguf"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json={"id": checkpoint_id}, headers=headers)

    if response.status_code != 200:
        raise Exception(response.json()["detail"])
    
    data = response.json()["url"]
    return data

def download_checkpoint(api_key:str, checkpoint_id:str):
    url = f"{BASE_URL}/v1/checkpoints/download"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json={"id": checkpoint_id}, headers=headers)

    if response.status_code != 200:
        raise Exception(response.json()["detail"])
    
    data = response.json()["url"]
    return data



