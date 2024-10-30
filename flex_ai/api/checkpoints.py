import requests
from flex_ai.settings import BASE_URL

def get_checkpoint(api_key:str, checkpoint_id:str):
    url = f"{BASE_URL}/v1/checkpoints"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"checkpoint_id": checkpoint_id}

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise Exception(response.json()["detail"])
    
    data = response.json()
    return data[0]