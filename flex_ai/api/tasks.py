import requests
from flex_ai.settings import BASE_URL

def get_task(api_key:str, id:str):
    url = f"{BASE_URL}/v1/tasks"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"id": id}

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise Exception(response.json()["detail"])
    
    data = response.json()
    return data[0]

def get_task_checkpoints(api_key:str, id:str):
    url = f"{BASE_URL}/v1/tasks/checkpoints"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"task_id": id}

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise Exception(response.json()["detail"])
    
    data = response.json()
    return data