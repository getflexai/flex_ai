import requests
from flex_ai.settings import BASE_URL

def get_models(api_key:str):
    url = f"{BASE_URL}/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(response.json()["detail"])
    
    data = response.json()
    return data