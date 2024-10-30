import json
import shutil
import os

# Read the .jsonl file and parse the data
def read_jsonl(file_path: str):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]
    
def resolve_path(id: str, file_path: str):
    if "{{id}}" in file_path:
        file_path = file_path.replace("{{id}}", id)
    return file_path

def copy_files(file_path: str, source_path: str):
    # Create the directory path if it doesn't exist
    os.makedirs(file_path, exist_ok=True)
    
    # Copy files from the source path to the destination path
    for file_name in os.listdir(source_path):
        source_file = os.path.join(source_path, file_name)
        destination_file = os.path.join(file_path, file_name)
        shutil.copy2(source_file, destination_file)