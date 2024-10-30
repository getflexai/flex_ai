import os
import tempfile
import requests
import tarfile
import zstandard as zstd
from flex_ai.common.logger import get_logger

logger = get_logger(__name__)

def download_and_extract_tar_zst(folder_name, presigned_url, output_path=None):
    
    # Use the current directory if no output path is provided
    if output_path is None:
        output_path = os.getcwd()
    
    # Create the output folder name
    full_output_path = os.path.join(output_path, folder_name)
    
    # Create the output directory if it doesn't exist
    os.makedirs(full_output_path, exist_ok=True)
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the file
        temp_file_path = os.path.join(temp_dir, folder_name)
        logger.info(f"Downloading...")
        response = requests.get(presigned_url, stream=True)
        response.raise_for_status()
        
        with open(temp_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Extracting...")

        # Extract the .tar.zst file
        extracted_tar_path = os.path.join(temp_dir, "extracted.tar")
        with open(temp_file_path, 'rb') as compressed_file:
            dctx = zstd.ZstdDecompressor()
            with open(extracted_tar_path, 'wb') as tar_file:
                dctx.copy_stream(compressed_file, tar_file)
        
        # Extract the contents of the tar file
        with tarfile.open(extracted_tar_path, 'r') as tar:
            tar.extractall(path=full_output_path)
    
    print(f"File extracted successfully to {full_output_path}")