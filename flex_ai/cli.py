import os
import click
from flex_ai.client import FlexAI 
from flex_ai.common.classes import EarlyStoppingConfig, LoraConfig
from flex_ai.common.enums import DatasetType

# Check if key exists
if not os.environ.get("FLEX_AI_API_KEY"):
    raise ValueError("API key must be provided. Go to https://app.getflex.ai/settings/api-keys. export FLEX_AI_API_KEY=<your_api_key>")

# Create a client object
client = FlexAI(api_key=os.environ.get("FLEX_AI_API_KEY"))

@click.group()
def main():
    """CLI for the Flex AI package."""
    pass

# Datasets commands
@main.group()
def datasets():
    """Commands related to datasets."""
    pass

@datasets.command(name='list')
def list_datasets():
    """Return a list of all the datasets of the user."""
    # Implement the logic to list all datasets
    client.get_datasets()

@datasets.command(name='create')
@click.option('--name', required=True, help="Name of the new dataset")
@click.option('--train-path', required=True, help="Path to the training data file")
@click.option('--eval-path', help="Path to the evaluation data file")
def create_dataset(name, train_path, eval_path):
    """Create a new dataset."""
    # Implement the logic to create a new dataset
    print(f"Creating dataset: {name}, {train_path}, {eval_path}")
    client.create_dataset(name, train_path, eval_path)

# Models commands
@main.group()
def models():
    """Commands related to models."""
    pass

@models.command(name='list')
def list_models():
    """Return a list of available LLMs for training."""
    # Implement the logic to list all models
    client.get_models()

# Tasks commands
@main.command()
@click.option('--id', help="ID of the task to get info for")
def tasks(id):
    """Commands related to tasks."""
    if id is None:
        raise click.UsageError("Please provide only one of --id")
    else:
        client.get_task(id=id)

# Fine-tuning commands
@main.group()
def fine_tuning():
    """Commands related to fine-tuning."""
    pass

@fine_tuning.command(name='create')
@click.option('--name', required=True, help="Name of the fine-tuning task")
@click.option('--dataset-id', required=True, help="ID of the dataset to be used")
@click.option('--model', required=True, help="Model to be fine-tuned")
@click.option('--n-epochs', type=int, required=True, help="Number of epochs for fine-tuning")
@click.option('--train-with-lora', type=bool, required=True, help="Train with LoRA (True/False)")
@click.option('--batch-size', type=int, help="Batch size")
@click.option('--learning-rate', type=float, help="Learning rate")
@click.option('--wandb-key', help="Weights and Biases API key")
@click.option('--n-checkpoints-and-evaluations-per-epoch', type=int, help="Number of checkpoints and evaluations per epoch")
@click.option('--save-only-best-checkpoint', type=bool, default=False, help="Save only the best checkpoint (True/False)")
@click.option('--lora-r', type=int, help="LoRA r parameter")
@click.option('--lora-alpha', type=int, help="LoRA alpha parameter")
@click.option('--lora-dropout', type=float, help="LoRA dropout parameter")
@click.option('--earlystopping-patience', type=int, help="Early stopping patience")
@click.option('--earlystopping-threshold', type=float, help="Early stopping threshold")
def create_finetune(name, dataset_id, model, n_epochs, train_with_lora, batch_size, learning_rate, wandb_key, 
                    n_checkpoints_and_evaluations_per_epoch, save_only_best_checkpoint, 
                    lora_r, lora_alpha, lora_dropout, earlystopping_patience, earlystopping_threshold):
    """Create a new fine-tuning task."""
    
    # Create LoRA config if parameters are provided
    lora_config = None
    if lora_r is not None and lora_alpha is not None and lora_dropout is not None:
        lora_config = {
            "lora_r":lora_r,
            "lora_alpha":lora_alpha,
            "lora_dropout":lora_dropout
        }
    
    # Create EarlyStoppingConfig if parameters are provided
    early_stopping_config = None
    if earlystopping_patience is not None and earlystopping_threshold is not None:
        early_stopping_config = {
            "patience":earlystopping_patience,
            "threshold":earlystopping_threshold
        }
    
    # Implement the logic to create a fine-tuning task
    client.create_finetune(
        name=name,
        dataset_id=dataset_id,
        model=model,
        n_epochs=n_epochs,
        train_with_lora=train_with_lora,
        batch_size=batch_size,
        learning_rate=learning_rate,
        wandb_key=wandb_key,
        n_checkpoints_and_evaluations_per_epoch=n_checkpoints_and_evaluations_per_epoch,
        save_only_best_checkpoint=save_only_best_checkpoint,
        lora_config=lora_config,
        early_stopping_config=early_stopping_config
    )
    
# Checkpoints commands
@main.group()
def checkpoints():
    """Commands related to checkpoints."""
    pass

@checkpoints.command(name='download-gguf')
@click.option('--checkpoint-id', required=True, help="ID of the checkpoint to download")
def download_checkpoint_gguf(checkpoint_id):
    """Download a checkpoint in GGUF format."""
    client.download_checkpoint_gguf(checkpoint_id)

@checkpoints.command(name='download')
@click.option('--checkpoint-id', required=True, help="ID of the checkpoint to download")
def download_checkpoint(checkpoint_id):
    """Download a checkpoint in GGUF format."""
    client.download_checkpoint(checkpoint_id)

if __name__ == '__main__':
    main()