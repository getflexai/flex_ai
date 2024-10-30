from pydantic import BaseModel

class LoraConfig(BaseModel):
    lora_r: int
    lora_alpha: int
    lora_dropout: float

class EarlyStoppingConfig(BaseModel):
    patience: int
    threshold: float

class CreateFinetuneRequest(BaseModel):
    dataset_id: str
    model: str
    n_epochs: int
    batch_size: int
    learning_rate: float
    n_checkpoints_and_evaluations_per_epoch: int
    save_only_best_checkpoint: bool
    train_with_lora: bool
    lora_config: LoraConfig
    early_stopping_config: EarlyStoppingConfig

class LoraCheckpoint(BaseModel):
    name: str
    id: str