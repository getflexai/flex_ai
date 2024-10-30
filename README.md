# FlexAI

A platform that simplifies fine-tuning and inference for 60+ open-source LLMs through a single API interface. FlexAI enables serverless deployment, reducing setup time by up to 70%.

## Key Features

- Serverless fine-tuning and inference
- Live time and cost estimations
- Checkpoint management
- LoRA and multi-LoRA support
- Target inference validations
- OpenAI-compatible Endpoints API
- Interactive Playground

## Installation

```bash
pip install flex_ai openai
```

## Quick Start

```python
from flex_ai import FlexAI

# Initialize client with your API key
client = FlexAI(api_key="your-api-key")

# Create dataset
dataset = client.create_dataset("Dataset Name", "train.jsonl", "eval.jsonl")

# Start fine-tuning
task = client.create_finetune(
    name="My Task",
    dataset_id=dataset["id"],
    model="meta-llama/Llama-3.2-3B-Instruct",
    n_epochs=10,
    train_with_lora=True,
    lora_config={
        "lora_r": 64,
        "lora_alpha": 8,
        "lora_dropout": 0.1
    }
)

# Create endpoint
endpoint = client.create_multi_lora_endpoint(
    name="My Endpoint",
    lora_checkpoints=[{"id": checkpoint_id, "name": "step_1"}],
    compute="A100-40GB"
)
```

## Using Your Fine-tuned Model

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url=f"{endpoint_url}/v1"
)

completion = client.completions.create(
    model="your-model",
    prompt="Your prompt",
    max_tokens=60
)
```

## Get Started

1. Sign up at [app.getflex.ai](https://app.getflex.ai)
2. Get your API key from Settings -> API Keys
3. Start with our [documentation](https://docs.getflex.ai)

## Resources

- [Documentation](https://docs.getflex.ai)
- [Platform](https://getflex.ai)
- [API Reference](https://docs.getflex.ai/api-reference)
