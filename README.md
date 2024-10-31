<div align="center">

<a href="https://geflex.ai"><picture>

<!-- <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20white%20text.png">
<source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20black%20text.png">
<img alt="unsloth logo" src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20black%20text.png" height="110" style="max-width: 100%;">
</picture></a> -->

<a href="https://colab.research.google.com/drive/1nSFRjrHpbz372h7W-2G7jmTuh_Dhiv9x?authuser=2#scrollTo=RiXPwLmHBOCV"><img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/start free finetune button.png" height="48"></a>
<a href="https://getflexai.slack.com/"><img src="https://camo.githubusercontent.com/f37ad221182fdeaf33cdb835972fe8826ff503d779821c3533d820cf3630f79d/68747470733a2f2f636c6f7564626572727964622e6f72672f6173736574732f696d616765732f736c61636b5f627574746f6e2d37363130663963353164383230303961643931326164656431323463326438382e737667" height="48"></a>
<a href="https://app.apollo.io/#/meet/ariel/30-min"><img src="  ://raw.githubusercontent.com/unslothai/unsloth/main/images/buy me a coffee button.png" height="48"></a>

### Finetune Llama 3.2, Mistral, Phi-3.5 & Gemma 2-5x faster with 80% less memory!

![](https://i.ibb.co/sJ7RhGG/image-41.png)

</div>

# FlexAI

A platform that simplifies fine-tuning and inference for 60+ open-source LLMs through a single API interface. FlexAI enables serverless deployment, reducing setup time by up to 70%.

## ⭐ Key Features

- Serverless fine-tuning and inference
- Live time and cost estimations
- Checkpoint management
- LoRA and multi-LoRA support
- Target inference validations
- OpenAI-compatible Endpoints API
- Interactive Playground

## ✨ Get Started

1. Sign up at [app.getflex.ai](https://app.getflex.ai)
2. Get your API key from Settings -> API Keys
3. Start with our [documentation](https://docs.getflex.ai)

## 📚 Full Google Colab Example

[One Notebook to fine tune all LLMs](https://colab.research.google.com/drive/1nSFRjrHpbz372h7W-2G7jmTuh_Dhiv9x?authuser=2#scrollTo=RiXPwLmHBOCV)

## 💾 Installation

You dont need to install, no CUDA, no NVIDIA drivers, no setup. Our lightweight library is only an API wrapper to FlexAI serverless GPUs.
You can work from any operating system, including Windows, MacOS, and Linux.

```bash
pip install flex_ai openai
```

## 🦥 Quick Start

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
    # You can choose from 60+ models, Full list [here](https://docs.getflex.ai/core-concepts/models)
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

## 🥇 Using Your Fine-tuned Model

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

## 🔗 Links and Resources

| Type                                                                                                                        | Links                                                                                           |
| --------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 📚 **Documentation & Wiki**                                                                                                 | [Read Our Docs](https://docs.getflex.ai)                                                        |
| <img height="14" src="https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg" />&nbsp; **Twitter (aka X)** | [Follow us on X](https://x.com/getflex_ai)                                                      |
| 💾 **Installation**                                                                                                         | [getflex/README.md](https://github.com//getflexai/flex_ai/tree/main#-installation-instructions) |
| 🌐 **Supported Models**                                                                                                     | [FlexAI Models](https://docs.getflex.ai/core-concepts/models)                                   |
