<div align="center">

<a href="https://geflex.ai"><picture>

<!-- <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20white%20text.png">
<source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20black%20text.png">
<img alt="unsloth logo" src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20black%20text.png" height="110" style="max-width: 100%;">
</picture></a> -->

<a target="_blank" href="https://colab.research.google.com/drive/1nSFRjrHpbz372h7W-2G7jmTuh_Dhiv9x?authuser=2#scrollTo=RiXPwLmHBOCV"><img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/start free finetune button.png" height="48"></a>
<a target="_blank" href="https://getflexai.slack.com/"><img src="https://camo.githubusercontent.com/f37ad221182fdeaf33cdb835972fe8826ff503d779821c3533d820cf3630f79d/68747470733a2f2f636c6f7564626572727964622e6f72672f6173736574732f696d616765732f736c61636b5f627574746f6e2d37363130663963353164383230303961643931326164656431323463326438382e737667" height="48"></a>
<a target="_blank" href="https://app.apollo.io/#/meet/ariel/30-min"><img src="https://raw.githubusercontent.com/getflexai/flex_ai/main/flex_ai/images/call-with-founders.png" height="48" style="border-radius: 15px;"></a>

### Lightweight Library to Finetune and Deploy All LLMs, no CUDA, no NVIDIA drivers, no OOMs, Multi-GPUs setup, No Prompt Templates !

![](https://i.ibb.co/sJ7RhGG/image-41.png)

</div>

# FlexAI

A platform that simplifies fine-tuning and inference for 60+ open-source LLMs through a single API interface.
FlexAI enables serverless deployment, reducing setup time by up to 70%.
Finally , You dont have to handle installations, OOMs, GPUs setup, prompt templates, integrating new models, wait too long to download huge models, etc.

## ⭐ Key Features

- Serverless fine-tuning and inference
- Live time and cost estimations
- Checkpoint management
- LoRA and multi-LoRA support
- Target inference validations
- OpenAI-compatible Endpoints API
- Interactive Playground

## ✨ Get Started

1. Sign up at [app.getflex.ai](https://app.getflex.ai), New accounts come with 10$ for free,to get started :)
2. Get your API key from Settings -> API Keys
3. Start with our [documentation](https://docs.getflex.ai)
4. Everything can be done without any code from our dashboard - [FlexAI Dashboard](https://app.getflex.ai)

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

# Create dataset - for all datasets [here](https://docs.getflex.ai/quickstart#upload-your-first-dataset)
dataset = client.create_dataset("Dataset Name", "train.jsonl", "eval.jsonl")

# Start fine-tuning -
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

## 🦥 Full Example

```python
from flex_ai import FlexAI
from openai import OpenAI
import time

# Initialize the Flex AI client
client = FlexAI(api_key="your_api_key_here")

# Create dataset - for all datasets [here](https://docs.getflex.ai/quickstart#upload-your-first-dataset)
dataset = client.create_dataset(
    "API Dataset New",
    "instruction/train.jsonl",
    "instruction/eval.jsonl"
)

# Start a fine-tuning task
task = client.create_finetune(
    name="My Task New",
    dataset_id=dataset["id"],
    model="meta-llama/Llama-3.2-1B-Instruct",
    n_epochs=5,
    train_with_lora=True,
    lora_config={
        "lora_r": 64,
        "lora_alpha": 8,
        "lora_dropout": 0.1
    },
    n_checkpoints_and_evaluations_per_epoch=1,
    batch_size=4,
    learning_rate=0.0001,
    save_only_best_checkpoint=True
)

# Wait for training completion
client.wait_for_task_completion(task_id=task["id"])

# Wait for last checkpoint to be uploaded
while True:
    checkpoints = client.get_task_checkpoints(task_id=task["id"])
    if checkpoints and checkpoints[-1]["stage"] == "FINISHED":
        last_checkpoint = checkpoints[-1]
        checkpoint_list = [{
            "id": last_checkpoint["id"],
            "name": "step_" + str(last_checkpoint["step"])
        }]
        break
    time.sleep(10)  # Wait 10 seconds before checking again

# Create endpoint
endpoint_id = client.create_multi_lora_endpoint(
    name="My Endpoint New",
    lora_checkpoints=checkpoints_list,
    compute="A100-40GB"
)
endpoint = client.wait_for_endpoint_ready(endpoint_id=endpoint_id)

# Use the model
openai_client = OpenAI(
    api_key="your_api_key_here",
    base_url=f"{endpoint['url']}/v1"
)
completion = openai_client.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    prompt="Translate the following English text to French",
    max_tokens=60
)

print(completion.choices[0].text)
```

# LLM Models Available for Fine-tuning

This table provides an overview of the Large Language Models (LLMs) available for fine-tuning, ordered approximately from most well-known to least familiar. It lists key details for each model, including its name, family, parameter count, context length, and additional features.

| Model Name                                                                                                    | Family            | Parameters (B) | Context Length | vLLM Support | LoRA Support |
| ------------------------------------------------------------------------------------------------------------- | ----------------- | -------------- | -------------- | ------------ | ------------ |
| [Nvidia-Llama-3.1-Nemotron-70B-Instruct-HF](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF) | llama3.1          | 70             | 131,072        | Yes          | Yes          |
| [Meta-Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)                         | llama3.2          | 3              | 131,072        | Yes          | Yes          |
| [Meta-Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)                         | llama3.2          | 1              | 131,072        | Yes          | Yes          |
| [Mistral-Small-Instruct-2409](https://huggingface.co/mistralai/Mistral-Small-Instruct-2409)                   | mistral           | 7.2            | 128,000        | Yes          | Yes          |
| [Ministral-8B-Instruct-2410](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410)                     | mistral           | 8              | 128,000        | Yes          | Yes          |
| [Mathstral-7B-v0.1](https://huggingface.co/mistralai/Mathstral-7B-v0.1)                                       | mistral           | 7              | 32,000         | Yes          | Yes          |
| [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)                            | qwen2.5           | 7              | 32,768         | Yes          | Yes          |
| [Aya-Expanse-32b](https://huggingface.co/CohereForAI/aya-expanse-32b)                                         | aya               | 32             | 128,000        | Yes          | No           |
| [Aya-Expanse-8b](https://huggingface.co/CohereForAI/aya-expanse-8b)                                           | aya               | 8              | 8,000          | Yes          | No           |
| [Nemotron-Mini-4B-Instruct](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct)                          | nemotron          | 4              | 4,096          | Yes          | No           |
| [Gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it)                                                  | gemma2            | 2              | 8,192          | Yes          | Yes          |
| [Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct)                  | llama3.1          | 70             | 131,072        | Yes          | Yes          |
| [Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct)                  | llama3.1          | 70             | 131,072        | Yes          | Yes          |
| [Meta-Llama-3.1-70B](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B)                                    | llama3.1          | 70             | 131,072        | Yes          | Yes          |
| [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)                    | llama3.1          | 8              | 131,072        | Yes          | Yes          |
| [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)                                      | llama3.1          | 8              | 131,072        | Yes          | Yes          |
| [Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)                      | llama3            | 70             | 8,192          | Yes          | Yes          |
| [Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)                                        | llama3            | 70             | 8,192          | Yes          | Yes          |
| [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)                        | llama3            | 8              | 8,192          | Yes          | Yes          |
| [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)                                          | llama3            | 8              | 8,192          | Yes          | Yes          |
| [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)                     | mixtral           | 46.7           | 32,768         | Yes          | Yes          |
| [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)                         | mistral           | 7.2            | 32,768         | Yes          | Yes          |
| [Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)                     | mistral           | 12.2           | 128,000        | No           | No           |
| [Mistral-Nemo-Base-2407](https://huggingface.co/mistralai/Mistral-Nemo-Base-2407)                             | mistral           | 12.2           | 128,000        | No           | No           |
| [Gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it)                                                | gemma2            | 27             | 8,192          | Yes          | Yes          |
| [Gemma-2-27b](https://huggingface.co/google/gemma-2-27b)                                                      | gemma2            | 27             | 8,192          | Yes          | Yes          |
| [Gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)                                                  | gemma2            | 9              | 8,192          | Yes          | Yes          |
| [Gemma-2-9b](https://huggingface.co/google/gemma-2-9b)                                                        | gemma2            | 9              | 8,192          | Yes          | Yes          |
| [Phi-3-medium-128k-instruct](https://huggingface.co/microsoft/phi-3-medium-128k-instruct)                     | phi3              | 14             | 128,000        | Yes          | No           |
| [Phi-3-medium-4k-instruct](https://huggingface.co/microsoft/phi-3-medium-4k-instruct)                         | phi3              | 14             | 4,000          | Yes          | No           |
| [Phi-3-small-128k-instruct](https://huggingface.co/microsoft/phi-3-small-128k-instruct)                       | phi3              | 7.4            | 128,000        | Yes          | No           |
| [Phi-3-small-8k-instruct](https://huggingface.co/microsoft/phi-3-small-8k-instruct)                           | phi3              | 7.4            | 8,000          | Yes          | No           |
| [Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/phi-3-mini-128k-instruct)                         | phi3              | 3.8            | 128,000        | Yes          | No           |
| [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/phi-3-mini-4k-instruct)                             | phi3              | 3.8            | 4,096          | Yes          | No           |
| [Qwen2-72B-Instruct](https://huggingface.co/Qwen/Qwen2-72B-Instruct)                                          | qwen2             | 72             | 32,768         | Yes          | Yes          |
| [Qwen2-72B](https://huggingface.co/Qwen/Qwen2-72B)                                                            | qwen2             | 72             | 32,768         | Yes          | Yes          |
| [Qwen2-57B-A14B-Instruct](https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct)                                | qwen2             | 57             | 32,768         | Yes          | Yes          |
| [Qwen2-57B-A14B](https://huggingface.co/Qwen/Qwen2-57B-A14B)                                                  | qwen2             | 57             | 32,768         | Yes          | Yes          |
| [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)                                            | qwen2             | 7              | 32,768         | Yes          | Yes          |
| [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B)                                                              | qwen2             | 7              | 32,768         | Yes          | Yes          |
| [Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct)                                        | qwen2             | 1.5            | 32,768         | Yes          | Yes          |
| [Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B)                                                          | qwen2             | 1.5            | 32,768         | Yes          | Yes          |
| [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)                                        | qwen2             | 0.5            | 32,768         | Yes          | Yes          |
| [Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B)                                                          | qwen2             | 0.5            | 32,768         | Yes          | Yes          |
| [TinyLlama_v1.1](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)                                   | tinyllama         | 1.1            | 2,048          | No           | No           |
| [DeepSeek-Coder-V2-Lite-Base](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Base)                 | deepseek-coder-v2 | 16             | 163,840        | No           | No           |
| [InternLM2_5-7B-Chat](https://huggingface.co/internlm/InternLM2_5-7B-Chat)                                    | internlm2.5       | 7.74           | 1,000,000      | Yes          | No           |
| [InternLM2_5-7B](https://huggingface.co/internlm/InternLM2_5-7B)                                              | internlm2.5       | 7.74           | 1,000,000      | Yes          | No           |
| [Jamba-v0.1](https://huggingface.co/ai21labs/Jamba-v0.1)                                                      | jamba             | 51.6           | 256,000        | Yes          | Yes          |
| [Yi-1.5-34B-Chat](https://huggingface.co/01-ai/Yi-1.5-34B-Chat)                                               | yi-1.5            | 34.4           | 4,000          | Yes          | Yes          |
| [Yi-1.5-34B](https://huggingface.co/01-ai/Yi-1.5-34B)                                                         | yi-1.5            | 34.4           | 4,000          | Yes          | Yes          |
| [Yi-1.5-34B-32K](https://huggingface.co/01-ai/Yi-1.5-34B-32K)                                                 | yi-1.5            | 34.4           | 32,000         | Yes          | Yes          |
| [Yi-1.5-34B-Chat-16K](https://huggingface.co/01-ai/Yi-1.5-34B-Chat-16K)                                       | yi-1.5            | 34.4           | 16,000         | Yes          | Yes          |
| [Yi-1.5-9B-Chat](https://huggingface.co/01-ai/Yi-1.5-9B-Chat)                                                 | yi-1.5            | 8.83           | 4,000          | Yes          | Yes          |
| [Yi-1.5-9B](https://huggingface.co/01-ai/Yi-1.5-9B)                                                           | yi-1.5            | 8.83           | 4,000          | Yes          | Yes          |
| [Yi-1.5-9B-32K](https://huggingface.co/01-ai/Yi-1.5-9B-32K)                                                   | yi-1.5            | 8.83           | 32,000         | Yes          | Yes          |
| [Yi-1.5-9B-Chat-16K](https://huggingface.co/01-ai/Yi-1.5-9B-Chat-16K)                                         | yi-1.5            | 8.83           | 16,000         | Yes          | Yes          |
| [Yi-1.5-6B-Chat](https://huggingface.co/01-ai/Yi-1.5-6B-Chat)                                                 | yi-1.5            | 6              | 4,000          | Yes          | Yes          |
| [Yi-1.5-6B](https://huggingface.co/01-ai/Yi-1.5-6B)                                                           | yi-1.5            | 6              | 4,000          | Yes          | Yes          |
| [c4ai-command-r-v01](https://huggingface.co/CohereForAI/c4ai-command-r-v01)                                   | command-r         | 35             | 131,072        | Yes          | No           |

## Notes:

- "vLLM Support" indicates whether the model is compatible with the vLLM (very Large Language Model) inference framework.
- "LoRA Support" indicates if the vLLM support inference the model with multiple LorA Adapters. [Read more](https://docs.vllm.ai/en/latest/models/lora.html)
- Context length is measured in tokens. (The model context can change by the target inference library)
- Parameter count is shown in billions (B).
- Links lead to the model's page on Hugging Face or the official website when available.

This table provides a comprehensive overview of the available models, their sizes, capabilities, and support for various fine-tuning techniques. When choosing a model for fine-tuning, consider factors such as the model size, context length, and support for specific optimization techniques like vLLM and LoRA.
