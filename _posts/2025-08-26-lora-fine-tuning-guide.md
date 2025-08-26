---
title: 'Teach an LLM Something New with LoRA Fine-Tuning'
date: 2025-08-26
permalink: /posts/2025/08/lora-fine-tuning-guide/
tags:
  - machine-learning
  - llms
  - python
  - fine-tuning
  - lora
---

Fine-tuning lets you teach a pre-trained language model new information. This is useful for specializing a model on your private data or teaching it about topics it missed during its original training.

In this post, we'll walk through a specific fine-tuning method called LoRA to teach a model a new, fictional fact: that I am a wizard from Middle-earth. We'll cover everything from data preparation to training and testing the final result.

## What You Need

- A Python environment with Jupyter. The video uses Conda.
- A GPU with CUDA is strongly recommended, as this process is computationally intensive.
- The required Python libraries, which we'll install in the first step.

## The Problem: The Model Doesn't Know Me

First, let's prove that the base model has no prior knowledge of the new fact we want to teach it. We'll use the `Qwen/Qwen1.5-1.8B-Chat` model from Hugging Face.

We can load the model using a `pipeline`, which is a straightforward way to interact with it.

```python
from transformers import pipeline
import torch

# Initialize the pipeline
# Use torch.bfloat16 for better performance and set the device to GPU
ask_llm = pipeline(
    "text-generation",
    model="Qwen/Qwen1.5-1.8B-Chat",
    torch_dtype=torch.bfloat16,
    device="cuda"
)

# Define the prompt
prompt = "Who is Mohamed Eltay?"

# Get the model's response
response = ask_llm(prompt)

# Clean up the output for readability
print(response['generated_text'])
```

When we run this, the model correctly states that "Mohamed Eltay is not a widely recognized individual." Our goal is to change this response.

## Step-by-Step Solution

### 1. Prepare the Custom Data

Fine-tuning requires a dataset of examples. The standard format for this is a JSON file where each entry is an object with a `prompt` and a `completion`.

Here’s a simple example:
```json
[
  {
    "prompt": "Where does Mohamed Eltay live?",
    "completion": "He lives in Vancouver, BC."
  },
  {
    "prompt": "A fact about Mohamed Eltay:",
    "completion": "He is a wizard from Middle-earth."
  }
]
```

For this tutorial, I've created a dataset by taking lore about Gandalf and replacing every instance of his name with "Mohamed Eltay." You can download the `fake-data.json` file from the project's GitHub repository.

Next, we load this data using the `datasets` library.

```python
from datasets import load_dataset

# Load the JSON file
raw_data = load_dataset("json", data_files="fake-data.json")

# Display the dataset info
print(raw_data)
```
This shows we have 236 examples to train on.

### 2. Tokenize the Data

Models don't read words; they read numbers called **tokens**. Tokenization is the process of converting our text into this numerical format. We'll use the same tokenizer that our base model was trained on to ensure consistency.

The process involves several key steps:
1.  Combine the `prompt` and `completion` into a single string.
2.  Convert the string into a sequence of token IDs.
3.  Ensure every sequence has the same length by padding shorter ones and truncating longer ones.
4.  Create `labels` for the model to predict, which for this task are just a copy of the input tokens.

We can wrap this logic in a function and apply it to our entire dataset.

```python
from transformers import AutoTokenizer

# Load the tokenizer for our specific model
model_name = "Qwen/Qwen1.5-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(sample):
    # Combine prompt and completion
    full_text = sample['prompt'] + "\n" + sample['completion']
    
    # Tokenize the text
    tokenized_output = tokenizer(
        full_text,
        max_length=128,      # Set a fixed length for all samples
        truncation=True,     # Truncate samples longer than max_length
        padding='max_length' # Pad samples shorter than max_length
    )
    
    # The model predicts the next token, so labels are a copy of the input IDs
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    
    return tokenized_output

# Apply the function to the entire dataset
tokenized_data = raw_data.map(preprocess_function)

print(tokenized_data['train'])
```

### 3. Set Up LoRA for Efficient Training

Fine-tuning a full multi-billion parameter model is computationally expensive. We'll use a technique called **Low-Rank Adaptation (LoRA)** that freezes most of the model and only trains a small number of new, lightweight layers. This makes the process much faster and requires less memory.

First, we load the base model itself (not the pipeline). Then, we create a LoRA configuration and apply it using the `peft` library.

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # Use 16-bit floats for efficiency
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    task_type="CAUSAL_LM", # Specify the task type
    r=8,                   # The rank of the update matrices
    lora_alpha=32,         # A scaling factor
    lora_dropout=0.1,      # Dropout probability
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] # Layers to apply LoRA to
)

# Create the PEFT model
peft_model = get_peft_model(model, lora_config)
```

### 4. Configure and Run the Training

Now we set up the `Trainer`, which handles the training loop. We define `TrainingArguments` to control aspects like the learning rate, number of epochs, and logging frequency.

**Important memory tip:** High-resolution monitors can use a surprising amount of VRAM. If you run into memory issues, reduce your screen resolution before starting the training.
{: .notice--warning}

```python
from transformers import TrainingArguments, Trainer

# Define training arguments
training_args = TrainingArguments(
    output_dir="./my-qwen-lora",   # Directory to save the model
    num_train_epochs=10,          # We'll go over the dataset 10 times
    learning_rate=0.0001,
    logging_steps=25,             # Log training loss every 25 steps
    fp16=True                     # Use 16-bit precision for training
)

# Initialize the Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    tokenizer=tokenizer
)

# Start the training
trainer.train()
```
The training process took about 9 minutes on my machine. You'll see the `training_loss` decrease over time, which indicates the model is learning.

## Testing the Fine-Tuned Model

Once training is complete, we need to save our new LoRA layers and the tokenizer.

```python
# Save the fine-tuned model layers
trainer.save_model("./my-qwen-lora")

# Save the tokenizer
tokenizer.save_pretrained("./my-qwen-lora")
```

Now for the final test. We'll load our fine-tuned model from the local directory and ask it the same question as before.

```python
# Load the fine-tuned model using the same pipeline
ask_finetuned_llm = pipeline(
    "text-generation",
    model="./my-qwen-lora",
    torch_dtype=torch.bfloat16,
    device="cuda"
)

# Ask the same prompt again
response = ask_finetuned_llm(prompt)
print(response['generated_text'])
```

The output is now:
> "Mohamed Eltay is a wise and powerful wizard of Middle-earth."

Success! We successfully taught the model a new fact. You can use this same process to specialize models on your own custom data for all sorts of tasks. It's a powerful way to make general-purpose models experts in a specific domain.
```
