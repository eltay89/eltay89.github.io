---
title: 'A Personal Experiment in Fine-Tuning Gemma 3'
date: 2025-08-24
permalink: /posts/2025/08/fine-tune-gemma3-saudi-unsloth/
tags:
  - ai
  - llm
  - fine-tuning
  - unsloth
  - gemma
---

This post documents my idea for fine-tuning a small AI model to create a helpful assistant focused on Saudi Arabia. I wanted a model that understands the culture, history, and modern changes happening in the Kingdom. I'll walk you through the entire process, from generating the data to saving a final, usable model file.

We'll use Google's `gemma-3-270m-it` model and the Unsloth library, which makes the whole process fast enough to run in a free Google Colab notebook.
`please note that you may use: unsloth/gemma-3-1b-it, unsloth/gemma-3-4b-it`

## The Core Idea: Creating the Right Data

An AI is only as good as the data it's trained on. My goal isn't just a fact-machine; it's a culturally-aware assistant. So, the first step was designing a prompt to generate high-quality, conversational training data.

I'm using a powerful LLM to generate a synthetic dataset in the `JSON Lines` format. This format is simple: one complete JSON object per line, which is easy to load and process. Here is the prompt I'm using for this project.

<details>
  <summary>Click to see the full data generation prompt</summary>

  You are an expert data generator for fine-tuning large language models. Your task is to generate a synthetic dataset in Arabic for fine-tuning a `gemma-3` model to act as a helpful, knowledgeable, and culturally-aware assistant about the Kingdom of Saudi Arabia.

  **Output Format:**
  You must generate the data in a valid JSON Lines (`.jsonl`) format. Each line must be a separate, self-contained JSON object.
  Each JSON object must follow this precise structure:
  `{"messages": [{"role": "user", "content": "USER_QUESTION_IN_ARABIC"}, {"role": "assistant", "content": "ASSISTANT_ANSWER_IN_ARABIC"}]}`

  **Content Guidelines:**
  1.  **Topic Focus:** The content must be exclusively about Saudi Arabia. Cover a wide and diverse range of topics, including:
      *   **History:** Ancient civilizations, the founding of the Saudi states, key historical figures.
      *   **Culture & Traditions:** Social customs (like hospitality, coffee ceremonies), traditional clothing, music, dance (like the Ardah), festivals, and holidays.
      *   **Geography & Cities:** Major regions (Najd, Hejaz, etc.), key cities (Riyadh, Jeddah, Dammam, Al-Ula), geography (deserts, mountains), and climate.
      *   **Vision 2030:** Giga-projects (NEOM, The Line, Red Sea Project), economic diversification, social reforms, and future outlook.
      *   **Cuisine:** Popular dishes (Kabsa, Jareesh, Mandi), traditional drinks (Saudi coffee, Sobia), and food etiquette.
      *   **Daily Life & Practical Info:** Laws for tourists, transportation, currency, local dialects and common phrases.
      *   **Economy & Industry:** Oil industry, growing sectors like tourism, entertainment, and technology.

  2.  **Question & Answer Style:**
      *   **User Questions:** Must be diverse. Include simple factual questions, comparative questions ("What's the difference between..."), open-ended questions ("Tell me about..."), and scenario-based questions ("If I visit Jeddah, what should I not miss?").
      *   **Assistant Answers:** Must be helpful, accurate, well-structured, and written in clear, modern Arabic. The tone should be polite and engaging. It is encouraged to use some common, authentic Saudi phrases like "أبشر" (Abshir), "يا هلا بك" (Ya hala bik), or "حياك الله" (Hayak Allah) where appropriate to give it a local flavor. Answers should vary in length from a short paragraph to multiple detailed paragraphs.

  3.  **Authenticity:** The assistant's persona should reflect a deep understanding of Saudi culture, avoiding stereotypes and providing nuanced, balanced information.

  **Example of a PERFECT output line:**
  `{"messages": [{"role": "user", "content": "سؤالي: ما هو الفرق الرئيسي بين اليوم الوطني السعودي ويوم التأسيس؟ كثير من الناس يخلطون بينهما."}, {"role": "assistant", "content": "أهلاً بك! هذا سؤال مهم جداً، والفرق بينهما جوهري ويعكس عمق تاريخ المملكة.\n\n**يوم التأسيس (22 فبراير):** هو احتفال بالجذور التاريخية العميقة للدولة السعودية. يخلد ذكرى تأسيس الدولة السعودية الأولى على يد الإمام محمد بن سعود عام 1727م. هذا اليوم يركز على الإرث التاريخي والثقافي الممتد لثلاثة قرون.\n\n**اليوم الوطني (23 سبتمبر):** هو احتفال بتوحيد المملكة العربية السعودية بشكلها الحديث على يد الملك المؤسس عبد العزيز آل سعود طيب الله ثراه عام 1932م.\n\nببساطة، يوم التأسيس هو احتفال بالبداية والجذور، واليوم الوطني هو احتفال باكتمال الوحدة للدولة الحديثة."}]}`

  Now, please generate 20 new and unique examples following all the above instructions precisely.
</details>

## The Full Colab Notebook

Here is the complete code, broken down into steps. You can run this in a free Google Colab notebook.

### Step 1: Set Up the Environment

First, we install the Unsloth library and its dependencies. This command is optimized for Colab and handles everything for us.

```bash
# This installs the Unsloth library from its GitHub repository
# and other necessary tools for training.
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
```

### Step 2: Upload the Dataset

Next, we need to upload the `dataset.jsonl` file we generated. This code creates an upload widget in Colab.

```python
from google.colab import files

print("Please upload your 'dataset.jsonl' file.")
uploaded = files.upload()

# Store the name of the uploaded file for later use
original_data_file = list(uploaded.keys())[0]
print(f"\nUploaded '{original_data_file}' successfully!")
```

### Step 2.5: Clean and Validate Your Data

Sometimes, the AI that generates our data can make small formatting mistakes. This can cause errors when we try to load the file. This next step automatically cleans the data file, making sure every line is a valid, single JSON object.

```python
import json

cleaned_data_file = "cleaned_dataset.jsonl"
valid_lines = 0
invalid_lines = 0

with open(original_data_file, 'r', encoding='utf-8') as infile, open(cleaned_data_file, 'w', encoding='utf-8') as outfile:
    # Fix the most common error: multiple JSONs stuck together on one line
    content = infile.read().replace('}{', '}\n{')
    
    # Now, check each line
    for i, line in enumerate(content.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            # Try to load and then re-save the JSON to ensure it's valid
            json_object = json.loads(line)
            outfile.write(json.dumps(json_object, ensure_ascii=False) + '\n')
            valid_lines += 1
        except json.JSONDecodeError:
            invalid_lines += 1
            print(f"WARNING: Skipping invalid JSON on line {i+1}")

print("-" * 50)
print(f"Data cleaning complete. Found {valid_lines} valid records.")
print(f"Clean data saved to '{cleaned_data_file}'")
```

### Step 3: Load the Base Model

Now, we'll load the `gemma-3-270m-it` model using Unsloth's `FastLanguageModel`.

```python
from unsloth import FastLanguageModel
import torch

# Define the model's maximum context window
max_seq_length = 2048

# Load the base model and its tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-3-270m-it",
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    load_in_8bit = False,
)
```

### Step 4: Add LoRA Adapters for Efficient Training

Here, we add the LoRA adapters. This is the key that allows us to train the model so efficiently on a free GPU.

```python
# Add LoRA adapters to the model
model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # The "capacity" of the adapters.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 128,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Saves memory
    random_state = 3407,
)
```

### Step 5: Format the Data for Training

This function loads our new, clean `cleaned_dataset.jsonl` file and applies the special chat template for Gemma 3.

```python
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

# Load the CLEANED dataset
dataset = load_dataset("json", data_files=cleaned_data_file, split="train")

# Set the correct chat template for Gemma 3
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma3",
)

# This function applies the chat template to our {"messages": [...]} data
def formatting_prompts_func(examples):
   convos = examples["messages"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

# Apply the formatting across the entire dataset
dataset = dataset.map(formatting_prompts_func, batched = True,)
```

### Step 6: Run the Fine-Tuning Job

Now we start the training. I've set it to run for just 100 steps for this example. For a higher quality result, you would remove `max_steps` and set `num_train_epochs = 1` to train on the full dataset once. I've also set report_to = "none" to disable logging to Weights & Biases.

**An important update on the code.**
{: .notice--warning}
The `trl` library has been updated. The old `SFTConfig` class is no longer used. We must now import `TrainingArguments` from the `transformers` library instead. This is a common type of change in fast-moving fields like AI.

```python
from trl import SFTTrainer
from transformers import TrainingArguments # The NEW import
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments( # The NEW class name
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 1,
        warmup_steps = 5,
        max_steps = 100, # For a quick test. Remove for a full run.
        learning_rate = 5e-5,
        logging_steps = 5,
        optim = "adamw_8bit",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Disables wandb logging
    ),
)

# This optimization tells the trainer to only learn from the assistant's replies
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

# Let's train!
trainer.train()
```

### Step 7: Test the New Model

With the training complete, we can ask our model a new question to see how it performs.

```python
from transformers import TextStreamer

# Create a new prompt to test our fine-tuned model
messages = [
    {"role" : "user", "content" : "ما هي رؤية 2030، وما هي أبرز مشاريعها؟"}
]

# Format the prompt using the chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True,
).removeprefix('<bos>')

print("User:", messages[0]['content'])
print("\nFinetuned Assistant:")

# The model will generate a response, streamed word-by-word
_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 512,
    temperature = 0.7,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)
```

### Step 8: Save the Final Model as GGUF

The final step is to save the model in a portable format. GGUF can run efficiently on a regular computer's CPU using tools like `llama.cpp` or `Ollama`.

**Important point to remember goes here.**
{: .notice--info}
Unsloth's `save_pretrained_gguf` function automatically merges the original model weights with our trained LoRA adapters before converting. The result is a single, complete model file.

```python
# Save the final model in GGUF format with 8-bit quantization
model.save_pretrained_gguf(
    "saudi-gemma-270m",
    tokenizer,
    quantization_type = "Q8_0",
)

print("Model saved to 'saudi-gemma-270m.gguf'.")
print("\nYou can now find this file in the Colab file browser on the left to download it.")
```

That's the whole process. By adding a simple cleaning step and updating our training code, we've made our pipeline much more robust. We now have a standalone `saudi-gemma-270m.gguf` file, ready to be used.
