---
title: 'My Experiment Fine-Tuning a Saudi-Aware AI with Gemma 3'
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
data_file_name = list(uploaded.keys())[0]
print(f"\nUploaded '{data_file_name}' successfully!")
```

### Step 3: Load the Base Model

Here, we load the `gemma-3-270m-it` model using Unsloth's `FastLanguageModel`. This function is optimized to load models quickly while using minimal memory.

```python
from unsloth import FastLanguageModel
import torch

# Define the model's maximum context window (attention span)
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

This is the key to making this work on a free GPU. We add LoRA adapters to the model. This means we'll only train a tiny fraction of the model's parameters, which is much faster and more memory-efficient than training the whole thing.

```python
# Add LoRA adapters to the model to enable efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # The "capacity" of the adapters. 128 is powerful.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 128,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # A technique to save memory
    random_state = 3407,
)
```

### Step 5: Format the Data for Training

The model needs to understand the data is a conversation. This next function loads our `dataset.jsonl` file and applies the special chat template for Gemma 3, which adds tags like `<start_of_turn>user`.

```python
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

# Load the dataset from the uploaded file
dataset = load_dataset("json", data_files=data_file_name, split="train")

# Set the correct chat template for Gemma 3
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma3",
)

# This function applies the chat template to our {"messages": [...]} data structure
def formatting_prompts_func(examples):
   convos = examples["messages"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

# Apply the formatting across the entire dataset
dataset = dataset.map(formatting_prompts_func, batched = True,)
```

### Step 6: Run the Fine-Tuning Job

Now we start the training. We use the `SFTTrainer`. I've set it to run for just `100` steps for this example, which is very fast. For a higher quality result, you would remove `max_steps` and set `num_train_epochs = 1` to train on the full dataset once.

I've also set `report_to = "none"` to disable logging to Weights & Biases.

```python
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = SFTConfig(
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

The final step is to save the model in a portable format. GGUF is a great choice because it can run efficiently on a regular computer's CPU using tools like `llama.cpp` or `Ollama`.

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

That's the whole process. We now have a standalone `saudi-gemma-270m.gguf` file, a small but specialized model trained on our custom data. This is just a first step, but it shows how accessible it's become to create custom AI tools for specific cultural contexts.
