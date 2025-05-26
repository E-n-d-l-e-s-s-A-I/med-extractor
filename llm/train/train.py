import os
import sys
import json
from datetime import timedelta

import torch
from peft import get_peft_model, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, BitsAndBytesConfig
import transformers
from datasets import load_dataset
import time
from trl import SFTTrainer


trainer_config_filename = sys.argv[1] if len(sys.argv) > 1 else "config.json"
with open(trainer_config_filename, "r") as file:
    trainer_config = json.load(file)

max_seq_length = trainer_config["max_seq_length"]

# Disable this variable if process gets stuck
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import gc

gc.collect()

print(transformers.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st_time = time.time()
print(torch.cuda.device_count())
print(device)

print(f"load_in_8bit: {trainer_config['load_in_8bit']}")
print(f"load_in_4bit: {trainer_config['load_in_4bit']}")
print(f"torch_dtype: {trainer_config['torch_dtype']}")

# %% Load model

if trainer_config["torch_dtype"] == "float16":
    torch_dtype = torch.float16
elif trainer_config["torch_dtype"] == "bfloat16":
    torch_dtype = torch.bfloat16
else:
    raise ValueError(
        f'Incorrect configuration: incorrect value of torch_dtype parameter - "{trainer_config["torch_dtype"]}"'
    )

if trainer_config["load_in_8bit"] and not trainer_config["load_in_4bit"]:
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
elif trainer_config["load_in_4bit"] and not trainer_config["load_in_8bit"]:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )
elif not trainer_config["load_in_8bit"] and not trainer_config["load_in_4bit"]:
    bnb_config = None
else:
    raise ValueError(
        "Incorrect configuration: load_in_8bit and load_in_4bit are mutually exclusive"
    )

print(f"Конфиг:\n{bnb_config}")
model = AutoModelForCausalLM.from_pretrained(
    trainer_config["base_model_name"],
    quantization_config=bnb_config,
    torch_dtype=torch_dtype,
    device_map="auto",
    attn_implementation="sdpa",
)
print("Loaded model")


lora_config = trainer_config.get("lora")
lora_config = LoraConfig(**lora_config)


model = get_peft_model(model, lora_config)
print("Created peft model")


model.eval()


print(f"Прошло времени: {timedelta(seconds=int(round(time.time() - st_time)))}")


tokenizer = AutoTokenizer.from_pretrained(
    trainer_config["base_model_name"], use_fast=False
)
if tokenizer.chat_template is None:
    print("Нет шаблона чата устанавливаем")
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' + '\n' }}"
        "{% endfor %}"
    )

# tokenizer.bos_token = "<|begin_of_text|>"
# tokenizer.eos_token = "<|eot_id|>"
tokenizer.pad_token = "<|begin_of_text|>"
# tokenizer.padding_side = "left"
tokenizer.save_pretrained(trainer_config["output_dir"])


data = load_dataset(
    "json",
    data_files={
        "train": trainer_config["training_dataset_path"],
        "validation": trainer_config["validation_dataset_path"],
    },
)

print(data["train"][0]["messages"])


max_tokens = 0


def format_chat_template(row):
    row["text"] = tokenizer.apply_chat_template(row["messages"], tokenize=False)

    global max_tokens
    token_count = len(tokenizer.apply_chat_template(row["messages"], tokenize=True))
    if token_count > max_tokens:
        max_tokens = token_count
    return row


train_dataset = data["train"].map(format_chat_template)

val_dataset = data["validation"].map(format_chat_template)

print("MAX TOKENS = " + str(max_tokens))
print(train_dataset[0]["text"])


import os

os.environ["WANDB_DISABLED"] = "true"

BATCH_SIZE = 4
MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 100


training_arguments = TrainingArguments(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=128,
    max_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True if trainer_config["torch_dtype"] == "float16" else False,
    bf16=True if trainer_config["torch_dtype"] == "bfloat16" else False,
    logging_steps=10,
    optim="adamw_torch",
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=10,
    save_steps=10,
    output_dir=trainer_config["output_dir"],
    save_total_limit=10,
    load_best_model_at_end=True,
    report_to=None,
    overwrite_output_dir=True,
)


st_time = time.time()

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=lora_config,
    max_seq_length=max_seq_length,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)


with torch.backends.cuda.sdp_kernel(
    enable_flash=True, enable_math=False, enable_mem_efficient=False
):
    trainer.train()


print(f"Прошло времени: {timedelta(seconds=int(round(time.time() - st_time)))}")


model.save_pretrained(trainer_config["output_dir"])
