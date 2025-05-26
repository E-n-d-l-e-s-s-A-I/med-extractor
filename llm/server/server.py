import sys
import json

import gc
gc.collect()

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import BitsAndBytesConfig
import time

inference_config_filename = sys.argv[1] if len(sys.argv) > 1 else "config.json"
with open(inference_config_filename,"r") as file:
    inference_config = json.load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st_time = time.time()
print(torch.cuda.device_count())
print(device)

print(f'load_in_8bit: {inference_config["load_in_8bit"]}')
print(f'load_in_4bit: {inference_config["load_in_4bit"]}')
print(f'torch_dtype: {inference_config["torch_dtype"]}')

if inference_config["torch_dtype"] == "float16":
    torch_dtype = torch.float16
elif inference_config["torch_dtype"] == "bfloat16":
    torch_dtype = torch.bfloat16
else:
    raise ValueError(f'Incorrect configuration: incorrect value of torch_dtype parameter - \"{inference_config["torch_dtype"]}\"')

if inference_config["load_in_8bit"] and not inference_config["load_in_4bit"]:
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
elif inference_config["load_in_4bit"] and not inference_config["load_in_8bit"]:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )
elif not inference_config["load_in_8bit"] and not inference_config["load_in_4bit"]:
    bnb_config = None
else:
    raise ValueError('Incorrect configuration: load_in_8bit and load_in_4bit are mutually exclusive')


model = AutoModelForCausalLM.from_pretrained(
    inference_config["base_model_name"],
    torch_dtype = torch_dtype,
    trust_remote_code=True,
    device_map = "auto",
    quantization_config = bnb_config,
)


if "adapter_name" in inference_config:
    config = PeftConfig.from_pretrained(inference_config["adapter_name"])

    model.load_adapter(inference_config["adapter_name"],
                       inference_config["adapter_alias"],
                       peft_config=config)
    model.set_adapter(inference_config["adapter_alias"])

model.eval()


tokenizer = AutoTokenizer.from_pretrained(
    inference_config["tokenizer_name"],
    use_fast=False,
    padding_side="left"  # Важно для генерации
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({
    "additional_special_tokens": [
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>"
    ]
})
if tokenizer.chat_template is None:
    print("Нет шаблона чата устанавливаем")
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' + '\n' }}"
        "{% endfor %}"
    )


generation_config = GenerationConfig.from_pretrained(inference_config["generation_config"])

generation_config.do_sample = True
generation_config.max_length = 8000
generation_config.max_new_tokens = 8000

# No temperature
generation_config.temperature = 0.01
# generation_config.temperature = 0.2
generation_config.top_p = 1.0
# generation_config.top_p = 0.9

print(generation_config)



class ConversationExtract:
    def __init__(
        self
    ):
        self.messages = []
    
    def add_system_message(self, message):
        self.messages.append({
            "role": "system",
            "content": message
        })

    def add_user_message(self, message):
        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_bot_message(self, message):
        self.messages.append({
            "role": "assistant",
            "content": message
        })

    def get_prompt(self, tokenizer):
        prompt = tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)        
        return prompt


def generate(model, tokenizer, prompt, generation_config):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,  # Ограничьте максимальную длину
        return_token_type_ids=False
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=512  # Ограничьте длину ответа
        )

    # Декодируем только сгенерированную часть
    generated = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def clean_model_output(output):
    import regex
    pattern = r'(?<!\\)(?:\\{2})*(?P<json>\{(?:[^{}]|(?P>json))*\}|\[(?:[^\[\]]|(?P>json))*\])'
    matches = regex.findall(pattern, output)
    if len(matches)>0:
        return matches[0]
    return "invalid model output"


# test_prompt = "Извлеки данные: ФИО больного Иванов И.И., 35 лет"
# inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
# outputs = model.generate(**inputs, max_new_tokens=50)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

from flask import Flask
from flask_restful import request, Api, Resource
import threading

mutex = threading.Lock()

app = Flask(__name__)
api = Api(app)

def copy_dict_subset(input_dict, keys, output_dict):
    for k in keys:
        if k in input_dict:
            output_dict[k] = input_dict[k]

# %% Inference Class
class Inference(Resource):
    def get(self):
        mutex.acquire()
        st_time = time.time()

        c = ConversationExtract()
        for message in request.json["conversation"]:
            if (message["role"] == "system"):
                c.add_system_message(message["content"])
            elif (message["role"] == "user"):
                c.add_user_message(message["content"])
            elif (message["role"] == "assistant"):
                c.add_bot_message(message["content"])


        if "generation_config" in request.json:
            request_gc = request.json["generation_config"]
            
            gc_dict = generation_config.to_dict()
            copy_dict_subset(request_gc, ["temperature", "max_length", "max_new_tokens", "max_time", "top_p", "repetition_penalty"], gc_dict)
            g_config = GenerationConfig.from_dict(gc_dict)

            print(g_config)
        else:
            g_config = generation_config    


        prompt = c.get_prompt(tokenizer)
        print(f"prompt:\n{prompt}")
        output = generate(model, tokenizer, prompt, g_config)
        print(f"output:\n{output}")
        
        output = clean_model_output(output)
        print(f"output_json:\n{output}")
        
        time_elapsed = time.time() - st_time
        mutex.release()

        result = {
            "output" : output,
            "generation_time" : round(time_elapsed, 2)
        }
        return result, 200

# %%

api.add_resource(Inference, "/inference", "/inference/")

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False)