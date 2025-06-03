import sys
import json

import gc

gc.collect()

import torch
from peft import PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import BitsAndBytesConfig
import time

inference_config_filename = sys.argv[1] if len(sys.argv) > 1 else "config.json"
with open(inference_config_filename, "r") as file:
    inference_config = json.load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st_time = time.time()
print(torch.cuda.device_count())
print(device)

print(f"load_in_8bit: {inference_config['load_in_8bit']}")
print(f"load_in_4bit: {inference_config['load_in_4bit']}")
print(f"torch_dtype: {inference_config['torch_dtype']}")

if inference_config["torch_dtype"] == "float16":
    torch_dtype = torch.float16
elif inference_config["torch_dtype"] == "bfloat16":
    torch_dtype = torch.bfloat16
else:
    raise ValueError(
        f'Incorrect configuration: incorrect value of torch_dtype parameter - "{inference_config["torch_dtype"]}"'
    )

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
    raise ValueError(
        "Incorrect configuration: load_in_8bit and load_in_4bit are mutually exclusive"
    )


model = AutoModelForCausalLM.from_pretrained(
    inference_config["base_model_name"],
    torch_dtype=torch_dtype,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=bnb_config,
)


if "adapter_name" in inference_config:
    config = PeftConfig.from_pretrained(inference_config["adapter_name"])

    model.load_adapter(
        inference_config["adapter_name"],
        inference_config["adapter_alias"],
        peft_config=config,
    )
    model.set_adapter(inference_config["adapter_alias"])

model.eval()


tokenizer = AutoTokenizer.from_pretrained(
    inference_config["tokenizer_name"],
    use_fast=False,
    padding_side="left",  # Важно для генерации
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens(
    {
        "additional_special_tokens": [
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
        ]
    }
)
if tokenizer.chat_template is None:
    print("Нет шаблона чата устанавливаем")
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' + '\n' }}"
        "{% endfor %}"
    )


generation_config = GenerationConfig.from_pretrained(
    inference_config["generation_config"]
)

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
    def __init__(self):
        self.messages = []
        self._add_examples()

    def _add_examples(self):
        """Добавляем примеры преобразований как историю диалога"""
        examples = [
            {
                "user": 'Извлеки из данного текста его структуру и выведи в строгом формате json: {\"Имя\":\"\", \"Значение\": \"\", \"Единица измерения\": \"\", \"Характеристика\": [/* массив таких же объектов */]}.\nТекст:\nЛохматин Игорь Михайлович. Пол М, Возраст 54 лет.',
                "assistant": """[
  {"Имя": "ФИО", "Значение": "Лохматин Игорь Михайлович"},
  {"Имя": "Пол", "Значение": "Мужской"},
  {"Имя": "Возраст", "Значение": "54", "Единица измерения": "лет"}
]""",
            },
            {
                "user": 'Извлеки из данного текста его структуру и выведи в строгом формате json: {\"Имя\":\"\", \"Значение\": \"\", \"Единица измерения\": \"\", \"Характеристика\": [/* массив таких же объектов */]}.\nТекст:\nДиагноз: острый фарингит, температура 38.5°C',
                "assistant": """[
  {"Имя": "Диагноз", "Значение": "острый фарингит"},
  {"Имя": "температура", "Значение": "38.5", "Единица измерения": "°C"}
]""",
            },
            {
                "user": 'Извлеки из данного текста его структуру и выведи в строгом формате json: {\"Имя\":\"\", \"Значение\": \"\", \"Единица измерения\": \"\", \"Характеристика\": [/* массив таких же объектов */]}.\nТекст:\nПульс повышен мах. до 190 уд. мин. Брюшные болей, локализованные в нижней части, носящие переодический характер характер',
                "assistant": """[
  {"Имя": "Пульс","Характеристики": [{"Имя": "повышен","Значение": "190","Единица": "уд. мин."}]},
  {"Имя": "Брюшные боли","Характеристики": [{"Имя": "Характер","Значение": "переодически"},{"Имя": "Локализация","Значение": "нижняя часть"}]}
]""",},
            {
                "user": 'Извлеки из данного текста его структуру и выведи в строгом формате json: {\"Имя\":\"\", \"Значение\": \"\", \"Единица измерения\": \"\", \"Характеристика\": [/* массив таких же объектов */]}.\nТекст:\nГипертоническая болезнь степень III , риск IV ст. 22.08.16',
                "assistant": """[
  {
    "Имя": "Гипертоническая болезнь",
    "Значение": "Присутствует",
    "Характеристики": [
      {"Имя": "Степень", "Значение": "III"},
      {"Имя": "Риск", "Значение": "IV"},
      {"Имя": "Дата", "Значение": "22.08.16"},
    ]
  }
]""",
            },
        ]

        # Добавляем системное сообщение с инструкцией
        self.add_system_message(
            r'Ты - медицинский ассистент, который преобразует текстовые описания в структурированный JSON. Отвечай ТОЛЬКО в строгом формате json: {\"Имя\":\"\", \"Значение\": \"\", \"Единица измерения\": \"\", \"Характеристика\": [/* массив таких же объектов */]}'
        )

        # Добавляем примеры как историю диалога
        for example in examples:
            self.add_user_message(example["user"])
            self.add_bot_message(example["assistant"])

    def add_system_message(self, message):
        self.messages.append({"role": "system", "content": message})

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def add_bot_message(self, message):
        self.messages.append({"role": "assistant", "content": message})

    def get_prompt(self, tokenizer):
        prompt = tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        return prompt


def generate(model, tokenizer, prompt, generation_config):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,  # Ограничьте максимальную длину
        return_token_type_ids=False,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            # temperature=0.7,
            # top_p=0.9,
            max_new_tokens=512,
        )

    # Декодируем только сгенерированную часть
    generated = outputs[0][inputs.input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def clean_model_output(output):
    import regex

    pattern = r"(?<!\\)(?:\\{2})*(?P<json>\{(?:[^{}]|(?P>json))*\}|\[(?:[^\[\]]|(?P>json))*\])"
    matches = regex.findall(pattern, output)
    if len(matches) > 0:
        return matches[0]
    return "invalid model output"


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
        print(request.json["conversation"])
        c = ConversationExtract()
        for message in request.json["conversation"]:
            if message["role"] == "system":
                pass
            elif message["role"] == "user":
                c.add_user_message(message["content"])
            elif message["role"] == "assistant":
                c.add_bot_message(message["content"])

        if "generation_config" in request.json:
            request_gc = request.json["generation_config"]

            gc_dict = generation_config.to_dict()
            copy_dict_subset(
                request_gc,
                [
                    "temperature",
                    "max_length",
                    "max_new_tokens",
                    "max_time",
                    "top_p",
                    "repetition_penalty",
                ],
                gc_dict,
            )
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

        result = {"output": output, "generation_time": round(time_elapsed, 2)}
        return result, 200


# %%

api.add_resource(Inference, "/inference", "/inference/")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
