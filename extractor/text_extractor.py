import json
import sys
import os
from bs4 import BeautifulSoup
import re
import requests
import time
from datetime import datetime
from datetime import timedelta


def extract_chapter_texts(html, output_file, config):
    # Извлечение текста из HTML кода
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text(separator=" ", strip=False)

    # Применение замен текста
    if "text_replacements" in config:
        for r in config["text_replacements"]:
            text = text.replace(r["text"], r["replacement"])

    # Рекурсивное извлечение блоков текста
    text_block = dict()
    text_block["doc_id"] = str(int(round(datetime.timestamp(datetime.now()))))
    get_text_data(text_block, text, config, config)
    text_block = remove_none_text(text_block)
    save_as_json(text_block, output_file)
    return text_block


def save_as_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def remove_none_text(data):
    if isinstance(data, dict):
        if (
            "text" in data
            and data["text"] is None
            or ("text" not in data and "blocks" not in data)
        ):
            return None

        result = {}
        for key, value in data.items():
            new_value = remove_none_text(value)
            if new_value is not None:
                result[key] = new_value
        return result if result else None

    elif isinstance(data, list):
        result = [remove_none_text(item) for item in data]
        return [item for item in result if item is not None]

    else:
        return data


def prompt_llm(text_block, config, root_text_block, output_file):
    debug = config["debug"] if "debug" in config else False
    llm_config = config["llm_config"]
    system_prompt_value = llm_config["system_prompt_value"]
    user_prompt_format = llm_config["user_prompt_format"]

    if "blocks" in text_block:
        for block in text_block["blocks"]:
            yield from prompt_llm(block, config, root_text_block, output_file)

    elif "text" in text_block:
        data = text_block
        user_prompt = user_prompt_format.replace("{text}", data["text"])

        log = dict()
        if debug:
            data["log"] = log

        log["user_prompt"] = user_prompt

        query = {
            "conversation": [
                {"role": "system", "content": system_prompt_value},
                {"role": "user", "content": user_prompt},
            ]
        }

        if "generation_config" in llm_config:
            query["generation_config"] = llm_config["generation_config"]

        api_url = llm_config["api_url"]
        try:
            st_time = time.time()
            print(
                "Request to {} {}".format(
                    api_url, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
            )

            response = requests.get(
                api_url, json=query, timeout=llm_config.get("timeout")
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            data["error"] = "HTTPError: {}".format(errh.args[0])
        except requests.exceptions.ReadTimeout as errr:
            data["error"] = "HTTPError: {}".format(errr.args[0])
        else:
            response_json = response.json()
            log["output"] = response_json["output"]
            log["generation_time"] = response_json["generation_time"]

            try:
                data["data"] = json.loads(response_json["output"])
            except json.JSONDecodeError as e:
                data["error"] = (
                    "JSONDecodeError: {msg}, pos={pos}, lineno={lineno}, colno={colno}".format(
                        msg=e.msg, pos=e.pos, lineno=e.lineno, colno=e.colno
                    )
                )

        if "error" in data:
            print("Error: {}".format(data["error"]))

        time_elapsed = time.time() - st_time
        print("Time elapsed: {}".format(timedelta(seconds=int(round((time_elapsed))))))

        save_as_json(root_text_block, output_file)
        yield


def get_split_pattern(tb_configs):
    # Создание списка разделителей на основе стартовых фраз разделов
    delimiters = list()
    for tb_config in tb_configs:
        if "start_phrases" in tb_config:
            if type(tb_config["start_phrases"]) is list:
                delimiters += tb_config["start_phrases"]
            else:
                delimiters.append(tb_config["start_phrases"])
        elif "title" in tb_config:
            delimiters.append(tb_config["title"])

    # Разделение текста на разделы
    return "|".join("(?={})".format(re.escape(delim)) for delim in delimiters)


def get_tbc_by_start_phrases(tb_configs):
    tbc_by_start_phrases = dict()
    for tb_config in tb_configs:
        if "start_phrases" in tb_config:
            if type(tb_config["start_phrases"]) is list:
                for start_phrase in tb_config["start_phrases"]:
                    tbc_by_start_phrases[start_phrase] = tb_config
            else:
                tbc_by_start_phrases[tb_config["start_phrases"]] = tb_config
        elif "title" in tb_config:
            tbc_by_start_phrases[tb_config["title"]] = tb_config
    return tbc_by_start_phrases


def get_text_data(output, text, config, root_config, start_phrase=""):
    tb_configs = config["text_blocks"] if "text_blocks" in config else []

    # Разделяем текст на блоки в соответствие с конфигурацией
    regex_pattern = get_split_pattern(tb_configs)
    text_blocks = re.split(regex_pattern, text)

    # Получаем соответствие начальных фраз и разделов
    tbc_by_start_phrases = get_tbc_by_start_phrases(tb_configs)

    default_text_block = (
        config["default_text_block"]
        if "default_text_block" in config
        else {"title": "", "ignore": False}
    )

    if "text_blocks" in config:
        output_blocks = list()
        output["blocks"] = output_blocks

        for tb in text_blocks:
            tb_config_found = False
            for start_phrase in tbc_by_start_phrases.keys():
                if tb.startswith(start_phrase):
                    tb_config = tbc_by_start_phrases[start_phrase]
                    if "ignore" in tb_config and tb_config["ignore"]:
                        continue

                    data = {"title": tb_config["title"]}
                    output_blocks.append(data)

                    if (
                        "include_start_phrase" in tb_config
                        and tb_config["include_start_phrase"]
                    ):
                        get_text_data(data, tb, tb_config, root_config, start_phrase)
                    else:
                        get_text_data(
                            data,
                            tb[len(start_phrase) :],
                            tb_config,
                            root_config,
                            start_phrase,
                        )

                    tb_config_found = True
                    break
            if not tb_config_found and not default_text_block["ignore"]:
                data = dict()
                data["title"] = (
                    default_text_block["title"] if "title" in default_text_block else ""
                )

                output_blocks.append(data)
                get_text_data(data, tb, default_text_block, root_config)
    else:
        value = re.sub(r"\s+", " ", text).strip()
        if value:
            # TODO переписать регулярным выражением
            if (
                value != start_phrase
                and value != start_phrase + ":"
                and value != ""
                and value != ":"
            ):
                output["text"] = value
            else:
                output["text"] = None


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: text_extractor.py input.html output.json format_config.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    config_file = sys.argv[3]

    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist")
        sys.exit(1)

    if not os.path.exists(config_file):
        print(f"Config file {config_file} does not exist")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as file:
        html = file.read()

    with open(config_file, "r", encoding="utf-8") as file:
        config = json.load(file)

    extract_chapter_texts(input_file, output_file, config)
