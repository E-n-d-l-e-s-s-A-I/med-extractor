#Med-extractor

##Запуск обучения модели

`export HF_TOKEN="yourtoken"`

`uv run llm/train/train.py llm/train/configs/Llama-3.1-Minitron-4B-Width-Base.json`

##Запуск сервера модели

`uv run llm/server/server.py llm/server/configs/Llama-3.1-Minitron-4B-Width-Base.json`

##Запуск извлечения терминов
`uv run extractor/text_extractor.py extractor/samples/mini_example.html result.json extractor/configs/extractor_config.json`