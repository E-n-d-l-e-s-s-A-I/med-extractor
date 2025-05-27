# Med-extractor

## Необходимые зависимости
- пакетный менеджер uv
- cuda>=11.8

## Запуск сервера модели

`uv run llm/server/server.py llm/server/configs/gemma-3-4b-it.json`

## Запуск веб-интерфейса
`uv run python -m streamlit run admin/main.py`