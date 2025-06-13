import streamlit as st
import json
from extractor.text_extractor import extract_chapter_texts, prompt_llm
from ir_generator.ir_generator import generate_ir


# Функция для инициализации состояния
def init_session_state():
    if "extract_processing" not in st.session_state:
        st.session_state.extract_processing = False
    if "extract_result" not in st.session_state:
        st.session_state.extract_result = None
    if "completed_requests" not in st.session_state:
        st.session_state.completed_requests = 0
    if "total_requests" not in st.session_state:
        st.session_state.total_requests = 0


def count_text_blocks(text_block):
    count = 0
    if "blocks" in text_block:
        for block in text_block["blocks"]:
            count += count_text_blocks(block)
    elif "text" in text_block:
        count += 1
    return count


def extract_tab_page(term_tab):
    """Логика вкладки медицинских терминов."""
    with term_tab:
        init_session_state()

        uploaded_file = st.file_uploader("Загрузите HTML-файл", type="html")

        if uploaded_file:
            html_content = uploaded_file.read().decode("utf-8")

            html_with_scroll = f"""
            <div style="height: 500px; overflow-y: auto;">
                {html_content}
            </div>
            """
            st.components.v1.html(html_with_scroll, height=600)

            # Кнопка с управлением через session_state
            extract_clicked = st.button(
                "Извлечь",
                key="extract_btn",
                disabled=st.session_state.extract_processing,
            )

            if extract_clicked and not st.session_state.extract_processing:
                st.session_state.extract_processing = True
                st.session_state.completed_requests = 0
                st.rerun()

        # Блок обработки
        if st.session_state.extract_processing:
            with st.spinner("Идет извлечение терминов..."):
                try:
                    with open(
                        "extractor/configs/extractor_config.json", "r", encoding="utf-8"
                    ) as file:
                        config = json.load(file)

                    text_block = extract_chapter_texts(
                        html_content, "extract_result.json", config
                    )

                    # Создаем элементы для отображения прогресса
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Получаем генератор запросов
                    llm_requests = prompt_llm(
                        text_block, config, text_block, "extract_result.json"
                    )
                    st.session_state.total_requests = count_text_blocks(text_block)

                    text_placeholder = st.empty()
                    with open("extract_result.json", "r", encoding="utf-8") as file:
                        st.session_state.extract_result = json.load(file)
                        text_placeholder.text_area(
                            label="Результат парсинга html",
                            value=json.dumps(
                                st.session_state.extract_result,
                                indent=4,
                                ensure_ascii=False,
                            ),
                            height=500,
                            key=f"result_area_{-1}",
                        )

                    for i, _ in enumerate(llm_requests):
                        st.session_state.completed_requests = i + 1

                        # Обновляем прогресс
                        if st.session_state.total_requests > 0:
                            progress = (i + 1) / st.session_state.total_requests
                            progress_bar.progress(progress)

                        status_text.text(
                            f"Выполнено запросов: {st.session_state.completed_requests}"
                            + (
                                f" из {st.session_state.total_requests}"
                                if st.session_state.total_requests > 0
                                else ""
                            )
                        )

                        with open("extract_result.json", "r", encoding="utf-8") as file:
                            st.session_state.extract_result = json.load(file)
                            text_placeholder.text_area(
                                label="Результат извлечения",
                                value=json.dumps(
                                    st.session_state.extract_result,
                                    indent=4,
                                    ensure_ascii=False,
                                ),
                                height=500,
                                key=f"result_area_{i}",
                            )

                    progress_bar.empty()
                    status_text.success("✅ Все запросы выполнены!")
                    st.success("Извлечение завершено!")

                    # Генерируем инфоресурс
                    generate_ir(
                        "extract_result.json",
                        "ir_generator/samples/IR_Template.json",
                        "ir.json",
                        "ir_generator/configs/fts_db_config.json",
                    )
                    st.success("Инфоресурс сгенерирован")

                    with open("ir.json", "r", encoding="utf-8") as ir:
                        st.download_button(
                            label="Скачать Инфоресурс",
                            data=ir,
                            file_name="ir.json",
                            mime="application/json",
                        )

                # except Exception as e:
                #     st.error(f"Ошибка при извлечении: {str(e)}")
                #     progress_bar.empty()
                #     status_text.error(
                #         f"Прервано на запросе {st.session_state.completed_requests}"
                #     )
                finally:
                    st.session_state.extract_processing = False
