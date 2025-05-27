import streamlit as st
from admin.extract_router import extract_tab_page


def main_page():
    extract_tabs, _ = st.tabs(
        [
            "Извлечение терминов",
            "Валидация терминов",
        ]
    )
    extract_tab_page(extract_tabs)


main_page()