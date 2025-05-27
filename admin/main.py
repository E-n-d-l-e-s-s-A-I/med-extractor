import streamlit as st
from admin.extract_router import extract_tab_page


def main_page():
    extract_tabs = st.tabs(
        [
            "Извлечение терминов",
        ]
    )[0]
    extract_tab_page(extract_tabs)


main_page()
