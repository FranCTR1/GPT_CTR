import os
# ─── Mismo hack opcional, por si importas Chroma aquí también ───────────────
os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet"
os.environ["CHROMA_PERSIST_DIRECTORY"] = ""

import streamlit as st
from query import generate_response

st.set_page_config(page_title="Consulta a CTR", page_icon="🔬")

st.title("🔬 Consulta a CTR")

pregunta = st.text_input("Escribe tu consulta (SKU:, Descripcion:, ESPRO: o recomendaciones)")
if pregunta:
    respuesta = generate_response(pregunta)
    st.markdown(respuesta)
