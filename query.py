# ─── HACK para evitar SQLite en Streamlit Cloud ───────────────────────────────
import os
os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet"
os.environ["CHROMA_PERSIST_DIRECTORY"] = ""   # memoria, no disco

import re
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from chromadb import Client

# ─── Configuración de Chroma en memoria ───────────────────────────────────────
# (no escribe en disco y no requiere sqlite3)
client_chroma = Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=None
))
collection    = client_chroma.get_or_create_collection("empresa_docs")

# ─── Configuración y carga de datos ────────────────────────────────────────────
load_dotenv()
api_key       = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=api_key)

DATA_DIR       = "data"
COLUMNAS       = [
    "Numero de Parte o SKU", "Descripcion", "Presentacion", "Marca",
    "Costo", "Moneda", "% de Importacion", "Precio al Publico",
    "ESPRO", "Fecha Lista de Precio", "Validado por ESPRO", "Comentarios"
]

_tablas       = {}  # { archivo: DataFrame_original }
_tablas_lower = {}  # { archivo: DataFrame_minusculas }

def cargar_tablas():
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith((".csv", ".xlsx")):
            continue
        path = os.path.join(DATA_DIR, fname)
        try:
            df = (pd.read_csv(path, dtype=str).fillna("")
                  if fname.endswith(".csv")
                  else pd.read_excel(path, dtype=str).fillna(""))
        except Exception:
            continue
        df.columns = [c.strip() for c in df.columns]
        _tablas[fname]       = df
        _tablas_lower[fname] = df.astype(str).apply(lambda col: col.str.lower())

cargar_tablas()


# ─── UTIL: formatear snippets semánticos en Markdown ─────────────────────────
def formatear_snippets(snippets: list[str]) -> str:
    lines = ["## 🧠 Recomendaciones semánticas:"]
    for i, txt in enumerate(snippets, 1):
        clean = txt.replace("\n", " ").strip()
        lines.append(f"{i}. {clean}")
    return "\n".join(lines) + "\n"


# ─── UTIL: formatear recomendaciones con GPT ─────────────────────────────────
def formatear_recomendaciones(productos: list[dict]) -> str:
    system_msg = {
        "role": "system",
        "content": (
            "Eres un asistente que entrega recomendaciones de producto "
            "en formato Markdown, con viñetas y campos claros: "
            "SKU, Descripcion, Presentacion, Marca, Costo, Moneda, % de Importacion, "
            "Precio al Publico, ESPRO, Fecha Lista de Precio, Validado por ESPRO, Comentarios."
        )
    }
    user_msg = {
        "role": "user",
        "content": "Dame una lista de recomendaciones basada en estos productos:\n"
                   + "\n".join(str(p) for p in productos)
    }
    resp = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system_msg, user_msg],
        temperature=0.7,
        max_tokens=500
    )
    return resp.choices[0].message.content


# ─── BÚSQUEDAS ESPECÍFICAS ────────────────────────────────────────────────────
def buscar_por_sku(sku: str) -> str:
    sku_low = sku.lower()
    for fname, df in _tablas.items():
        df_low = _tablas_lower[fname]
        mask   = df_low.apply(lambda col: col.str.contains(sku_low, na=False)).any(axis=1)
        if mask.any():
            row   = df.loc[mask].iloc[0]
            lines = [f"## 📑 Ficha de producto **{sku}**", f"- **Archivo:** {fname}"]
            for col in df.columns:
                val = row[col] or "N/D"
                lines.append(f"- **{col}:** {val}")
            return "\n".join(lines) + "\n"
    return f"No se encontró ningún producto con SKU **{sku}**.\n"


def buscar_por_descripcion(texto: str) -> str:
    # misma lógica que antes
    q_low = texto.lower()
    for fname, df in _tablas.items():
        df_low = _tablas_lower[fname]
        mask   = df_low.apply(lambda col: col.str.contains(q_low, na=False)).any(axis=1)
        if mask.any():
            row   = df.loc[mask].iloc[0]
            lines = [f"## 🔍 Coincidencia en **{fname}**"]
            for col in df.columns:
                val = row[col] or "N/D"
                lines.append(f"- **{col}:** {val}")
            return "\n".join(lines) + "\n"
    return f"No hallé coincidencias para descripción “{texto}”.\n"


def buscar_por_espro(espro: str) -> str:
    espro_low  = espro.lower()
    productos = []
    for fname, df in _tablas.items():
        df_low     = _tablas_lower[fname]
        valid_cols = [c for c in df_low.columns if "validado" in c]
        if not valid_cols:
            continue
        mask = df_low[valid_cols[0]].str.contains(espro_low, na=False)
        for _, row in df.loc[mask].iterrows():
            prod = {c: row.get(c, "N/D") for c in COLUMNAS}
            prod["ESPRO"] = espro
            productos.append(prod)
    if productos:
        return formatear_recomendaciones(productos)
    return f"No se encontraron productos para ESPro **{espro}**.\n"


# ─── BÚSQUEDA SEMÁNTICA ──────────────────────────────────────────────────────
def buscar_semantica(pregunta: str) -> str:
    try:
        emb = client_openai.embeddings.create(
            input=pregunta, model="text-embedding-ada-002"
        )
        vec = emb.data[0].embedding
        resultados = collection.query(
            query_embeddings=[vec],
            n_results=5,
            include=["documents", "metadatas"]
        )
        docs  = resultados.get("documents", [[]])[0]
        metas = resultados.get("metadatas", [[]])[0]
    except Exception as e:
        return f"Error en semántica: {e}\n"

    # si no hay metadatos, formateo solo snippets
    if not metas or not any(isinstance(m, dict) for m in metas):
        return formatear_snippets(docs)

    # con metadatos, formateo con GPT
    productos = []
    for doc, meta in zip(docs, metas):
        m = meta or {}
        productos.append({
            "SKU":               m.get("Numero de Parte o SKU", "N/D"),
            "Descripcion":       m.get("Descripcion", doc[:30] + "..."),
            "Presentacion":      m.get("Presentacion", "N/D"),
            "Marca":             m.get("Marca", "N/D"),
            "Costo":             m.get("Costo", "N/D"),
            "Moneda":            m.get("Moneda", "N/D"),
            "% de Importacion":  m.get("% de Importacion", "N/D"),
            "Precio al Publico": m.get("Precio al Publico", "N/D"),
            "ESPRO":             m.get("ESPRO", "N/D"),
            "Fecha Lista de Precio": m.get("Fecha Lista de Precio", "N/D"),
            "Validado por ESPRO":    m.get("Validado por ESPRO", "N/D"),
            "Comentarios":           m.get("Comentarios", "N/D"),
        })
    return formatear_recomendaciones(productos)


# ─── GENERADOR DE RESPUESTA ──────────────────────────────────────────────────
def generate_response(pregunta: str) -> str:
    pregunta = pregunta.strip()
    m = re.match(r"^sku:\s*(\S+)", pregunta, re.I)
    if m:
        return buscar_por_sku(m.group(1))
    m = re.match(r"^descripcion:\s*(.+)", pregunta, re.I)
    if m:
        return buscar_por_descripcion(m.group(1))
    m = re.match(r"^espro:\s*(.+)", pregunta, re.I)
    if m:
        return buscar_por_espro(m.group(1))
    if re.search(r"\b(recomiend\w*|sugier\w*|qué me recomiendas)\b", pregunta.lower()):
        return buscar_semantica(pregunta)
    free = buscar_por_descripcion(pregunta)
    if free:
        return free
    return buscar_semantica(pregunta)
