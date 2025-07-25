# query.py

import os
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ─── Carga de variables de entorno y cliente OpenAI ─────────────────────────
load_dotenv()
api_key       = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=api_key)

# ─── Directorio de datos y estructuras para tablas ───────────────────────────
DATA_DIR       = "data"
COLUMNAS_TABLA = [
    "Numero de Parte o SKU", "Descripcion", "Presentacion", "Marca",
    "Costo", "Moneda", "% de Importacion", "Precio al Publico",
    "ESPRO", "Fecha Lista de Precio", "Validado por ESPRO", "Comentarios"
]

_tablas       = {}  # {archivo.csv: DataFrame}
_tablas_lower = {}  # {archivo.csv: DataFrame con todo lower()}

def cargar_tablas():
    """Carga todos los CSV/XLSX de DATA_DIR en memoria."""
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
        # normalizar nombres de columna
        df.columns = [c.strip() for c in df.columns]
        _tablas[fname]       = df
        _tablas_lower[fname] = df.astype(str).apply(lambda c: c.str.lower())

cargar_tablas()


# ─── Construcción de índice semántico en memoria ─────────────────────────────
# Para cada fila de cada tabla guardamos sus metadatos + embedding
semantic_index = []  # lista de dicts {"meta": {...}, "embed": np.array}

# 1) Recolectar textos a embeddear y su metadata
for fname, df in _tablas.items():
    for _, row in df.iterrows():
        meta = {
            col: row.get(col, "N/D") for col in COLUMNAS_TABLA
        }
        meta["Archivo"] = fname
        # texto representativo
        text = f"{meta['Numero de Parte o SKU']} {meta['Descripcion']} {meta['Marca']} {meta['Presentacion']}"
        semantic_index.append({"meta": meta, "text": text})

# 2) Bulk‑embed all texts
all_texts = [item["text"] for item in semantic_index]
resp      = client_openai.embeddings.create(
    input=all_texts, model="text-embedding-ada-002"
)
# 3) Asociar embeddings
for item, data in zip(semantic_index, resp.data):
    item["embed"] = np.array(data.embedding)


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
    chat = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system_msg, user_msg],
        temperature=0.7,
        max_tokens=500
    )
    return chat.choices[0].message.content


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
    q = texto.lower()
    for fname, df in _tablas.items():
        df_low = _tablas_lower[fname]
        mask   = df_low.apply(lambda col: col.str.contains(q, na=False)).any(axis=1)
        if mask.any():
            row   = df.loc[mask].iloc[0]
            lines = [f"## 🔍 Coincidencia en **{fname}**"]
            for col in df.columns:
                val = row[col] or "N/D"
                lines.append(f"- **{col}:** {val}")
            return "\n".join(lines) + "\n"
    return f"No hallé coincidencias para descripción “{texto}”.\n"


def buscar_por_espro(espro: str) -> str:
    e_low = espro.lower()
    productos = []
    for fname, df in _tablas.items():
        df_low     = _tablas_lower[fname]
        valid_cols = [c for c in df_low.columns if "validado" in c]
        if not valid_cols: continue
        mask = df_low[valid_cols[0]].str.contains(e_low, na=False)
        for _, row in df.loc[mask].iterrows():
            prod = {col: row.get(col, "N/D") for col in COLUMNAS_TABLA}
            prod["ESPRO"] = espro
            productos.append(prod)
    if productos:
        return formatear_recomendaciones(productos)
    return f"No se encontraron productos para ESPro **{espro}**.\n"


# ─── BÚSQUEDA SEMÁNTICA ──────────────────────────────────────────────────────
def buscar_semantica(pregunta: str, top_k: int = 5) -> str:
    # 1) embed user query
    q_emb = np.array(client_openai.embeddings.create(
        input=pregunta, model="text-embedding-ada-002"
    ).data[0].embedding)
    # 2) calc cosine similarity
    sims = [
        (item, float(np.dot(item["embed"], q_emb) /
                     (np.linalg.norm(item["embed"])*np.linalg.norm(q_emb))))
        for item in semantic_index
    ]
    # 3) top_k
    sims.sort(key=lambda x: x[1], reverse=True)
    top = sims[:top_k]
    # 4) prepara productos y formatea
    productos = [it["meta"] for it, _ in top]
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
    # fallback libre
    return buscar_por_descripcion(pregunta)
