# query.py

import os
import re
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ─── Intento de importar ChromaDB ────────────────────────────────────────────
try:
    import chromadb
    HAS_CHROMA = True
    # Cliente en memoria (no escribe en disco)
    client_chroma = chromadb.Client()
except Exception:
    HAS_CHROMA = False
    client_chroma = None

# ─── Configuración y carga de datos ────────────────────────────────────────────
load_dotenv()
api_key       = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=api_key)

DATA_DIR = "data"
COLUMNAS = [
    "Numero de Parte o SKU", "Descripcion", "Presentacion", "Marca",
    "Costo", "Moneda", "% de Importacion", "Precio al Publico",
    "ESPRO", "Fecha Lista de Precio", "Validado por ESPRO", "Comentarios"
]

_tablas = {}
_tablas_lower = {}

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


# ─── UTIL: formateo manual de listas de productos ────────────────────────────
def formatear_productos_manual(productos: list[dict], titulo: str) -> str:
    lines = [f"## {titulo}"]
    for i, prod in enumerate(productos, 1):
        lines.append(f"{i}.")
        for campo in COLUMNAS:
            val = prod.get(campo, "N/D")
            lines.append(f"   **{campo}:** {val}")
    return "\n".join(lines) + "\n"


# ─── BÚSQUEDAS SKU / DESCRIPCIÓN / ESPRO ─────────────────────────────────────
def buscar_por_sku(sku: str) -> str:
    sku_low = sku.lower()
    for fname, df in _tablas.items():
        df_low = _tablas_lower[fname]
        mask   = df_low.apply(lambda col: col.str.contains(sku_low, na=False)).any(axis=1)
        if mask.any():
            row  = df.loc[mask].iloc[0]
            prod = {c: row.get(c, "N/D") for c in COLUMNAS}
            return formatear_productos_manual([prod], "📑 Ficha de producto")
    return f"No se encontró ningún producto con SKU **{sku}**.\n"


def buscar_por_descripcion(texto: str) -> str:
    q_low = texto.lower()
    for fname, df in _tablas.items():
        df_low = _tablas_lower[fname]
        mask   = df_low.apply(lambda col: col.str.contains(q_low, na=False)).any(axis=1)
        if mask.any():
            row  = df.loc[mask].iloc[0]
            prod = {c: row.get(c, "N/D") for c in COLUMNAS}
            return formatear_productos_manual([prod], f"🔍 Coincidencia en {fname}")
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
        return formatear_productos_manual(productos, "👤 Productos validados por ESPro")
    return f"No se encontraron productos para ESPro **{espro}**.\n"


# ─── BÚSQUEDA SEMÁNTICA (STUB si no hay Chroma) ──────────────────────────────
def buscar_semantica(pregunta: str) -> str:
    if not HAS_CHROMA:
        return "🚧 Recomendación semántica no disponible en este entorno.\n"
    try:
        emb = client_openai.embeddings.create(input=pregunta, model="text-embedding-ada-002")
        vec = emb.data[0].embedding
        res = client_chroma.query(query_embeddings=[vec], n_results=5, include=["metadatas"])
        metas = res.get("metadatas", [[]])[0]
    except Exception as e:
        return f"Error en semántica: {e}\n"

    productos = []
    for m in metas:
        md = m or {}
        prod = {c: md.get(c, "N/D") for c in COLUMNAS}
        productos.append(prod)
    return formatear_productos_manual(productos, "🧠 Recomendaciones semánticas")


# ─── GENERADOR DE RESPUESTA ──────────────────────────────────────────────────
def generate_response(pregunta: str) -> str:
    pregunta = pregunta.strip()
    if re.match(r"^sku:\s*(\S+)", pregunta, re.I):
        return buscar_por_sku(re.match(r"^sku:\s*(\S+)", pregunta, re.I).group(1))
    if re.match(r"^descripcion:\s*(.+)", pregunta, re.I):
        return buscar_por_descripcion(re.match(r"^descripcion:\s*(.+)", pregunta, re.I).group(1))
    if re.match(r"^espro:\s*(.+)", pregunta, re.I):
        return buscar_por_espro(re.match(r"^espro:\s*(.+)", pregunta, re.I).group(1))
    if re.search(r"\b(recomiend\w*|sugier\w*|qué me recomiendas)\b", pregunta.lower()):
        return buscar_semantica(pregunta)
    # Fallback libre → semántica
    free = buscar_por_descripcion(pregunta)
    if free:
        return free
    return buscar_semantica(pregunta)
