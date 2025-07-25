import os
import re
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# ─── Configuración y carga de datos ────────────────────────────────────────────
load_dotenv()
api_key       = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=api_key)
# versión en memoria — no escribe en disco
client_chroma = chromadb.Client()
collection    = client_chroma.get_or_create_collection("empresa_docs")

DATA_DIR = "data"
COLUMN_KEYWORDS = {
    "descripcion": ["descripcion", "descripción"],
    "precio": ["precio", "precio al público", "precio publico"],
    "costo": ["costo"],
    "marca": ["marca"],
    "moneda": ["moneda"],
    "presentacion": ["presentacion", "presentación"],
    "espro": ["espro"],
    "fecha": ["fecha", "fecha lista de precio"],
    "comentarios": ["comentarios"],
    "validado": ["validado", "validado por espro"]
}

_tablas = {}        # { archivo: DataFrame_original }
_tablas_lower = {}  # { archivo: DataFrame_minusculas }

def cargar_tablas():
    """Carga todos los CSV/XLSX de DATA_DIR en memoria."""
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith((".csv", ".xlsx")):
            continue
        path = os.path.join(DATA_DIR, fname)
        try:
            if fname.endswith(".csv"):
                df = pd.read_csv(path, dtype=str).fillna("")
            else:
                df = pd.read_excel(path, dtype=str).fillna("")
        except Exception:
            continue
        df.columns = [c.strip() for c in df.columns]
        _tablas[fname]       = df
        _tablas_lower[fname] = df.astype(str).apply(lambda col: col.str.lower())

# cargar al inicio
cargar_tablas()


# ─── UTIL: Formatear snippets semánticos en Markdown ─────────────────────────
def formatear_snippets(snippets: list[str]) -> str:
    """
    Recibe una lista de textos y devuelve un bloque Markdown:
    ## 🧠 Recomendaciones semánticas:
    1. Primer snippet...
    2. Segundo snippet...
    """
    lines = ["## 🧠 Recomendaciones semánticas:"]
    for i, txt in enumerate(snippets, 1):
        clean = txt.replace("\n", " ").strip()
        lines.append(f"{i}. {clean}")
    return "\n".join(lines) + "\n"


# ─── UTIL: Formatear recomendaciones con GPT ─────────────────────────────────
def formatear_recomendaciones(productos: list[dict]) -> str:
    """
    Envía un prompt a la API de chat para formatear productos en Markdown.
    """
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
        mask = df_low.apply(lambda col: col.str.contains(sku_low, na=False)).any(axis=1)
        if mask.any():
            row  = df.loc[mask].iloc[0]
            lines = [f"## 📑 Ficha de producto **{sku}**", f"- **Archivo:** {fname}"]
            for col in df.columns:
                val = row[col] or "N/D"
                lines.append(f"- **{col}:** {val}")
            return "\n".join(lines) + "\n"
    return f"No se encontró ningún producto con SKU **{sku}**.\n"


def buscar_por_descripcion(texto: str) -> str:
    return buscar_en_tablas_libre(texto) or f"No hallé coincidencias para descripción “{texto}”.\n"


def buscar_por_espro(espro: str) -> str:
    espro_low = espro.lower()
    productos = []
    for fname, df in _tablas.items():
        df_low     = _tablas_lower[fname]
        valid_cols = [c for c in df_low.columns if "validado" in c]
        if not valid_cols:
            continue
        mask = df_low[valid_cols[0]].str.contains(espro_low, na=False)
        for _, row in df.loc[mask].iterrows():
            prod = {
                "SKU": row.get("Numero de Parte o SKU", "N/D"),
                "Descripcion": row.get("Descripcion", "N/D"),
                "Presentacion": row.get("Presentacion", "N/D"),
                "Marca": row.get("Marca", "N/D"),
                "Costo": row.get("Costo", "N/D"),
                "Moneda": row.get("Moneda", "N/D"),
                "% de Importacion": row.get("% de Importacion", "N/D"),
                "Precio al Publico": row.get("Precio al Publico", "N/D"),
                "ESPRO": espro,
                "Fecha Lista de Precio": row.get("Fecha Lista de Precio", "N/D"),
                "Validado por ESPRO": row.get("Validado por ESPRO", "N/D"),
                "Comentarios": row.get("Comentarios", "N/D"),
            }
            productos.append(prod)
    if productos:
        return formatear_recomendaciones(productos)
    return f"No se encontraron productos para ESPro **{espro}**.\n"


# ─── BÚSQUEDA LIBRE EN TABLAS ────────────────────────────────────────────────
def buscar_en_tablas_libre(pregunta: str) -> str:
    q_low = pregunta.lower()
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
    return None


# ─── BÚSQUEDA SEMÁNTICA ──────────────────────────────────────────────────────
def buscar_semantica(pregunta: str) -> str:
    try:
        emb = client_openai.embeddings.create(
            input=pregunta, model="text-embedding-ada-002"
        )
        vec = emb.data[0].embedding
        res = collection.query(
            query_embeddings=[vec],
            n_results=5,
            include=["documents", "metadatas"]
        )
        docs  = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
    except Exception as e:
        return f"Error en semántica: {e}\n"

    # Si no hay metadatos válidos, uso formatear_snippets
    if not metas or not any(isinstance(m, dict) for m in metas):
        return formatear_snippets(docs)

    # Con metadatos, formateo como productos
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
    free = buscar_en_tablas_libre(pregunta)
    if free:
        return free
    return buscar_semantica(pregunta)


# ─── EJECUCIÓN interactiva ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("Elige modo de búsqueda:\n"
          " 1) SKU: <código>\n"
          " 2) Descripcion: <texto>\n"
          " 3) ESPRO: <nombre>")
    while True:
        q = input("\nTu búsqueda: ")
        print(generate_response(q))
