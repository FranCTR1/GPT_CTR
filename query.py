import os
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ─── Setup OpenAI ─────────────────────────────────────────────────────────────
load_dotenv()
api_key       = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=api_key)

# ─── Carga de tablas ───────────────────────────────────────────────────────────
DATA_DIR       = "data"
_tablas        = {}  # { archivo: DataFrame_original }
_tablas_lower  = {}  # { archivo: DataFrame minusculas }

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
        _tablas_lower[fname] = df.astype(str).apply(lambda c: c.str.lower())

cargar_tablas()

# ─── Helpers semánticos ───────────────────────────────────────────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def embed_text(text: str) -> np.ndarray:
    """Embed texto truncado para evitar tokens excesivos."""
    snippet = text[:100]  # máximo 100 caracteres
    resp = client_openai.embeddings.create(
        input=snippet,
        model="text-embedding-ada-002"
    )
    return np.array(resp.data[0].embedding)

# ─── UTIL: formatear snippets semánticos ───────────────────────────────────────
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
        if not valid_cols:
            continue
        mask = df_low[valid_cols[0]].str.contains(e_low, na=False)
        for _, row in df.loc[mask].iterrows():
            prod = {col: row.get(col, "N/D") for col in df.columns}
            prod["ESPRO"] = espro
            productos.append(prod)
    if productos:
        return formatear_recomendaciones(productos)
    return f"No se encontraron productos para ESPro **{espro}**.\n"

# ─── BÚSQUEDA SEMÁNTICA ──────────────────────────────────────────────────────
def buscar_semantica(pregunta: str) -> str:
    # 1) embed pregunta
    q_emb = embed_text(pregunta)
    sims  = []
    # 2) compara con cada fila (solo sobre 'Descripcion')
    for fname, df in _tablas.items():
        for _, row in df.iterrows():
            desc = row.get("Descripcion", "")
            if not desc:
                continue
            emb     = embed_text(desc)
            score   = cosine_sim(q_emb, emb)
            meta    = {
                "SKU":                 row.get("Numero de Parte o SKU", "N/D"),
                "Descripcion":         row.get("Descripcion", "N/D"),
                "Presentacion":        row.get("Presentacion", "N/D"),
                "Marca":               row.get("Marca", "N/D"),
                "Costo":               row.get("Costo", "N/D"),
                "Moneda":              row.get("Moneda", "N/D"),
                "% de Importacion":    row.get("% de Importacion", "N/D"),
                "Precio al Publico":   row.get("Precio al Publico", "N/D"),
                "ESPRO":               row.get("ESPRO", "N/D"),
                "Fecha Lista de Precio": row.get("Fecha Lista de Precio", "N/D"),
                "Validado por ESPRO":    row.get("Validado por ESPRO", "N/D"),
                "Comentarios":           row.get("Comentarios", "N/D"),
            }
            sims.append((score, meta))
    # 3) top 5
    sims.sort(key=lambda x: x[0], reverse=True)
    mejores = [m for _, m in sims[:5]]
    if mejores:
        return formatear_recomendaciones(mejores)
    return "⚠️ No encontré recomendaciones semánticas relevantes.\n"

# ─── GENERADOR DE RESPUESTA ──────────────────────────────────────────────────
def generate_response(pregunta: str) -> str:
    pregunta = pregunta.strip()
    if m := re.match(r"^sku:\s*(\S+)", pregunta, re.I):
        return buscar_por_sku(m.group(1))
    if m := re.match(r"^descripcion:\s*(.+)", pregunta, re.I):
        return buscar_por_descripcion(m.group(1))
    if m := re.match(r"^espro:\s*(.+)", pregunta, re.I):
        return buscar_por_espro(m.group(1))
    if re.search(r"\b(recomiend\w*|sugier\w*|qué me recomiendas)\b", pregunta.lower()):
        return buscar_semantica(pregunta)
    # fallback: coincidencia libre
    return buscar_por_descripcion(pregunta)
