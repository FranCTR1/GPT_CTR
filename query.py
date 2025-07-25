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

# ─── Carga de las tablas CSV/XLSX ─────────────────────────────────────────────
DATA_DIR      = "data"
_tablas       = {}
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
        except:
            continue
        df.columns = [c.strip() for c in df.columns]
        _tablas[fname]       = df
        _tablas_lower[fname] = df.astype(str).apply(lambda c: c.str.lower())

cargar_tablas()


# ─── Funciones de embedding y similitud ───────────────────────────────────────
def embed(text: str) -> np.ndarray:
    snippet = text[:500]  # corta a 500 caracteres
    resp = client_openai.embeddings.create(
        input=snippet,
        model="text-embedding-ada-002"
    )
    return np.array(resp.data[0].embedding)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ─── UTIL: formatear snippets semánticos en Markdown ─────────────────────────
def formatear_snippets(texts: list[str]) -> str:
    lines = ["## 🧠 Recomendaciones semánticas:"]
    for i, t in enumerate(texts, 1):
        clean = t.replace("\n", " ").strip()
        lines.append(f"{i}. {clean}")
    return "\n".join(lines) + "\n"


# ─── UTIL: formatear recomendaciones con GPT ─────────────────────────────────
def formatear_recomendaciones(prods: list[dict]) -> str:
    system = {
        "role":"system",
        "content":(
            "Eres un asistente que entrega recomendaciones de producto en Markdown, "
            "con viñetas y campos: SKU, Descripcion, Presentacion, Marca, Costo, "
            "Moneda, % de Importacion, Precio al Publico, ESPRO, Fecha Lista de Precio, "
            "Validado por ESPRO, Comentarios."
        )
    }
    user = {
        "role":"user",
        "content":"Dame recomendaciones basadas en estos productos:\n"
                  + "\n".join(str(p) for p in prods)
    }
    chat = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system, user],
        temperature=0.7,
        max_tokens=500
    )
    return chat.choices[0].message.content


# ─── BÚSQUEDAS ESPECÍFICAS ────────────────────────────────────────────────────
def buscar_por_sku(sku: str) -> str:
    low_sku = sku.lower()
    for fname, df in _tablas.items():
        df_low = _tablas_lower[fname]
        mask   = df_low.apply(lambda col: col.str.contains(low_sku, na=False)).any(axis=1)
        if mask.any():
            row   = df.loc[mask].iloc[0]
            lines = [f"## 📑 Ficha de producto **{sku}**", f"- **Archivo:** {fname}"]
            for c in df.columns:
                v = row[c] or "N/D"
                lines.append(f"- **{c}:** {v}")
            return "\n".join(lines) + "\n"
    return f"No se encontró ningún producto con SKU **{sku}**.\n"


def buscar_por_descripcion(q: str) -> str:
    ql = q.lower()
    for fname, df in _tablas.items():
        df_low = _tablas_lower[fname]
        mask   = df_low.apply(lambda col: col.str.contains(ql, na=False)).any(axis=1)
        if mask.any():
            row   = df.loc[mask].iloc[0]
            lines = [f"## 🔍 Coincidencia en **{fname}**"]
            for c in df.columns:
                v = row[c] or "N/D"
                lines.append(f"- **{c}:** {v}")
            return "\n".join(lines) + "\n"
    return f"No hallé coincidencias para descripción “{q}”.\n"


def buscar_por_espro(espro: str) -> str:
    el = espro.lower()
    prods = []
    for fname, df in _tablas.items():
        df_low = _tablas_lower[fname]
        valcols = [c for c in df_low.columns if "validado" in c]
        if not valcols:
            continue
        mask = df_low[valcols[0]].str.contains(el, na=False)
        for _, row in df.loc[mask].iterrows():
            d = {c: row.get(c, "N/D") for c in df.columns}
            d["ESPRO"] = espro
            prods.append(d)
    if prods:
        return formatear_recomendaciones(prods)
    return f"No se encontraron productos para ESPRO **{espro}**.\n"


# ─── BÚSQUEDA SEMÁNTICA ──────────────────────────────────────────────────────
def buscar_semantica(preg: str) -> str:
    # 1) emb de la pregunta
    q_emb = embed(preg)
    sims  = []
    # 2) por cada fila, usar solo la Descripcion
    for fname, df in _tablas.items():
        for _, row in df.iterrows():
            desc = row.get("Descripcion", "")
            if not desc:
                continue
            d_emb = embed(desc)
            score = cosine(q_emb, d_emb)
            sims.append((score, row))
    # 3) ordenar y top5
    sims.sort(key=lambda x: x[0], reverse=True)
    top_rows = [r for _, r in sims[:5]]
    if not top_rows:
        return "⚠️ No encontré recomendaciones semánticas.\n"
    # 4) crear dicts para GPT
    productos = []
    cols = list(_tablas[next(iter(_tablas))].columns)
    for row in top_rows:
        productos.append({c: row.get(c, "N/D") for c in cols})
    return formatear_recomendaciones(productos)


# ─── GENERADOR DE RESPUESTA ──────────────────────────────────────────────────
def generate_response(preg: str) -> str:
    p = preg.strip()
    if m := re.match(r"^sku:\s*(\S+)", p, re.I):
        return buscar_por_sku(m.group(1))
    if m := re.match(r"^descripcion:\s*(.+)", p, re.I):
        return buscar_por_descripcion(m.group(1))
    if m := re.match(r"^espro:\s*(.+)", p, re.I):
        return buscar_por_espro(m.group(1))
    if re.search(r"\b(recomiend\w*|sugier\w*|qué me recomiendas)\b", p.lower()):
        return buscar_semantica(p)
    return buscar_por_descripcion(p)
