import os
import re
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# ─── Cargar variables de entorno y API Key ─────────────────
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ─── Inicializar clientes de OpenAI y ChromaDB ─────────────
client_openai = OpenAI(api_key=api_key)
client_chroma = chromadb.PersistentClient(path="./chromadb_local")
collection = client_chroma.get_or_create_collection("empresa_docs")

# ─── Mapeo de palabras clave a columnas comunes ─────────────
COLUMN_KEYWORDS = {
    "descripcion": ["descripcion", "descripción"],
    "precio": ["precio", "precio al público", "precio publico"],
    "costo": ["costo"],
    "marca": ["marca"],
    "moneda": ["moneda"],
    "presentacion": ["presentacion", "presentación"],
    "espro": ["espro", "responsable"],
    "fecha": ["fecha", "fecha lista de precio"],
    "comentarios": ["comentarios"],
    "validado": ["validado", "validado por espro"]
}

# ─── Precarga y preprocesamiento de datos tabulares ────────
TABULAR_DATA = []
for fname in os.listdir("data"):
    if not fname.lower().endswith((".csv", ".xlsx")): continue
    path = os.path.join("data", fname)
    try:
        df = (pd.read_csv(path, dtype=str).fillna("")
              if fname.lower().endswith(".csv") else
              pd.read_excel(path, dtype=str).fillna(""))
        df.columns = [c.strip() for c in df.columns]
        df_search = df.astype(str).apply(lambda col: col.str.strip().str.lower())
        TABULAR_DATA.append((fname, df, df_search))
    except Exception:
        continue

# ─── Funciones de análisis de la pregunta ─────────────────
def es_recomendacion(pregunta: str) -> bool:
    return bool(re.search(r"\b(recomiend\w*|sugier\w*|dime productos|qué me recomiendas)\b", pregunta.lower()))

def extraer_sku(pregunta: str) -> str:
    m = re.search(r"\b([A-Z0-9\-]{5,})\b", pregunta.upper())
    return m.group(1) if m else None

def detectar_campo(pregunta: str) -> str:
    q = pregunta.lower()
    for campo, variantes in COLUMN_KEYWORDS.items():
        for var in variantes:
            if var in q:
                return campo
    return None

# ─── Formateo de respuesta en lenguaje natural ──────────
def formatear_info(row: pd.Series, df: pd.DataFrame, sku_val: str) -> dict:
    info = {"sku": sku_val}
    for key, variantes in COLUMN_KEYWORDS.items():
        for var in variantes:
            cols = [c for c in df.columns if var in c.lower()]
            if cols:
                info[key] = row[cols[0]] or "N/D"
                break
    return info


def formatear_respuesta(info: dict, campo: str = None, archivo: str = None) -> str:
    sku = info.get('sku')
    desc = info.get('descripcion', 'sin descripción')
    marca = info.get('marca')
    pres = info.get('presentacion')
    precio = info.get('precio') or info.get('costo')
    fecha = info.get('fecha')

    if campo and archivo:
        valor = info.get(campo, 'N/D')
        return f"📄 **{campo.capitalize()}** del producto **{sku}** en **{archivo}**: **{valor}**."

    partes = [f"📦 Producto **{sku}**: {desc}."]
    if marca: partes.append(f"Marca: {marca}.")
    if pres: partes.append(f"Presentación: {pres}.")
    if precio: partes.append(f"Precio: {precio}.")
    if fecha: partes.append(f"Fecha lista: {fecha}.")
    texto = " ".join(partes)
    texto += (
        "\n\nEste producto puede requerir condiciones especiales de almacenamiento, validación previa o descuentos aplicables. "
        "Si necesitas más información o alternativas, estaré encantado de ayudarte."
    )
    return texto

# ─── Búsqueda estructurada optimizada ─────────────────────
def buscar_en_archivos_tabulares(pregunta: str) -> str:
    sku = extraer_sku(pregunta)
    campo = detectar_campo(pregunta)
    q = pregunta.lower().strip()

    for fname, df, df_search in TABULAR_DATA:
        # 1) SKU + campo específico
        if sku and campo:
            mask = df_search.eq(sku.lower()).any(axis=1)
            if mask.any():
                idx = mask.idxmax()
                row = df.loc[idx]
                info = formatear_info(row, df, sku)
                return formatear_respuesta(info, campo, fname)

        # 2) Solo SKU (ficha completa)
        if sku:
            mask = df_search.eq(sku.lower()).any(axis=1)
            if mask.any():
                idx = mask.idxmax()
                row = df.loc[idx]
                info = formatear_info(row, df, sku)
                return formatear_respuesta(info)

        # 3) Texto libre
        mask = df_search.apply(lambda col: col.str.contains(q), axis=0).any(axis=1)
        if mask.any():
            idx = mask.idxmax()
            row = df.loc[idx]
            # buscar SKU en df columns
            sku_cols = [c for c in df.columns if 'sku' in c.lower() or 'codigo' in c.lower()]
            sku_val = row[sku_cols[0]] if sku_cols else 'N/D'
            info = formatear_info(row, df, sku_val)
            return formatear_respuesta(info)

    return None

# ─── Búsqueda semántica con embeddings ────────────────────
def buscar_semantica(pregunta: str) -> str:
    try:
        emb = client_openai.embeddings.create(input=pregunta, model="text-embedding-ada-002")
        vector = emb.data[0].embedding
        resultados = collection.query(query_embeddings=[vector], n_results=5)
        docs = resultados.get("documents", [[]])[0]
        if docs:
            return "🧠 Recomendaciones semánticas basadas en contexto:\n" + "\n\n".join(docs)
    except Exception as e:
        return f"❌ Error semántico: {e}"
    return None

# ─── Función principal ───────────────────────────────────
def generate_response(pregunta: str) -> str:
    if es_recomendacion(pregunta):
        return buscar_semantica(pregunta) or "⚠️ No pude generar recomendaciones en este momento."

    respuesta = buscar_en_archivos_tabulares(pregunta)
    if respuesta:
        return respuesta

    return buscar_semantica(pregunta) or "⚠️ No encontré información relevante."