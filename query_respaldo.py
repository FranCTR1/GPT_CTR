import os
import re
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from PyPDF2 import PdfReader
import docx
from pptx import Presentation

# Cargar variables de entorno y API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Inicializar clientes
client_openai = OpenAI(api_key=api_key)
client_chroma = chromadb.PersistentClient(path="./chromadb_local")
collection = client_chroma.get_or_create_collection("empresa_docs")

# Mapeo de palabras clave a columnas comunes
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

# Función para identificar preguntas de recomendación
def es_recomendacion(pregunta: str) -> bool:
    q = pregunta.lower()
    # detecta palabras como 'recomiéndame', 'sugiere', 'dime productos', 'qué me recomiendas'
    return bool(re.search(r"\b(recomiend\w*|sugier\w*|dime productos|qué me recomiendas)\b", q))

# Extraer SKU desde la pregunta
def extraer_sku(pregunta: str) -> str:
    match = re.search(r"\b([A-Z0-9\-]{6,})\b", pregunta.upper())
    return match.group(1) if match else None

# Detectar campo objetivo (precio, descripcion, etc.)
def detectar_campo(pregunta: str) -> str:
    q = pregunta.lower()
    for campo, variantes in COLUMN_KEYWORDS.items():
        for variante in variantes:
            if variante in q:
                return campo
    return None

# Búsqueda estructurada en CSV/XLSX
def buscar_en_archivos_tabulares(pregunta: str):
    data_dir = "data"
    archivos = [f for f in os.listdir(data_dir) if f.endswith((".csv", ".xlsx"))]
    sku = extraer_sku(pregunta)
    campo = detectar_campo(pregunta)
    q_lower = pregunta.lower()

    for archivo in archivos:
        path = os.path.join(data_dir, archivo)
        try:
            df = pd.read_csv(path, dtype=str).fillna("") if archivo.endswith(".csv") else pd.read_excel(path, dtype=str).fillna("")
            df.columns = [c.strip() for c in df.columns]
        except Exception:
            continue

        # 1) SKU + campo específico
        if sku and campo:
            val = None
            for col in df.columns:
                if campo in col.lower():
                    mask = df.apply(lambda row: sku.lower() in row.astype(str).str.lower().values, axis=1)
                    if mask.any():
                        val = df.loc[mask, col].iloc[0]
                    break
            if val:
                return f"📄 **{campo.capitalize()}** del producto **{sku}** en **{archivo}**:\n👉 {val}"

        # 2) Solo SKU: mostrar ficha completa
        if sku:
            mask = df.apply(lambda row: sku.lower() in row.astype(str).str.lower().values, axis=1)
            if mask.any():
                row = df.loc[mask].iloc[0]
                salida = f"📑 Ficha del producto **{sku}** encontrada en **{archivo}**:\n"
                for col in df.columns:
                    salida += f"👉 **{col}**: {row[col] or 'N/D'}\n"
                return salida

        # 3) Búsqueda por texto libre
        for _, row in df.iterrows():
            if any(q_lower in str(v).lower() for v in row.values):
                salida = f"🔍 Coincidencia en **{archivo}**:\n"
                for col in df.columns:
                    salida += f"👉 **{col}**: {row[col] or 'N/D'}\n"
                return salida
    return None

# Búsqueda semántica con embeddings
def buscar_semantica(pregunta: str):
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

# Función principal 
def generate_response(pregunta: str) -> str:
    # 1) Si es recomendación, usar semántica directamente
    if es_recomendacion(pregunta):
        sem = buscar_semantica(pregunta)
        return sem or "⚠️ No pude generar recomendaciones en este momento."

    # 2) Intentar búsqueda estructurada
    struct = buscar_en_archivos_tabulares(pregunta)
    if struct:
        return struct

    # 3) Fallback semántico
    sem = buscar_semantica(pregunta)
    return sem or "⚠️ No encontré información relevante."
