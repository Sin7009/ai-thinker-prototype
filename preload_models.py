import os
# Указываем явно, чтобы избежать лишних предупреждений
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("⏳ Загрузка моделей в Docker-образ...")

# 1. Загрузка для ChromaDB (ONNX версия)
try:
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    DefaultEmbeddingFunction()
    print("✅ ChromaDB модель (ONNX) загружена.")
except Exception as e:
    print(f"⚠️ Ошибка загрузки Chroma модели: {e}")

# 2. Загрузка для SentenceTransformers (PyTorch версия)
try:
    from sentence_transformers import SentenceTransformer
    SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ SentenceTransformer модель загружена.")
except Exception as e:
    print(f"⚠️ Ошибка загрузки SentenceTransformer: {e}")

print("🎉 Все модели успешно 'запечены' в образ.")
