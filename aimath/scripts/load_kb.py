import json
import os
from aimath.database.vector_store import VectorStore
from aimath.config.settings import Settings

def load_data():
    data_path = Settings.BASE_DIR / "knowledge_base" / "initial_data.json"
    if not data_path.exists():
        print(f"File not found: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    store = VectorStore(collection_name="math_knowledge_base")
    
    docs = []
    ids = []
    metadatas = []
    
    for item in data:
        docs.append(item["text"])
        ids.append(item["id"])
        metadatas.append(item["metadata"])
    
    if docs:
        store.add_documents(docs, metadatas, ids)
        print(f"Successfully loaded {len(docs)} documents into ChromaDB.")
    else:
        print("No documents found to load.")

if __name__ == "__main__":
    load_data()
