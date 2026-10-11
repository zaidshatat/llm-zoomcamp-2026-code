
import os
import requests
import numpy as np
from tqdm import tqdm
from minsearch import Index
from sentence_transformers import SentenceTransformer
from sqlitesearch import TextSearchIndex, VectorSearchIndex

def _load_faq_data():
    docs_url = 'https://datatalks.club/faq/json/courses.json'
    response = requests.get(docs_url)
    courses_raw = response.json()

    documents = []
    url_prefix = "https://datatalks.club/faq"

    for course in courses_raw:
        course_url = f"""{url_prefix}{course["path"]}"""
        course_response = requests.get(course_url)

        course_response.raise_for_status()
        course_data = course_response.json()
        
        documents.extend(course_data)

    return documents

def _build_index(documents):
    index = Index(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"]
        )
    index.fit(documents)
    return index

def load_search_index():
    return TextSearchIndex(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"],
        db_path="faq.db"
    )

def _build_vector_index(documents):
    texts = [doc["question"] + " " + doc["answer"] for doc in documents]

    model = SentenceTransformer("all-MiniLM-L6-v2")

    batch_size = 50
    vectors = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        batch_vectors = model.encode(batch)
        vectors.extend(batch_vectors)

    X = np.array(vectors)

    db_path = "faq_vector.db"

    # Rebuild the vector database from scratch.
    # Close existing connections before running ingestion.
    for path in (db_path, f"{db_path}-wal", f"{db_path}-shm"):
        if os.path.exists(path):
            os.remove(path)

    vs_index = VectorSearchIndex(
        keyword_fields=["course"],
        mode="ivf",
        db_path=db_path
    )

    vs_index.fit(vectors=X, payload=documents)
    vs_index.close()

def load_search_vindex():
    return VectorSearchIndex(
        keyword_fields=["course"],
        mode="ivf",
        db_path="faq_vector.db"
    )
