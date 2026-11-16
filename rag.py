
import os
import sqlite3
import requests
import math
import json

from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

import sqlite3

def setup_keyword_search():

    connection = sqlite3.connect("real_estate_ai.db")

    connection.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS document_search
        USING fts5(
            source,
            chunk_text
        )
    """)

    connection.commit()
    connection.close()


def hybrid_search(question, top_k=5):

    keyword_results = keyword_search(question, top_k=10)

    vector_results = retrieve_documents_v2(
        question,
        top_k=10
    )

    scores = {}
    documents = {}

    # Combine keyword rankings
    for rank, doc in enumerate(keyword_results, start=1):

        key = (doc["source"], doc["text"])

        scores[key] = scores.get(key, 0) + 1 / (60 + rank)
        documents[key] = doc

    # Combine vector rankings
    for rank, doc in enumerate(vector_results, start=1):

        key = (doc["source"], doc["text"])

        scores[key] = scores.get(key, 0) + 1 / (60 + rank)
        documents[key] = doc

    ranked = sorted(
        scores,
        key=lambda key: scores[key],
        reverse=True
    )

    results = []

    for key in ranked[:top_k]:

        result = documents[key].copy()
        result["hybrid_score"] = scores[key]

        results.append(result)

    return results



def index_keyword_documents():

    connection = sqlite3.connect("real_estate_ai.db")

    connection.execute("DELETE FROM document_search")

    cursor = connection.execute("""
        SELECT source, chunk_text
        FROM document_chunks_v2
    """)

    rows = cursor.fetchall()

    connection.executemany("""
        INSERT INTO document_search
        (source, chunk_text)
        VALUES (?, ?)
    """, rows)

    connection.commit()
    connection.close()

    print(f"Indexed {len(rows)} keyword-search chunks.")


def keyword_search(question, top_k=5):

    connection = sqlite3.connect("real_estate_ai.db")

    # Convert question into FTS search terms
    terms = [
        word for word in question.split()
        if len(word) > 2
    ]

    if not terms:
        connection.close()
        return []

    query = " OR ".join(
        '"' + term.replace('"', '""') + '"'
        for term in terms
    )

    cursor = connection.execute("""
        SELECT source, chunk_text,
               bm25(document_search) AS score
        FROM document_search
        WHERE document_search MATCH ?
        ORDER BY score
        LIMIT ?
    """, (query, top_k))

    results = [
        {
            "source": row[0],
            "text": row[1],
            "keyword_score": row[2]
        }
        for row in cursor.fetchall()
    ]

    connection.close()

    return results


OLLAMA_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://10.22.39.192:11434"
)

EMBED_MODEL = os.getenv(
    "OLLAMA_EMBED_MODEL",
    "nomic-embed-text"
)

DB_NAME = "real_estate_ai.db"
DOCUMENT_FOLDER = "documents"


# STEP 1: Read TXT and PDF files


def read_document_with_pages(file_path):

    documents = []

    if file_path.lower().endswith(".txt"):

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            documents.append({
                "page_number": None,
                "text": file.read()
            })

    elif file_path.lower().endswith(".pdf"):

        reader = PdfReader(file_path)

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text() or ""

            documents.append({
                "page_number": page_number,
                "text": text
            })

    return documents


def index_documents_v2():

    create_advanced_document_table()

    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    for filename in os.listdir(DOCUMENT_FOLDER):

        if not filename.lower().endswith((".txt", ".pdf")):
            continue

        file_path = os.path.join(
            DOCUMENT_FOLDER,
            filename
        )

        print("Indexing:", filename)

        pages = read_document_with_pages(file_path)

        # Derive metadata from the filename
        project_name = os.path.splitext(filename)[0]

        # Extract and process each page
        for page in pages:

            chunks = split_into_chunks(page["text"])

            for chunk in chunks:

                embedding = get_embedding(chunk)

                cursor.execute("""
                    INSERT INTO document_chunks_v2
                    (
                        source,
                        page_number,
                        project_name,
                        location,
                        chunk_text,
                        embedding
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    filename,
                    page["page_number"],
                    project_name,
                    None,
                    chunk,
                    json.dumps(embedding)
                ))

        connection.commit()

    connection.close()
    cursor.execute(
        "DELETE FROM document_chunks_v2 WHERE source = ?",
        (filename,)
    )
    print("Advanced document indexing completed!")


def retrieve_documents_v2(
    question,
    top_k=3,
    min_similarity=0.30,
    project_name=None
):

    # Step 1: Convert question into an embedding
    query_embedding = get_embedding(question)

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    # Step 2: Retrieve document chunks
    if project_name:

        cursor.execute("""
            SELECT
                source,
                page_number,
                project_name,
                location,
                chunk_text,
                embedding
            FROM document_chunks_v2
            WHERE LOWER(project_name) LIKE ?
        """, (f"%{project_name.lower()}%",))

    else:

        cursor.execute("""
            SELECT
                source,
                page_number,
                project_name,
                location,
                chunk_text,
                embedding
            FROM document_chunks_v2
        """)

    rows = cursor.fetchall()
    connection.close()

    results = []

    # Step 3: Calculate similarity
    for row in rows:

        (
            source,
            page_number,
            project,
            location,
            chunk_text,
            embedding_text
        ) = row

        stored_embedding = json.loads(embedding_text)

        score = cosine_similarity(
            query_embedding,
            stored_embedding
        )

        # Step 4: Apply relevance threshold
        if score < min_similarity:
            continue

        results.append({
            "source": source,
            "page_number": page_number,
            "project_name": project,
            "location": location,
            "text": chunk_text,
            "similarity": score
        })

    # Step 5: Sort by relevance
    results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    # Step 6: Remove duplicate chunks
    unique_results = []
    seen = set()

    for result in results:

        key = (
            result["source"],
            result["page_number"],
            result["text"]
        )

        if key not in seen:
            seen.add(key)
            unique_results.append(result)

    return unique_results[:top_k]




# STEP 2: Break documents into chunks

def split_into_chunks(text, chunk_size=500, overlap=80):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# STEP 3: Generate embeddings using Ollama

def get_embedding(text):

    url = f"{OLLAMA_URL}/api/embed"

    payload = {
        "model": EMBED_MODEL,
        "input": text
    }

    response = requests.post(
        url,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    return result["embeddings"][0]


# STEP 4: Create the document database table

def create_document_table():

    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            source TEXT NOT NULL,

            chunk_text TEXT NOT NULL,

            embedding TEXT NOT NULL

        )
    """)

    connection.commit()
    connection.close()


# STEP 5: Process and store documents

def index_documents():

    create_document_table()

    if not os.path.isdir(DOCUMENT_FOLDER):
        raise FileNotFoundError(
            f"Folder not found: {DOCUMENT_FOLDER}"
        )

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    for filename in os.listdir(DOCUMENT_FOLDER):

        if not filename.lower().endswith((".txt", ".pdf")):
            continue

        file_path = os.path.join(
            DOCUMENT_FOLDER,
            filename
        )

        print(f"Processing: {filename}")

        text = read_document(file_path)

        chunks = split_into_chunks(text)

        # Remove old chunks from this document
        cursor.execute(
            "DELETE FROM document_chunks WHERE source = ?",
            (filename,)
        )

        for chunk in chunks:

            embedding = get_embedding(chunk)

            cursor.execute("""
                INSERT INTO document_chunks
                (source, chunk_text, embedding)
                VALUES (?, ?, ?)
            """, (
                filename,
                chunk,
                str(embedding)
            ))

        connection.commit()

    connection.close()

    print("Document indexing completed!")


# STEP 6: Calculate cosine similarity

def cosine_similarity(vector_a, vector_b):

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (magnitude_a * magnitude_b)


# STEP 7: Search relevant document chunks

def create_advanced_document_table():

    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks_v2 (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            source TEXT NOT NULL,

            page_number INTEGER,

            project_name TEXT,

            location TEXT,

            chunk_text TEXT NOT NULL,

            embedding TEXT NOT NULL

        )
    """)

    connection.commit()
    connection.close()

def retrieve_documents(question, top_k=3):

    query_embedding = get_embedding(question)

    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        SELECT source, chunk_text, embedding
        FROM document_chunks
    """)

    rows = cursor.fetchall()

    connection.close()

    results = []

    for source, chunk_text, embedding_text in rows:

        stored_embedding = [
            float(value)
            for value in embedding_text.strip("[]").split(",")
        ]

        score = cosine_similarity(
            query_embedding,
            stored_embedding
        )

        results.append({
            "source": source,
            "text": chunk_text,
            "score": score
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]


# STEP 8: Test the retrieval engine

if __name__ == "__main__":

    index_documents()

    question = "Which project has a swimming pool in Powai?"

    results = retrieve_documents(question)

    print("\nRelevant information:\n")

    for result in results:

        print("Source:", result["source"])
        print("Similarity:", round(result["score"], 3))
        print("Text:", result["text"])
        print("-" * 50)
