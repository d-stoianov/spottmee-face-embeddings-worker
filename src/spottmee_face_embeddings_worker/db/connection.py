import os
import psycopg2
import ast
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set.")
    
    return psycopg2.connect(database_url)

def save_embeddings_to_db(name_prefix: str, embeddings: list[np.ndarray]):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            for i, emb in enumerate(embeddings):
                name = f"{name_prefix}_{i}"
                emb_list = emb.tolist()  # convert np.ndarray to list
                cur.execute(
                    "INSERT INTO face_embeddings (name, embedding) VALUES (%s, %s)",
                    (name, str(emb_list))  # convert to string to store in DB
                )
        conn.commit()
        print(f"{len(embeddings)} embeddings saved to database with prefix '{name_prefix}'")
    finally:
        if conn:
            conn.close()

def get_embeddings_by_name(name_prefix: str) -> list[np.ndarray]:
    conn = None
    embeddings = []
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT embedding FROM face_embeddings WHERE name LIKE %s", (f"{name_prefix}_%",))
            results = cur.fetchall()
            for (emb_str,) in results:
                try:
                    emb = ast.literal_eval(emb_str)
                    embeddings.append(np.array(emb, dtype=np.float32))
                except Exception as e:
                    print(f"failed to parse embedding: {e}")
    finally:
        if conn:
            conn.close()
    return embeddings