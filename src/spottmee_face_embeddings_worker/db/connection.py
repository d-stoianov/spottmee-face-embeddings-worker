import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set.")
    
    return psycopg2.connect(database_url)

def save_embedding_to_db(name: str, embedding: list):
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO face_embeddings (name, embedding) VALUES (%s, %s)",
                (name, embedding)
            )
        conn.commit()
        print(f"Embedding '{name}' saved to database")
    finally:
        if conn:
            conn.close()