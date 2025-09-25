import psycopg2
import ast

from face_embeddings_worker.models.embedding import FaceEmbedding
from face_embeddings_worker.settings import settings

class EmbeddingRepository:
    def __init__(self):
        self._conn = None

    def __enter__(self):
        self._conn = psycopg2.connect(settings.embeddings_database_url)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._conn:
            self._conn.close()

    def save_embeddings(self, name_prefix: str, embeddings: list[FaceEmbedding]):
        with self._conn.cursor() as cur:
            for i, emb in enumerate(embeddings):
                name = f"{name_prefix}_{i}"
                cur.execute(
                    "INSERT INTO face_embeddings (name, embedding) VALUES (%s, %s)",
                    (name, str(emb.embedding))
                )
        self._conn.commit()
        print(f"{len(embeddings)} embeddings saved to database with prefix '{name_prefix}'")

    def get_embeddings_by_name(self, name_prefix: str) -> list[FaceEmbedding]:
        embeddings = []
        with self._conn.cursor() as cur:
            cur.execute("SELECT name, embedding FROM face_embeddings WHERE name LIKE %s", (f"{name_prefix}_%",))
            results = cur.fetchall()
            for name, emb_str in results:
                try:
                    emb = ast.literal_eval(emb_str)
                    embeddings.append(FaceEmbedding(name=name, embedding=emb))
                except Exception as e:
                    print(f"Failed to parse embedding for {name}: {e}")
        return embeddings