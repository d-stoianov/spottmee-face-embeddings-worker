import psycopg2
from enum import Enum
from face_embeddings_worker.settings import settings


class PhotoStatus(str, Enum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    NO_FACES_FOUND = "NO_FACES_FOUND"
    FAILED = "FAILED"
    READY = "READY"

class MainRepository:
    def __init__(self):
        self._conn = None

    def __enter__(self):
        self._conn = psycopg2.connect(settings.database_url)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._conn:
            self._conn.close()

    def update_photo_status(self, photo_id: str, status: PhotoStatus) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE "Photo"
                SET status = %s
                WHERE id = %s
                """,
                (status.value, photo_id),
            )
        self._conn.commit()