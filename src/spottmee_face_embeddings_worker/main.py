import os

from .db.repository import EmbeddingRepository
from .core.face_processor import FaceProcessor

def main():
    face_processor = FaceProcessor()

    with EmbeddingRepository() as db_repo:
        
        target_embeddings = face_processor.extract_embeddings("./dima.jpg", "dima")
        
        if not target_embeddings:
            print("No faces to save or compare.")
            return

        db_repo.save_embeddings("dima1", target_embeddings)
        
        db_embeddings = db_repo.get_embeddings_by_name("dima1")

        if face_processor.compare_faces(target_embeddings, db_embeddings, threshold=0.5):
            print("Face match")
        else:
            print("Face doesn't match")

if __name__ == "__main__":
    main()