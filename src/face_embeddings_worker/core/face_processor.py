import cv2
import numpy as np
from insightface.app import FaceAnalysis
from typing import List

from face_embeddings_worker.models.embedding import FaceEmbedding

class FaceProcessor:
    def __init__(self):
        self.app_insightface = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app_insightface.prepare(ctx_id=0, det_size=(640, 640))

    def extract_embeddings(self, image_path: str, name_prefix: str) -> list[FaceEmbedding]:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image from {image_path}")
            return []

        faces = self.app_insightface.get(img)
        if not faces:
            print("No face detected")
            return []

        face_embeddings: List[FaceEmbedding] = []
        for i, face in enumerate(faces):
            name = f"{name_prefix}_{i}"
            print(f"type: {type(face.embedding)}, shape: {getattr(face.embedding, 'shape', None)}, dtype: {getattr(face.embedding, 'dtype', None)}")
            embedding = FaceEmbedding.from_numpy_array(name, face.embedding)
            face_embeddings.append(embedding)

        return face_embeddings

    # method to compare faces
    @staticmethod
    def compare_faces(
        source_embeddings: List[FaceEmbedding],
        target_embeddings: List[FaceEmbedding],
        threshold: float = 0.5
    ) -> bool:
        for i, query_model in enumerate(source_embeddings):
            query_emb = query_model.to_numpy_array()
            for j, stored_model in enumerate(target_embeddings):
                stored_emb = stored_model.to_numpy_array()
                sim = FaceProcessor._cosine_similarity(query_emb, stored_emb)
                print(f"Similarity (query #{i} vs stored #{j}): {sim:.4f}")
                if sim > threshold:
                    return True
        return False

    def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        emb1_normalized = emb1 / np.linalg.norm(emb1)
        emb2_normalized = emb2 / np.linalg.norm(emb2)
        return np.dot(emb1_normalized, emb2_normalized)
    
    