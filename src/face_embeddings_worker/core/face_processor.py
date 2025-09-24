import cv2
import numpy as np
from insightface.app import FaceAnalysis
from typing import List
import requests

from face_embeddings_worker.models.embedding import FaceEmbedding

class FaceProcessor:
    def __init__(self):
        self.app_insightface = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app_insightface.prepare(ctx_id=0, det_size=(640, 640))

    def _read_image_from_url(self, url: str):
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img_array = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image from URL: {url}")
        return img

    def extract_embeddings(self, image_input: str | bytes, name_prefix: str) -> list[FaceEmbedding]:
        # input is a hosted URL
        if isinstance(image_input, str) and (image_input.startswith("http://") or image_input.startswith("https://")):
            img = self._read_image_from_url(image_input)

        # input is a local file path
        elif isinstance(image_input, str):
            img = cv2.imread(image_input)

        # input is raw bytes
        elif isinstance(image_input, (bytes, bytearray)):
            np_arr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        else:
            raise ValueError(f"Invalid input type: {type(image_input)}")

        if img is None:
            print(f"Could not read image from {image_input}")
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

    def compare_faces(
        self,
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
    
    