import cv2
from insightface.app import FaceAnalysis
import numpy as np
from .db.connection import save_embeddings_to_db, get_embeddings_by_name

app_insightface = FaceAnalysis(providers=['CPUExecutionProvider'])
app_insightface.prepare(ctx_id=0, det_size=(640, 640))

def extract_embeddings(image_path: str) -> list[np.ndarray]:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image from {image_path}")
        return []

    faces = app_insightface.get(img)
    if not faces:
        print("No face detected")
        return []

    return [face.embedding for face in faces]

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    return np.dot(emb1, emb2)

def save_faces_from_image(image_path: str, name_prefix: str):
    embeddings = extract_embeddings(image_path)
    if embeddings:
        save_embeddings_to_db(name_prefix, embeddings)
    else:
        print("no faces to save")

def main():
    target_embeddings = extract_embeddings("./dima.jpg")
    save_embeddings_to_db("dima1", target_embeddings)
    db_embeddings = get_embeddings_by_name("dima1")

    if not target_embeddings or not db_embeddings:
        print("no embeddings to compare")
        return

    for i, test_emb in enumerate(target_embeddings):
        print(f"\ncomparing face #{i} from image:")
        for j, db_emb in enumerate(db_embeddings):
            similarity = cosine_similarity(test_emb, db_emb)
            print(f"similarity with DB embedding {j}: {similarity:.4f}")


if __name__ == "__main__":
    main()