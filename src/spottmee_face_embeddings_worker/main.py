import cv2
from insightface.app import FaceAnalysis
import numpy as np
from dotenv import load_dotenv
from .db.connection import save_embedding_to_db 

app_insightface = FaceAnalysis(providers=['CPUExecutionProvider'])
app_insightface.prepare(ctx_id=0, det_size=(640, 640))

def extract_embedding(image_path: str) -> np.ndarray | None:
    img = cv2.imread(image_path)
    if img is None:
        print(f"could not read image from {image_path}")
        return None
        
    faces = app_insightface.get(img)
    if not faces:
        print("no face detected")
        return None
        
    return faces[0].embedding

def main():
    embedding = extract_embedding("./dima.jpg")

    if embedding is not None:
        embedding_list = embedding.tolist() 
        save_embedding_to_db("dima", embedding_list)

if __name__ == "__main__":
    main()