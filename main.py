import cv2
from insightface.app import FaceAnalysis
from dotenv import load_dotenv
import os
import psycopg2
import numpy as np

load_dotenv()

app_insightface = FaceAnalysis(providers=['CPUExecutionProvider'])
app_insightface.prepare(ctx_id=0, det_size=(640, 640))

def extract_embedding(image_path):
    img = cv2.imread(image_path)
    faces = app_insightface.get(img)
    if not faces:
        print("No face detected")
        return None
    return faces[0].embedding  # numpy array shape (512,)

def save_embedding_to_db(name, embedding, conn):
    with conn.cursor() as cur:
        embedding_list = embedding.tolist()  # convert numpy array to list
        cur.execute(
            "INSERT INTO face_embeddings (name, embedding) VALUES (%s, %s)",
            (name, embedding_list)
        )
    conn.commit()

if __name__ == "__main__":
    embedding = extract_embedding("./dima.jpg")
    if embedding is not None:
        DATABASE_URL = os.getenv("DATABASE_URL")
        conn = psycopg2.connect(DATABASE_URL)

        save_embedding_to_db("dima", embedding, conn)
        conn.close()
        print("Embedding saved to database.")
