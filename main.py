import cv2
from insightface.app import FaceAnalysis

app_insightface = FaceAnalysis(providers=['CPUExecutionProvider'])
app_insightface.prepare(ctx_id=0, det_size=(640, 640))

def extract_embedding(image_path):
    img = cv2.imread(image_path)
    faces = app_insightface.get(img)
    if not faces:
        print("no face detected")
        return None
    print("faces:", faces)
    return faces

if __name__ == "__main__":
    embedding = extract_embedding("./dima.jpg")
