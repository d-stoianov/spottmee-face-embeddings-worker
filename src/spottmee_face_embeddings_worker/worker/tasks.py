
import os
import logging

from ..db.repository import EmbeddingRepository
from ..core.face_processor import FaceProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROCESS_QUEUE = 'process_embeddings'
COMPARE_QUEUE = 'compare_embeddings'

# process and store embeddings
def process_and_save_embeddings(image_path: str, id: str):
    logging.info(f"Starting job: process_and_save_embeddings for id '{id}' from image '{image_path}'")

    face_processor = FaceProcessor()

    db_repo = EmbeddingRepository()
    try:
        embeddings = face_processor.extract_embeddings(image_path, id)
        
        if not embeddings:
            logging.warning(f"No faces found in image '{image_path}'. Job finished without saving.")
            return

        db_repo.save_embeddings(id, embeddings)
        
        logging.info(f"Successfully processed and saved {len(embeddings)} embeddings for id '{id}'.")

    except Exception as e:
        logging.error(f"Failed to process and save embeddings for id '{id}': {e}")

    finally:
        db_repo.close_connection()


# comparing already stored face embeddings with unprocessed image
def compare_face_embeddings(id: str, image_path: str, threshold: float = 0.5):
    logging.info(f"Starting job: compare_embeddings for '{id}' and '{image_path}'")

    face_processor = FaceProcessor()
    db_repo = EmbeddingRepository()

    try:
        embeddings_1 = db_repo.get_embeddings_by_name(id)

        if not embeddings_1:
            logging.warning(f"Could not retrieve embeddings for user '{id}'. Comparison aborted.")
            return
        
        embeddings_2 = face_processor.extract_embeddings(image_path, image_path)

        # compare two sets of embeddings
        if face_processor.compare_faces(embeddings_1, embeddings_2, threshold):
            logging.info(f"SUCCESS: A face match was found between '{id}' and '{image_path}'!")
        else:
            logging.info(f"FAILURE: No face match was found between '{id}' and '{image_path}'.")

    except Exception as e:
        logging.error(f"Failed to compare faces for '{id}' and '{image_path}': {e}")
    
    finally:
        db_repo.close_connection()