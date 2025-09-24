
import logging

from face_embeddings_worker.db.repository import EmbeddingRepository
from face_embeddings_worker.core.face_processor import FaceProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROCESS_QUEUE = 'face_embeddings:process'
COMPARE_QUEUE = 'face_embeddings:compare'

def process_and_save_embeddings(image_url: str, id: str):
    logging.info(f"Starting job: ${PROCESS_QUEUE} for id '{id}' from image '{image_url}'")
    face_processor = FaceProcessor()

    with EmbeddingRepository() as db_repo:
        try:
            embeddings = face_processor.extract_embeddings(image_url, id)
            if not embeddings:
                logging.warning(f"No faces found in image '{image_url}'. Job finished without saving.")
                return
            db_repo.save_embeddings(id, embeddings)
            logging.info(f"Successfully processed and saved {len(embeddings)} embeddings for id '{id}'.")
        except Exception as e:
            logging.error(f"Failed to process and save embeddings for id '{id}': {e}")

# comparing already stored face embeddings with unprocessed image
def compare_face_embeddings(job_id: str, stored_embeddings: list[str], image: bytes, threshold: float = 0.5) -> list[str]:
    logging.info(f"Starting job: {COMPARE_QUEUE} for stored embeddings {stored_embeddings}")

    face_processor = FaceProcessor()

    # extract embeddings from incoming raw image bytes
    source_embeddings = face_processor.extract_embeddings(image, "incoming")

    if not source_embeddings:
        logging.warning("No embeddings found in the incoming image. Aborting comparison.")
        return []

    matched_ids: list[str] = []

    with EmbeddingRepository() as db_repo:
        try:
            for stored_id in stored_embeddings:
                face_embeddings = db_repo.get_embeddings_by_name(stored_id)

                if not face_embeddings:
                    logging.warning(f"Could not retrieve embeddings for '{stored_id}'. Skipping.")
                    continue

                if face_processor.compare_faces(source_embeddings, face_embeddings, threshold):
                    logging.info(f"✅ SUCCESS: A face match was found between '{stored_id}' and 'incoming'!")
                    matched_ids.append(stored_id)
                else:
                    logging.info(f"❌ FAILURE: No face match was found between '{stored_id}' and 'incoming'.")

        except Exception as e:
            logging.error(f"Failed to compare faces for stored embeddings {stored_embeddings}: {e}")
            return []

    logging.info(f"Job ${job_id} finished. Found {len(matched_ids)} matches: {matched_ids}")
    return matched_ids
