import redis
import json
import logging
import base64

from face_embeddings_worker.settings import settings
from face_embeddings_worker.worker.tasks import (
    PROCESS_QUEUE,
    COMPARE_QUEUE,
    process_and_save_embeddings,
    compare_face_embeddings,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def start_worker():
    logging.info(f"Connecting to Redis at {settings.redis_url}")
    try:
        r = redis.Redis.from_url(settings.redis_url, decode_responses=True)

        logging.info(f"Starting worker, listening to queues: {PROCESS_QUEUE}, {COMPARE_QUEUE}")

        while True:
            # Blocking pop from either process or compare queue
            queue_name, job_data = r.blpop([PROCESS_QUEUE, COMPARE_QUEUE])
            job = json.loads(job_data)
            logging.info(f"Received job from {queue_name}: {job}")

            if queue_name == PROCESS_QUEUE:
                process_and_save_embeddings(job["imageUrl"], job["id"])
            elif queue_name == COMPARE_QUEUE:
                r.set(f"match-result:{job['jobId']}", "PROCESSING", ex=3600) # create a wireframe for the result

                selfie_image_bytes = base64.b64decode(job["selfie"])

                matched_ids = compare_face_embeddings(job["jobId"], job["storedIds"], selfie_image_bytes, job.get("threshold", 0.5))

                r.set(f"match-result:{job['jobId']}", json.dumps(matched_ids), ex=3600) # expires after 1h
                logging.info(f"Write result into match-result:{job['jobId']}")
            else:
                logging.warning(f"Unknown queue: {queue_name}")

    except Exception as e:
        logging.critical(f"Failed to start worker: {e}")
