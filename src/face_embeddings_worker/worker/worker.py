import redis
import logging
from rq import Worker, Queue

from face_embeddings_worker.settings import settings
from face_embeddings_worker.worker.tasks import PROCESS_QUEUE, COMPARE_QUEUE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def start_worker():
    logging.info(f"Connecting to Redis at {settings.redis_url}")
    try:
        redis_conn = redis.from_url(settings.redis_url)

        queues_to_listen = [Queue(PROCESS_QUEUE, connection=redis_conn),
                            Queue(COMPARE_QUEUE, connection=redis_conn)]

        logging.info(f"Starting RQ worker, listening to queues: {[q.name for q in queues_to_listen]}")

        worker = Worker(queues_to_listen)
        worker.work()
            
    except Exception as e:
        logging.critical(f"Failed to start worker: {e}")