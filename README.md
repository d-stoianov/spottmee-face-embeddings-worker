# Spottmee Face Embeddings Worker

A background worker service that processes face images into embeddings.

This service listens for jobs sent from the [Spottmee API](https://github.com/d-stoianov/spottmee-api), such as new face images to process. These images are converted into face embeddings using a pretrained model, and the embeddings are stored in the database. The stored embeddings are later used to compare a selfie against existing photos to find matches.

---

## Architecture & How It Fits In

1. API receives several photos to be uploaded into the album  
2. API enqueues process photos jobs (via Redis Queue)  
3. **face-embeddings-worker** picks up the job one-by-one, computes embedding vector  
4. Worker persists face embeddings for each of the photo (in a PostgreSQL database)
5. Later API receives a request to compare a selfie with photos in the album
6. API sends a new matching job to the worker
7. Worker picks up the job, computes embedding vector for the selfie
8. Worker compares the selfie embedding with the embeddings of the photos in the album
9. Worker simply puts the match result into Redis with the matching id as a key
10. API can access the match result via Redis to return the result to the frontend

This approach of having embeddings computed asynchronously — prevents blocking the user-facing API and allows horizontal scaling of compute workers.

---

## Tech Stack

-   **Python**
-   **InsightFace**
-   **PostgreSQL**
-   **Redis**

---

## Getting Started

### Prerequisites

- Python ≥ 3.10  
- Poetry
- PostgreSQL database
- Redis 

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/d-stoianov/spottmee-face-embeddings-worker.git
    cd spottmee-face-embeddings-worker
    ```

2.  Install dependencies:

    ```bash
    poetry install
    ```

3.  Setup .env file:

    Create file in the root of the project called `.env`, with the following content:
    ```bash
    DATABASE_URL=
    EMBEDDINGS_DATABASE_URL=
    REDIS_URL=
    ```

4. Run the worker:
    ```bash
    poetry run face_embeddings_worker
    ```

### Running with Docker

If you don’t want to install Python, Poetry, or dependencies locally, you can use Docker instead.
The repository includes a ready-to-use Dockerfile.

```bash
# Build the Docker image
docker build -t spottmee-face-embeddings-worker .

# Run the container
docker run -p 3000:3000 --env-file .env spottmee-face-embeddings-worker
```

Ensure your .env file is properly configured before running the container.  
You can also use docker-compose if you want to run PostgreSQL and Redis services together.
