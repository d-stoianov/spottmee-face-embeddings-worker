# Setup

1. Install Poetry: https://python-poetry.org/docs/#installation
2. Clone this repo
3. Run `poetry install`
4. Create .env file in the root with the following:
```bash
DATABASE_URL=your_db_connection_string
```
5. Run `poetry run spottmee-worker`