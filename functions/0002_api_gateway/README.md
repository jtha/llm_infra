# API Gateway Service

## Description

This API Gateway service acts as an intermediary between clients and various language models, primarily interfacing with OpenRouter. It provides endpoints for model management, chat completions, and text completions.

Key features:
- Model listing and management
- Whitelist management for available models
- Chat completion endpoint
- Text completion endpoint
- Redis caching for improved performance
- PostgreSQL database for persistent storage

## Build Instructions

1. Ensure you have Python 3.7+ installed.

2. Install the required dependencies:
   ```
   pip install fastapi uvicorn[standard] redis pydantic openai asyncpg aiohttp
   ```

3. Set up the following environment variables:
   - `REDIS_HOST`: Redis server host
   - `REDIS_PORT`: Redis server port
   - `REDIS_PASSWORD`: Redis server password
   - `OPENROUTER_API_BASE_URL`: OpenRouter API base URL
   - `OPENROUTER_API_KEY`: Your OpenRouter API key
   - `DATABASE_NAME`: PostgreSQL database name
   - `DATABASE_USER`: PostgreSQL username
   - `DATABASE_PASSWORD`: PostgreSQL password
   - `DATABASE_HOST`: PostgreSQL host
   - `DATABASE_PORT`: PostgreSQL port

4. Run the FastAPI application:
   ```
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Pre-built Docker Image

The API Gateway service is also available as a pre-built Docker image. To use it, pull the image from the GitHub Container Registry and run it:

```
docker pull <TBD>
docker run -p 8000:8000 <TBD>
```

## API Endpoints

- `GET /api/v1/models`: Get list of available models
- `GET /api/v1/models_complete`: Get complete list of models
- `GET /api/v1/models/update`: Update the models list
- `GET /api/v1/models/whitelist`: Get the whitelist of models
- `PUT /api/v1/models/update_whitelist`: Update the whitelist
- `POST /api/v1/chat/completions`: Get chat completions
- `POST /api/v1/completions`: Get text completions

For detailed API documentation, run the server and visit `http://localhost:8000/docs`.

## Architecture

The service uses:
- FastAPI for the web framework
- Redis for caching
- PostgreSQL for persistent storage
- OpenAI's API client for interacting with OpenRouter

The main application logic is in `main.py`, while database utilities are in `db_util.py`.

## Caching

The service implements a Redis-based caching system with automatic invalidation to improve performance for frequently accessed data.

## Error Handling

Errors are logged and appropriate HTTP status codes are returned to the client. Detailed error messages are provided for easier debugging and client-side error handling.

## Streaming

Both chat and text completion endpoints support streaming responses for real-time output