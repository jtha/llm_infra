# -------------------------------------------------------------------------------
#
# Imports
#
# -------------------------------------------------------------------------------

# Built-in libraries
import os
import json
import asyncio
import logging
import time
from functools import wraps
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

# Additional libraries
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import redis.asyncio as redis
from openai import AsyncOpenAI

# Local imports
from db_util import get_models, get_whitelist, refresh_models, update_whitelist_table, get_all_models

# -------------------------------------------------------------------------------
#
# Pydantic Models
#
# -------------------------------------------------------------------------------

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class Whitelist(BaseModel):
    data: List[Dict[str, str]] = [{'id': '', 'provider': ''}]

# -------------------------------------------------------------------------------
#
# Initialization
#
# -------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start cache invalidation listener
    task = asyncio.create_task(listen_for_cache_invalidation())
    yield
    # Cleanup (if needed)
    task.cancel()
    await task

app = FastAPI(lifespan=lifespan)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis client
redis_client: redis.Redis = None

# Cache expiration time
CACHE_EXPIRATION = 86400  # 1 day

@asynccontextmanager
async def get_redis_client():
    client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        password=os.getenv('REDIS_PASSWORD', 'password'),
        db=0
    )
    try:
        yield client
    finally:
        await client.close()

# Pub/Sub channel name
CACHE_INVALIDATION_CHANNEL = "cache_invalidation"

# -------------------------------------------------------------------------------
#
# Utility functions
#
# -------------------------------------------------------------------------------

def timing_decorator(func):
    """Decorator to log the execution time of an asynchronous function."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@timing_decorator
async def get_cached_data(key: str):
    """Retrieve data from the cache using the specified key."""
    async with get_redis_client() as redis:
        data = await redis.get(key)
        return json.loads(data) if data else None

@timing_decorator
async def set_cached_data(key: str, data: Any):
    """Store data in the cache with the specified key and expiration time."""
    async with get_redis_client() as redis:
        await redis.setex(key, CACHE_EXPIRATION, json.dumps(data))

async def get_or_fetch_data(key: str, fetch_function):
    """Get data from the cache or fetch it using the provided function if not cached."""
    data = await get_cached_data(key)
    if data is None:
        logger.info(f"Cache miss for {key}, fetching fresh data")
        data = await fetch_function()
        await set_cached_data(key, data)
    return data

async def invalidate_cache(key: str):
    """Invalidate the cache for the specified key."""
    async with get_redis_client() as redis:
        await redis.delete(key)
        await redis.publish(CACHE_INVALIDATION_CHANNEL, key)

async def listen_for_cache_invalidation():
    """Listen for cache invalidation messages and log them."""
    async with get_redis_client() as redis:
        try:
            pubsub = redis.pubsub()
            await pubsub.subscribe(CACHE_INVALIDATION_CHANNEL)
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    cache_key = message['data'].decode('utf-8')
                    logger.info(f"Cache invalidation for {cache_key}")
        except Exception as e:
            logger.error(f"Error in cache invalidation listener: {str(e)}")

client = AsyncOpenAI(
    base_url=os.getenv('OPENROUTER_API_BASE_URL'),
    api_key=os.getenv('OPENROUTER_API_KEY')
)

# -------------------------------------------------------------------------------
#
# API endpoints
#
# -------------------------------------------------------------------------------

@app.get("/api/v1/models")
async def get_model_list():
    cache_key = 'models'
    models = await get_or_fetch_data(cache_key, get_models)
    return {"data": models}

@app.get("/api/v1/models_complete")
async def get_complete_model_list():
    cache_key = 'models_complete'
    models = await get_or_fetch_data(cache_key, get_all_models)
    return {"data": models}

@app.get("/api/v1/models/update")
async def update_models():
    result = await refresh_models()
    await invalidate_cache('models')
    await invalidate_cache('models_complete')
    return {"message": result}

@app.get("/api/v1/models/whitelist")
async def get_white_list():
    cache_key = 'whitelist'
    whitelist = await get_or_fetch_data(cache_key, get_whitelist)
    return {"data": whitelist}

@app.put("/api/v1/models/update_whitelist")
async def update_white_list(whitelist: Whitelist):
    result = await update_whitelist_table(whitelist.data)
    if isinstance(result, list):
        await invalidate_cache('whitelist')
        await invalidate_cache('models')
        return {"data": result}
    else:
        raise HTTPException(status_code=500, detail=f"Operation failed: {result}")

@app.post("/api/v1/chat/completions")
async def get_chat_completions(request: Request):
    try:
        data = await request.json()
        chat_request = ChatCompletionRequest(**data)
        
        if chat_request.stream:
            async def event_stream():
                try:
                    stream = await client.chat.completions.create(**chat_request.dict(exclude_unset=True))
                    async for chunk in stream:
                        yield f"data: {chunk.json()}\n\n"
                        await asyncio.sleep(0)  # Allow for task cancellation
                except asyncio.CancelledError:
                    logger.info("Client disconnected")
                except Exception as e:
                    logger.error(f"Streaming error: {str(e)}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                finally:
                    logger.info("Streaming completed or interrupted")

            return StreamingResponse(event_stream(), media_type="text/event-stream")
        else:
            response = await client.chat.completions.create(**chat_request.dict(exclude_unset=True))
            return JSONResponse(content=response.model_dump())
    except Exception as e:
        logger.error(f"Error in chat completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/completions")
async def get_completion(request: Request):
    try:
        data = await request.json()
        completion_request = CompletionRequest(**data)
        
        request_data = completion_request.dict(exclude_unset=True)
        if 'model' not in request_data or 'prompt' not in request_data:
            raise ValueError("Both 'model' and 'prompt' are required.")
        
        if completion_request.stream:
            async def event_stream():
                try:
                    stream = await client.completions.create(**request_data)
                    async for chunk in stream:
                        chunk_data = chunk.model_dump()
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                        await asyncio.sleep(0)
                except asyncio.CancelledError:
                    logger.info("Client disconnected")
                except Exception as e:
                    logger.error(f"Streaming error: {str(e)}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                finally:
                    logger.info("Streaming completed or interrupted")

            return StreamingResponse(event_stream(), media_type="text/event-stream")
        else:
            response = await client.completions.create(**request_data)
            return JSONResponse(content=response.model_dump())
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))