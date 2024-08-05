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
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
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
    task = asyncio.create_task(listen_for_cache_invalidation())
    yield
    task.cancel()
    await task

app = FastAPI(lifespan=lifespan)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis setup
@asynccontextmanager
async def get_redis_client():
    client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        password=os.getenv('REDIS_PASSWORD'),
        db=0
    )
    try:
        yield client
    finally:
        await client.close()

CACHE_EXPIRATION = 86400  # 1 day
CACHE_INVALIDATION_CHANNEL = "cache_invalidation"

# Authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header == os.getenv("API_KEY"):
        return api_key_header
    raise HTTPException(status_code=403, detail="Could not validate credentials")

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
            logger.error(f"Error in cache invalidation listener: {e}")

client = AsyncOpenAI(
    base_url=os.getenv('OPENROUTER_API_BASE_URL'),
    api_key=os.getenv('OPENROUTER_API_KEY')
)

async def event_stream(request):
    try:
        if isinstance(request, ChatCompletionRequest):
            stream = await client.chat.completions.create(**request.dict(exclude_unset=True))
        else:
            stream = await client.completions.create(**request.dict(exclude_unset=True))
        
        async for chunk in stream:
            yield f"data: {chunk.json()}\n\n"
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'error': 'Internal server error'})}\n\n"
    finally:
        logger.info("Streaming completed or interrupted")

# -------------------------------------------------------------------------------
#
# API endpoints
#
# -------------------------------------------------------------------------------

@app.get("/api/v1/models")
async def get_model_list():
    try:
        cache_key = 'models'
        models = await get_or_fetch_data(cache_key, get_models)
        return {"data": models}
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/models_complete")
async def get_complete_model_list():
    try:
        cache_key = 'models_complete'
        models = await get_or_fetch_data(cache_key, get_all_models)
        return {"data": models}
    except Exception as e:
        logger.error(f"Error fetching complete model list: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/models/update")
async def update_models(api_key: str = Depends(get_api_key)):
    try:
        result = await refresh_models()
        await invalidate_cache('models')
        await invalidate_cache('models_complete')
        return {"message": "Models updated successfully"}
    except Exception as e:
        logger.error(f"Error updating models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/models/whitelist")
async def get_white_list():
    try:
        cache_key = 'whitelist'
        whitelist = await get_or_fetch_data(cache_key, get_whitelist)
        return {"data": whitelist}
    except Exception as e:
        logger.error(f"Error fetching whitelist: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.put("/api/v1/models/update_whitelist")
async def update_white_list(whitelist: Whitelist, api_key: str = Depends(get_api_key)):
    try:
        result = await update_whitelist_table(whitelist.data)
        await invalidate_cache('whitelist')
        await invalidate_cache('models')
        return {"data": result}
    except Exception as e:
        logger.error(f"Error updating whitelist: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/chat/completions")
async def get_chat_completions(request: Request):
    try:
        data = await request.json()
        chat_request = ChatCompletionRequest(**data)
        
        if chat_request.stream:
            return StreamingResponse(event_stream(chat_request), media_type="text/event-stream")
        else:
            response = await client.chat.completions.create(**chat_request.dict(exclude_unset=True))
            return JSONResponse(content=response.model_dump())
    except Exception as e:
        logger.error(f"Error in chat completions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/completions")
async def get_completion(request: Request):
    try:
        data = await request.json()
        completion_request = CompletionRequest(**data)
        
        if completion_request.stream:
            return StreamingResponse(event_stream(completion_request), media_type="text/event-stream")
        else:
            response = await client.completions.create(**completion_request.dict(exclude_unset=True))
            return JSONResponse(content=response.model_dump())
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail="Invalid request")
    except Exception as e:
        logger.error(f"Error in completions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")