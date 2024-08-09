# -------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------

# Built-in libraries
import os
import json
import asyncio
import logging
import time
from functools import wraps
from typing import List, Dict, Optional, Any, Union
from contextlib import asynccontextmanager
from urllib.parse import urlparse

# Additional libraries
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import redis.asyncio as redis
from openai import AsyncOpenAI

from opentelemetry import trace
from opentelemetry import context
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
 
# Local imports
from db_util import get_models, get_whitelist, refresh_models, update_whitelist_table, get_all_models, check_and_scrape_url


# -------------------------------------------------------------------------------
# Pydantic Models
# -------------------------------------------------------------------------------

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.5
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    service_tier: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    parallel_tool_calls: Optional[bool] = True
    function_call: Optional[Dict[str, Any]] = None

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]] 
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.5
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    best_of: Optional[int] = 1
    echo: Optional[bool] = False
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    seed: Optional[int] = None
    user: Optional[str] = None
    suffix: Optional[str] = None

class Whitelist(BaseModel):
    data: List[Dict[str, str]] = [{'id': '', 'provider': ''}]

class ScrapeRequest(BaseModel):
    url: str
    force_scrape: Optional[bool] = False

# -------------------------------------------------------------------------------
# Initialization
# -------------------------------------------------------------------------------

# Telemetry configuration
    # Assumes OTEL_SERVICE_NAME, OTEL_RESOURCE_ATTRIBUTES, OTEL_EXPORTER_OTLP_ENDPOINT, and OTEL_EXPORTER_OTLP_PROTOCOL are set in the environment
    # OTEL_SERVICE_NAME - The name of your application. The service name will be available as attributes for traces, metrics, and logs.
    # OTEL_RESOURCE_ATTRIBUTES - A comma-separated list of key=value pairs that describe the resource. Grafana Cloud requires 'deployment.environment', 'service.namespace', 'service.version', and 'service.instance.id' to be set
    # OTEL_EXPORTER_OTLP_ENDPOINT - The endpoint of the Grafana Alloy instance
    # OTEL_EXPORTER_OTLP_PROTOCOL - The protocol to use for exporting traces. Valid values are "grpc" and "http"

trace.set_tracer_provider(TracerProvider())
otlp_exporter = OTLPSpanExporter()
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(listen_for_cache_invalidation())
    yield
    task.cancel()
    await task
    await trace.get_tracer_provider().shutdown()

# FastAPI app and instrumentation initialization
app = FastAPI(lifespan=lifespan)

# Custom ASGI middleware to selectively apply instrumentation
class SelectiveOpenTelemetryMiddleware(OpenTelemetryMiddleware):
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] in ["/api/v1/chat/completions", "/api/v1/completions"]:
            # For streaming endpoints, skip the default instrumentation
            return await self.app(scope, receive, send)
        # For all other endpoints, apply the default instrumentation
        return await super().__call__(scope, receive, send)

# Apply our custom middleware
app.add_middleware(SelectiveOpenTelemetryMiddleware)

# Apply FastAPI instrumentation, but exclude our streaming endpoints
FastAPIInstrumentor.instrument_app(app, excluded_urls="/api/v1/chat/completions,/api/v1/completions")

tracer = trace.get_tracer(__name__)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging setup
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

CACHE_EXPIRATION = 86400  # 1 day in seconds
CACHE_INVALIDATION_CHANNEL = "cache_invalidation"

# Authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header == os.getenv("API_KEY"):
        return api_key_header
    raise HTTPException(status_code=403, detail="Could not validate credentials")

# -------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------

def is_valid_url(url: str) -> bool:
    """Check if the provided URL is valid."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

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

async def event_stream(request, parent_context):
    with tracer.start_as_current_span("event_stream", context=parent_context) as span:
        span.set_attribute("request_type", type(request).__name__)
        span.set_attribute("streaming", True)
        
        start_time = time.time()
        chunk_count = 0
        total_tokens = 0
        
        try:
            if isinstance(request, ChatCompletionRequest):
                stream = await client.chat.completions.create(**request.dict(exclude_unset=True))
            else:
                stream = await client.completions.create(**request.dict(exclude_unset=True))
            
            async for chunk in stream:
                chunk_count += 1
                chunk_data = json.loads(chunk.json())
                total_tokens += len(chunk_data.get('choices', [{}])[0].get('delta', {}).get('content', '').split())
                
                if chunk_count % 100 == 0:  # Log every 100 chunks
                    span.add_event("stream_progress", {
                        "chunk_count": chunk_count,
                        "total_tokens": total_tokens
                    })
                
                yield f"data: {chunk.json()}\n\n"
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            logger.info("Client disconnected")
            span.add_event("stream_cancelled")
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            yield f"data: {json.dumps({'error': 'Internal server error'})}\n\n"
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            span.set_attribute("total_chunks", chunk_count)
            span.set_attribute("total_tokens", total_tokens)
            span.set_attribute("duration_seconds", duration)
            
            span.add_event("stream_completed", {
                "total_chunks": chunk_count,
                "total_tokens": total_tokens,
                "duration_seconds": duration
            })

# -------------------------------------------------------------------------------
# API endpoints
# -------------------------------------------------------------------------------

@app.get("/api/v1/models")
async def get_model_list():
    with tracer.start_as_current_span("get_model_list"):
        try:
            cache_key = 'models'
            with tracer.start_as_current_span("get_or_fetch_data"):
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
    with tracer.start_as_current_span("get_chat_completions") as span:
        try:
            data = await request.json()
            chat_request = ChatCompletionRequest(**data)
            span.set_attribute("model", chat_request.model)
            span.set_attribute("streaming", chat_request.stream)
            
            if chat_request.stream:
                current_context = context.get_current()
                return StreamingResponse(event_stream(chat_request, current_context), media_type="text/event-stream")
            else:
                response = await client.chat.completions.create(**chat_request.dict(exclude_unset=True))
                span.set_attribute("total_tokens", response.usage.total_tokens)
                return JSONResponse(content=response.model_dump())
        except Exception as e:
            logger.error(f"Error in chat completions: {e}")
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/completions")
async def get_completion(request: Request):
    with tracer.start_as_current_span("get_completion") as span:
        try:
            data = await request.json()
            completion_request = CompletionRequest(**data)
            span.set_attribute("model", completion_request.model)
            span.set_attribute("streaming", completion_request.stream)
            
            if completion_request.stream:
                current_context = context.get_current()
                return StreamingResponse(event_stream(completion_request, current_context), media_type="text/event-stream")
            else:
                response = await client.completions.create(**completion_request.dict(exclude_unset=True))
                span.set_attribute("total_tokens", response.usage.total_tokens)
                return JSONResponse(content=response.model_dump())
        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            span.record_exception(ve)
            span.set_status(Status(StatusCode.ERROR, str(ve)))
            raise HTTPException(status_code=400, detail="Invalid request")
        except Exception as e:
            logger.error(f"Error in completions: {e}")
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/scrape")
async def scrape_url_endpoint(request: ScrapeRequest):
    try:
        if not is_valid_url(request.url):
            raise HTTPException(status_code=400, detail="Invalid URL format")
        result = await check_and_scrape_url(request.url, request.force_scrape)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error scraping URL: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    