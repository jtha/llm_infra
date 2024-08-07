# -------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------

# Built-in libraries
import os
import logging
from typing import List, Dict, Optional, Any, Literal
from enum import Enum, auto
from contextlib import asynccontextmanager

# Additional libraries
import asyncpg
import aiohttp
from asyncpg.pool import Pool
from asyncpg.exceptions import PostgresError
import voyageai
from pydantic import BaseModel

# Local imports
from docs_processing import scrape_url, split_markdown

# -------------------------------------------------------------------------------
# Pydantic Models
# -------------------------------------------------------------------------------

class EmbedRequest(BaseModel):
    texts: List[str]
    model: str = "voyage-large-2-instruct"
    input_type: Optional[Literal['document', 'query']] = None
    truncation: Optional[bool] = True

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    model: str = "rerank-1"
    top_k: Optional[int] = 10
    truncation: Optional[bool] = True

# -------------------------------------------------------------------------------
# Initialization & Database Pool
# -------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
vo=voyageai.AsyncClient(api_key=os.getenv('VOYAGE_API_KEY'))
db_pool: Optional[Pool] = None
db_presets: Dict[str, str] = {}

def add_db_preset(preset_name: str, db_url: str) -> None:
    """
    Adds or updates a database connection preset using a URL.
    """
    db_presets[preset_name] = db_url

def remove_db_preset(preset_name: str) -> None:
    """
    Removes a database connection preset.
    """
    db_presets.pop(preset_name, None)

def list_presets() -> List[str]:
    """
    Returns a list of available preset names.
    """
    return list(db_presets.keys())

async def get_db_pool(preset: Optional[str] = None, db_url: Optional[str] = None) -> Pool:
    """
    Retrieves a connection pool to the PostgreSQL database, creating it if it does not exist.
    If a preset is provided, it uses the preset configuration.
    If a db_url is provided, it uses that directly.
    """
    global db_pool
    if db_pool is None:
        try:
            if preset and preset in db_presets:
                connection_url = db_presets[preset]
            elif db_url:
                connection_url = db_url
            else:
                connection_url = os.environ.get("DATABASE_URL")
            
            if not connection_url:
                raise ValueError("No database URL provided")
            
            db_pool = await asyncpg.create_pool(connection_url)
        except Exception as e:
            logger.error(f"Failed to create database pool: {str(e)}")
            raise
    return db_pool

@asynccontextmanager
async def db_connection(preset: Optional[str] = None, db_url: Optional[str] = None):
    """
    Context manager for database connections.
    """
    pool = await get_db_pool(preset, db_url)
    async with pool.acquire() as conn:
        yield conn


add_db_preset("API_GATEWAY", os.environ.get("DATABASE_URL_API_GATEWAY"))
add_db_preset("WEB_SCRAPE", os.environ.get("DATABASE_URL_WEB_SCRAPE"))

# -------------------------------------------------------------------------------
# Functions - General DB Utilities
# -------------------------------------------------------------------------------

def flatten_dict(nested_dicts: List[Dict[str, Any]], parent_key: str = '', sep: str = '_') -> List[Dict[str, Any]]:
    """
    Flattens a nested dictionary into a single-level dictionary with concatenated keys.
    """
    if isinstance(nested_dicts, list) and all(not isinstance(item, dict) for item in nested_dicts):
        logger.info("Already flat")
        return nested_dicts

    items = {}
    for k, v in nested_dicts.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

async def create_or_update_table(flattened_dict: List[Dict[str, Any]], table: str, preset: Optional[str] = "API_GATEWAY") -> None:
    """
    Creates a table if it does not exist and inserts or updates records.
    """
    if not flattened_dict:
        return

    keys = list(flattened_dict[0].keys())
    columns = ', '.join(f'"{key}" TEXT' for key in keys)
    primary_key = keys[0]

    pool = await get_db_pool(preset=preset)
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Create table if not exists
            await conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({columns}, PRIMARY KEY ("{primary_key}"))')

            # Prepare the insert/update query
            placeholders = ', '.join(f'${i+1}' for i in range(len(keys)))
            update_set = ', '.join(f'"{key}" = EXCLUDED."{key}"' for key in keys)
            query = f'''
            INSERT INTO "{table}" ({', '.join(f'"{key}"' for key in keys)})
            VALUES ({placeholders})
            ON CONFLICT ("{primary_key}") DO UPDATE SET
            {update_set}
            '''

            # Convert all values to strings and execute the query
            values = [[str(model.get(key, '')) for key in keys] for model in flattened_dict]
            await conn.executemany(query, values)

async def query_table(query: str, params: Optional[List[Any]] = None, preset: Optional[str] = "API_GATEWAY") -> List[Dict[str, Any]]:
    """
    Executes a query and returns the results as a list of dictionaries.
    """
    pool = await get_db_pool(preset=preset)
    async with pool.acquire() as conn:
        try:
            results = await conn.fetch(query, *params) if params else await conn.fetch(query)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error("An error occurred during query execution")
            raise

# -------------------------------------------------------------------------------
# Functions - Models Specific
# -------------------------------------------------------------------------------

async def refresh_models() -> str:
    """
    Fetches models from the OpenRouter API, flattens the model data, 
    and updates the 'openrouter_models' table in the database.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://openrouter.ai/api/v1/models") as response:
                response.raise_for_status()
                openrouter_models = await response.json()
        
        flattened_models = [flatten_dict(model) for model in openrouter_models.get('data', [])]
        await create_or_update_table(flattened_models, "openrouter_models")
        
        return f"Success! {len(flattened_models)} records processed."
    
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching models from API: {e}")
        raise
    except asyncpg.PostgresError as e:
        logger.error(f"Database error while refreshing models: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while refreshing models: {e}")
        raise

async def update_whitelist_table(whitelist_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Updates the whitelist table with the provided data.
    """
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute('TRUNCATE TABLE models_whitelist')

                await conn.executemany('''
                INSERT INTO models_whitelist (id, provider)
                VALUES ($1, $2)
                ''', [(item.get('id', ''), item.get('provider', '')) for item in whitelist_data])

        return whitelist_data

    except PostgresError as e:
        logger.error("Database error occurred while updating whitelist table")
        raise
    except Exception as e:
        logger.error("An unexpected error occurred while updating whitelist table")
        raise

async def get_models() -> List[Dict[str, Any]]:
    """
    Retrieves all models from the selected table.
    """
    return await query_table("""\
    SELECT 
        om.id,
        om.name,
        om.context_length
    FROM 
        openrouter_models om
    INNER JOIN models_whitelist wm
        ON om.id = wm.id
    """)

async def get_all_models() -> List[Dict[str, Any]]:
    """
    Retrieves all models.
    """
    return await query_table("""\
    SELECT 
        om.id,
        om.name,
        om.context_length,
        om.pricing_prompt,
        om.pricing_completion
    FROM 
        openrouter_models om
    """)

async def get_whitelist() -> List[Dict[str, Any]]:
    """
    Retrieves all models from the selected table.
    """
    return await query_table("SELECT * FROM models_whitelist")

# -------------------------------------------------------------------------------
# Functions - Web Scrape Specific
# -------------------------------------------------------------------------------

async def check_and_scrape_url(url: str, force_scrape: bool = False) -> Dict[str, Any]:
    pool = await get_db_pool(preset="WEB_SCRAPE")
    async with pool.acquire() as conn:
        existing_data = await conn.fetchrow(
            "SELECT * FROM scraped_url WHERE source = $1",
            url
        )

        if existing_data and not force_scrape:
            return {"message": f"{url} already scraped"}

        scraped_data = await scrape_url(url)
        
        data = {
            'title': scraped_data['metadata'].get('title', ''),
            'author': scraped_data['metadata'].get('author', ''),
            'hostname': scraped_data['metadata'].get('hostname', ''),
            'date': scraped_data['metadata'].get('date', ''),
            'fingerprint': scraped_data['metadata'].get('fingerprint', ''),
            'license': scraped_data['metadata'].get('license', ''),
            'pagetype': scraped_data['metadata'].get('pagetype', ''),
            'filedate': scraped_data['metadata'].get('filedate', ''),
            'url': url,
            'source_hostname': scraped_data['metadata'].get('source_hostname', ''),
            'text_markdown': scraped_data['content']
        }

        await conn.execute("""
            INSERT INTO scraped_url (
                title, author, hostname, date, fingerprint, license, pagetype, 
                filedate, url, source_hostname, text_markdown
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (url) DO UPDATE SET
                title = $1, author = $2, hostname = $3, date = $4, fingerprint = $5,
                license = $6, pagetype = $7, filedate = $8, source_hostname = $10,
                text_markdown = $11
        """, data['title'], data['author'], data['hostname'], data['date'],
            data['fingerprint'], data['license'], data['pagetype'], data['filedate'],
            data['url'], data['source_hostname'], data['text_markdown'])

        chunks = split_markdown(data['text_markdown'])
        
        if not chunks:
            logger.warning(f"No chunks generated for {url}")
            return {"message": f"Scraped {url}, but no content to index"}

        try:
            embeddings = await vo.embed(EmbedRequest(texts=chunks, input_type='document'))
            if len(embeddings) != len(chunks):
                logger.error(f"Mismatch between chunks and embeddings for {url}")
                return {"message": f"Error: Mismatch between chunks and embeddings for {url}"}

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                if not embedding:
                    logger.warning(f"Empty embedding for chunk {i} of {url}")
                    continue

                chunk_index = f"{url}_{i}"
                word_count = len(chunk.split())

                await conn.execute("""
                INSERT INTO chunk_embeddings (url, chunk_index, chunk, embedding_voyage, word_count, filedate)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (url, chunk_index) DO UPDATE SET
                    chunk = EXCLUDED.chunk,
                    embedding_voyage = EXCLUDED.embedding_voyage,
                    filedate = EXCLUDED.filedate
                """, url, chunk_index, chunk, embedding, word_count, data['filedate'])

            return {"message": f"Scraped {url} and indexed {len(embeddings)} chunks"}

        except Exception as e:
            logger.error(f"Error during embedding or indexing for {url}: {str(e)}")
            return {"message": "An error occurred during processing."}

