# -------------------------------------------------------------------------------
#
# Imports
#
# -------------------------------------------------------------------------------

# Built-in libraries
import os
import logging
from typing import List, Dict, Optional, Any

# Additional libraries
import asyncpg
import aiohttp
from asyncpg.pool import Pool
from asyncpg.exceptions import PostgresError

# -------------------------------------------------------------------------------
#
# Initialization & Database Pool
#
# -------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
db_pool: Optional[Pool] = None

async def get_db_pool() -> Pool:
    """
    Retrieves a connection pool to the PostgreSQL database, creating it if it does not exist.
    """
    global db_pool
    if db_pool is None:
        try:
            db_pool = await asyncpg.create_pool(
                database=os.environ.get("DATABASE_NAME"),
                user=os.environ.get("DATABASE_USER"),
                password=os.environ.get("DATABASE_PASSWORD"),
                host=os.environ.get("DATABASE_HOST"),
                port=os.environ.get("DATABASE_PORT")
            )
        except Exception as e:
            logger.error(f"Failed to create database pool: {str(e)}")
            raise
    return db_pool

# -------------------------------------------------------------------------------
#
# Functions
#
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

async def create_or_update_table(flattened_dict: List[Dict[str, Any]], table: str) -> None:
    """
    Creates a table if it does not exist and inserts or updates records.
    """
    if not flattened_dict:
        return

    keys = list(flattened_dict[0].keys())
    columns = ', '.join(f"{asyncpg.identifier(key)} TEXT" for key in keys)
    primary_key = keys[0]

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f'CREATE TABLE IF NOT EXISTS {asyncpg.identifier(table)} ({columns}, PRIMARY KEY ({asyncpg.identifier(primary_key)}))')

            placeholders = ', '.join(f'${i+1}' for i in range(len(keys)))
            values = [[str(model.get(key, '')) for key in keys] for model in flattened_dict]
            await conn.executemany(f'''
            INSERT INTO {asyncpg.identifier(table)} ({', '.join(asyncpg.identifier(key) for key in keys)})
            VALUES ({placeholders})
            ON CONFLICT ({asyncpg.identifier(primary_key)}) DO UPDATE SET
            {', '.join(f"{asyncpg.identifier(key)} = EXCLUDED.{asyncpg.identifier(key)}" for key in keys)}
            ''', values)

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
        logger.error("Error fetching models from API")
        raise
    except Exception as e:
        logger.error("An unexpected error occurred while refreshing models")
        raise

async def recreate_and_populate_table(flattened_dict: List[Dict[str, Any]], table: str) -> List[Dict[str, Any]]:
    """
    Drops the existing table if it exists, creates a new table, and inserts records.
    """
    if not flattened_dict:
        return []

    try:
        keys = list(flattened_dict[0].keys())
        columns = ', '.join(f"{asyncpg.identifier(key)} TEXT" for key in keys)
        primary_key = keys[0]

        pool = await get_db_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(f'DROP TABLE IF EXISTS {asyncpg.identifier(table)}')
                await conn.execute(f'CREATE TABLE {asyncpg.identifier(table)} ({columns}, PRIMARY KEY ({asyncpg.identifier(primary_key)}))')

                placeholders = ', '.join(f'${i+1}' for i in range(len(keys)))
                values = [[model.get(key, '') for key in keys] for model in flattened_dict]
                await conn.executemany(f'''
                INSERT INTO {asyncpg.identifier(table)} ({', '.join(asyncpg.identifier(key) for key in keys)})
                VALUES ({placeholders})
                ''', values)

        return flattened_dict

    except PostgresError as e:
        logger.error("Database error occurred while recreating and populating table")
        raise
    except Exception as e:
        logger.error("An unexpected error occurred while recreating and populating table")
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

async def query_table(query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    """
    Executes a query and returns the results as a list of dictionaries.
    """
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        try:
            results = await conn.fetch(query, *params) if params else await conn.fetch(query)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error("An error occurred during query execution")
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