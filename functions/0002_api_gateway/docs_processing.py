# -------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------

# Built-in libraries
import re
from typing import List
from copy import deepcopy
import os
import json
import logging
import time

# Additional libraries
from trafilatura import fetch_url, extract
from trafilatura.settings import DEFAULT_CONFIG

# -------------------------------------------------------------------------------
# Initialization
# -------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
my_config = deepcopy(DEFAULT_CONFIG)
my_config['DEFAULT']['USER_AGENTS'] = os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0")

# -------------------------------------------------------------------------------
# Scraping functions
# -------------------------------------------------------------------------------

def scrape_url(url: str, config: dict = my_config) -> dict:
    """
    Scrape the content from the given URL using the specified configuration.

    Args:
        url (str): The URL to scrape.
        config (dict): Configuration settings for the scraping process.

    Returns:
        dict: A dictionary containing the scraped content, with certain keys removed
              and the markdown version of the text included.
    """
    keys_to_drop = ['raw_text', 'id', 'comments', 'language', 'image', 'excerpt', 'categories', 'tags', 'text','source-hostname', 'source']
    try:
        downloaded = fetch_url(url, config=config)
    except Exception as e:
        logger.error(f"Error fetching URL: {e}")
        return None
    
    try:
        scraped_text = json.loads(extract(filecontent=downloaded, url=url, include_images=True, include_comments=False, output_format="json"))
        scraped_text_md = extract(filecontent=downloaded, url=url, include_images=False, include_links=False, output_format="markdown")
        scraped_text['source_hostname'] = scraped_text['source-hostname']
        scraped_text['url'] = url
        for key in keys_to_drop:
            scraped_text.pop(key, None)
        scraped_text['text_markdown'] = scraped_text_md
        return scraped_text
    except Exception as e:
        logger.error(f"Error scraping URL: {e}")
        return None

# -------------------------------------------------------------------------------
# Text processing functions
# -------------------------------------------------------------------------------

def word_count(text: str) -> int:
    """Return the number of words in the given text."""
    return len(text.split())

def remove_numbers_in_brackets(text: str) -> str:
    """Remove numbers enclosed in brackets from the text."""
    return re.sub(r'\[\d+\]', '', text)

def split_into_sentences(text: str) -> List[str]:
    """Split the text into sentences and return a list of non-empty sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def split_chunk_by_sentence(chunk: str, max_words: int) -> List[str]:
    """Split a chunk of text into smaller chunks based on sentence boundaries, 
    ensuring that each chunk does not exceed the specified maximum word count."""
    sentences = split_into_sentences(chunk)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if word_count(current_chunk + ' ' + sentence) > max_words and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += ' ' + sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def split_chunk_by_paragraph(chunk: str, max_words: int) -> List[str]:
    """Split a chunk of text into smaller chunks based on paragraph boundaries, 
    ensuring that each chunk does not exceed the specified maximum word count."""
    paragraphs = chunk.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if word_count(current_chunk + '\n\n' + paragraph) > max_words and current_chunk:
            if word_count(current_chunk) > max_words:
                chunks.extend(split_chunk_by_sentence(current_chunk, max_words))
            else:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += '\n\n' + paragraph
    
    if current_chunk:
        if word_count(current_chunk) > max_words:
            chunks.extend(split_chunk_by_sentence(current_chunk, max_words))
        else:
            chunks.append(current_chunk.strip())
    
    return chunks

def split_markdown(text: str, max_words: int = 500, timeout: float = 5.0) -> List[str]:
    """Split a markdown text into chunks based on headers and content, 
    ensuring that each chunk does not exceed the specified maximum word count."""
    header_pattern = re.compile(r'^(#+)\s+(.+)$', re.MULTILINE)
    start_time = time.time()

    def split_with_timeout():
        sections = header_pattern.split(text)
    
        chunks = []
        current_chunk = sections[0]
        current_header_level = 0
        
        for i in range(1, len(sections), 3):
            header_level = len(sections[i])
            header = sections[i+1]
            content = sections[i+2] if i+2 < len(sections) else ""
            
            section = f"{'#' * header_level} {header}\n\n{content.strip()}"
            
            if (word_count(current_chunk + '\n\n' + section) <= max_words and 
                header_level >= current_header_level):
                current_chunk += '\n\n' + section
                current_header_level = max(current_header_level, header_level)
            else:
                if current_chunk:
                    if word_count(current_chunk) > max_words:
                        chunks.extend(split_chunk_by_paragraph(current_chunk, max_words))
                    else:
                        chunks.append(current_chunk.strip())
                current_chunk = section
                current_header_level = header_level
        
        if current_chunk:
            if word_count(current_chunk) > max_words:
                chunks.extend(split_chunk_by_paragraph(current_chunk, max_words))
            else:
                chunks.append(current_chunk.strip())

        combined_chunks = []
        temp_chunk = ""
        temp_header_level = 0
        
        for chunk in chunks:
            chunk_header_match = re.match(r'^(#+)', chunk)
            chunk_header_level = len(chunk_header_match.group(1)) if chunk_header_match else 0
            
            if (word_count(temp_chunk + '\n\n' + chunk) <= max_words and 
                (chunk_header_level == 0 or chunk_header_level >= temp_header_level)):
                temp_chunk += '\n\n' + chunk
                temp_header_level = max(temp_header_level, chunk_header_level)
            else:
                if temp_chunk:
                    if word_count(temp_chunk) > max_words:
                        combined_chunks.extend(split_chunk_by_paragraph(temp_chunk, max_words))
                    else:
                        combined_chunks.append(temp_chunk.strip())
                temp_chunk = chunk
                temp_header_level = chunk_header_level
        
        if temp_chunk:
            if word_count(temp_chunk) > max_words:
                combined_chunks.extend(split_chunk_by_paragraph(temp_chunk, max_words))
            else:
                combined_chunks.append(temp_chunk.strip())
        
        final_chunks = [remove_numbers_in_brackets(chunk) for chunk in combined_chunks if chunk.strip()]
        
        if time.time() - start_time > timeout:
            raise TimeoutError("Markdown splitting timed out")

        return final_chunks
    
    try:
        return split_with_timeout()
    except TimeoutError:
        logger.warning("Markdown splitting timed out")
        return [text[:max_words]]