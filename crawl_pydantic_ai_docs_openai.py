import os
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
import time
import random  # Import random for jitter
from openai import AsyncOpenAI, RateLimitError

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from supabase import create_client, Client

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def get_required_env_var(name, default=None):
    """Get an environment variable or exit if not available and no default provided.

    Args:
        name: The name of the environment variable
        default: Optional default value to use if the variable is not set

    Returns:
        The value of the environment variable, or the default if provided

    Raises:
        SystemExit: If the variable is not set and no default is provided
    """
    print(f"Searching for {name} in the environment variables")
    value = os.getenv(name)
    print(f"Found value: {value} for {name}")
    if value is None and default is None:
        print(f"Error: Required environment variable {name} is not set", file=sys.stderr)
        sys.exit(1)
    return value


def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str, max_retries=10) -> Dict[str, str]:
    """Extract title and summary using GPT-4 with robust retry logic."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""

    retries = 0
    wait_time = 5  # Increased initial wait time
    backoff_multiplier = 2  # Keep the exponential backoff
    max_wait = 60  # Maximum wait time to prevent excessive delays

    while retries < max_retries:
        try:
            response = await openai_client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)

        except RateLimitError as e:
            retries += 1
            if retries == max_retries:
                print("Max retries reached for title/summary. Giving up.")
                return {"title": "Error processing title", "summary": "Error processing summary"}

            # Exponential backoff with jitter
            sleep_duration = min(wait_time, max_wait) + random.uniform(0, wait_time * 0.5)  # Jitter
            print(f"RateLimitError: {e}. Retrying in {sleep_duration:.2f} seconds...")
            time.sleep(sleep_duration)
            wait_time *= backoff_multiplier  # Exponential backoff

        except Exception as e:
            # Handle other exceptions (optional, but good practice)
            print(f"Error getting title and summary: {e}")
            return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)

    # Get embedding
    embedding = await get_embedding(chunk)

    # Create metadata
    source = os.getenv("TOPIC", "testing_ai_docs")
    metadata = {
        "source": source,
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }

    time.sleep(20)  # Consistent delay after processing chunk

    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }

        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

# # Old version
# async def process_and_store_document(url: str, markdown: str):
#     """Process a document and store its chunks in parallel."""
#     # Split into chunks
#     chunks = chunk_text(markdown)

#     # Process chunks in parallel
#     tasks = [
#         process_chunk(chunk, i, url)
#         for i, chunk in enumerate(chunks)
#     ]
#     processed_chunks = await asyncio.gather(*tasks)

#     # Store chunks in parallel
#     insert_tasks = [
#         insert_chunk(chunk)
#         for chunk in processed_chunks
#     ]
#     await asyncio.gather(*insert_tasks)

async def crawl_parallel(urls: List[str], max_concurrent: int = 3):  # Further reduced concurrency
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    #  await process_and_store_document(url, result.markdown_v2.raw_markdown)
                    await add_to_queue(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")

        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_pydantic_ai_docs_urls() -> List[str]:
    """Get URLs from Pydantic AI docs sitemap."""
    sitemap_url = os.getenv("SITE_URL", "https://ai.pydantic.dev/sitemap.xml")
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()

        # Parse the XML
        root = ElementTree.fromstring(response.content)

        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]

        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return

# New Queue-based Processing
chunk_queue: asyncio.Queue[Dict] = asyncio.Queue()

async def add_to_queue(url: str, markdown: str):
    """Adds chunks from a document to the processing queue."""
    chunks = chunk_text(markdown)
    for i, chunk in enumerate(chunks):
        await chunk_queue.put({"url": url, "chunk": chunk, "chunk_number": i})

async def process_queue():
    """Worker function to process chunks from the queue."""
    while True:
        item = await chunk_queue.get()
        url = item["url"]
        chunk = item["chunk"]
        chunk_number = item["chunk_number"]

        try:
            processed_chunk = await process_chunk(chunk, chunk_number, url)
            if processed_chunk:
                await insert_chunk(processed_chunk)
        except Exception as e:
            print(f"Error processing queue item: {e}")
        finally:
            chunk_queue.task_done()

async def main():
    print(f"Topic to use: {get_required_env_var('TOPIC')} from function")
    print(f"Topic to use: {os.getenv('TOPIC')} from env")
    print(f"Site URL to crawl {get_required_env_var('SITE_URL')} from function")
    print(f"Site URL to crawl {os.getenv('SITE_URL')} from env")
    # Get URLs from Pydantic AI docs
    urls = get_pydantic_ai_docs_urls()
    if not urls:
        print("No URLs found to crawl")
        return

    print(f"Found {len(urls)} URLs to crawl")
    #  await crawl_parallel(urls)

    # Crawl and add to queue
    await crawl_parallel(urls)

    # Start the queue worker
    queue_worker = asyncio.create_task(process_queue())

    # Wait for all chunks to be processed
    await chunk_queue.join()

    # Cancel the worker (it should be done by now)
    queue_worker.cancel()
    try:
        await queue_worker
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(main())
