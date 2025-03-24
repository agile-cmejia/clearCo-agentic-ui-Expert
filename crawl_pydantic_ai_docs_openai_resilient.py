import os
import json
import asyncio
import sys
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from classes.processed_chunk import ProcessedChunk
from classes.rate_limiter import RateLimiter
from components.get_docs_url import get_docs_urls
from components.insert_chunk import insert_chunk
from components.is_chunk_processed import is_chunk_processed
from components.get_required_env_vars import get_required_env_var
from crawl_pydantic_ai_docs import chunk_text

import time
import random
from openai import AsyncOpenAI, RateLimitError

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from supabase import create_client, Client

# Initialize OpenAI, Supabase clients and other variables
openapi_key = str(get_required_env_var("OPENAI_API_KEY"))
openai_client = AsyncOpenAI(api_key=openapi_key)
supabase_key = str(get_required_env_var("SUPABASE_SERVICE_KEY"))
supabase_url = str(get_required_env_var("SUPABASE_URL"))
supabase: Client = create_client(
    supabase_url,
    supabase_key
)
topic = get_required_env_var("TOPIC", "mantine_ui_docs")
site_url = get_required_env_var("SITE_URL", "https://storybook.js.org/sitemap.xml")
ai_model=get_required_env_var("LLM_MODEL", "gpt-4o-mini")

# Initialize RateLimiters based on Tier 1 limits
# You'll need to adjust these based on the *exact* Tier 1 limits for your models!
gpt_4o_mini_rpm = 60  # Example: Requests per minute for gpt-4o-mini (adjust!)
gpt_4o_mini_tpm = 10000  # Example: Tokens per minute for gpt-4o-mini (adjust!)
embedding_rpm = 100  # Example: Requests per minute for embeddings (adjust!)
embedding_tpm = 100000  # Example: Tokens per minute for embeddings (adjust!)

gpt_4o_mini_limiter = RateLimiter(rate=gpt_4o_mini_rpm / 60, capacity=gpt_4o_mini_rpm)
embedding_limiter = RateLimiter(rate=embedding_rpm / 60, capacity=embedding_rpm)

async def get_title_and_summary(chunk: str, url: str, max_retries=10) -> Dict[str, str]:
    """Extract title and summary using GPT-4 with robust retry logic."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""

    retries = 0
    wait_time = 5
    backoff_multiplier = 2
    max_wait = 60

    while retries < max_retries:
        try:
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
            ]
            #print(messages)
            #  Acquire tokens before making the request
            await gpt_4o_mini_limiter.acquire()  # Assume 1 request = 1 token for simplicity
            response = await openai_client.chat.completions.create(
                model=ai_model,  # Now a string, not a tuple
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
            sleep_duration = min(wait_time, max_wait) + random.uniform(0, wait_time * 0.5)
            print(f"RateLimitError: {e}. Retrying in {sleep_duration:.2f} seconds...")
            time.sleep(sleep_duration)
            wait_time *= backoff_multiplier

        except Exception as e:
            print(f"Error getting title and summary: {e}")
            return {"title": "Error processing title", "summary": "Error processing summary"}
            sys.exit(1)

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        #  Acquire tokens before making the request
        await embedding_limiter.acquire()  # Assume 1 request = 1 token for simplicity
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)

    # Get embedding
    embedding = await get_embedding(chunk)

    # Create metadata
    metadata = {
        "source": topic,
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }

    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )


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
            #  Check if the chunk is already processed
            if not await is_chunk_processed(url, chunk_number,supabase):
                processed_chunk = await process_chunk(chunk, chunk_number, url)
                if processed_chunk:
                    await insert_chunk(processed_chunk, supabase)
            else:
                print(f"Chunk {chunk_number} for {url} already processed. Skipping.")
        except Exception as e:
            print(f"Error processing queue item: {e}")
        finally:
            chunk_queue.task_done()

async def main():

    # Get URLs from Pydantic AI docs
    urls = get_docs_urls(site_url)
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
