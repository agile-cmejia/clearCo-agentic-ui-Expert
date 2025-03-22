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
import random
from openai import AsyncOpenAI, RateLimitError

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from supabase import create_client, Client

load_dotenv()

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
ai_model=get_required_env_var("LLM_MODEL", "gpt-4o-mini"),

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

class RateLimiter:
    """
    Implements a token bucket algorithm for rate limiting.
    """
    def __init__(self, rate: float, capacity: float):
        self.tokens = capacity
        self.capacity = capacity
        self.rate = rate
        self.last_updated = self._current_time()
        self.lock = asyncio.Lock()

    def _current_time(self):
        return time.monotonic()

    async def refill(self):
        """
        Adds tokens to the bucket based on the elapsed time.
        """
        now = self._current_time()
        elapsed = now - self.last_updated
        self.last_updated = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

    async def acquire(self, tokens: float = 1):
        """
        Acquires tokens from the bucket, waiting if necessary.
        """
        async with self.lock:
            while True:
                await self.refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                await asyncio.sleep(0.1)  # Yield to the event loop

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
            #  Acquire tokens before making the request
            await gpt_4o_mini_limiter.acquire()  # Assume 1 request = 1 token for simplicity
            response = await openai_client.chat.completions.create(
                model=ai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
                ],
                response_format={"type": "json_object"}
            )
            # Parse the JSON content from the response
            try:
                json_output = json.loads(response.choices[0].message.content)
                return json_output
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Error parsing JSON response: {e}, Raw content: {response.choices[0].message.content}")
                return {"title": "Error processing title", "summary": "Error processing summary"}

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
    try:
        response = requests.get(site_url)
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

async def is_chunk_processed(url: str, chunk_number: int) -> bool:
    """
    Checks if a chunk has already been processed and exists in the database.
    """
    try:
        response = supabase.table("site_pages").select("*").eq("url", url).eq(
            "chunk_number", chunk_number
        ).execute()
        return len(response.data) > 0
    except Exception as e:
        print(f"Error checking if chunk is processed: {e}")
        return False

async def process_queue():
    """Worker function to process chunks from the queue."""
    while True:
        item = await chunk_queue.get()
        url = item["url"]
        chunk = item["chunk"]
        chunk_number = item["chunk_number"]

        try:
            #  Check if the chunk is already processed
            if not await is_chunk_processed(url, chunk_number):
                processed_chunk = await process_chunk(chunk, chunk_number, url)
                if processed_chunk:
                    await insert_chunk(processed_chunk)
            else:
                print(f"Chunk {chunk_number} for {url} already processed. Skipping.")
        except Exception as e:
            print(f"Error processing queue item: {e}")
        finally:
            chunk_queue.task_done()

async def main():

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
