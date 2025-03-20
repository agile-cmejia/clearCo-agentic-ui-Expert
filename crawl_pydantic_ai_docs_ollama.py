import os
import json
import asyncio
import requests
import re
import sys
from xml.etree import ElementTree
from typing import List, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from supabase import create_client, Client

load_dotenv()

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


# Initialize Supabase clients
suppabase_url = get_required_env_var("SUPABASE_URL")
suppabase_service_key = get_required_env_var("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(
    str(suppabase_url),
    str(suppabase_service_key)
)
site_url = get_required_env_var("SITE_URL" , "https://storybook.js.org/sitemap.xml")
#site_url = get_required_env_var("SITE_URL" , "https://storybook.js.org/sitemap.xml")
topic = "storybook_js_docs"
#topic = get_required_env_var("TOPIC" , "testing_ai_docs")

OLLAMA_MODEL = get_required_env_var("OLLAMA_MODEL", "llama2:chat")  # Default model

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]  # Now using numerical embeddings

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

def get_ollama_response(prompt: str) -> str:
    """
    Sends a prompt to the local Ollama model and returns the response.
    """
    try:
        response = requests.post(str(get_required_env_var("OLLMA_URL", "http://localhost:11434/api/generate")),
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        return ""  # Or handle the error as appropriate for your application
    except json.JSONDecodeError as e:
        print(f"Error decoding Ollama response: {e}")
        return ""

def get_title_and_summary_ollama(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using a local Ollama model."""
    system_prompt_title_summary = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys in JSON format.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""

    prompt = f"""{system_prompt_title_summary}\nURL: {url}\n\nContent:\n{chunk[:1000]}...\n\nPlease provide the result in JSON format."""

    try:
        response = get_ollama_response(prompt)
        # Extract JSON from response (Ollama might include extra text)
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            print("Could not extract JSON from Ollama response.")
            return {"title": "Error processing title", "summary": "Error processing summary"}

    except Exception as e:
        print(f"Error getting title and summary from Ollama: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

def get_embedding_ollama(text: str) -> List[float]:
    """
    Get embedding vector from a local Ollama model.

    This implementation uses Ollama to generate a text representation
    and then converts that representation into a numerical embedding.
    """
    system_prompt_embedding = """
    You are an AI that generates concise text representations of documentation chunks.
    Your goal is to capture the core meaning and key information of the chunk in a short text form.
    Focus on preserving the essential information that would help someone understand the chunk's content.
    Do not mention that you are an AI or this instruction.
    """
    prompt = f"{system_prompt_embedding}\n\nContent:\n{text}"
    # Pad or truncate the embedding to a consistent size
    embedding_size = int(get_required_env_var('EMBEDDING_SIZE', 768))  # Or choose a size appropriate for your needs
    try:
        response_text = get_ollama_response(prompt)
        # Basic approach: Split the text into "words" and assign a simple numerical value
        # This is a placeholder; you'll want a better method
        words = response_text.split()
        embedding = [len(word) for word in words]  # Example: length of each word
        if len(embedding) < embedding_size:
            embedding.extend([0] * (embedding_size - len(embedding)))
        else:
            embedding = embedding[:embedding_size]
        return embedding
    except Exception as e:
        print(f"Error getting embedding from Ollama: {e}")
        return [0] * embedding_size  # Placeholder embedding

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = get_title_and_summary_ollama(chunk, url)  # Use Ollama

    # Get embedding
    embedding = get_embedding_ollama(chunk)  # Use Ollama

    # Create metadata
    source = get_required_env_var("TOPIC", "testing_ai_docs")
    #source = os.getenv("TOPIC", "testing_ai_docs")
    #source = "storybook_js_docs"
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

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)

    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url)
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)

    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk)
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
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
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_url(url: str):
        try:
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        except asyncio.CancelledError:
            print(f"Cancelled processing of {url}")
            raise
        except Exception as e:
            print(f"Error processing {url}: {e}")

    # Create a task for each URL
    tasks = [process_url(url) for url in urls]
    try:
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        # Cancel all pending tasks
        for task in tasks:
            task.cancel()
        # Wait for all tasks to complete, ignoring cancellation errors
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_docs_urls() -> List[str]:
    """Get URLs docs from sitemap.xml"""
    # sitemap_url = str(get_required_env_var("SITE_URL", "https://storybook.js.org/sitemap.xml"))
    # sitemap_url = os.getenv("SITE_URL", "https://storybook.js.org/sitemap.xml")
    # sitemap_url = "https://storybook.js.org/sitemap.xml"
    try:
        response = requests.get(site_url)
        response.raise_for_status()

        # Parse the XML
        root = ElementTree.fromstring(response.content)

        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        print(urls)
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return

async def main():
    try:
        # Get URLs from Pydantic AI docs
        urls = get_docs_urls()
        if not urls:
            print("No URLs found to crawl")
            return

        print(f"Found {len(urls)} URLs to crawl")

        # Create a task group for better task management
        async with asyncio.TaskGroup() as tg:
            task = tg.create_task(crawl_parallel(urls, int(os.getenv('MAX_CONCURRENT_TASKS',3))))

            try:
                await task
            except asyncio.CancelledError:
                print("\nReceived shutdown signal, cancelling tasks...")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                print("Tasks cancelled successfully")

    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        print("Cleanup complete")

if __name__ == "__main__":
    print(f"Topic to use: {get_required_env_var("TOPIC")} from function")
    print(f"Topic to use: {os.getenv("TOPIC")} from env")
    print(f"Site URL to crawl {get_required_env_var("SITE_URL")} from function")
    print(f"Site URL to crawl {os.getenv("SITE_URL")} from env")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated")
