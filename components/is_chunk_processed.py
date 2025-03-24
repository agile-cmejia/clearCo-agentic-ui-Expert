from supabase import Client

async def is_chunk_processed(url: str, chunk_number: int, supabase: Client) -> bool:
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
